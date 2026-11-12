"""Cross-encoder rerank: query-side wiring, soft failure, and the service's
GPU-lease / idle behaviour. No model and no GPU: the scorer is a fake."""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mcp_server  # noqa: E402
import reranker as RR  # noqa: E402
import reranker_service as SVC  # noqa: E402
from test_v3_search import db, parent_rows, search, v2_row, vec  # noqa: E402,F401


def _corpus(db, k=12):
    # c1 is the closest by cosine, c{k} the farthest; one parent each.
    rows = [v2_row(i, f"B{i % 4}", f"B{i % 4}/c{i}.md", f"[B]\npassage {i}", vec(0, 1, i / 4),
                   parent=f"p{i}") for i in range(1, k + 1)]
    db.create_table("textbook_chunks", data=rows)
    db.create_table("textbook_parents", data=parent_rows([f"p{i}" for i in range(1, k + 1)]))


@pytest.fixture
def fake_rr(monkeypatch):
    calls = []

    def score(query, docs, max_length=512, timeout=60.0):
        calls.append((query, list(docs)))
        # Reverse the cosine order: the last doc of the pool scores highest.
        return [float(i) for i in range(len(docs))], {"start": "warm", "ms": 1.0}

    monkeypatch.setattr(RR, "enabled", lambda: True)
    monkeypatch.setattr(RR, "score", score)
    monkeypatch.setattr(mcp_server, "TEXTBOOK_RERANK", True)
    return calls


def test_rerank_reorders_pool_and_keeps_n(db, fake_rr):
    _corpus(db)
    plain = search(n_results=3, retrieval="vector", rerank=False)
    res = search(n_results=3, retrieval="vector")
    assert len(fake_rr) == 1
    query, docs = fake_rr[0]
    assert query == "q" and len(docs) == 12  # whole pool, not just n
    assert len(res) == 3
    assert [r["rerank_score"] for r in res] == sorted((r["rerank_score"] for r in res), reverse=True)
    assert res[0]["file"] != plain[0]["file"]  # the fake reversed the order
    assert all("rerank_score" not in r and "rank_before_rerank" not in r for r in plain)
    assert res[0]["rank_before_rerank"] == 12  # the fake put the pool's last item first
    # similarity stays the cosine, not the cross-encoder score
    assert all(0.0 <= r["similarity"] <= 1.0 for r in res)


def test_rerank_doc_is_the_parent_text(db, fake_rr):
    _corpus(db, k=2)
    search(n_results=2, retrieval="vector")
    assert fake_rr[0][1] == ["parent text", "parent text"]


def test_rerank_failure_falls_back(db, monkeypatch):
    _corpus(db)
    monkeypatch.setattr(RR, "enabled", lambda: True)
    monkeypatch.setattr(mcp_server, "TEXTBOOK_RERANK", True)

    def boom(*a, **k):
        raise RR.RerankUnavailable("gpu claimed by another job")

    monkeypatch.setattr(RR, "score", boom)
    logged = []
    monkeypatch.setattr(mcp_server, "_log", lambda logger, entry, ctx: logged.append(entry))
    res = search(n_results=3, retrieval="vector")
    plain = search(n_results=3, retrieval="vector", rerank=False)
    assert [r["file"] for r in res] == [r["file"] for r in plain]
    assert logged[0]["rerank"] == {"applied": False, "reason": "gpu claimed by another job"}
    assert logged[1]["rerank"] is None


def test_rerank_off_by_config(db, fake_rr, monkeypatch):
    _corpus(db)
    monkeypatch.setattr(mcp_server, "TEXTBOOK_RERANK", False)
    search(n_results=3, retrieval="vector")
    assert fake_rr == []
    search(n_results=3, retrieval="vector", rerank=True)  # per-call override
    assert len(fake_rr) == 1


def test_lecture_is_never_reranked(fake_rr, monkeypatch):
    seen = {}

    def tables(corpus):
        seen["corpus"] = corpus
        return None, None

    monkeypatch.setattr(mcp_server, "get_corpus_tables", tables)
    with pytest.raises(mcp_server.TextbookIndexMissing):
        mcp_server._textbook_search_v2_raw({"query": "q"}, corpus="lecture")
    assert fake_rr == []


def test_default_retrieval_is_hybrid():
    import config
    assert config.TEXTBOOK_RETRIEVAL == "hybrid" or "VAULT_SEARCH_TEXTBOOK_RETRIEVAL" in __import__("os").environ


# --- GPU lease gating --------------------------------------------------------

def test_gpu_claimed_flags(tmp_path, monkeypatch):
    assert SVC.gpu_claimed(None) is False
    assert SVC.gpu_claimed(tmp_path) is False
    (tmp_path / "queue").mkdir()
    assert SVC.gpu_claimed(tmp_path) is False
    (tmp_path / "queue" / "ticket").write_text("x")
    assert SVC.gpu_claimed(tmp_path) is True
    (tmp_path / "queue" / "ticket").unlink()
    (tmp_path / "PAUSE").write_text("x")
    assert SVC.gpu_claimed(tmp_path) is True
    monkeypatch.setattr(RR, "GPU_LEASE_STATE_DIR", tmp_path)
    assert RR.gpu_claimed() is True


def test_client_refuses_when_gpu_claimed(tmp_path, monkeypatch):
    (tmp_path / "PAUSE").write_text("x")
    monkeypatch.setattr(RR, "GPU_LEASE_STATE_DIR", tmp_path)
    monkeypatch.setattr(RR, "enabled", lambda: True)
    spawned = []
    monkeypatch.setattr(RR, "_spawn", lambda: spawned.append(1))
    with pytest.raises(RR.RerankUnavailable, match="claimed"):
        RR.score("q", ["d"])
    assert spawned == []  # never starts a model while another job wants the card


# --- the service with a fake scorer -----------------------------------------

class FakeScorer:
    name = "fake"

    def score(self, query, docs, max_length=512):
        return [float(len(d)) for d in docs]


def _serve(tmp_path, idle=60.0):
    server = HTTPServer(("127.0.0.1", 0), None)
    st = SVC.State(FakeScorer(), idle, tmp_path)
    server.RequestHandlerClass = SVC.make_handler(st)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server, st, f"http://127.0.0.1:{server.server_address[1]}"


def _post(base, obj):
    req = urllib.request.Request(base + "/rerank", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_service_scores_and_yields(tmp_path):
    server, st, base = _serve(tmp_path)
    try:
        out = _post(base, {"query": "q", "docs": ["a", "abc"]})
        assert out["scores"] == [1.0, 3.0]
        (tmp_path / "PAUSE").write_text("x")
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, {"query": "q", "docs": ["a"]})
        assert e.value.code == 503 and st.stopping
    finally:
        server.shutdown()
        server.server_close()


def test_watcher_exits_on_idle_and_on_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(SVC, "WATCH_INTERVAL_S", 0.05)
    for trigger in ("idle", "claim"):
        server, st, _ = _serve(tmp_path, idle=0.1 if trigger == "idle" else 60.0)
        if trigger == "claim":
            (tmp_path / "PAUSE").write_text("x")
        th = threading.Thread(target=SVC.watcher, args=(st, server), daemon=True)
        th.start()
        th.join(timeout=5)
        assert not th.is_alive() and st.stopping, trigger
        server.server_close()
        (tmp_path / "PAUSE").unlink(missing_ok=True)


def test_source_prior_lifts_guideline_and_newer(monkeypatch):
    monkeypatch.setattr(mcp_server, "RERANK_GUIDELINE_PRIOR", 3.0)
    monkeypatch.setattr(mcp_server, "RERANK_YEAR_PRIOR", 0.5)
    sp = mcp_server._source_prior
    assert sp({"meta": {"corpus_type": "guideline", "year": 2026}}) == 3.5
    assert sp({"meta": {"corpus_type": "textbook", "year": 2010}}) == 0.0
    assert sp({"meta": {"corpus_type": "textbook", "year": 2020}}) == 0.25
    assert sp({"meta": {}}) == 0.0
    monkeypatch.setattr(mcp_server, "RERANK_GUIDELINE_PRIOR", 0.0)
    monkeypatch.setattr(mcp_server, "RERANK_YEAR_PRIOR", 0.0)
    assert sp({"meta": {"corpus_type": "guideline", "year": 2026}}) == 0.0


def test_prior_reorders_close_scores(monkeypatch):
    monkeypatch.setattr(mcp_server, "RERANK_GUIDELINE_PRIOR", 3.0)
    monkeypatch.setattr(mcp_server, "RERANK_YEAR_PRIOR", 0.0)
    monkeypatch.setattr(mcp_server, "fetch_parents_by_id", lambda *a, **k: {})
    monkeypatch.setattr(RR, "score", lambda q, docs, **k: ([2.0, 0.5, -5.0], {"start": "warm", "ms": 1}))
    pool = [{"parent_id": "a", "text": "a", "meta": {"corpus_type": "textbook"}},
            {"parent_id": "b", "text": "b", "meta": {"corpus_type": "guideline"}},
            {"parent_id": "c", "text": "c", "meta": {"corpus_type": "guideline"}}]
    out, info = mcp_server._rerank_pool(pool, None, "q")
    assert [r["parent_id"] for r in out] == ["b", "a", "c"]  # prior beats a 1.5 gap, not a 7 gap
    assert out[0]["rerank_score"] == 0.5 and out[0]["rerank_prior"] == 3.0
    assert "rerank_prior" not in out[1]
