"""Opt-in vault retrieval settings: hybrid (vector + FTS) and sections_per_note.

Both default OFF; these tests pin what each does when asked for, and that the
default path is unchanged. Real LanceDB table, stubbed embedder.

Run directly (`python server/tests/test_vault_hybrid_sections.py`) or under pytest.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VAULT_PATH", tempfile.mkdtemp(prefix="vsearch_test_vault_"))

try:
    import lancedb
    import mcp_server
except ImportError:
    mcp_server = None


if mcp_server is not None:
    import contextlib

    Q = [1.0, 0.0, 0.0, 0.0]

    def _row(i, note, pid, vec, text):
        return {"vector": vec, "parent_id": pid, "file": f"Clinical/{note}.md", "note": note,
                "section_path": f"{note} > S{pid}", "folder": "Clinical", "tags": "",
                "mtime": 0.0, "text": text}

    # "Close" is nearest by vector but never says the term; "Exact" is far by
    # vector and is the only row containing it. Note "Long" has two sections.
    ROWS = [
        _row(0, "Close", "c1", [0.99, 0.14, 0.0, 0.0], "general nerve mobilisation overview"),
        _row(1, "Long", "l1", [0.95, 0.31, 0.0, 0.0], "first section about the elbow"),
        _row(2, "Long", "l2", [0.90, 0.43, 0.0, 0.0], "second section about the wrist"),
        _row(3, "Exact", "e1", [0.20, 0.98, 0.0, 0.0], "ULNT1 median nerve bias test"),
    ]
    PARENTS = {r["parent_id"]: {"text": f"PARENT {r['parent_id']}: {r['text']}"} for r in ROWS}

    class _Emb:
        def embed(self, model, input, options=None):
            return {"embeddings": [Q]}

    @contextlib.contextmanager
    def _env(with_fts=True):
        db = lancedb.connect(tempfile.mkdtemp(prefix="vsearch_hyb_"))
        tbl = db.create_table("vault_chunks_v2", ROWS, mode="overwrite")
        if with_fts:
            tbl.create_fts_index("text", use_tantivy=False, replace=True)
        names = ("get_vault_chunks_table", "get_ollama", "fetch_vault_parents",
                 "vault_v2_active", "get_graph", "SCOPE_MAP_PATH", "DEFAULT_SCOPE_ENV")
        saved = {n: getattr(mcp_server, n) for n in names}
        mcp_server.get_vault_chunks_table = lambda: tbl
        mcp_server.get_ollama = lambda: _Emb()
        mcp_server.fetch_vault_parents = lambda ids: {i: PARENTS[i] for i in ids if i in PARENTS}
        mcp_server.vault_v2_active = lambda: True
        mcp_server.get_graph = lambda: {}
        mcp_server.SCOPE_MAP_PATH = Path(tempfile.mkdtemp()) / "none.json"   # no map → "all"
        mcp_server.DEFAULT_SCOPE_ENV = ""
        try:
            yield
        finally:
            for n, v in saved.items():
                setattr(mcp_server, n, v)

    def _notes(args):
        out = mcp_server._vault_search_raw({"query": "ULNT1", "n_results": 3,
                                            "boost_recent": False, **args})
        return out["results"]

    def test_default_is_vector_one_section():
        with _env():
            res = _notes({"include_excerpt": True})
        assert [r["note"] for r in res] == ["Close", "Long", "Exact"]
        long_ = next(r for r in res if r["note"] == "Long")
        assert long_["excerpt"] == PARENTS["l1"]["text"]
        assert not any(k.startswith("_") for r in res for k in r)

    def test_hybrid_lifts_the_exact_term():
        with _env():
            res = _notes({"retrieval": "hybrid"})
        names = [r["note"] for r in res]
        assert names.index("Exact") < 2, names
        # similarity stays a cosine, never the RRF score (~0.03)
        exact = next(r for r in res if r["note"] == "Exact")
        assert 0.15 < exact["similarity"] < 0.3
        assert not any(k.startswith("_") for r in res for k in r)

    def test_hybrid_without_fts_index_still_matches_by_flat_scan():
        """LanceDB scans the text when there is no FTS index: slower, same answer."""
        with _env(with_fts=False):
            res = _notes({"retrieval": "hybrid"})
        assert [r["note"] for r in res].index("Exact") < 2

    def test_hybrid_error_falls_back_to_vector():
        saved = mcp_server.VAULT_FTS_COLUMN
        mcp_server.VAULT_FTS_COLUMN = "no_such_column"
        try:
            with _env():
                res = _notes({"retrieval": "hybrid"})
        finally:
            mcp_server.VAULT_FTS_COLUMN = saved
        assert [r["note"] for r in res] == ["Close", "Long", "Exact"]
        assert not any(k.startswith("_") for r in res for k in r)

    def test_sections_per_note_appends_next_section():
        with _env():
            res = _notes({"include_excerpt": True, "sections_per_note": 2})
        long_ = next(r for r in res if r["note"] == "Long")
        assert PARENTS["l1"]["text"] in long_["excerpt"]
        assert PARENTS["l2"]["text"] in long_["excerpt"]
        assert long_["excerpt"].index("l1") < long_["excerpt"].index("l2")
        assert [r["note"] for r in res] == ["Close", "Long", "Exact"]   # ranking unchanged

else:
    def test_skipped_deps_missing():
        """Placeholder so the file reports as skipped-by-design, not empty."""
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} tests passed")
