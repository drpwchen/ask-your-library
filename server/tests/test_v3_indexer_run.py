"""textbook_indexer_v3 end to end on a temp corpus, with a fake embedder.

Covers the I3 rules on a real LanceDB: --book is never skipped by the hash
cache, a file that became empty loses its rows, removed files are cleaned up,
re-runs resume from the hash cache, and the coverage audit is per chapter.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

lancedb = pytest.importorskip("lancedb")
import chunker_v3 as C  # noqa: E402
import textbook_indexer_v3 as V  # noqa: E402

DIM = 8


class FakeEmbedder:
    calls = 0

    def __init__(self, model, host):
        self.model = model
        self.failed = 0

    def probe(self):
        return DIM, "fakedigest"

    def embed(self, texts):
        FakeEmbedder.calls += len(texts)
        out = []
        for t in texts:
            v = [0.0] * DIM
            v[hash(t) % DIM] = 1.0
            out.append(v)
        return out

    def unload(self):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "Embedder", FakeEmbedder)
    monkeypatch.setattr(C, "qwen_tok", C.whitespace_tok)
    monkeypatch.setattr(V, "truncate_tokens", lambda t, n: t)
    corpus = tmp_path / "corpus"
    words = " ".join(f"w{i}" for i in range(120))
    for book in ("Book_7e_2020", "Old_1e_2000"):
        d = corpus / book
        d.mkdir(parents=True)
        (d / "ch01_A.md").write_text(f"<!-- page 1 -->\n## Intro\n{words}\n", encoding="utf-8")
        (d / "ch02_B.md").write_text(f"<!-- page 5 -->\n## Next\n{words}\n"
                                     "**[Table on page 5]**\n| a | b |\n| --- | --- |\n| 1 | 2 |\n",
                                     encoding="utf-8")
    g = corpus / "_Guidelines" / "G_2022"
    g.mkdir(parents=True)
    (g / "full_text.md").write_text(f"<!-- page 1 -->\n{words}\n", encoding="utf-8")
    (corpus / "Book_7e_2020" / "_manifest.json").write_text(json.dumps({
        "schema": 1, "split_method": "bookmarks", "page_labels": {"1": "xi", "5": "3"},
        "files": [{"file": "ch02_B.md", "chapter_number": 2, "title": "Chapter 2 B"}]}))
    meta = tmp_path / "books.json"
    meta.write_text(json.dumps({"Old_1e_2000": {"status": "superseded", "superseded_by": "Book_7e_2020"}}))
    db = tmp_path / "lance_db_v3"
    base = ["--corpus", str(corpus), "--db", str(db), "--books-meta", str(meta), "--no-gpu-lease"]
    return corpus, db, base


def rows(db, cols=("file",)):
    t = lancedb.connect(str(db)).open_table("textbook_chunks")
    return t.search().select(list(cols)).limit(10 ** 6).to_pandas()


def test_build_resume_book_force_empty_orphan_audit(env):
    corpus, db, base = env
    assert V.main(base) == 0
    df = rows(db, ("file", "source", "chapter_number", "printed_page_start", "page_confidence",
                   "chunk_kind", "pdf_page_start", "status"))
    assert set(df["source"]) == {"Book_7e_2020", "G_2022"}          # superseded book not indexed
    b = df[df["file"] == "Book_7e_2020/ch02_B.md"]
    assert set(b["chapter_number"]) == {2} and "table" in set(b["chunk_kind"])
    assert set(b["printed_page_start"]) == {"3"} and set(b["page_confidence"]) == {"printed"}
    t = lancedb.connect(str(db)).open_table("textbook_chunks")
    assert any("text" in str(ix) for ix in t.list_indices())          # FTS index built
    meta = json.loads((db / "_v3_state" / "index_meta.json").read_text(encoding="utf-8"))
    assert meta["model_digest"] == "fakedigest" and meta["normalization"] == "l2"

    # resume: nothing changed → nothing embedded
    FakeEmbedder.calls = 0
    assert V.main(base) == 0
    assert FakeEmbedder.calls == 0

    # --book: re-embedded even though hashes match (I3)
    assert V.main(base + ["--book", "Book_7e_2020"]) == 0
    assert FakeEmbedder.calls > 0
    assert set(rows(db)["file"]) >= {"Book_7e_2020/ch01_A.md", "Book_7e_2020/ch02_B.md"}

    # emptied file loses its rows; deleted file is cleaned up
    (corpus / "Book_7e_2020" / "ch01_A.md").write_text("<!-- page 1 -->\n", encoding="utf-8")
    (corpus / "_Guidelines" / "G_2022" / "full_text.md").unlink()
    assert V.main(base) == 0
    files = set(rows(db)["file"])
    assert "Book_7e_2020/ch01_A.md" not in files
    assert not any(f.startswith("_Guidelines/G_2022") for f in files)

    # per-chapter audit passes, then catches a chapter with no rows
    assert V.main(base + ["--audit"]) == 0
    (corpus / "Book_7e_2020" / "ch03_C.md").write_text("<!-- page 9 -->\n" + "text " * 100)
    assert V.main(base + ["--audit"]) == 1


def test_signature_change_requires_rebuild(env, monkeypatch):
    corpus, db, base = env
    assert V.main(base) == 0
    monkeypatch.setattr(FakeEmbedder, "probe", lambda self: (DIM, "otherdigest"))
    assert V.main(base) == 2
    assert V.main(base + ["--rebuild"]) == 0


def test_live_db_refused(env, monkeypatch, tmp_path):
    corpus, db, base = env
    live = tmp_path / "live"
    monkeypatch.setattr(V, "live_paths", lambda: [live])
    with pytest.raises(SystemExit):
        V.main(["--corpus", str(corpus), "--db", str(live), "--no-gpu-lease"])
    assert not live.exists()
