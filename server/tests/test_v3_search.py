"""textbook_search over a v3 index and over a v2 index (query side, I6/I7/I8).

Builds tiny LanceDB tables in the test data dir (conftest points
VAULT_SEARCH_DATA_DIR at a temp folder) and replaces the Ollama client with a
fixed-vector fake, so nothing here needs a model.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")
import mcp_server  # noqa: E402

DIM = 8


def vec(i: int, j: int | None = None, w: float = 0.0):
    v = [0.0] * DIM
    v[i] = 1.0
    if j is not None:
        v[j] = w
    s = sum(x * x for x in v) ** 0.5
    return [x / s for x in v]


class FakeClient:
    def __init__(self, *a, **k):
        pass

    def embed(self, model, input, options=None, keep_alive=None):
        return {"embeddings": [vec(0) for _ in input]}


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(mcp_server.ollama, "Client", FakeClient)
    path = mcp_server.TEXTBOOK_DB_PATH
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield lancedb.connect(str(path))
    shutil.rmtree(path, ignore_errors=True)


def v2_row(i, book, file, text, v, parent="p0", kind="body"):
    return {"id": f"c{i}", "vector": v, "text": text, "parent_id": parent, "file": file,
            "book": book, "chapter": "Ch", "section_path": "", "section_path_raw": "",
            "page_start": 10, "page_end": 90, "chunk_kind": kind, "chunk_idx": 0,
            "n_siblings": 1, "token_count": 50, "heading_origin": "page_fallback",
            "figure_ids": [], "has_figure_ref": False, "chunking_version": "v2.1",
            "embedding_model": "m", "query_template_version": "q", "indexing_signature": "s",
            "mtime": 0.0}


def parent_rows(ids, text="parent text"):
    return [{"parent_id": p, "text": text, "file": "f", "book": "b", "chapter": "c",
             "section_path": "", "section_path_raw": "", "page_start": 1, "page_end": 2,
             "token_count": 5, "heading_origin": "H2", "figure_ids": [], "has_figure_ref": False,
             "chunking_version": "v", "embedding_model": "m", "query_template_version": "q",
             "indexing_signature": "s", "mtime": 0.0, "vector": vec(1)} for p in ids]


def search(**args):
    args.setdefault("query", "q")
    return mcp_server._textbook_search_v2_raw(args)


def test_v2_index_gets_pdf_only_and_flag_from_text(db):
    flagged = ("[B — Ch]\n**[Table on page 3]**\n<!-- ⚠️ table needs out-of-band review (dosage) -->\n"
               "| a | b |\n| --- | --- |\n| 1 | 2 |")
    db.create_table("textbook_chunks", data=[
        v2_row(1, "Book_7e_2021", "Book_7e_2021/ch01.md", flagged, vec(0), kind="table"),
        v2_row(2, "Book_7e_2021", "Book_7e_2021/ch02.md", "[B — Ch]\nplain", vec(0, 1, 0.5), parent="p1")])
    db.create_table("textbook_parents", data=parent_rows(["p0", "p1"]))
    res = search(n_results=5)
    assert res[0]["page_confidence"] == "pdf_only"
    assert res[0]["table_review_flag"] == "review"
    assert "table_review_flag" not in res[1]
    assert res[0]["year"] == 2021 and res[0]["status"] == "current" and res[0]["corpus_type"] == "textbook"
    assert "printed_page_start" not in res[0]
    md = mcp_server.format_textbook_markdown(res, "q")
    assert "PDF p.10-90" in md and "⚠️ table needs checking" in md


def _v3_rows():
    import textbook_indexer_v3 as V
    cs, ps = V.schemas(DIM)
    base = dict(book="Book_2e", source="Book_2e", lineage="Book", chapter="Chapter 1 Gait",
                chapter_id="Book_2e#ch1", chapter_number=1, section_path="Gait cycle",
                section_path_raw="Gait cycle", section_kind="", page_start=12, page_end=13,
                pdf_page_start=12, pdf_page_end=13, printed_page_start="2", printed_page_end="3",
                page_confidence="printed", char_start=10, char_end=900, token_count=40,
                heading_origin="H2", figure_ids=[], has_figure_ref=False, edition="2e",
                year=2014, tier=1, status="current", corpus_type="textbook",
                split_method="bookmarks", source_hash="abc", chunking_version="v3.0",
                embedding_model="m", query_template_version="q", indexing_signature="s", mtime=0.0)
    table_text = ("[Book_2e — Chapter 1 Gait > Gait cycle]\n**[Table on page 12]**\n"
                  "<!-- ⚠️ table needs out-of-band review (dosage) -->\n| Drug | Dose |\n| --- | --- |\n"
                  "| Dantrolene | 25 mg |")
    chunks = [
        {**base, "id": "t1", "vector": vec(0), "text": table_text, "parent_id": "P1",
         "chunk_kind": "table", "chunk_idx": 0, "n_siblings": 2, "table_id": "tbl_x",
         "table_review_flag": "review", "table_embed_truncated": False, "file": "Book_2e/ch02.md"},
        {**base, "id": "b1", "vector": vec(0, 1, 0.3), "text": "[Book_2e — x]\nbody text",
         "parent_id": "P1", "chunk_kind": "body", "chunk_idx": 1, "n_siblings": 2, "table_id": "",
         "table_review_flag": "", "table_embed_truncated": False, "file": "Book_2e/ch02.md",
         "printed_page_start": "", "printed_page_end": "", "page_confidence": "pdf_only"},
    ]
    parents = [{**base, "parent_id": "P1", "vector": vec(1), "has_table": True,
                "text": "the whole parent section", "file": "Book_2e/ch02.md"}]
    return cs, ps, chunks, parents


def test_v3_index_returns_provenance_and_whole_table(db):
    cs, ps, chunks, parents = _v3_rows()
    db.create_table("textbook_chunks", data=chunks, schema=cs)
    db.create_table("textbook_parents", data=parents, schema=ps)
    res = search(n_results=5)
    t = res[0]
    assert t["chunk_kind"] == "table"
    assert t["parent_text"].startswith("**[Table on page 12]**") and "Dantrolene" in t["parent_text"]
    assert t["printed_page_start"] == "2" and t["page_confidence"] == "printed"
    assert t["pdf_page_start"] == 12 and t["chapter_number"] == 1 and t["table_review_flag"] == "review"
    assert t["tier"] == 1 and t["edition"] == "2e" and t["chunk_id"] == "t1"
    b = res[1]
    assert b["page_confidence"] == "pdf_only" and "printed_page_start" not in b
    assert "table_review_flag" not in b
    md = mcp_server.format_textbook_markdown(res, "q")
    assert "p.2-3 · PDF p.12-13" in md


def test_lineage_diversity_and_backfill(db):
    rows = []
    for i in range(5):
        rows.append(v2_row(i, "Braddom_7e_2021", f"Braddom_7e_2021/c{i}.md", "x", vec(0, 1, 0.01 * i), parent=f"a{i}"))
    for i in range(5):
        rows.append(v2_row(10 + i, "Braddom_6e_2011", f"Braddom_6e_2011/c{i}.md", "x",
                           vec(0, 1, 0.05 + 0.01 * i), parent=f"b{i}"))
    rows.append(v2_row(20, "Other_1e_2020", "Other_1e_2020/c.md", "x", vec(0, 1, 0.5), parent="o"))
    db.create_table("textbook_chunks", data=rows)
    db.create_table("textbook_parents", data=parent_rows([r["parent_id"] for r in rows]))
    res = search(n_results=6)
    assert len(res) == 6                                   # backfilled, not short
    books = [r["book"] for r in res]
    # editions share one cap of 3 before the backfill kicks in
    assert books[:4].count("Other_1e_2020") == 1
    assert sum(b.startswith("Braddom") for b in books[:4]) == 3


def test_hybrid_without_fts_index_falls_back_to_vector(db):
    db.create_table("textbook_chunks", data=[v2_row(1, "B", "B/c.md", "x", vec(0))])
    db.create_table("textbook_parents", data=parent_rows(["p0"]))
    res = search(retrieval="hybrid", n_results=3)
    assert len(res) == 1


def test_fts_uses_query_en(db):
    rows = [v2_row(1, "B", "B/a.md", "[B]\nbaclofen intrathecal pump", vec(0), parent="pa"),
            v2_row(2, "B", "B/b.md", "[B]\ndantrolene sodium dose", vec(2), parent="pb")]
    t = db.create_table("textbook_chunks", data=rows)
    t.create_fts_index("text", use_tantivy=False, replace=True)
    db.create_table("textbook_parents", data=parent_rows(["pa", "pb"]))
    res = search(query="丹曲林劑量", query_en="dantrolene dose", retrieval="fts", n_results=3)
    assert res and res[0]["file"] == "B/b.md"
    res = search(query="丹曲林劑量", query_en="dantrolene dose", retrieval="hybrid", n_results=3)
    assert {r["file"] for r in res} == {"B/a.md", "B/b.md"}


def test_query_en_in_tool_schema():
    tb = next(t for t in mcp_server.TOOLS if t["name"] == "textbook_search")
    assert "query_en" in tb["inputSchema"]["properties"]
