"""books.json query-time join + the chat context built from the shared MCP path.

Run directly (`python server/tests/test_books_meta_chat.py`) or under pytest.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "eval"))

_tmp = tempfile.mkdtemp(prefix="vsearch_books_")
os.environ.setdefault("VAULT_PATH", _tmp)
os.environ.setdefault("VAULT_SEARCH_DATA_DIR", _tmp)

import books_meta as B  # noqa: E402


def test_infer_and_override():
    meta = {"Book_PMR_6e_2015": {"status": "superseded", "superseded_by": "Book_PMR_7e_2021"},
            "AHA_Stroke_2016": {"status": "superseded", "superseded_by": "VA_Stroke_2024"}}
    m = B.lookup("Book_PMR_7e_2021", "Book_PMR_7e_2021/ch1.md", meta)
    assert (m["edition"], m["year"], m["corpus_type"], m["status"]) == ("7e", 2021, "textbook", "current")
    m = B.lookup("Book_PMR_6e_2015", "Book_PMR_6e_2015/ch1.md", meta)
    assert m["status"] == "superseded"
    # guidelines share one `book`; the key is the sub-folder
    m = B.lookup("_Guidelines", "_Guidelines/AHA_Stroke_2016/full_text.md", meta)
    assert (m["source_key"], m["corpus_type"], m["status"], m["year"]) == \
        ("AHA_Stroke_2016", "guideline", "superseded", 2016)


def test_visible_and_tie_break():
    rs = [{"book": "Old_1e_2001", "file": "Old_1e_2001/a.md", "similarity": 0.80},
          {"book": "New_2e_2020", "file": "New_2e_2020/a.md", "similarity": 0.803},
          {"book": "Far_1e_2022", "file": "Far_1e_2022/a.md", "similarity": 0.70}]
    meta = {"Old_1e_2001": {"tier": 1}, "New_2e_2020": {"tier": 2}}
    B.annotate(rs, meta)
    # same 0.01 bucket: tier 1 wins the tie even though it is older
    assert [r["book"] for r in B.tie_break(rs)] == ["Old_1e_2001", "New_2e_2020", "Far_1e_2022"]
    meta["Old_1e_2001"]["status"] = "superseded"
    B.annotate(rs, meta)
    assert [r["book"] for r in rs if B.visible(r, False, None)] == ["New_2e_2020", "Far_1e_2022"]
    assert len([r for r in rs if B.visible(r, True, None)]) == 3
    assert B.describe(rs[0]["meta"]).startswith("1e · 2001 · tier 1 · superseded")


def test_prefilter():
    assert B.prefilter(None, {}) is None
    assert B.prefilter("guideline", {}) == "book = '_Guidelines'"
    assert "OR book IN ('X')" in B.prefilter("guideline", {"X": {"corpus_type": "guideline"}})


def test_load_reads_file_and_ignores_comments():
    p = Path(_tmp) / "books.json"
    p.write_text(json.dumps({"_comment": "x", "A_1e_2000": {"tier": 1}}), encoding="utf-8")
    assert B.load(p) == {"A_1e_2000": {"tier": 1}}
    assert B.load(Path(_tmp) / "nope.json") == {}


# --- chat context through the shared path ------------------------------------

try:
    import api_server
except ImportError:  # fastapi / lancedb not installed
    api_server = None

if api_server is not None:
    from adapters import parse_chat_context

    def test_chat_uses_shared_path_with_quotas():
        calls = []

        def fake_vault(args, ctx=None):
            calls.append(("vault", args["n_results"], ctx))
            return {"results": [{"note": "Stroke rehab", "section": "Tx", "similarity": 0.71,
                                 "excerpt": "note body"}]}

        def fake_tb(args, ctx=None, corpus="textbook"):
            calls.append((corpus, args["n_results"], ctx))
            return [{"book": "B_7e_2021", "file": "B_7e_2021/ch12.md", "section_path": "Dosing",
                     "page_start": 341, "page_end": 342, "similarity": 0.8, "parent_text": "5 mg tid",
                     "meta": {"edition": "7e", "year": 2021, "corpus_type": "textbook", "status": "current"}}]

        saved = (api_server._vault_search_raw, api_server._textbook_search_v2_raw, api_server.lecture_enabled)
        api_server._vault_search_raw = fake_vault
        api_server._textbook_search_v2_raw = fake_tb
        api_server.lecture_enabled = lambda: True
        try:
            ctx = api_server._search_vault_context("q", 10, {"caller": "chat", "session_id": "r1"})
        finally:
            api_server._vault_search_raw, api_server._textbook_search_v2_raw, api_server.lecture_enabled = saved

        assert [(c, n) for c, n, _ in calls] == [("vault", 3), ("textbook", 4), ("lecture", 2)]
        assert all(c[2]["caller"] == "chat" for c in calls)
        parsed = parse_chat_context(ctx)
        tb = [r for r in parsed if r["corpus"] == "textbook"][0]
        assert (tb["book"], tb["file"], tb["page_start"], tb["page_end"]) == \
            ("B_7e_2021", "B_7e_2021/ch12.md", 341, 342)
        assert [r["corpus"] for r in parsed] == ["vault", "textbook", "lecture"]
        # passages are not vault notes
        assert api_server._extract_note_names(ctx) == {"Stroke rehab"}

    def test_chat_survives_a_retrieval_error():
        saved = api_server.chat_retrieve
        api_server.chat_retrieve = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone"))
        try:
            assert api_server._search_vault_context("q") == ""
        finally:
            api_server.chat_retrieve = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} books/chat tests passed")
