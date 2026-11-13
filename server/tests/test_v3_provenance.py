"""v3 index: provenance, table units, chunker, file choice, safety rails.

Pure tests — no Ollama, no tokenizer download (chunker runs on a whitespace
tokenizer).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunker_v3 as C  # noqa: E402
import provenance as P  # noqa: E402
import table_units as T  # noqa: E402
import vault_v3 as V3  # noqa: E402

TOK = C.whitespace_tok()


def words(n, w="word"):
    return " ".join(f"{w}{i}" for i in range(n))


# --------------------------------------------------------------------------- manifest

def write_manifest(d: Path, **over):
    m = {"schema": 1, "book": "Book_2e", "source_pdf": "Book_2e.pdf", "split_method": "bookmarks",
         "page_labels": {str(i): str(i - 10) for i in range(11, 60)},
         "page_confidence": "printed",
         "files": [{"file": "ch02_Gait.md", "seq": 2, "chapter_number": 1, "title": "Chapter 1 Gait",
                    "pdf_page_start": 12, "pdf_page_end": 29, "kind": "body"},
                   {"file": "ch09_References.md", "seq": 9, "chapter_number": None,
                    "title": "References", "pdf_page_start": 50, "pdf_page_end": 55,
                    "kind": "references"}],
         "sections": [{"level": 3, "title": "Further Reading", "pdf_page": 28,
                       "file": "ch02_Gait.md", "kind": "references"}]}
    m.update(over)
    (d / "_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return m


def test_manifest_contract_and_draft_key_names(tmp_path):
    write_manifest(tmp_path)
    m = P.load_manifest(tmp_path)
    assert m["split_method"] == "bookmarks" and m["source_pdf"] == "Book_2e.pdf"
    assert P.printed_label(m, 12) == "2"
    assert P.file_entry(m, "ch02_Gait.md")["chapter_number"] == 1
    assert P.file_entry(m, "CH02_gait.md")["title"] == "Chapter 1 Gait"   # casefolded
    assert P.page_confidence(m) == "printed"
    # draft names from the converter proposal still read
    (tmp_path / "_manifest.json").write_text(json.dumps(
        {"schema": 1, "source": "x.pdf", "split": "pattern", "page_labels": None, "files": []}))
    m2 = P.load_manifest(tmp_path)
    assert m2["source_pdf"] == "x.pdf" and m2["split_method"] == "pattern"
    assert m2["page_confidence"] == "pdf_only"


def test_manifest_claims_printed_without_labels_is_downgraded(tmp_path):
    write_manifest(tmp_path, page_labels=None, page_confidence="printed")
    assert P.page_confidence(P.load_manifest(tmp_path)) == "pdf_only"


def test_missing_or_broken_manifest_is_none(tmp_path):
    assert P.load_manifest(tmp_path) is None
    (tmp_path / "_manifest.json").write_text("{not json")
    assert P.load_manifest(tmp_path) is None
    assert P.file_entry(None, "x.md") == {}
    assert P.printed_label(None, 3) == ""


def test_basename_lookup_refuses_ambiguity(tmp_path):
    (tmp_path / "_manifest.json").write_text(json.dumps({"schema": 1, "files": [
        {"file": "A/full_text.md", "title": "A"}, {"file": "B/full_text.md", "title": "B"}]}))
    m = P.load_manifest(tmp_path)
    assert P.file_entry(m, "B/full_text.md")["title"] == "B"
    assert P.file_entry(m, "full_text.md") == {}


# --------------------------------------------------------------------------- lines / misc

def test_line_table_initial_page_and_offsets():
    content = "tail of page 11\n<!-- page 12 -->\nline a\nline b"
    lines = P.line_table(content, initial_page=11)
    assert [ln["page"] for ln in lines] == [11, 12, 12, 12]
    assert lines[1]["marker"] and lines[1]["text"] == ""
    a, b = P.char_span(lines, 2, 4)
    assert content[a:b] == "line a\nline b"


def test_lineage_and_source_keys():
    assert P.lineage_key("Braddom_PM-and-R_6e_2021") == P.lineage_key("Braddom_PM-and-R_7e_2021") \
        == "Braddom_PM-and-R"
    assert P.lineage_key("ACSM_Exercise-Testing-Prescription_11e_2021") == \
        "ACSM_Exercise-Testing-Prescription"
    assert P.lineage_key("Whatever_2e", {"lineage": "X"}) == "X"
    assert P.source_key("_Guidelines", "_Guidelines/AAOS_ACL_2022/ch01.md") == "AAOS_ACL_2022"
    assert P.source_key("Book", "Book/ch01.md") == "Book"


def test_watermarks_stripped():
    t, n = P.strip_watermarks("Continued on following page booksmedicos.org")
    assert t == "Continued on following page" and n == 1
    assert P.strip_watermarks("see https://t.me/MBS_MedicalBooksStore now")[1] == 1


def test_reference_detection_is_conservative():
    refs = "\n".join(f"{i}. Smith AB, Lee C, et al. Title. J Rehabil. 20{10 + i};{i}:1-9."
                     for i in range(1, 9))
    assert P.looks_like_reference_list(refs)
    prose = "\n".join(["Spasticity is velocity dependent (Lance 1980) and is common."] * 8)
    assert not P.looks_like_reference_list(prose)
    assert P.is_reference_title("References") and P.is_reference_title("12.4 Suggested Readings")
    assert not P.is_reference_title("Reference values for gait")


# --------------------------------------------------------------------------- tables

TABLE_DOC = """<!-- page 20 -->
Table 3-1 Starting doses
Some prose before the table.

**[Table on page 20]**

<!-- ⚠️ table needs out-of-band review (dosage/units) — verify against PDF page 20 -->
| Drug | Dose |
| --- | --- |
| Baclofen | 5 mg tid |

<!-- page 21 -->

| Tizanidine | 2 mg |
| Dantrolene | 25 mg |

Prose after the table on page 21.
"""


def test_cross_page_table_is_one_unit_with_both_pages():
    lines = P.line_table(TABLE_DOC)
    units = T.find_table_units(lines)
    assert len(units) == 1
    u = units[0]
    assert u["pages"] == [20, 21]
    assert u["review_flag"] == T.REVIEW
    assert u["caption"].startswith("Table 3-1")
    assert u["n_rows"] == 3
    text = T.unit_text(lines, u)
    assert "Dantrolene" in text and "<!-- page" not in text


def test_stitched_continuation_comment_extends_pages():
    doc = ("<!-- page 5 -->\n**[Table on page 5]**\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
           "<!-- table continues from page 6 -->\n\nafter\n")
    u = T.find_table_units(P.line_table(doc))[0]
    assert u["pages"] == [5, 6] and u["review_flag"] is None


def test_table_with_different_columns_after_page_break_is_separate():
    doc = ("<!-- page 5 -->\n| a | b |\n| --- | --- |\n| 1 | 2 |\n<!-- page 6 -->\n"
           "| x | y | z |\n| --- | --- | --- |\n")
    assert len(T.find_table_units(P.line_table(doc))) == 2


# --------------------------------------------------------------------------- chunker

def _ctx(**kw):
    base = dict(book="Book_2e", rel_file="Book_2e/ch02_Gait.md", chapter="Chapter 1 Gait")
    base.update(kw)
    return C.FileContext(**base)


def test_children_carry_their_own_pages_not_the_file_range():
    pages = "\n".join(f"<!-- page {p} -->\n{words(150, f'p{p}w')}\n" for p in range(1, 11))
    settings = C.Settings(child_target=200, child_overlap=0, parent_target=400, parent_max=600)
    parents, kids, _ = C.chunk_file(pages, _ctx(), settings, TOK)
    assert len(kids) >= 5
    for k in kids:
        assert k["pdf_page_end"] - k["pdf_page_start"] <= 1, k["pdf_page_start"]
        first_word = k["core_text"].split()[0]
        assert first_word.startswith(f"p{k['pdf_page_start']}w")
        assert k["page_confidence"] == "pdf_only" and k["printed_page_start"] == ""
    for p in parents:
        assert p["pdf_page_end"] - p["pdf_page_start"] <= 2


def test_printed_pages_and_chapter_from_manifest(tmp_path):
    m = write_manifest(tmp_path)
    m = P.load_manifest(tmp_path)
    doc = "continuation of page 12\n<!-- page 13 -->\n" + words(50)
    _, kids, _ = C.chunk_file(doc, _ctx(manifest=m, manifest_rel="ch02_Gait.md"),
                              C.Settings(child_overlap=0, min_child_tokens=1), TOK)
    k = kids[0]
    assert (k["pdf_page_start"], k["pdf_page_end"]) == (12, 13)
    assert (k["printed_page_start"], k["printed_page_end"]) == ("2", "3")
    assert k["page_confidence"] == "printed"


def test_offsets_point_into_the_source():
    doc = "<!-- page 1 -->\n" + words(30) + "\n\n" + words(30, "x")
    _, kids, _ = C.chunk_file(doc, _ctx(), C.Settings(child_target=35, child_overlap=0,
                                                      min_child_tokens=1), TOK)
    for k in kids:
        assert doc[k["char_start"]:k["char_end"]].strip() == k["core_text"].strip()


def test_table_is_its_own_child_whole_and_flagged():
    doc = TABLE_DOC + "\n" + words(40)
    _, kids, st = C.chunk_file(doc, _ctx(), C.Settings(child_overlap=0, min_child_tokens=1), TOK)
    tables = [k for k in kids if k["chunk_kind"] == "table"]
    assert len(tables) == 1 and st.tables == 1
    t = tables[0]
    assert (t["pdf_page_start"], t["pdf_page_end"]) == (20, 21)
    assert t["table_review_flag"] == "review" and t["table_id"].startswith("tbl_")
    assert "Baclofen" in t["text"] and "Dantrolene" in t["text"]
    body = [k for k in kids if k["chunk_kind"] != "table"]
    assert all("Baclofen" not in k["core_text"] for k in body)


def test_long_table_is_not_split_but_embeds_a_digest():
    rows = "\n".join(f"| drug{i} | {i} mg {words(30)} |" for i in range(200))
    doc = f"<!-- page 1 -->\nTable 9-9 Huge\n**[Table on page 1]**\n| Drug | Dose |\n| --- | --- |\n{rows}\n"
    s = C.Settings(child_overlap=0, embed_max=500, table_digest_tokens=100, min_child_tokens=1)
    _, kids, st = C.chunk_file(doc, _ctx(), s, TOK)
    t = [k for k in kids if k["chunk_kind"] == "table"]
    assert len(t) == 1
    assert "drug199" in t[0]["text"]                       # returned whole
    assert t[0]["table_embed_truncated"] and "drug199" not in t[0]["embed_text"]
    assert "| Drug | Dose |" in t[0]["embed_text"] and "Table 9-9" in t[0]["embed_text"]
    assert st.tables_digest_embedded == 1


def test_references_section_and_file_are_dropped(tmp_path):
    write_manifest(tmp_path)
    m = P.load_manifest(tmp_path)
    refs = "\n".join(f"{i}. Doe J, et al. Paper. Arch Phys Med Rehabil. 2019;{i}:1-5." for i in range(12))
    doc = f"<!-- page 27 -->\n## Gait cycle\n{words(60)}\n\n## Further Reading\n{refs}\n"
    s = C.Settings(child_overlap=0, min_child_tokens=1)
    _, kids, st = C.chunk_file(doc, _ctx(manifest=m, manifest_rel="ch02_Gait.md"), s, TOK)
    assert kids and all("Arch Phys" not in k["core_text"] for k in kids)
    assert st.reference_parents_dropped == 1
    # the same block without any heading is caught by citation density
    _, kids2, st2 = C.chunk_file(f"<!-- page 3 -->\n{words(60)}\n\n{refs}\n", _ctx(), s, TOK)
    assert all("Arch Phys" not in k["core_text"] for k in kids2)
    assert st2.reference_children_dropped == 1
    # whole file marked references in the manifest
    p, k, _ = C.chunk_file(doc, _ctx(manifest=m, rel_file="Book_2e/ch09_References.md",
                                     manifest_rel="ch09_References.md"), s, TOK)
    assert p == [] and k == []
    # --keep-references keeps and labels them
    _, kids3, _ = C.chunk_file(doc, _ctx(manifest=m, manifest_rel="ch02_Gait.md"),
                               C.Settings(child_overlap=0, min_child_tokens=1, keep_references=True), TOK)
    assert any(k["section_kind"] == "references" for k in kids3)


def test_reference_named_file_without_manifest_is_cut_at_plain_heading():
    # Guideline shape: no manifest, "REFERENCES" is a plain line, citations wrap so
    # per-line density stays low. Body before the heading must survive.
    wrapped = "\n".join(f"{i}. Doe J, Roe K. A long title that wraps onto\nthe next line. Some Journal "
                        f"of Things, vol {i}." for i in range(20))
    doc = f"<!-- page 65 -->\nDisclosure {words(40)}\nREFERENCES \n \n{wrapped}\n"
    s = C.Settings(child_overlap=0, min_child_tokens=1)
    ctx = _ctx(book="_Guidelines", rel_file="_Guidelines/AAOS_X_2025/ch74_REFERENCES.md")
    _, kids, st = C.chunk_file(doc, ctx, s, TOK)
    assert kids and all("Doe J" not in k["text"] for k in kids)
    assert any("Disclosure" in k["core_text"] for k in kids)
    assert st.reference_parents_dropped == 1
    # zero-width joiners inside the heading and "Select References" spelling
    doc2 = f"<!-- page 3 -->\n{words(40)}\nRefer‌‌ences\n{wrapped}\n"
    ctx2 = _ctx(rel_file="Book_2e/ch11_Select_References.md")
    _, kids2, _ = C.chunk_file(doc2, ctx2, s, TOK)
    assert kids2 and all("Doe J" not in k["text"] for k in kids2)
    # no heading line found -> file kept whole (body loss is worse than a leak)
    _, kids3, _ = C.chunk_file(f"<!-- page 3 -->\n{words(40)}\n{wrapped}\n", ctx2, s, TOK)
    assert any("Doe J" in k["text"] for k in kids3)
    # an ordinary chapter file with a plain "References" line is untouched by this rule
    _, kids4, _ = C.chunk_file(doc, _ctx(), s, TOK)
    assert any("Doe J" in k["text"] for k in kids4)
    # --keep-references disables it
    _, kids5, _ = C.chunk_file(doc, ctx, C.Settings(child_overlap=0, min_child_tokens=1,
                                                    keep_references=True), TOK)
    assert any("Doe J" in k["text"] for k in kids5)


def test_title_from_filename():
    assert P.title_from_filename("_Guidelines/A/ch74_REFERENCES.md") == "REFERENCES"
    assert P.title_from_filename("B\\ch104_Suggested_Reading.md") == "Suggested Reading"
    assert P.is_reference_title(P.title_from_filename("B/ch12a_Select_References.md"))
    assert not P.is_reference_title(P.title_from_filename("B/ch02_Gait.md"))


def test_deep_headings_fill_section_path():
    doc = f"# Book title\n<!-- page 4 -->\n## Epidemiology\n{words(40)}\n### Incidence\n{words(40, 'i')}\n"
    parents, kids, _ = C.chunk_file(doc, _ctx(), C.Settings(child_overlap=0, min_child_tokens=1,
                                                            parent_min=1), TOK)
    paths = {tuple(k["section_path_parts"]) for k in kids}
    assert ("Epidemiology",) in paths and ("Epidemiology", "Incidence") in paths
    assert kids[0]["text"].startswith("[Book_2e — Chapter 1 Gait > Epidemiology")


def test_chunk_ids_are_stable():
    doc = "<!-- page 1 -->\n" + words(300)
    a = [k["chunk_id"] for k in C.chunk_file(doc, _ctx(), C.Settings(child_target=100), TOK)[1]]
    b = [k["chunk_id"] for k in C.chunk_file(doc, _ctx(), C.Settings(child_target=100), TOK)[1]]
    assert a == b and len(set(a)) == len(a)


def test_watermark_removed_from_indexed_text():
    doc = "<!-- page 1 -->\n" + words(30) + " booksmedicos.org\n"
    _, kids, st = C.chunk_file(doc, _ctx(), C.Settings(min_child_tokens=1), TOK)
    assert "booksmedicos" not in kids[0]["text"] and st.watermarks == 1


# --------------------------------------------------------------------------- indexer plumbing

def _v3():
    import textbook_indexer_v3 as V
    return V


def test_full_text_or_chapters_never_both(tmp_path):
    V = _v3()
    g = tmp_path / "_Guidelines" / "AAOS_ACL_2022"
    g.mkdir(parents=True)
    (g / "full_text.md").write_text("x" * 1000)
    (g / "ch01_A.md").write_text("x" * 400)
    (g / "ch02_B.md").write_text("x" * 400)
    files, dec = V.choose_files(g)
    assert sorted(f.name for f in files) == ["ch01_A.md", "ch02_B.md"]
    (g / "ch02_B.md").write_text("")  # split is mostly empty → full_text wins
    files, dec = V.choose_files(g)
    assert [f.name for f in files] == ["full_text.md"] and dec[0]["chose"] == "full_text"


def test_plan_units_guidelines_and_exclusions(tmp_path):
    V = _v3()
    for d in ("Book_6e_2010", "Book_7e_2020", "_Guidelines/G1", "_Guidelines/G2", "_figure_remap"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "full_text.md").write_text("<!-- page 1 -->\nhello")
    meta = {"Book_6e_2010": {"status": "superseded", "superseded_by": "Book_7e_2020"},
            "G2": {"status": "low_trust", "corpus_type": "guideline"}}
    items, excluded, _ = V.plan_corpus(tmp_path, meta, book=None, include_superseded=False)
    assert sorted(i["source"] for i in items) == ["Book_7e_2020", "G1"]
    assert {e["source"] for e in excluded} == {"Book_6e_2010", "G2"}
    items, _, _ = V.plan_corpus(tmp_path, meta, book=None, include_superseded=True)
    assert len(items) == 4


def test_refuses_the_live_database(tmp_path):
    V = _v3()
    live = tmp_path / "lance_db"
    with pytest.raises(SystemExit):
        V.check_target(live, None, [live])
    with pytest.raises(SystemExit):
        V.check_target(live / "sub", None, [live])
    with pytest.raises(SystemExit):
        V.check_target(tmp_path / "corpus" / "db", tmp_path / "corpus", [live])
    V.check_target(tmp_path / "lance_db_v3", tmp_path / "corpus", [live])  # fine


def test_model_tables_and_signature():
    V = _v3()
    assert V.table_names("qwen3-embedding:0.6b") == ("textbook_chunks", "textbook_parents")
    assert V.table_names("qwen3-embedding:4b") == ("textbook_chunks__qwen3_embedding_4b",
                                                   "textbook_parents__qwen3_embedding_4b")
    s = C.Settings()
    a = V.compute_signature(V.signature_parts("m", "abc", 1024, s))
    assert a != V.compute_signature(V.signature_parts("m", "abd", 1024, s))      # model digest
    assert a != V.compute_signature(V.signature_parts("m", "abc", 2560, s))      # dimension
    assert any(p.startswith("doc_instruction=") for p in V.signature_parts("m", "x", 1, s))
    assert "norm=l2" in V.signature_parts("m", "x", 1, s)


def test_holding_report_only_suggests(tmp_path):
    V = _v3()
    corpus = tmp_path / "corpus"
    for d in ("Book_6e_2010", "Book_7e_2020", "Other_1e_2000", "Dup_2e_2014", "Dup_2e_2015"):
        (corpus / d).mkdir(parents=True)
        (corpus / d / "full_text.md").write_text("x")
    meta_p = tmp_path / "books.json"
    meta_p.write_text(json.dumps({"Other_1e_2000": {"status": "low_trust"}}))
    out = tmp_path / "holding.md"
    V.main(["--corpus", str(corpus), "--books-meta", str(meta_p), "--holding-report", str(out)])
    rows = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    by = {r["source"]: r for r in rows}
    assert by["Other_1e_2000"]["basis"] == "decided"
    assert by["Book_6e_2010"]["replacement"] == "Book_7e_2020"
    assert "Dup_2e_2014" in by and by["Dup_2e_2014"]["basis"].startswith("duplicate")
    assert all((corpus / d).exists() for d in ("Book_6e_2010", "Other_1e_2000"))  # nothing moved


# --------------------------------------------------------------------------- vault v3 helpers

def test_vault_v3_exclusions():
    assert V3.excluded("Tooling/backups/CLAUDE.md.bak-2026-07-03.md", set()) == "backup"
    assert V3.excluded("98Templates/Disease.md", set()) == "templates"
    assert V3.excluded("Templates/x.md", set()) == "templates"
    assert V3.excluded("Tooling/notes.md", {"tooling"}) == "excluded_folder"
    assert V3.excluded("52Medicine/Stroke.md", {"tooling"}) == ""


def test_vault_v3_yaml_frontmatter():
    from indexer import parse_frontmatter
    content = ('---\ntitle: "A: quoted"\ntags:\n  - stroke\n  - "rehab"\nmeta:\n  nested: 1\n'
               'summary: |\n  line one\n  line two\n---\nBody here\n')
    meta, body = V3.parse_frontmatter_yaml(content, parse_frontmatter)
    assert meta["title"] == "A: quoted" and meta["meta"] == {"nested": 1}
    assert V3.as_list(meta["tags"]) == ["stroke", "rehab"] and body == "Body here"
    bad = "---\ntags: [unclosed\n---\nBody\n"
    meta2, body2 = V3.parse_frontmatter_yaml(bad, parse_frontmatter)
    assert body2.strip() == "Body"


def test_vault_v3_hard_caps():
    kids = [{"text": "[F — N]\n" + "\n".join(words(50) for _ in range(40)), "token_count": 2000,
             "chunk_idx": 0, "n_siblings": 1, "parent_id": "p"}]
    out = V3.cap_children(kids, 300, TOK.count, TOK.window)
    assert len(out) > 1 and all(c["token_count"] <= 300 for c in out)
    assert all(c["text"].startswith("[F — N]\n") for c in out)
    one_line = [{"text": words(5000), "token_count": 5000, "chunk_idx": 0, "n_siblings": 1}]
    assert all(c["token_count"] <= 300 for c in V3.cap_children(one_line, 300, TOK.count, TOK.window))


def test_scope_of():
    import scope as S
    smap = {"medical": ["52Medicine", "00Inbox"]}
    assert S.scope_of("52Medicine", smap) == "medical"
    assert S.scope_of("12Journal", smap) == "personal"
    assert S.scope_of("x", {}) == ""


def test_vault_v3_profile_needs_its_own_db(tmp_path, monkeypatch):
    import vault_indexer_v2 as VI
    import textbook_indexer_v3 as V
    monkeypatch.setattr(V, "live_paths", lambda: [tmp_path / "lance_db"])
    for name in ("DB_PATH", "HASH_CACHE_PATH", "ERROR_LOG_PATH", "READY_MARKER_DIR", "PROFILE"):
        monkeypatch.setattr(VI, name, getattr(VI, name))  # restored after the test
    try:
        with pytest.raises(SystemExit):
            VI.configure(None, "v3")
        with pytest.raises(SystemExit):
            VI.configure(str(tmp_path / "lance_db"), "v3")
        VI.configure(str(tmp_path / "lance_db_v3"), "v3")
        assert VI.DB_PATH == tmp_path / "lance_db_v3"
        assert VI.HASH_CACHE_PATH.parent == tmp_path / "lance_db_v3" / "_vault_state"
        assert "scope" in VI.chunks_schema().names and "profile=v3" in VI._signature_parts()
        # A v2-profile run must never write into that v3 vault index.
        with pytest.raises(SystemExit, match="v3-profile vault index"):
            VI.configure(None, "v2")
    finally:
        VI.PROFILE = "v2"
    assert "scope" not in VI.chunks_schema().names and "profile=v3" not in VI._signature_parts()


def test_live_v3_db_is_writable_by_v3_only(tmp_path, monkeypatch):
    """After the switch-over the live textbook DB IS the v3 DB: the v3 indexer
    may update it; the v2 textbook indexer must refuse it."""
    import textbook_indexer_v3 as V
    import textbook_indexer as T2
    live = tmp_path / "lance_db_v3"
    (live / "_v3_state").mkdir(parents=True)
    V.check_target(live, tmp_path / "corpus", [live])          # allowed
    old = tmp_path / "lance_db"
    old.mkdir()
    with pytest.raises(SystemExit):
        V.check_target(old, tmp_path / "corpus", [old])         # live v2: refused
    with pytest.raises(SystemExit):
        V.check_target(live / "sub", tmp_path / "corpus", [live])  # nested: refused
    monkeypatch.setattr(T2, "DB_PATH", live)
    with pytest.raises(SystemExit, match="v3 database"):
        T2.index_textbooks(incremental=True)
