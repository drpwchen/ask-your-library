"""corpus.py: `check` on the shipped fixtures and on broken copies, `add --dry-run`
command building and refusals, and `add` end to end with a fake embedder."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
FIXTURES = SERVER.parent / "examples" / "corpus-fixture"

import corpus  # noqa: E402
import lecture_meta  # noqa: E402

TEXTBOOK_SRC = "Example_Shoulder-Rehab_2e_2024"
COURSE = "20250301_Example_Shoulder-Ultrasound"


@pytest.fixture
def tb(tmp_path):
    return Path(shutil.copytree(FIXTURES / "textbook", tmp_path / "textbook"))


@pytest.fixture
def lec(tmp_path):
    return Path(shutil.copytree(FIXTURES / "lecture", tmp_path / "lecture"))


def codes(items):
    return {i["code"] for i in items}


# ----------------------------------------------------------------------------- check

def test_fixtures_pass_clean():
    for kind in ("textbook", "lecture"):
        rep = corpus.check(FIXTURES / kind)
        assert rep.errors == [] and rep.warnings == [], (kind, rep.errors, rep.warnings)
        assert rep.kind == kind and rep.kind_source == "corpus.json"
    assert corpus.check(FIXTURES / "textbook").stats == {
        "sources": 1, "guidelines": 1, "files": 3, "manifests": 1}
    assert corpus.check(FIXTURES / "lecture").stats == {"courses": 1, "entries": 2, "meta_files": 1}


def test_missing_marker_warns_and_infers(tb, lec):
    (tb / "corpus.json").unlink()
    (lec / "corpus.json").unlink()
    for root, kind in ((tb, "textbook"), (lec, "lecture")):
        rep = corpus.check(root)
        assert rep.errors == []
        assert codes(rep.warnings) == {"no_marker"}
        assert (rep.kind, rep.kind_source) == (kind, "inferred")


@pytest.mark.parametrize("content,code", [
    ("{not json", "marker_json"),
    ("[1, 2]", "marker_json"),
    ('{"format": "t2n-corpus/2", "kind": "textbook"}', "marker_format"),
    ('{"format": "t2n-corpus/1", "kind": "slides"}', "marker_kind"),
    ('{"format": "t2n-corpus/1", "kind": "textbook", "name": 3}', "marker_name"),
])
def test_bad_marker_is_an_error(tb, content, code):
    (tb / "corpus.json").write_text(content, encoding="utf-8")
    assert code in codes(corpus.check(tb).errors)


def test_empty_and_missing_roots(tmp_path):
    assert codes(corpus.check(tmp_path / "nope").errors) == {"no_root"}
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "corpus.json").write_text('{"format": "t2n-corpus/1", "kind": "textbook"}', encoding="utf-8")
    assert codes(corpus.check(empty).errors) == {"no_sources"}
    (empty / "Book_1e_2020").mkdir()
    assert codes(corpus.check(empty).errors) == {"no_files"}
    (empty / "corpus.json").write_text('{"format": "t2n-corpus/1", "kind": "lecture"}', encoding="utf-8")
    assert codes(corpus.check(empty).errors) == {"no_entries"}


def test_textbook_warnings(tb):
    src = tb / TEXTBOOK_SRC
    (tb / "stray.md").write_text("x", encoding="utf-8")
    (tb / "_Guidelines" / "loose.md").write_text("x", encoding="utf-8")
    (tb / "Empty_Book_1e_2020").mkdir()
    (src / "ch01_Examination.md").write_text(
        (src / "ch01_Examination.md").read_text(encoding="utf-8").replace("<!-- page", "<!-- pg"),
        encoding="utf-8")
    (src / "ch02_Exercise.md").write_text("# Ch 2\n\nno markers\n", encoding="utf-8")
    m = json.loads((src / "_manifest.json").read_text(encoding="utf-8"))
    m["files"].append({"file": "ch09_Gone.md"})
    (src / "_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    rep = corpus.check(tb)
    assert rep.errors == []
    assert codes(rep.warnings) == {"root_md", "guideline_loose_md", "empty_source",
                                   "no_page_markers", "manifest_stale"}


def test_unreadable_manifest_and_full_text_choice(tb):
    src = tb / TEXTBOOK_SRC
    (src / "_manifest.json").write_text("{broken", encoding="utf-8")
    (src / "full_text.md").write_text("<!-- page 1 -->\n" + "word " * 5000, encoding="utf-8")
    assert codes(corpus.check(tb).warnings) == {"manifest_unreadable", "chapters_ignored"}


def test_lecture_entry_warnings_and_strict(lec, capsys):
    course = lec / COURSE
    (course / "_meta.md").unlink()
    (course / "No_source.md").write_text("# Title\n\n## Body\n\nSource: too late\n", encoding="utf-8")
    (course / "No_h1.md").write_text("Intro\n\nSource: 2025-03-01 A《B》\n", encoding="utf-8")
    (course / "No_date.md").write_text("# T\n\nSource: Dr. A《Course》· x\n", encoding="utf-8")
    (course / "No_course.md").write_text("# T\n\nSource: 2025-03-01\n", encoding="utf-8")
    rep = corpus.check(lec)
    assert rep.errors == []
    assert codes(rep.warnings) == {"no_meta", "entry_no_source", "entry_no_h1",
                                   "source_no_date", "source_no_course"}
    assert corpus.main(["check", str(lec)]) == 0
    assert corpus.main(["check", str(lec), "--strict"]) == 1


def test_books_meta_checked_for_textbooks_only(tb, lec, tmp_path):
    bm = tmp_path / "books.json"
    bm.write_text(json.dumps({"_comment": "x", TEXTBOOK_SRC: {"tier": 1},
                              "ExampleSociety_LowBackPain_2023": {"corpus_type": "guideline"},
                              "Typo_Book": {"tier": 2}}), encoding="utf-8")
    rep = corpus.check(tb, bm)
    assert [w["message"] for w in rep.warnings if w["code"] == "books_meta_unknown"] == [
        "key 'Typo_Book' matches no source folder in this corpus (typo, or an entry for another corpus)"]
    assert corpus.check(lec, bm).warnings == []
    bm.write_text("{bad", encoding="utf-8")
    assert codes(corpus.check(tb, bm).errors) == {"books_meta_json"}


def test_check_cli_json_and_exit_codes(tb):
    out = subprocess.run([sys.executable, str(SERVER / "corpus.py"), "check", str(tb), "--json"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["ok"] is True and data["format"] == "t2n-corpus/1" and data["kind"] == "textbook"
    (tb / "corpus.json").write_text("{bad", encoding="utf-8")
    out = subprocess.run([sys.executable, str(SERVER / "corpus.py"), "check", str(tb)],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 1 and "marker_json" in out.stdout


# ----------------------------------------------------------------------------- add

def _settings(tmp_path, **kw):
    s = {"ok": True, "data_dir": tmp_path / "data", "textbook_db": tmp_path / "data" / "lance_db",
         "textbook_db_set": False, "lecture_db": tmp_path / "data" / "lecture_db",
         "lecture_db_set": False, "books_meta": tmp_path / "data" / "books.json",
         "env_file": tmp_path / ".env", "error": ""}
    s.update(kw)
    return s


def _mark_v3(db: Path, corpus_root: Path | None = None):
    st = db / "_v3_state"
    st.mkdir(parents=True)
    if corpus_root is not None:
        (st / "index_meta.json").write_text(json.dumps({"corpus": str(corpus_root)}), encoding="utf-8")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr(corpus, "load_settings", lambda: s)
    import textbook_indexer_v3 as V
    monkeypatch.setattr(V, "live_paths", lambda: [s["textbook_db"], s["lecture_db"]])
    return s


def test_default_db_per_kind(tmp_path):
    s = _settings(tmp_path)
    assert corpus.default_db("textbook", s) == tmp_path / "data" / "textbook_db"
    assert corpus.default_db("lecture", s) == tmp_path / "data" / "lecture_db_v3"
    live = tmp_path / "live_v3"
    _mark_v3(live)
    s = _settings(tmp_path, textbook_db=live, textbook_db_set=True, lecture_db=live, lecture_db_set=True)
    assert corpus.default_db("textbook", s) == live
    assert corpus.default_db("lecture", s) == live
    # Configured but not (yet) v3: never chosen implicitly.
    s = _settings(tmp_path, textbook_db=tmp_path / "plain", textbook_db_set=True)
    assert corpus.default_db("textbook", s) == tmp_path / "data" / "textbook_db"


def test_add_dry_run_prints_commands_and_env(tb, lec, settings, capsys):
    assert corpus.main(["add", str(tb), "--dry-run"]) == 0
    out = capsys.readouterr().out
    db = (settings["data_dir"] / "textbook_db").resolve()
    assert f"--corpus {tb.resolve()} --db {db} --books-meta {settings['books_meta']}" in out
    assert f"--db {db} --books-meta {settings['books_meta']} --audit" in out
    assert f"VAULT_SEARCH_TEXTBOOK_DB_PATH={db}" in out
    assert "nothing indexed" in out
    assert not db.exists()

    assert corpus.main(["add", str(lec), "--dry-run", "--no-gpu-lease"]) == 0
    out = capsys.readouterr().out
    ldb = (settings["data_dir"] / "lecture_db_v3").resolve()
    assert f"--corpus {lec.resolve()} --db {ldb} --books-meta {settings['books_meta']} --no-gpu-lease" in out
    assert f"VAULT_SEARCH_LECTURE_DB_PATH={ldb}" in out


def test_indexer_argv():
    assert corpus.indexer_argv(Path("/c"), Path("/d"), None, False) == ["--corpus", str(Path("/c")),
                                                                        "--db", str(Path("/d"))]
    assert corpus.indexer_argv(Path("/c"), Path("/d"), "/b.json", True)[-3:] == [
        "--books-meta", "/b.json", "--no-gpu-lease"]


def test_add_refusals(tb, settings, tmp_path, capsys):
    # check errors stop add
    bad = tmp_path / "bad"
    bad.mkdir()
    assert corpus.main(["add", str(bad), "--dry-run"]) == 1
    # a db inside the corpus (the indexer's own guard)
    assert corpus.main(["add", str(tb), "--db", str(tb / "db"), "--dry-run"]) == 2
    assert "inside the corpus" in capsys.readouterr().out
    # a live, non-v3 database
    assert corpus.main(["add", str(tb), "--db", str(settings["textbook_db"]), "--dry-run"]) == 2
    # a v3 database built from another corpus root
    other = tmp_path / "other_db"
    _mark_v3(other, tmp_path / "some_other_corpus")
    assert corpus.main(["add", str(tb), "--db", str(other), "--dry-run"]) == 2
    assert "built from another corpus root" in capsys.readouterr().out
    # same root: fine
    same = tmp_path / "same_db"
    _mark_v3(same, tb.resolve())
    assert corpus.main(["add", str(tb), "--db", str(same), "--dry-run"]) == 0
    # --strict refuses on warnings
    (tb / "stray.md").write_text("x", encoding="utf-8")
    assert corpus.main(["add", str(tb), "--dry-run", "--strict"]) == 1


def test_add_hint_when_env_points_at_unbuilt_db(tb, tmp_path, monkeypatch, capsys):
    target = tmp_path / "data" / "textbook_db"
    s = _settings(tmp_path, textbook_db=target, textbook_db_set=True)
    monkeypatch.setattr(corpus, "load_settings", lambda: s)
    import textbook_indexer_v3 as V
    monkeypatch.setattr(V, "live_paths", lambda: [target])
    assert corpus.main(["add", str(tb), "--dry-run"]) == 2
    assert "Build first, then set it" in capsys.readouterr().out


def test_env_lines_in_effect(tmp_path):
    db = tmp_path / "db"
    s = _settings(tmp_path, textbook_db=db, textbook_db_set=True)
    assert corpus.env_lines("textbook", db, str(s["books_meta"]), s) == (
        [f"VAULT_SEARCH_TEXTBOOK_DB_PATH={db}"], True)
    lines, ok = corpus.env_lines("textbook", db, str(tmp_path / "x.json"), s)
    assert not ok and lines[1] == f"VAULT_SEARCH_BOOKS_META={tmp_path / 'x.json'}"
    assert corpus.env_lines("lecture", db, None, s)[1] is False


# ----------------------------------------------------------------------------- add, end to end

lancedb = pytest.importorskip("lancedb")


@pytest.fixture
def fake_index(monkeypatch):
    import chunker_v3 as C
    import textbook_indexer_v3 as V
    from test_v3_indexer_run import FakeEmbedder
    monkeypatch.setattr(V, "Embedder", FakeEmbedder)
    monkeypatch.setattr(C, "qwen_tok", C.whitespace_tok)
    monkeypatch.setattr(V, "truncate_tokens", lambda t, n: t)
    return V


@pytest.mark.parametrize("kind", ["textbook", "lecture"])
def test_add_indexes_fixture_and_reruns_incrementally(kind, tmp_path, settings, fake_index, capsys):
    root = Path(shutil.copytree(FIXTURES / kind, tmp_path / kind))
    db = tmp_path / f"{kind}_db"
    assert corpus.main(["add", str(root), "--db", str(db), "--no-gpu-lease"]) == 0
    out = capsys.readouterr().out
    assert "missing: 0" in out and "indexed but not selected: 0" in out
    assert f"{corpus.ENV_FOR_KIND[kind]}={db.resolve()}" in out
    files = set(lancedb.connect(str(db)).open_table("textbook_chunks").search().select(["file"])
                .limit(1000).to_pandas()["file"])
    if kind == "textbook":
        assert files == {f"{TEXTBOOK_SRC}/ch01_Examination.md", f"{TEXTBOOK_SRC}/ch02_Exercise.md",
                         "_Guidelines/ExampleSociety_LowBackPain_2023/full_text.md"}
    else:
        assert f"{COURSE}/Biceps_tendon_and_subscapularis.md" in files
    assert corpus.built_from(db) == str(root.resolve())
    # second run: nothing changed, nothing re-embedded
    assert corpus.main(["add", str(root), "--db", str(db), "--no-gpu-lease"]) == 0
    assert "'skipped': 3" in capsys.readouterr().out


def test_lecture_source_read_from_corpus_when_index_has_none(tmp_path):
    """The v3 chunker drops `Source:` lines, so the parents table has none;
    the provenance comes from the entry file under the recorded corpus root."""
    root = FIXTURES / "lecture"
    f = f"{COURSE}/Biceps_tendon_and_subscapularis.md"
    got = lecture_meta.source_by_file(None, [f, f"{COURSE}/_meta.md"], lambda s: s, corpus_root=str(root))
    assert set(got) == {f}
    assert got[f]["date"] == "2025-03-01"
    assert got[f]["speaker"] == "Dr. Example"
    assert got[f]["course_title"] == "Shoulder Ultrasound Workshop"
    assert lecture_meta.source_by_file(None, [f], lambda s: s) == {}


# ----------------------------------------------------------------------------- shared guideline manifest

def _shared_guideline_manifest(root: Path, own: bool = False) -> Path:
    """textbook-to-note --resplit splits `_Guidelines` as ONE book: one manifest at
    `_Guidelines/_manifest.json` whose entries are `<Name>/<file>`."""
    g = root / "_Guidelines" / "ExampleSociety_LowBackPain_2023"
    (g / "full_text.md").rename(g / "ch01_Recommendations.md")
    (root / "_Guidelines" / "_manifest.json").write_text(json.dumps({
        "schema": 1, "book": "_Guidelines", "split_method": "bookmarks",
        "page_labels": {"1": "S1", "2": "S2"}, "page_confidence": "printed",
        "files": [{"file": "ExampleSociety_LowBackPain_2023/ch01_Recommendations.md",
                   "chapter_number": 1, "title": "Recommendations", "pdf_page_start": 1,
                   "kind": "body"}]}), encoding="utf-8")
    if own:
        (g / "_manifest.json").write_text(json.dumps({
            "schema": 1, "files": [{"file": "ch01_Recommendations.md", "chapter_number": 7,
                                    "title": "Own"}]}), encoding="utf-8")
    return g


def test_find_manifest_falls_back_to_shared_guideline_manifest(tb):
    import textbook_indexer_v3 as V
    g = _shared_guideline_manifest(tb)
    f = g / "ch01_Recommendations.md"
    m, rel = V.find_manifest(f, g, {})
    assert rel == "ExampleSociety_LowBackPain_2023/ch01_Recommendations.md"
    import provenance as P
    assert P.file_entry(m, rel)["chapter_number"] == 1
    # the guideline's own manifest still wins
    g = _shared_guideline_manifest(Path(shutil.copytree(FIXTURES / "textbook", tb.parent / "tb2")), own=True)
    m, rel = V.find_manifest(g / "ch01_Recommendations.md", g, {})
    assert rel == "ch01_Recommendations.md" and P.file_entry(m, rel)["chapter_number"] == 7
    # an ordinary book never climbs above its own folder
    book = tb / TEXTBOOK_SRC
    (tb / "_manifest.json").write_text('{"files": []}', encoding="utf-8")
    (book / "_manifest.json").unlink()
    assert V.find_manifest(book / "ch01_Examination.md", book, {}) == (None, "ch01_Examination.md")


def test_check_reads_shared_guideline_manifest(tb):
    _shared_guideline_manifest(tb)
    rep = corpus.check(tb)
    assert rep.errors == [] and rep.warnings == []
    assert rep.stats["manifests"] == 2
    m = json.loads((tb / "_Guidelines" / "_manifest.json").read_text(encoding="utf-8"))
    m["files"][0]["file"] = "ExampleSociety_LowBackPain_2023/gone.md"
    (tb / "_Guidelines" / "_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert codes(corpus.check(tb).warnings) == {"manifest_stale"}


def test_add_indexes_guideline_with_shared_manifest(tb, tmp_path, settings, fake_index):
    _shared_guideline_manifest(tb)
    db = tmp_path / "gdb"
    assert corpus.main(["add", str(tb), "--db", str(db), "--no-gpu-lease"]) == 0
    df = (lancedb.connect(str(db)).open_table("textbook_chunks").search()
          .where("book = '_Guidelines'")
          .select(["chapter", "chapter_number", "printed_page_start", "page_confidence", "split_method"])
          .limit(100).to_pandas())
    assert len(df) and set(df["chapter"]) == {"Recommendations"}
    assert set(df["chapter_number"]) == {1}
    assert set(df["page_confidence"]) == {"printed"} and set(df["split_method"]) == {"bookmarks"}
    assert set(df["printed_page_start"]) <= {"S1", "S2"}


def test_add_prints_the_whole_config_error(tb, tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, ok=False, error="RuntimeError: line one, e.g.\n  line two\nline three")
    monkeypatch.setattr(corpus, "load_settings", lambda: s)
    import textbook_indexer_v3 as V
    monkeypatch.setattr(V, "live_paths", lambda: [])
    assert corpus.main(["add", str(tb), "--db", str(tmp_path / "db")]) == 2
    out = capsys.readouterr().out
    assert "line one, e.g.\n  line two\nline three" in out
