"""lecture_search: _meta.md exclusion and Source-line provenance.

Run directly (`python server/tests/test_lecture_meta.py`) or under pytest.
"""

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

_tmp = tempfile.mkdtemp(prefix="vsearch_lecture_")
os.environ.setdefault("VAULT_PATH", _tmp)
os.environ.setdefault("VAULT_SEARCH_DATA_DIR", _tmp)

import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

import lecture_meta as L  # noqa: E402


def test_parse_plain():
    p = L.parse_source("2025-02-15 Dr. Wu《Upper-limb nerve US》· course_dir · c01, c03")
    assert p == {"date": "2025-02-15", "speaker": "Dr. Wu", "course": "Upper-limb nerve US"}


def test_parse_with_prefix_and_range():
    p = L.parse_source("Source: 2026-04-18/19 多位講者《2026 年會》· catalog `Ortho/x`")
    assert (p["date"], p["speaker"], p["course"]) == ("2026-04-18", "多位講者", "2026 年會")


def test_parse_unknowns_become_empty():
    p = L.parse_source("日期未載明 ⚠️ 講者未載明 ⚠️《DAI STOB》· mc_PT_DAI · c94")
    assert p == {"date": "", "speaker": "", "course": "DAI STOB"}
    p = L.parse_source("2024-11-24 連教授（全名未載明）《進階班》· x · c05")
    assert p["speaker"] == "連教授"


def test_parse_without_title():
    p = L.parse_source("2024-06-23 台灣復健醫學會顱磁刺激治療研討會暨工作坊 · d01/d07")
    assert p == {"date": "2024-06-23", "speaker": "", "course": "台灣復健醫學會顱磁刺激治療研討會暨工作坊"}
    assert L.parse_source("") == {"date": "", "speaker": "", "course": ""}


def test_find_source_line_and_describe():
    text = "# Title\n\nSource: 2025-01-01 A《B》· c\nDomain: x\n"
    assert L.find_source_line(text) == "2025-01-01 A《B》· c"
    assert L.find_source_line("# no source\n") == ""
    assert L.describe({"date": "2025-01-01", "speaker": "A", "course_title": "B"}) == "2025-01-01 · A · 《B》"
    assert L.describe({}) == ""


def test_is_meta_file():
    assert L.is_meta_file("course/_meta.md")
    assert L.is_meta_file("course\_meta.md")
    assert L.is_meta_file("_meta.md")
    assert not L.is_meta_file("course/x_meta.md")
    assert not L.is_meta_file("course/entry.md")


def _db():
    db = lancedb.connect(tempfile.mkdtemp(prefix="vsearch_lecture_db_"))
    rows = [
        {"parent_id": "p1", "file": "c1/_meta.md", "text": "# c1\n\nSource: 2025-01-01 Meta《M》"},
        {"parent_id": "p2", "file": "c1/entry.md", "text": "# Entry\n\nSource: 2025-02-15 吳醫師《課》· c1 · c01\n"},
        {"parent_id": "p3", "file": "c1/entry.md", "text": "## 技術\n- bullet [c01 03:21]"},
        {"parent_id": "p4", "file": "c1/x_meta.md", "text": "## no source here"},
    ]
    return db.create_table("parents", pa.Table.from_pylist(rows))


def test_meta_where_filter_on_real_lancedb():
    t = _db()
    files = sorted(t.search().where(L.META_EXCLUDE_WHERE).select(["file"]).to_pandas()["file"])
    assert files == ["c1/entry.md", "c1/entry.md", "c1/x_meta.md"]


def test_source_by_file_reads_h1_parent():
    t = _db()
    esc = lambda s: s.replace("'", "''")  # noqa: E731
    out = L.source_by_file(t, ["c1/entry.md", "c1/_meta.md", "c1/x_meta.md"], esc)
    assert set(out) == {"c1/entry.md"}
    assert (out["c1/entry.md"]["speaker"], out["c1/entry.md"]["date"],
            out["c1/entry.md"]["course_title"]) == ("吳醫師", "2025-02-15", "課")
    assert L.source_by_file(None, ["c1/entry.md"], esc) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
