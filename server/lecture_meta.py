"""Lecture corpus helpers: the course-level `_meta.md` exclusion and the
`Source:` line parser.

A lecture corpus is textbook-format Markdown, one folder per course. Each
course folder carries a `_meta.md` (speaker list, clip table, material
locations) that is bookkeeping, not lecture content; it is kept out of
`lecture_search` results. Each entry file starts with an H1 and a fixed
`Source:` line, for example

    Source: 2025-02-15 Dr. Wu《Upper-limb nerve ultrasound》· course_dir · c01, c03

from which the speaker, date and course title are parsed. The line is free
text written by people and agents, so every field is optional: a field that
cannot be read is returned as "" rather than guessed.
"""

from __future__ import annotations

import re

META_FILENAME = "_meta.md"

# LanceDB (DataFusion) filter that drops the course-level meta files.
META_EXCLUDE_WHERE = (
    f"file != '{META_FILENAME}' AND NOT ends_with(file, '/{META_FILENAME}')"
    f" AND NOT ends_with(file, '\\{META_FILENAME}')"
)

_SOURCE_LINE = re.compile(r"^\s*Source:\s*(.+?)\s*$", re.M)
# 2025-02-15, 2026-04-18/19, 2026-07-25~26, 2026-09-12（六）
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:\s*[/~–-]\s*\d{1,2}(?:-\d{2})?)?(?:[（(][^）)]*[）)])?")
_TITLE = re.compile(r"《([^》]+)》")
_UNKNOWN = re.compile(r"(未載明|未知|unknown)", re.I)  # whole value is a placeholder
_SEP = re.compile(r"\s*[·・]\s*")


def is_meta_file(path: str) -> bool:
    p = (path or "").replace("\\", "/")
    return p == META_FILENAME or p.endswith("/" + META_FILENAME)


def find_source_line(text: str) -> str:
    """The value of the first `Source:` line in a block of Markdown, or ""."""
    m = _SOURCE_LINE.search(text or "")
    return m.group(1) if m else ""


_UNKNOWN_NOTE = re.compile(r"[（(][^）)]*(未載明|未知|unknown)[^）)]*[）)]", re.I)


def _clean_name(s: str) -> str:
    # "連教授（全名未載明）" keeps "連教授"; a bare "講者未載明" becomes "".
    s = _UNKNOWN_NOTE.sub("", s)
    s = s.replace("⚠️", "").strip(" ，,;；:：")
    if not s or _UNKNOWN.search(s):
        return ""
    return s


def parse_source(value: str) -> dict:
    """Split a `Source:` value into date / speaker / course.

    Returns {"date": "YYYY-MM-DD" or "", "speaker": str, "course": str}.
    - date: the first ISO date (a range keeps its start date).
    - course: the first 《…》 title; without one, the text between the date and
      the first `·` when there is no speaker part in front of it.
    - speaker: the text between the date and 《; "" when absent or marked
      unknown (未載明 / ⚠️ only).
    """
    out = {"date": "", "speaker": "", "course": ""}
    v = (value or "").strip()
    if not v:
        return out
    if v.lower().startswith("source:"):
        v = v[7:].strip()
    head = v
    dm = _DATE.search(v)
    if dm and dm.start() <= 2:
        out["date"] = dm.group(1)
        head = v[dm.end():].strip()
    elif v.startswith("日期未載明"):
        head = v[len("日期未載明"):].strip()
    tm = _TITLE.search(head)
    if tm:
        out["course"] = tm.group(1).replace("⚠️", "").strip()
        out["speaker"] = _clean_name(head[:tm.start()])
    else:
        first = _SEP.split(head, maxsplit=1)[0]
        out["course"] = _clean_name(first)
    return out


def describe(r: dict) -> str:
    """One provenance line for a lecture result: 'date · speaker · 《course》'."""
    parts = [r.get("date") or "", r.get("speaker") or ""]
    if r.get("course_title"):
        parts.append(f"《{r['course_title']}》")
    return " · ".join(p for p in parts if p)


def source_by_file(parents_table, files, escape) -> dict[str, dict]:
    """{file: parsed Source} for the given entry files, read from the parents
    table (the H1 parent of an entry holds its Source line). One query; files
    without a Source line are simply absent from the result."""
    uniq = [f for f in dict.fromkeys(files) if f and not is_meta_file(f)]
    if parents_table is None or not uniq:
        return {}
    quoted = ",".join("'" + escape(f) + "'" for f in uniq)
    df = (parents_table.search()
          .where(f"file IN ({quoted}) AND text LIKE '%Source:%'")
          .select(["file", "text"])
          .limit(max(50, len(uniq) * 20))
          .to_pandas())
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        f = row["file"]
        if f in out:
            continue
        line = find_source_line(str(row["text"]))
        if line:
            parsed = parse_source(line)
            out[f] = {"date": parsed["date"], "speaker": parsed["speaker"],
                      "course_title": parsed["course"], "source_line": line}
    return out
