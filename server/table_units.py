"""
Tables as their own retrieval units (v3 textbook index) — pure, no lancedb.

The converter writes every table as::

    **[Table on page 21]**
    <!-- ⚠️ table structure unverified: ... -->            (optional QC flag)
    <!-- ⚠️ table needs out-of-band review (...) ... -->   (optional review flag)
    | header | ... |
    | --- | --- |
    | row | ... |
    <!-- table continues from page 22 -->                  (stitched continuation)

A table that runs past a page break can also arrive as two runs of ``|`` lines
with only blank lines and a ``<!-- page N -->`` marker between them. Both
shapes are one table here: the unit never stops at a page marker, and its page
range covers every page it touches.

A table is atomic: it is never cut at a child-length limit. When it is too
long to embed, the caller embeds the caption + header + first rows and still
returns the whole table.
"""

from __future__ import annotations

import hashlib
import re

LABEL_RE = re.compile(r"^\s*\*\*\[Table(?: on page (\d+))?[^\]]*\]\*\*\s*$")
REVIEW_RE = re.compile(r"<!--\s*⚠️\s*table needs out-of-band review")
UNVERIFIED_RE = re.compile(r"<!--\s*⚠️\s*table structure unverified")
CONTINUES_RE = re.compile(r"<!--\s*table continues from page (\d+)\s*-->")
CAPTION_RE = re.compile(r"^\s*(?:\*\*)?(?:Table|TABLE|表)\s*[A-Z]?\d+(?:[.\-–]\d+)*[A-Za-z]?\b")
SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")

REVIEW = "review"          # needs out-of-band review (misbinding risk)
UNVERIFIED = "unverified"  # structural QC flag


def is_table_line(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("|") and "|" in s[1:]


def n_cols(text: str) -> int:
    s = text.strip()
    if not s.startswith("|"):
        return 0
    return max(0, s.count("|") - 1)


def _is_comment(text: str) -> bool:
    s = text.strip()
    return s.startswith("<!--") and s.endswith("-->")


def find_table_units(lines: list[dict], *, caption_lookback: int = 60) -> list[dict]:
    """Locate table units in a ``provenance.line_table()`` list.

    Returns dicts with ``start``/``end`` (line indices, end exclusive, covering
    label, flag comments, rows and continuation comments), ``rows_start``,
    ``pages`` (sorted unique physical pages), ``review_flag``, ``caption``,
    ``header`` (first row + separator), ``n_rows``, ``anchor_page``.
    """
    n = len(lines)
    units: list[dict] = []
    i = 0
    while i < n:
        t = lines[i]["text"]
        start = None
        label_page = 0
        if LABEL_RE.match(t):
            # label → (blank | flag comments)* → table rows
            j = i + 1
            while j < n and (not lines[j]["text"].strip() or _is_comment(lines[j]["text"])) \
                    and not lines[j]["marker"]:
                j += 1
            if j < n and is_table_line(lines[j]["text"]):
                start = i
                m = LABEL_RE.match(t)
                label_page = int(m.group(1)) if m and m.group(1) else 0
                rows_start = j
            else:
                i += 1
                continue
        elif is_table_line(t) and i + 1 < n and is_table_line(lines[i + 1]["text"]):
            start = i
            rows_start = i
        else:
            i += 1
            continue

        # Absorb flag comments between label and rows (already skipped over).
        j = rows_start
        last_row = j
        ncol = n_cols(lines[j]["text"])
        pages = set()
        while j < n:
            txt = lines[j]["text"]
            if is_table_line(txt):
                last_row = j
                pages.add(lines[j]["page"])
                j += 1
                continue
            if CONTINUES_RE.search(txt):
                pages.add(int(CONTINUES_RE.search(txt).group(1)))
                last_row = j
                j += 1
                continue
            # Look ahead across blank lines / page markers: the table may resume
            # on the next page with the same column count.
            k = j
            crossed_marker = False
            while k < n and (not lines[k]["text"].strip() or lines[k]["marker"]):
                crossed_marker = crossed_marker or lines[k]["marker"]
                k += 1
            if (crossed_marker and k < n and is_table_line(lines[k]["text"])
                    and n_cols(lines[k]["text"]) == ncol and not LABEL_RE.match(lines[k]["text"])):
                j = k
                continue
            break
        end = last_row + 1

        flag = None
        block = [lines[x]["text"] for x in range(start, end)]
        if any(REVIEW_RE.search(b) for b in block):
            flag = REVIEW
        elif any(UNVERIFIED_RE.search(b) for b in block):
            flag = UNVERIFIED

        rows = [lines[x]["text"] for x in range(rows_start, end) if is_table_line(lines[x]["text"])]
        header = rows[:2] if len(rows) >= 2 and SEPARATOR_RE.match(rows[1]) else rows[:1]
        anchor = label_page or (min(pages) if pages else lines[start]["page"])
        pages.add(anchor)
        pages.discard(0)

        units.append({
            "start": start,
            "end": end,
            "rows_start": rows_start,
            "pages": sorted(pages),
            "anchor_page": anchor,
            "review_flag": flag,
            "caption": _find_caption(lines, start, anchor, caption_lookback),
            "header": header,
            "n_rows": max(0, len(rows) - len(header)),
        })
        i = end
    return units


def _find_caption(lines: list[dict], start: int, anchor_page: int, lookback: int) -> str:
    """The nearest 'Table 12-3 …' line above the table on its anchor page."""
    for x in range(start - 1, max(-1, start - 1 - lookback), -1):
        ln = lines[x]
        if ln["marker"] and ln["page"] < anchor_page:
            break
        if CAPTION_RE.match(ln["text"]) and not ln["text"].strip().startswith("|"):
            return ln["text"].strip()[:200]
    return ""


def table_id(source: str, rel_file: str, unit: dict) -> str:
    raw = f"{source}|{rel_file}|{unit['start']}|{unit['anchor_page']}"
    return "tbl_" + hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def unit_text(lines: list[dict], unit: dict) -> str:
    """The table exactly as written (label, flags, rows, continuation notes);
    page markers inside a cross-page table are dropped."""
    out = [lines[x]["text"] for x in range(unit["start"], unit["end"]) if not lines[x]["marker"]]
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip("\n")


def embed_digest(lines: list[dict], unit: dict, max_tokens: int, tok_len) -> str:
    """Caption + header + as many leading rows as fit in ``max_tokens``."""
    head = [unit["caption"]] if unit["caption"] else []
    head += unit["header"]
    used = tok_len("\n".join(head))
    rows = [lines[x]["text"] for x in range(unit["rows_start"], unit["end"])
            if is_table_line(lines[x]["text"])][len(unit["header"]):]
    kept = []
    for r in rows:
        t = tok_len(r)
        if used + t > max_tokens:
            break
        kept.append(r)
        used += t
    return "\n".join(head + kept)
