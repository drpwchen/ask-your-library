"""
Provenance for the v3 textbook index — pure functions, no lancedb / ollama.

Where a chunk came from, precisely enough to cite it:

* ``pdf_page_start/end``      physical PDF pages of the chunk's OWN lines,
                              read from the converter's ``<!-- page N -->``
                              markers (not the whole file's range)
* ``printed_page_start/end``  the page number printed on the paper, from the
                              book's ``_manifest.json`` ``page_labels``; empty
                              when the PDF defines no labels, and then
                              ``page_confidence = "pdf_only"``
* ``chapter_number``          the real chapter number from the manifest (not
                              the converter's bookmark sequence)
* ``source_hash``             sha256 of the markdown file the chunk was cut from
* ``char_start/char_end``     offsets of the chunk's own lines in that file

The manifest is written by the converter (textbook-to-note ``--resplit``,
``T2N_SPLIT_MANIFEST=1``). Schema 1::

    {"schema": 1, "book": "...", "source_pdf": "...",
     "split_method": "bookmarks|config|pattern|per_chapter_pdf|carried",
     "page_labels": {"<pdf_page>": "<printed label>"} | null,
     "page_confidence": "printed|pdf_only",
     "files": [{"file": "ch02_x.md", "seq": 2, "chapter_number": 1 | null,
                "title": "...", "pdf_page_start": 12, "pdf_page_end": 29,
                "kind": "body|references|front|back"}],
     "sections": [{"level": 3, "title": "...", "pdf_page": 28,
                   "file": "ch02_x.md", "kind": "..."}]}

A missing or unreadable manifest is not an error: every field above that needs
it is left empty and the page confidence is ``pdf_only``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

MANIFEST_NAME = "_manifest.json"
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")

PRINTED = "printed"
PDF_ONLY = "pdf_only"

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest(book_dir: Path) -> dict | None:
    """Read and normalise ``<book_dir>/_manifest.json``. None when absent/broken.

    Tolerates the draft key names (``source``/``split``) next to the contract's
    (``source_pdf``/``split_method``), and string or int page keys.
    """
    p = Path(book_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    labels = raw.get("page_labels")
    if isinstance(labels, dict) and labels:
        labels = {int(k): str(v) for k, v in labels.items() if str(k).strip().isdigit()}
    else:
        labels = None
    files: dict = {}
    base_seen: dict = {}
    for f in raw.get("files") or []:
        if isinstance(f, dict) and f.get("file"):
            rel = _norm_rel(f["file"])
            files[rel] = f
            b = _norm_name(f["file"])
            base_seen[b] = base_seen.get(b, 0) + 1
            files.setdefault("::base::" + b, f)
    for b, cnt in base_seen.items():
        if cnt > 1:  # same basename in two sub-folders: basename lookup is ambiguous
            files["::base::" + b] = None
    sections = [s for s in (raw.get("sections") or []) if isinstance(s, dict)]
    conf = raw.get("page_confidence") or (PRINTED if labels else PDF_ONLY)
    if not labels:
        conf = PDF_ONLY  # never claim printed pages we cannot look up
    return {
        "schema": raw.get("schema", 1),
        "book": raw.get("book", ""),
        "source_pdf": raw.get("source_pdf") or raw.get("source") or "",
        "split_method": raw.get("split_method") or raw.get("split") or "",
        "page_labels": labels,
        "page_confidence": conf,
        "files": files,
        "sections": sections,
    }


def _norm_name(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name.casefold()


def _norm_rel(name: str) -> str:
    return str(name).replace("\\", "/").strip("/").casefold()


def file_entry(manifest: dict | None, filename: str) -> dict:
    """The manifest's ``files[]`` entry for one markdown file ({} if none).

    ``filename`` is the path relative to the manifest's folder; an exact
    relative-path match wins, a bare basename matches only when unambiguous.
    """
    if not manifest:
        return {}
    files = manifest["files"]
    hit = files.get(_norm_rel(filename))
    if hit is None:
        hit = files.get("::base::" + _norm_name(filename))
    return hit or {}


def printed_label(manifest: dict | None, pdf_page: int) -> str:
    """Printed page label for a physical page; '' when unknown."""
    if not manifest or not manifest.get("page_labels") or not pdf_page:
        return ""
    return manifest["page_labels"].get(int(pdf_page), "")


def page_confidence(manifest: dict | None) -> str:
    return (manifest or {}).get("page_confidence") or PDF_ONLY


# ---------------------------------------------------------------------------
# Lines → pages / offsets
# ---------------------------------------------------------------------------


def line_table(content: str, initial_page: int = 0) -> list[dict]:
    """One record per line: text, page, char offset, is_marker.

    Lines before the first ``<!-- page N -->`` marker get ``initial_page``:
    the manifest's ``pdf_page_start`` for the file when known (a chapter file
    cut mid-page starts with the tail of that page), else 0.
    """
    out = []
    page = initial_page
    off = 0
    for line in content.split("\n"):
        m = PAGE_MARKER_RE.search(line)
        if m:
            page = int(m.group(1))
        out.append({"text": "" if m else line, "page": page, "off": off,
                    "end": off + len(line), "marker": bool(m)})
        off += len(line) + 1
    return out


def first_marker_page(content: str) -> int:
    m = PAGE_MARKER_RE.search(content)
    return int(m.group(1)) if m else 0


def page_span(lines: list[dict], start: int, end: int) -> tuple[int, int]:
    """(first, last) physical page of lines[start:end]; (0, 0) if none known."""
    pages = [ln["page"] for ln in lines[start:end] if ln["page"] > 0 and not ln["marker"]
             and ln["text"].strip()]
    if not pages:
        pages = [ln["page"] for ln in lines[start:end] if ln["page"] > 0]
    if not pages:
        return 0, 0
    return min(pages), max(pages)


def char_span(lines: list[dict], start: int, end: int) -> tuple[int, int]:
    if end <= start:
        return 0, 0
    return lines[start]["off"], lines[end - 1]["end"]


def printed_span(manifest: dict | None, ps: int, pe: int) -> tuple[str, str]:
    return printed_label(manifest, ps), printed_label(manifest, pe)


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Watermarks
# ---------------------------------------------------------------------------

# Download-site stamps that PDF rips print on every page. Measured on the
# author's corpus (2026-11-09): `booksmedicos.org` ~5,000 lines, the t.me
# store link 14. Removed from the indexed text only — offsets still point into
# the untouched markdown.
WATERMARK_RE = re.compile(
    r"\s*(?:https?://)?(?:www\.)?(?:booksmedicos\.org|ebooksmedicine\.net|medbooksvn\.org"
    r"|t\.me/[A-Za-z0-9_]+|afkebooks\.com|freemedicalbooks\.org|ebookmedicine\.net)\b/?",
    re.IGNORECASE,
)


def strip_watermarks(text: str) -> tuple[str, int]:
    """(text without watermark stamps, number removed)."""
    new, n = WATERMARK_RE.subn("", text)
    return new, n


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

REFERENCE_TITLE_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*)?(?:references?|bibliography|suggested\s+readings?|"
    r"further\s+readings?|literature\s+cited|select(?:ed)?\s+references|參考文獻|参考文献)\s*[:.]?\s*$",
    re.IGNORECASE,
)

# One reference line: "12. Smith AB, Jones C. Title. J Foo. 2019;12:34-56."
_CITE_LINE_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}\s*[;:,]\s*\d+|\bet al\b\.?|\bdoi:\s*10\.|\b\d+\s*\(\d+\)\s*:\s*\d+"
    r"|\bJ\s+[A-Z][a-z]+|\bPMID\b)",
)
_ENUM_RE = re.compile(r"^\s*(?:\[?\d{1,4}[\].)]|•)\s+\S")


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = re.sub(r"^[\d.\s]+", "", s)
    s = re.sub(r"[\s]+", " ", s).strip(" .:;-")
    return s


_ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\u2060\ufeff]")


def is_reference_title(title: str) -> bool:
    # Zero-width joiners inside a word ("Refer\u200c\u200cences", Rinaldi 2023) must not hide it.
    return bool(REFERENCE_TITLE_RE.match(_ZERO_WIDTH_RE.sub("", title or "")))


def title_from_filename(rel_file: str) -> str:
    """'_Guidelines/X/ch74_REFERENCES.md' -> 'REFERENCES' (chapter prefix and underscores dropped)."""
    stem = re.sub(r"\.md$", "", (rel_file or "").replace("\\", "/").rsplit("/", 1)[-1], flags=re.I)
    stem = re.sub(r"^(?:ch|pages?_)[0-9a-z]*?_", "", stem, flags=re.I)
    return stem.replace("_", " ").strip()


def reference_heading_offset(content: str) -> int | None:
    """Character offset of the first line that is nothing but a References-type heading
    (markdown ``#`` optional — guideline PDFs carry it as a plain text line), else None."""
    pos = 0
    for line in (content or "").split("\n"):
        if is_reference_title(line.lstrip("#").strip()):
            return pos
        pos += len(line) + 1
    return None


def citation_density(text: str) -> float:
    """Share of non-empty lines that look like bibliography entries."""
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _CITE_LINE_RE.search(ln))
    return hits / len(lines)


def looks_like_reference_list(text: str, min_lines: int = 6, threshold: float = 0.6) -> bool:
    """A block that is mostly citations (used when no heading says 'References').

    Conservative on purpose: at least ``min_lines`` non-empty lines and at least
    ``threshold`` of them citation-shaped. Body prose that cites "(Smith et al.,
    2019)" now and then stays well below that.
    """
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if len(lines) < min_lines:
        return False
    return citation_density(text) >= threshold


def section_kind_for(manifest: dict | None, filename: str, section_titles: list[str]) -> str:
    """Kind of a parent from its heading path: manifest section kind, else title regex.

    The innermost heading that the manifest knows wins; a heading that reads
    "References" counts as references even without a manifest.
    """
    known = {}
    if manifest:
        fname = _norm_name(filename)
        for s in manifest["sections"]:
            if s.get("file") and _norm_name(s["file"]) != fname:
                continue
            if s.get("title") and s.get("kind"):
                known[norm_title(s["title"])] = s["kind"]
    for t in reversed([t for t in section_titles if t]):
        k = known.get(norm_title(t))
        if k:
            return k
        if is_reference_title(t):
            return "references"
    return ""


# ---------------------------------------------------------------------------
# Lineage / source keys
# ---------------------------------------------------------------------------

GUIDELINE_BOOK = "_Guidelines"

_EDITION_TAIL = re.compile(r"(?:[_\- ](?:\d{1,2}(?:e|ed|th|nd|rd|st)|ed\d+|v\d+))?"
                           r"(?:[_\- ](?:19|20)\d{2}[a-z]*)?$", re.IGNORECASE)


def source_key(book: str, rel_file: str) -> str:
    """The unit a citation names: the book folder, or for guidelines the
    guideline's own sub-folder (they all share one ``_Guidelines`` book)."""
    if book == GUIDELINE_BOOK:
        parts = (rel_file or "").replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] == GUIDELINE_BOOK:
            return parts[1]
    return book


def lineage_key(key: str, meta: dict | None = None) -> str:
    """Editions of one book share a lineage: ``Braddom_PM-and-R_6e_2021`` and
    ``Braddom_PM-and-R_7e_2021`` → ``Braddom_PM-and-R``. ``books.json`` may set
    ``lineage`` explicitly; that always wins over the name rule."""
    if meta and meta.get("lineage"):
        return str(meta["lineage"])
    k = key or ""
    for _ in range(2):  # "_7e_2021" is two tails
        new = _EDITION_TAIL.sub("", k)
        if new == k or not new:
            break
        k = new
    return k or key
