"""
v3 textbook chunker — line-accurate parents and children, tables as units.

Why a new chunker instead of patching textbook_indexer's: that one turns a
parent into plain text before cutting children, so a child cannot know which
lines (and therefore which pages) it came from; every child inherited its
parent's page range, and a parent without H2 headings spanned the whole file
(25% of children spanned 50+ pages). Here every piece of text stays attached
to its line numbers from the first step, so pages, printed pages and character
offsets are computed from the chunk's OWN lines.

Pure apart from the tokenizer, which is injected (``Tok``) so tests run with
a trivial whitespace tokenizer and no model download.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import provenance as P
import table_units as T

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？；])\s+")


@dataclass
class Tok:
    """Token counter + hard splitter. ``count(text) -> int``;
    ``window(text, max) -> list[str]`` cuts text into pieces of ≤ max tokens."""
    count: Callable[[str], int]
    window: Callable[[str, int], list]


def qwen_tok() -> Tok:
    from textbook_indexer import get_tokenizer, tok_len

    def window(text: str, max_tokens: int) -> list[str]:
        tk = get_tokenizer()
        ids = tk.encode(text, add_special_tokens=False)
        return [tk.decode(ids[i:i + max_tokens], skip_special_tokens=True)
                for i in range(0, len(ids), max_tokens)] or [""]

    return Tok(count=tok_len, window=window)


def whitespace_tok() -> Tok:
    """Test tokenizer: one token per whitespace-separated word."""
    def count(text: str) -> int:
        return len(text.split())

    def window(text: str, max_tokens: int) -> list[str]:
        w = text.split()
        return [" ".join(w[i:i + max_tokens]) for i in range(0, len(w), max_tokens)] or [""]

    return Tok(count=count, window=window)


@dataclass
class Settings:
    child_target: int = 512
    child_overlap: int = 75
    parent_target: int = 1024
    parent_max: int = 1800
    parent_min: int = 100
    table_in_parent_max: int = 1200   # bigger tables appear in the parent as a pointer line
    embed_max: int = 3800             # model context is 4096; leave room for the prefix
    table_digest_tokens: int = 1024   # what an over-long table embeds
    min_child_tokens: int = 20
    keep_references: bool = False
    strip_watermarks: bool = True

    def signature_parts(self) -> list[str]:
        return [f"CT={self.child_target}", f"CO={self.child_overlap}", f"PT={self.parent_target}",
                f"PX={self.parent_max}", f"PN={self.parent_min}", f"TP={self.table_in_parent_max}",
                f"EM={self.embed_max}", f"TD={self.table_digest_tokens}",
                f"MC={self.min_child_tokens}", f"KR={int(self.keep_references)}",
                f"WM={int(self.strip_watermarks)}"]


@dataclass
class FileContext:
    book: str
    rel_file: str
    chapter: str
    manifest: dict | None = None
    file_kind: str = ""          # manifest files[].kind
    chapter_number: int | None = None
    source_hash: str = ""
    mtime: float = 0.0
    source: str = ""             # citation unit (guideline sub-folder for _Guidelines)
    manifest_rel: str = ""       # path relative to the manifest's folder

    def __post_init__(self):
        if not self.source:
            self.source = P.source_key(self.book, self.rel_file)


@dataclass
class Stats:
    watermarks: int = 0
    reference_parents_dropped: int = 0
    reference_children_dropped: int = 0
    tables: int = 0
    tables_digest_embedded: int = 0
    short_children_dropped: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------

def _piece(lines, a, b, text, kind="prose", **kw) -> dict:
    ps, pe = P.page_span(lines, a, b)
    cs, ce = P.char_span(lines, a, b)
    d = {"kind": kind, "text": text, "ls": a, "le": b, "ps": ps, "pe": pe, "cs": cs, "ce": ce}
    d.update(kw)
    return d


def _split_long_text(text: str, max_tokens: int, tok: Tok) -> list[str]:
    if tok.count(text) <= max_tokens:
        return [text]
    out, cur = [], ""
    for s in _SENT_SPLIT.split(text):
        if not s:
            continue
        if tok.count(s) > max_tokens:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(tok.window(s, max_tokens))
            continue
        cand = (cur + " " + s) if cur else s
        if cur and tok.count(cand) > max_tokens:
            out.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def build_pieces(lines: list[dict], a: int, b: int, units_by_start: dict, in_table: list,
                 settings: Settings, tok: Tok, stats: Stats) -> list[dict]:
    """Cut lines[a:b] into pieces: prose runs of ≤ child_target tokens (split
    at lines, then sentences, then token windows) and whole tables."""
    pieces: list[dict] = []
    i = a
    run: list[int] = []
    block_id = 0

    def flush_run():
        nonlocal run, block_id
        if not run:
            return
        cur: list[int] = []
        cur_tok = 0
        for li in run:
            txt = lines[li]["text"]
            if settings.strip_watermarks:
                txt, nw = P.strip_watermarks(txt)
                stats.watermarks += nw
            t = tok.count(txt)
            if t > settings.child_target:
                if cur:
                    pieces.append(_piece(lines, cur[0], cur[-1] + 1, _join(lines, cur, settings, stats),
                                         block=block_id))
                    cur, cur_tok = [], 0
                for part in _split_long_text(txt, settings.child_target, tok):
                    pieces.append(_piece(lines, li, li + 1, part, block=block_id))
                continue
            if cur and cur_tok + t > settings.child_target:
                pieces.append(_piece(lines, cur[0], cur[-1] + 1, _join(lines, cur, settings, stats),
                                     block=block_id))
                cur, cur_tok = [], 0
            cur.append(li)
            cur_tok += t
        if cur:
            pieces.append(_piece(lines, cur[0], cur[-1] + 1, _join(lines, cur, settings, stats),
                                 block=block_id))
        run = []
        block_id += 1

    while i < b:
        if i in units_by_start:
            flush_run()
            u = units_by_start[i]
            pieces.append(_piece(lines, i, u["end"], T.unit_text(lines, u), kind="table", unit=u,
                                 block=block_id))
            block_id += 1
            i = u["end"]
            continue
        if in_table[i]:
            i += 1
            continue
        ln = lines[i]
        if ln["marker"] or not ln["text"].strip():
            flush_run()
            i += 1
            continue
        if _is_noise_line(ln["text"]):
            i += 1
            continue
        run.append(i)
        i += 1
    flush_run()
    return [p for p in pieces if p["text"].strip()]


_COMMENT_LINE = re.compile(r"^\s*<!--.*-->\s*$")
_HEADER_LINE = re.compile(r"^(?:#\s+\S|Source:\s)")


def _is_noise_line(text: str) -> bool:
    """Converter bookkeeping that is not book text: the file-title ``# `` line,
    the ``Source: <local path>`` line, and HTML comments other than figure
    ``REF`` markers (ligature repairs, rejected pseudo-tables, BOOK/SOURCE tags).
    Table flag comments live inside table units and are kept there."""
    if _COMMENT_LINE.match(text):
        return "REF:" not in text
    return bool(_HEADER_LINE.match(text))


def _join(lines, idxs, settings, stats) -> str:
    out = []
    for li in idxs:
        t = lines[li]["text"]
        if settings.strip_watermarks:
            t, _ = P.strip_watermarks(t)  # counted once in flush_run
        out.append(t)
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# sections → parents → children
# ---------------------------------------------------------------------------

def find_sections(lines: list[dict], in_table: list) -> list[dict]:
    """Heading-delimited sections. ``#`` (level 1) is the file title and does
    not open a section; ``##`` and ``###`` do (deep bookmarks write these)."""
    heads = []
    fenced = False
    for i, ln in enumerate(lines):
        if FENCE_RE.match(ln["text"]):
            fenced = not fenced
            continue
        if fenced or in_table[i]:
            continue
        m = HEADING_RE.match(ln["text"])
        if m:
            lvl = len(m.group(1))
            if lvl >= 2:
                heads.append((i, min(lvl, 3), m.group(2).strip()))
    sections = []
    h2 = h3 = ""
    starts = [0] + [h[0] for h in heads]
    labels = [("", "")]
    for _, lvl, title in heads:
        if lvl == 2:
            h2, h3 = title, ""
        else:
            h3 = title
        labels.append((h2, h3))
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        if e > s:
            sections.append({"start": s, "end": e, "h2": labels[k][0], "h3": labels[k][1]})
    return sections


def _path(h2: str, h3: str) -> list[str]:
    return [x for x in (h2, h3) if x]


def chunk_file(content: str, ctx: FileContext, settings: Settings | None = None,
               tok: Tok | None = None) -> tuple[list[dict], list[dict], Stats]:
    """(parents, children, stats) for one markdown file. Children include
    table units (``chunk_kind='table'``). Nothing here embeds anything."""
    settings = settings or Settings()
    tok = tok or qwen_tok()
    stats = Stats()
    m = ctx.manifest
    mrel = ctx.manifest_rel or ctx.rel_file
    entry = P.file_entry(m, mrel)

    # A chapter FILE named "References" with no manifest kind (guidelines have no
    # manifest; their "REFERENCES" is a plain text line, not a # heading, and wrapped
    # citations stay under the per-child density threshold). 2026-11-13: 1,119 such
    # rows across 31 files were indexed, three of them in one T1 top-5. Cut from the
    # heading line to EOF; body text before it (a split landing mid-page) is kept.
    # No heading line found -> keep the file: losing body text is worse than a leak.
    if (not settings.keep_references
            and (ctx.file_kind or entry.get("kind") or "") in ("", "body")
            and P.is_reference_title(P.title_from_filename(ctx.rel_file))):
        cut = P.reference_heading_offset(content)
        if cut is not None:
            content = content[:cut]
            stats.reference_parents_dropped += 1

    initial = int(entry.get("pdf_page_start") or 0)
    lines = P.line_table(content, initial_page=initial)
    units = T.find_table_units(lines)
    stats.tables = len(units)
    in_table = [False] * len(lines)
    units_by_start = {}
    for u in units:
        units_by_start[u["start"]] = u
        for x in range(u["start"], u["end"]):
            in_table[x] = True

    file_kind = ctx.file_kind or entry.get("kind") or ""
    if file_kind == "references" and not settings.keep_references:
        stats.reference_parents_dropped += 1
        return [], [], stats

    parents: list[dict] = []
    for sec in find_sections(lines, in_table):
        path = _path(sec["h2"], sec["h3"])
        kind = P.section_kind_for(m, mrel, path) or (file_kind if file_kind != "body" else "")
        pieces = build_pieces(lines, sec["start"], sec["end"], units_by_start, in_table,
                              settings, tok, stats)
        if not pieces:
            continue
        for grp in _pack_parent(pieces, settings, tok):
            parents.append({"pieces": grp, "path": path, "kind": kind})

    parents = _merge_small(parents, settings, tok)

    out_parents, out_children = [], []
    for ordinal, par in enumerate(parents):
        if par["kind"] == "references" and not settings.keep_references:
            stats.reference_parents_dropped += 1
            continue
        rec = _parent_record(par, ordinal, lines, ctx, settings, tok)
        if rec is None:
            continue
        kids = _children(par, rec, lines, ctx, settings, tok, stats)
        if not kids:
            continue
        out_parents.append(rec)
        out_children.extend(kids)
    return out_parents, out_children, stats


def _ptok(p: dict, settings: Settings, tok: Tok) -> int:
    if p["kind"] == "table":
        t = tok.count(p["text"])
        return t if t <= settings.table_in_parent_max else 20
    return tok.count(p["text"])


def _pack_parent(pieces, settings, tok):
    groups, cur, cur_tok = [], [], 0
    for p in pieces:
        t = _ptok(p, settings, tok)
        if cur and cur_tok + t > settings.parent_target:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        groups.append(cur)
    return groups


def _merge_small(parents, settings, tok):
    out = []
    for par in parents:
        if out:
            prev = out[-1]
            pt = sum(_ptok(p, settings, tok) for p in prev["pieces"])
            ct = sum(_ptok(p, settings, tok) for p in par["pieces"])
            same_h2 = prev["path"][:1] == par["path"][:1]
            if (pt < settings.parent_min and same_h2 and pt + ct <= settings.parent_max
                    and prev["kind"] == par["kind"]):
                common = [a for a, b in zip(prev["path"], par["path"]) if a == b]
                out[-1] = {"pieces": prev["pieces"] + par["pieces"], "path": common,
                           "kind": prev["kind"], "merged": True}
                continue
        out.append(par)
    return out


def _render(pieces, settings, tok) -> str:
    parts, prev = [], None
    for p in pieces:
        if p["kind"] == "table" and tok.count(p["text"]) > settings.table_in_parent_max:
            u = p["unit"]
            txt = f"[Table on PDF page {u['anchor_page']}" + (f": {u['caption']}" if u["caption"] else "") \
                  + " — returned as its own passage]"
        else:
            txt = p["text"]
        sep = "\n" if prev is not None and prev.get("block") == p.get("block") else "\n\n"
        parts.append(txt if prev is None else sep + txt)
        prev = p
    return "".join(parts).strip()


def _prefix(ctx: FileContext, path: list[str]) -> str:
    s = f"[{ctx.source} — {ctx.chapter}"
    if path:
        s += " > " + " > ".join(path)
    return s + "]\n"


def _page_fields(m, ps, pe):
    pps, ppe = P.printed_span(m, ps, pe)
    conf = P.PRINTED if (pps and ppe) else P.PDF_ONLY
    return {"pdf_page_start": ps, "pdf_page_end": pe, "page_start": ps, "page_end": pe,
            "printed_page_start": pps, "printed_page_end": ppe, "page_confidence": conf}


def _span(pieces):
    ps = [p["ps"] for p in pieces if p["ps"]]
    pe = [p["pe"] for p in pieces if p["pe"]]
    return (min(ps) if ps else 0, max(pe) if pe else 0,
            min(p["cs"] for p in pieces), max(p["ce"] for p in pieces),
            min(p["ls"] for p in pieces), max(p["le"] for p in pieces))


def _parent_record(par, ordinal, lines, ctx, settings, tok):
    text = _render(par["pieces"], settings, tok)
    if not text:
        return None
    ps, pe, cs, ce, ls, le = _span(par["pieces"])
    path = par["path"]
    pid = hashlib.md5("|".join([ctx.source, ctx.rel_file, " > ".join(path), str(ls),
                                str(ordinal)]).encode("utf-8", errors="replace")).hexdigest()
    rec = {
        "parent_id": pid, "text": text, "section_path_parts": path,
        "section_kind": par["kind"], "char_start": cs, "char_end": ce,
        "line_start": ls, "line_end": le,
        "heading_origin": ("H3" if len(path) == 2 else "H2" if path else "page_block")
        + ("_merged" if par.get("merged") else ""),
        "token_count": tok.count(text),
        "embed_text": _prefix(ctx, path) + text,
        "has_table": any(p["kind"] == "table" for p in par["pieces"]),
    }
    rec.update(_page_fields(ctx.manifest, ps, pe))
    return rec


def _children(par, prec, lines, ctx, settings, tok, stats):
    prefix = _prefix(ctx, par["path"])
    kids: list[dict] = []
    prose: list[list[dict]] = []
    cur, cur_tok = [], 0
    for p in par["pieces"]:
        if p["kind"] == "table":
            if cur:
                prose.append(cur)
                cur, cur_tok = [], 0
            kids.append(_table_child(p, prec, lines, ctx, settings, tok, prefix, stats))
            continue
        t = tok.count(p["text"])
        if cur and cur_tok + t > settings.child_target:
            prose.append(cur)
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        prose.append(cur)

    prev_tail = ""
    body_kids = []
    for grp in prose:
        core = _render(grp, settings, tok)
        if not settings.keep_references and P.looks_like_reference_list(core):
            stats.reference_children_dropped += 1
            prev_tail = ""
            continue
        kind = "references" if P.looks_like_reference_list(core) else ""
        text = (prev_tail + "\n" + core) if prev_tail else core
        ps, pe, cs, ce, ls, le = _span(grp)
        ntok = tok.count(prefix + text)
        if tok.count(core) < settings.min_child_tokens and len(prose) > 1:
            stats.short_children_dropped += 1
            continue
        rec = {
            "chunk_kind": "figure_caption" if "<!-- REF:" in core[:400] else "body",
            "text": prefix + text, "embed_text": prefix + text, "core_text": core,
            "parent_id": prec["parent_id"], "section_path_parts": par["path"],
            "section_kind": kind or prec["section_kind"], "char_start": cs, "char_end": ce,
            "line_start": ls, "line_end": le, "token_count": ntok,
            "heading_origin": prec["heading_origin"], "table_id": "", "table_review_flag": "", "table_embed_truncated": False,
        }
        rec.update(_page_fields(ctx.manifest, ps, pe))
        body_kids.append(rec)
        words = tok.window(core, max(1, settings.child_overlap))
        prev_tail = words[-1] if settings.child_overlap and words else ""
    kids.extend(body_kids)
    # Keep document order: tables were appended as met; sort by line.
    kids.sort(key=lambda k: (k["line_start"], k["chunk_kind"] != "table"))
    for i, k in enumerate(kids):
        k["chunk_idx"] = i
        k["n_siblings"] = len(kids)
        k["chunk_id"] = hashlib.md5(f"{ctx.source}|{ctx.rel_file}|{k['char_start']}|{k['char_end']}|"
                                    f"{k['chunk_kind']}|{k['core_text']}".encode("utf-8", errors="replace")).hexdigest()
    return kids


def _table_child(p, prec, lines, ctx, settings, tok, prefix, stats):
    u = p["unit"]
    full = p["text"]
    pages = u["pages"] or [p["ps"]]
    embed = prefix + full
    truncated = False
    if tok.count(embed) > settings.embed_max:
        embed = prefix + T.embed_digest(lines, u, settings.table_digest_tokens, tok.count)
        truncated = True
        stats.tables_digest_embedded += 1
    rec = {
        "chunk_kind": "table", "text": prefix + full, "embed_text": embed, "core_text": full,
        "parent_id": prec["parent_id"], "section_path_parts": prec["section_path_parts"],
        "section_kind": prec["section_kind"], "char_start": p["cs"], "char_end": p["ce"],
        "line_start": p["ls"], "line_end": p["le"], "token_count": tok.count(prefix + full),
        "heading_origin": prec["heading_origin"],
        "table_id": T.table_id(ctx.source, ctx.rel_file, u),
        "table_review_flag": u["review_flag"] or "", "table_embed_truncated": truncated,
        "table_caption": u["caption"],
    }
    rec.update(_page_fields(ctx.manifest, min(pages), max(pages)))
    return rec
