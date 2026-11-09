"""
Vault index profile "v3" — pure helpers used by ``vault_indexer_v2.py --profile v3``.

The default profile ("v2") is untouched, so the nightly incremental run of an
existing index behaves exactly as before. v3 is for a NEW database (Phase 4
rebuild) and changes four things:

* **Real YAML frontmatter.** The v2 parser understood ``key: value`` and simple
  lists only; quoted values, nested maps and multi-line strings fell apart.
  ``yaml.safe_load`` reads them; a broken block falls back to the old parser.
* **Exclusions.** Backup copies (``*.bak*.md``), template folders and any folder
  named in ``VAULT_SEARCH_V3_EXCLUDE_FOLDERS`` are not indexed.
* **Hard length caps.** A note with no blank lines used to become one child of
  tens of thousands of tokens, of which the embedder saw only the first 4,096.
  Children are cut to ``CHILD_HARD_MAX`` and parents to ``PARENT_HARD_MAX``.
* **A ``scope`` column**, written from the scope map at index time (queries
  still filter on ``folder`` so a changed map needs no reindex).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

CHILD_HARD_MAX = 1024
PARENT_HARD_MAX = 2000

BAK_RE = re.compile(r"\.bak\b|\.bak-|\.backup\b|~$", re.IGNORECASE)
TEMPLATE_FOLDER_RE = re.compile(r"(?:^|\d|_|\s)templates?$", re.IGNORECASE)


def extra_excluded_folders() -> set[str]:
    raw = os.environ.get("VAULT_SEARCH_V3_EXCLUDE_FOLDERS", "")
    return {s.strip().casefold() for s in raw.split(",") if s.strip()}


def excluded(rel_path: str, extra: set[str] | None = None) -> str:
    """Reason a vault file is not indexed under v3, or '' to index it."""
    extra = extra_excluded_folders() if extra is None else extra
    parts = Path(rel_path.replace("\\", "/")).parts
    name = parts[-1] if parts else rel_path
    if BAK_RE.search(Path(name).stem) or ".bak" in name.casefold():
        return "backup"
    for folder in parts[:-1]:
        if folder.casefold() in extra:
            return "excluded_folder"
        if TEMPLATE_FOLDER_RE.search(folder):
            return "templates"
    return ""


def parse_frontmatter_yaml(content: str, fallback) -> tuple[dict, str]:
    """(meta, body) with a real YAML parser; ``fallback`` on malformed YAML."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end]
    body = content[end + 4:]
    try:
        import yaml
        meta = yaml.safe_load(fm_text)
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            return fallback(content)
    except Exception:
        return fallback(content)
    return meta, body.strip()


def as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    return [s.strip() for s in str(v).split(",") if s.strip()]


def cap_text(text: str, max_tokens: int, tok_len, window) -> list[str]:
    """Cut text into pieces of ≤ max_tokens: by lines, then by token windows."""
    if tok_len(text) <= max_tokens:
        return [text]
    out, cur, cur_t = [], [], 0
    for line in text.split("\n"):
        t = tok_len(line)
        if t > max_tokens:
            if cur:
                out.append("\n".join(cur))
                cur, cur_t = [], 0
            out.extend(window(line, max_tokens))
            continue
        if cur and cur_t + t > max_tokens:
            out.append("\n".join(cur))
            cur, cur_t = [], 0
        cur.append(line)
        cur_t += t
    if cur:
        out.append("\n".join(cur))
    return [o for o in out if o.strip()]


def cap_children(children: list[dict], max_tokens: int, tok_len, window) -> list[dict]:
    """Split any child above the cap; the heading prefix line is repeated."""
    out: list[dict] = []
    for c in children:
        if c["token_count"] <= max_tokens:
            out.append(c)
            continue
        text = c["text"]
        prefix = ""
        if text.startswith("[") and "\n" in text:
            first, rest = text.split("\n", 1)
            if first.endswith("]"):
                prefix, text = first + "\n", rest
        budget = max(64, max_tokens - tok_len(prefix))
        for piece in cap_text(text, budget, tok_len, window):
            out.append({**c, "text": prefix + piece, "token_count": tok_len(prefix + piece)})
    for i, c in enumerate(out):
        c["chunk_idx"] = i
        c["n_siblings"] = len(out)
    return out


def cap_parents(parents: list[dict], max_tokens: int, tok_len, window) -> list[dict]:
    """Split any parent above the cap into numbered parts with distinct ids."""
    import hashlib
    out: list[dict] = []
    for p in parents:
        if p["token_count"] <= max_tokens:
            out.append(p)
            continue
        for k, piece in enumerate(cap_text(p["text"], max_tokens, tok_len, window)):
            pid = hashlib.md5(f"{p['parent_id']}|part{k}".encode()).hexdigest()
            out.append({**p, "parent_id": pid, "text": piece, "token_count": tok_len(piece),
                        "heading_origin": p["heading_origin"] + "_capped"})
    return out
