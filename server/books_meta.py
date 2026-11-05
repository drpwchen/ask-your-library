"""
Book metadata joined onto textbook results at QUERY time.

    {"_comment": "...",
     "Braddom_PM-and-R_7e_2021": {"tier": 1, "corpus_type": "textbook"},
     "Braddom_PM-and-R_6e_2021": {"status": "superseded", "superseded_by": "Braddom_PM-and-R_7e_2021"},
     "AHA-ASA_StrokeRehab_2016": {"corpus_type": "guideline", "status": "superseded",
                                  "superseded_by": "VADoD_StrokeRehab_2024"}}

Keys are the source's folder name. Guidelines all live under one `book`
(`_Guidelines`) in the index, so for them the key is the sub-folder taken from
the file path (`_Guidelines/<key>/full_text.md`).

Fields (all optional)
  edition        "7e"           inferred from `_7e_` in the folder name
  year           2021           inferred from the last `_YYYY` in the folder name
  tier           1..3           1 = primary reference; used only as a tie-break
  corpus_type    textbook | guideline | handbook | review | exam
                                inferred: guideline under _Guidelines, else textbook
  status         current | superseded | derivative | low_trust
  superseded_by  folder name of the replacement

Nothing here needs a reindex: the join reads `book`/`file`, which every chunk
already carries. A rebuild can later copy these fields into the index itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GUIDELINE_BOOK = "_Guidelines"
HIDDEN_STATUSES = {"superseded", "derivative"}
TIE_EPSILON = 0.01  # similarities closer than this count as a tie

_cache: dict = {"path": None, "mtime": None, "data": {}}


def load(path: Path | None) -> dict[str, dict]:
    """Read books.json (re-read when it changes). Missing file → {}."""
    if path is None or not Path(path).exists():
        return {}
    p = Path(path)
    mtime = p.stat().st_mtime
    if _cache["path"] == str(p) and _cache["mtime"] == mtime:
        return _cache["data"]
    raw = json.loads(p.read_text(encoding="utf-8"))
    data = {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}
    _cache.update(path=str(p), mtime=mtime, data=data)
    return data


def book_key(book: str, file: str) -> str:
    if book == GUIDELINE_BOOK:
        parts = (file or "").replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] == GUIDELINE_BOOK:
            return parts[1]
    return book


def infer(key: str, book: str) -> dict:
    m = re.search(r"_(\d+)e_", key)
    years = re.findall(r"(?:^|_)((?:19|20)\d{2})(?=_|$|[a-z])", key)
    return {
        "edition": f"{m.group(1)}e" if m else "",
        "year": int(years[-1]) if years else None,
        "tier": None,
        "corpus_type": "guideline" if book == GUIDELINE_BOOK else "textbook",
        "status": "current",
        "superseded_by": "",
    }


def lookup(book: str, file: str, meta: dict[str, dict]) -> dict:
    key = book_key(book, file)
    out = infer(key, book)
    out.update({k: v for k, v in meta.get(key, {}).items() if v is not None})
    out["source_key"] = key
    return out


def annotate(results: list[dict], meta: dict[str, dict]) -> list[dict]:
    for r in results:
        r["meta"] = lookup(r.get("book", ""), r.get("file", ""), meta)
    return results


def visible(r: dict, include_superseded: bool, corpus_type: str | None) -> bool:
    m = r["meta"]
    if not include_superseded and m.get("status") in HIDDEN_STATUSES:
        return False
    if corpus_type and m.get("corpus_type") != corpus_type:
        return False
    return True


def prefilter(corpus_type: str | None, meta: dict[str, dict]) -> str | None:
    """Coarse SQL prefilter so a corpus_type filter does not starve the
    candidate list. Exact filtering still happens in `visible()`."""
    if not corpus_type:
        return None
    tagged = [k for k, v in meta.items() if v.get("corpus_type") == corpus_type]
    quoted = ", ".join("'" + k.replace("'", "''") + "'" for k in tagged)
    if corpus_type == "guideline":
        return f"book = '{GUIDELINE_BOOK}'" + (f" OR book IN ({quoted})" if quoted else "")
    base = f"book != '{GUIDELINE_BOOK}'"
    return base


def tie_break(results: list[dict]) -> list[dict]:
    """Within a similarity tie, prefer the better tier, then the newer year.

    Ordering by similarity is kept: a lower-tier source never overtakes a
    clearly better match, it only wins against an equally good one.
    """
    def key(r):
        m = r.get("meta", {})
        tier = m.get("tier") or 9
        year = m.get("year") or 0
        return (-round(float(r.get("similarity", 0)) / TIE_EPSILON), tier, -year)
    return sorted(results, key=key)


def describe(m: dict) -> str:
    """Compact one-line provenance for the Markdown output ('' when nothing to say)."""
    bits = []
    if m.get("corpus_type") and m["corpus_type"] != "textbook":
        bits.append(m["corpus_type"])
    if m.get("edition"):
        bits.append(m["edition"])
    if m.get("year"):
        bits.append(str(m["year"]))
    if m.get("tier"):
        bits.append(f"tier {m['tier']}")
    if m.get("status") and m["status"] != "current":
        s = m["status"]
        if m.get("superseded_by"):
            s += f" → {m['superseded_by']}"
        bits.append(s)
    return " · ".join(bits)
