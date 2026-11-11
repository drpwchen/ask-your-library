"""
vault-search eval — relevance matching, per-question metrics, paired statistics.

Stdlib only, so it runs (and is tested) on a machine with no index at all.

Data shapes
-----------
A *question* (one line of a question-set .jsonl):

    {"id": "T1-001", "type": "T1", "tier": "gold", "corpus": "textbook",
     "query": "...", "cluster": "BookFolder/ch12",
     "relevant": [{"corpus": "textbook", "book": "BookFolder", "file": "BookFolder/ch12.md",
                   "page_start": 341, "page_end": 342, "span": "...", "grade": 2}],
     "contaminants": [{"corpus": "textbook", "book": "OldEdition"}],
     "answer": {"text": "...", "critical": ["dose", "unit", "route"]},
     "expect_abstain": false}

`corpus` names the ONE result list the question is scored on. The three corpora
are never merged into one ranking: their similarity scores come from different
models and tables and are not comparable.

A *result* (one item of a system's ranked list for one corpus):

    {"corpus": "textbook", "rank": 1, "book": "...", "file": "...",
     "page_start": 341, "page_end": 342, "sim": 0.71, "text": "..."}
    {"corpus": "vault", "rank": 1, "note": "...", "folder": "...", "section": "...", "sim": 0.66}

A *judgment* is a human/LLM verdict on a result the question's gold list did not
anticipate: {"qid": ..., "key": result_key(r), "rel": 0|1|2, "conflict": bool}.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable

PAGE_WINDOW = 2  # ±pages for the page-window hit


# --- keys & matching ---------------------------------------------------------

def _norm(s) -> str:
    return " ".join(str(s or "").replace("\\", "/").split()).casefold()


def result_key(r: dict) -> str:
    """Stable identity of a result, used to store judgments."""
    c = r.get("corpus", "")
    if c == "vault":
        return f"vault::{_norm(r.get('note'))}"
    return f"{c}::{_norm(r.get('book'))}::{_norm(r.get('file'))}::{int(r.get('page_start') or 0)}"


def doc_match(r: dict, g: dict) -> bool:
    """Same document as the gold item (note / book+file / book when no file given)."""
    if r.get("corpus") != g.get("corpus"):
        return False
    if g["corpus"] == "vault":
        return _norm(r.get("note")) == _norm(g.get("note"))
    if g.get("book") and _norm(r.get("book")) != _norm(g.get("book")):
        return False
    files = g.get("files") or ([g["file"]] if g.get("file") else [])
    if not files:
        return bool(g.get("book"))
    rf = _norm(r.get("file"))
    return any(rf == _norm(f) or rf.endswith("/" + _norm(f)) or _norm(f).endswith("/" + rf)
               for f in files)


def page_match(r: dict, g: dict, window: int = PAGE_WINDOW) -> bool:
    """doc_match AND the result's page range overlaps the gold range ±window."""
    if not doc_match(r, g):
        return False
    gs, ge = int(g.get("page_start") or 0), int(g.get("page_end") or g.get("page_start") or 0)
    rs, re_ = int(r.get("page_start") or 0), int(r.get("page_end") or r.get("page_start") or 0)
    if not (gs and rs):
        return False
    return rs <= ge + window and re_ >= gs - window


def is_contaminant(r: dict, q: dict, global_books: set[str]) -> bool:
    """Global entries are book folder names; "vault:<folder>" marks a vault folder."""
    if r.get("corpus") == "vault":
        if f"vault:{_norm(r.get('folder'))}" in global_books:
            return True
    elif _norm(r.get("book")) in global_books:
        return True
    return any(doc_match(r, c) for c in q.get("contaminants", []))


def build_judgment_index(judgments: Iterable[dict]) -> dict[tuple[str, str], dict]:
    idx = {}
    for j in judgments:
        idx[(j["qid"], j["key"])] = j  # last verdict wins
    return idx


def relevance(r: dict, q: dict, jidx: dict) -> int | None:
    """Graded relevance of one result. None = nobody has judged it yet."""
    best = 0
    for g in q.get("relevant", []):
        if doc_match(r, g):
            best = max(best, int(g.get("grade", 1)))
    if best:
        return best
    j = jidx.get((q["id"], result_key(r)))
    if j is None:
        return None
    return int(j.get("rel", 0))


# --- per-question metrics ----------------------------------------------------

def score_question(q: dict, results: list[dict], *, jidx: dict | None = None,
                   global_contaminants: set[str] | None = None, k_hit: int = 3,
                   k_recall: int = 10) -> dict:
    """Metrics for one question on its target corpus list.

    Unjudged results count as NOT relevant here, and the count of unjudged
    results inside the top-k is reported next to the score, so a run whose
    pool was never completed cannot pass itself off as a measured one.
    """
    jidx = jidx or {}
    gcont = {_norm(b) for b in (global_contaminants or set())}
    lst = sorted((r for r in results if r.get("corpus") == q["corpus"]),
                 key=lambda r: r.get("rank", 0))
    rels = [relevance(r, q, jidx) for r in lst]
    out = {
        "qid": q["id"], "type": q.get("type", ""), "cluster": q.get("cluster") or q["id"],
        "n_results": len(lst), "zero": len(lst) == 0,
        "unjudged_at_hit": sum(1 for x in rels[:k_hit] if x is None),
        "unjudged_at_recall": sum(1 for x in rels[:k_recall] if x is None),
        "top_sim": lst[0].get("sim") if lst else None,
    }
    if q.get("expect_abstain"):
        out["abstain_question"] = True
        return out

    def rel_ok(x):
        return (x or 0) >= 1

    first = next((i for i, x in enumerate(rels[:k_recall]) if rel_ok(x)), None)
    out["hit@1"] = int(bool(rels) and rel_ok(rels[0]))
    out[f"hit@{k_hit}"] = int(any(rel_ok(x) for x in rels[:k_hit]))
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)

    golds = q.get("relevant", [])
    if golds:
        found = sum(1 for g in golds if any(doc_match(r, g) for r in lst[:k_recall]))
        out[f"recall@{k_recall}"] = found / len(golds)
    paged = [g for g in golds if g.get("page_start")]
    if paged:
        out[f"page_hit@{k_hit}"] = int(any(page_match(r, g) for r in lst[:k_hit] for g in paged))

    conflict = any(
        (jidx.get((q["id"], result_key(r))) or {}).get("conflict") for r in lst[:k_hit]
    )
    out[f"contamination@{k_hit}"] = int(
        conflict or any(is_contaminant(r, q, gcont) for r in lst[:k_hit])
    )
    return out


# --- aggregation -------------------------------------------------------------

METRIC_KEYS = ("hit@1", "hit@3", "mrr", "recall@10", "page_hit@3", "contamination@3", "zero")


def aggregate(rows: list[dict], latencies: list[float] | None = None) -> dict:
    """Mean of every metric present, overall and per question type."""

    def summarise(rs):
        d = {"n": len(rs)}
        for k in METRIC_KEYS:
            vals = [float(r[k]) for r in rs if k in r]
            if vals:
                d[k] = round(sum(vals) / len(vals), 4)
                d[f"{k}_n"] = len(vals)
        d["unjudged_at_hit"] = sum(r.get("unjudged_at_hit", 0) for r in rs)
        return d

    out = {"all": summarise([r for r in rows if not r.get("abstain_question")])}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("type") or "?"].append(r)
    out["by_type"] = {t: summarise(rs) for t, rs in sorted(by_type.items())}
    if latencies:
        s = sorted(latencies)
        out["latency_ms"] = {"p50": _pct(s, 50), "p90": _pct(s, 90), "n": len(s)}
    return out


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    i = (len(sorted_vals) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo), 1)


# --- paired statistics -------------------------------------------------------

def mcnemar_exact(a: list[int], b: list[int]) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes (e.g. hit@3).

    Only the discordant pairs carry information: b10 = A hit / B miss,
    b01 = A miss / B hit. Under H0 each discordant pair is a fair coin.
    """
    assert len(a) == len(b)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b10 + b01
    if n == 0:
        return {"b10": 0, "b01": 0, "p": 1.0}
    k = min(b10, b01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b10": b10, "b01": b01, "p": round(min(1.0, 2 * tail), 6)}


def cluster_bootstrap_diff(a: list[float], b: list[float], clusters: list[str],
                           n_boot: int = 5000, seed: int = 20261106) -> dict:
    """Paired bootstrap of mean(b - a), resampling whole clusters.

    Silver questions bunch up in a few books/chapters; resampling single
    questions would treat ten questions from one chapter as ten independent
    observations and give a falsely narrow interval.
    """
    assert len(a) == len(b) == len(clusters)
    groups: dict[str, list[float]] = defaultdict(list)
    for x, y, c in zip(a, b, clusters):
        groups[c].append(y - x)
    keys = list(groups)
    if not keys:
        return {"diff": 0.0, "ci95": [0.0, 0.0], "p": 1.0, "clusters": 0}
    observed = sum(sum(v) for v in groups.values()) / sum(len(v) for v in groups.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        tot, cnt = 0.0, 0
        for _ in keys:
            v = groups[keys[rng.randrange(len(keys))]]
            tot += sum(v)
            cnt += len(v)
        stats.append(tot / cnt if cnt else 0.0)
    stats.sort()
    lo, hi = stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]
    # two-sided p: how often the bootstrap distribution crosses zero
    below = sum(1 for s in stats if s <= 0) / n_boot
    above = sum(1 for s in stats if s >= 0) / n_boot
    return {"diff": round(observed, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p": round(min(1.0, 2 * min(below, above)), 4), "clusters": len(keys)}


def cluster_permutation_diff(a: list[float], b: list[float], clusters: list[str],
                             n_perm: int = 100_000, seed: int = 20261111) -> dict:
    """Paired cluster sign-flip permutation test of mean(b - a).

    Under H0 (A and B exchangeable) the sign of each cluster's summed
    difference is a fair coin; all questions of a cluster flip together.
    With few clusters (~20) this is the primary test: the cluster bootstrap's
    percentile p runs optimistic there (I9 pilot: 0.008 vs 0.022). Exact
    enumeration up to 16 clusters, Monte Carlo above. Works for binary
    outcomes (hit@3) and continuous ones (MRR) alike.
    """
    assert len(a) == len(b) == len(clusters)
    sums: dict[str, float] = defaultdict(float)
    for x, y, c in zip(a, b, clusters):
        sums[c] += y - x
    v = [d for d in sums.values()]
    n = len(a)
    if not n:
        return {"diff": 0.0, "p": 1.0, "clusters": 0, "method": "none"}
    obs = abs(sum(v))
    eps = 1e-12
    if len(v) <= 16:
        ge = 0
        total = 1 << len(v)
        for mask in range(total):
            s = sum(-d if (mask >> i) & 1 else d for i, d in enumerate(v))
            if abs(s) >= obs - eps:
                ge += 1
        p, method = ge / total, "exact"
    else:
        rng = random.Random(seed)
        ge = 0
        for _ in range(n_perm):
            s = sum(d if rng.random() < 0.5 else -d for d in v)
            if abs(s) >= obs - eps:
                ge += 1
        p, method = (ge + 1) / (n_perm + 1), f"monte_carlo_{n_perm}"
    return {"diff": round(sum(v) / n, 4), "p": round(min(1.0, p), 4),
            "clusters": len(v), "method": method}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a pre-registered family of tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = round(running, 4)
    return out
