"""
Claim-level end-to-end eval: retrieval → fixed answer prompt → per-claim grading.

hit@3 says the right passage was retrieved; it does not say the answer written
from it is right. This step writes an answer from exactly what a system
retrieved (saved run, so the retrieval is frozen), then grades it:

  per claim   supported by a cited source? numbers/units/population/frequency
              consistent with the source? citation points at the right
              source, edition and page?
  per answer  correct vs the gold answer; abstained when it should (T7) and
              not when it should not; T1 critical error = a wrong dose, unit,
              route or frequency (zero tolerance, reported separately).

    python eval/e2e.py --set gold.jsonl --run runs/mcp.json --out runs/mcp.e2e.json
    python eval/e2e.py --summarize runs/mcp.e2e.json

Answer and grade prompts are fixed in this file; changing either makes runs
incomparable, so both carry a version string that is written into the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
from harness import load_jsonl  # noqa: E402

ANSWER_VERSION = "answer-v1"
GRADE_VERSION = "grade-v1"

ANSWER_SYSTEM = (
    "You answer a physician's clinical question using ONLY the numbered sources provided. "
    "Every factual sentence must cite its source as [S#]. Quote numbers (doses, units, "
    "frequencies, cut-offs) exactly as the source gives them, with the population they apply to. "
    "If the sources disagree, say so and cite both. If the sources do not contain the answer, "
    "reply with the single line NOT_FOUND followed by one sentence on what is missing. "
    "Answer in the language of the question; keep medical terms in English."
)

GRADE_SYSTEM = (
    "You grade an answer written from retrieved sources against a reference answer for a "
    "clinical retrieval benchmark. Be strict about numbers: dose, unit, route, frequency, "
    "population. A claim is 'supported' only if the cited source text actually states it. "
    "Reply with JSON only."
)


def _src_label(r: dict) -> str:
    if r.get("corpus") == "vault":
        return f"note: {r.get('note')} › {r.get('section', '')}"
    pg = f", p.{r.get('page_start')}-{r.get('page_end')}" if r.get("page_start") else ", page unknown"
    return f"{r.get('corpus')}: {r.get('book')} — {r.get('file')}{pg}"


def select_sources(ans: dict, k: int) -> list[dict]:
    """Top-k of every corpus the system searched, in corpus order."""
    out = []
    for corpus in ("vault", "textbook", "lecture"):
        lst = sorted((r for r in ans["results"] if r["corpus"] == corpus), key=lambda r: r["rank"])
        out.extend(lst[:k])
    return out


def answer(q: dict, sources: list[dict], model: str) -> str:
    blocks = [f"[S{i}] ({_src_label(r)})\n{(r.get('text') or '').strip()[:3000]}"
              for i, r in enumerate(sources, 1)]
    prompt = f"QUESTION: {q['query']}\n\nSOURCES:\n\n" + "\n\n".join(blocks)
    return llm.ask(prompt, ANSWER_SYSTEM, model=model)


def grade(q: dict, sources: list[dict], reply: str, model: str) -> dict:
    gold = q.get("answer") or {}
    rel = [{k: g.get(k) for k in ("corpus", "book", "file", "note", "page_start", "page_end", "span")}
           for g in q.get("relevant", [])]
    src = "\n\n".join(f"[S{i}] ({_src_label(r)})\n{(r.get('text') or '').strip()[:3000]}"
                      for i, r in enumerate(sources, 1))
    prompt = (
        f"QUESTION ({q.get('type', '')}): {q['query']}\n\n"
        f"REFERENCE ANSWER: {gold.get('text') or '(none — the correct behaviour is to say it is not found)' if q.get('expect_abstain') else gold.get('text') or '(not given; judge support only)'}\n"
        f"CRITICAL FIELDS: {', '.join(gold.get('critical', [])) or '-'}\n"
        f"GOLD SOURCES: {json.dumps(rel, ensure_ascii=False)}\n\n"
        f"SOURCES SHOWN TO THE ANSWERER:\n{src}\n\n"
        f"ANSWER TO GRADE:\n{reply}\n\n"
        "Return JSON: {\"abstained\": bool, \"answer_correct\": bool|null, "
        "\"critical_error\": bool, \"critical_detail\": str, "
        "\"citation_correct\": bool|null, \"edition_correct\": bool|null, "
        "\"claims\": [{\"claim\": str, \"cites\": [\"S1\"], \"supported\": bool, "
        "\"numbers_ok\": bool|null, \"citation_ok\": bool|null}]}\n"
        "answer_correct: matches the reference answer (null if no reference). "
        "critical_error: true if any critical field (dose/unit/route/frequency) contradicts the reference. "
        "citation_correct: cited sources include a gold source (null if no gold source). "
        "edition_correct: no claim relies on a superseded edition/guideline when a current one was shown."
    )
    return llm.ask_json(prompt, GRADE_SYSTEM, model=model)


def run_one(q: dict, ans: dict, k: int, model: str) -> dict:
    srcs = select_sources(ans, k)
    t0 = time.time()
    reply = answer(q, srcs, model)
    g = grade(q, srcs, reply, model)
    return {"qid": q["id"], "type": q.get("type"), "n_sources": len(srcs),
            "reply": reply, "grade": g, "seconds": round(time.time() - t0, 1)}


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r.get("type") or "?"].append(r)

    def s(rs):
        g = [r["grade"] for r in rs if isinstance(r.get("grade"), dict)]
        claims = [c for x in g for c in x.get("claims", []) if isinstance(c, dict)]

        def rate(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None
        return {
            "n": len(rs),
            "answer_correct": rate([x.get("answer_correct") for x in g]),
            "abstained": rate([x.get("abstained") for x in g]),
            "critical_errors": sum(1 for x in g if x.get("critical_error")),
            "citation_correct": rate([x.get("citation_correct") for x in g]),
            "edition_correct": rate([x.get("edition_correct") for x in g]),
            "claims": len(claims),
            "claim_supported": rate([c.get("supported") for c in claims]),
            "claim_numbers_ok": rate([c.get("numbers_ok") for c in claims]),
            "claim_citation_ok": rate([c.get("citation_ok") for c in claims]),
        }
    return {"all": s([r for r in rows if r.get("type") != "T7"]),
            "by_type": {t: s(rs) for t, rs in sorted(by.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set")
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=5, help="sources per searched corpus")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", metavar="E2E_JSON")
    a = ap.parse_args(argv)

    if a.summarize:
        d = json.loads(Path(a.summarize).read_text(encoding="utf-8"))
        print(json.dumps(summarize(d["rows"]), ensure_ascii=False, indent=1))
        return 0

    qs = load_jsonl(Path(a.set))
    if a.limit:
        qs = qs[: a.limit]
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, q, run["answers"][q["id"]], a.k, a.model): q["id"]
                for q in qs if q["id"] in run["answers"]}
        for f in futs:
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({"qid": futs[f], "error": f"{type(e).__name__}: {e}"})
    out = {"run": a.run, "label": run.get("label"), "model": a.model, "k": a.k,
           "answer_version": ANSWER_VERSION, "grade_version": GRADE_VERSION,
           "rows": rows, "summary": summarize([r for r in rows if "grade" in r])}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"]["all"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
