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

Each finished question is appended to <out>.partial.jsonl as soon as it is done.
Re-running the same command skips those questions, so a stop (Ctrl-C, killed task,
usage limit) loses only the questions in flight. A usage-limit reply stops the run
instead of being graded as a wrong answer. --fresh discards the checkpoint.

Answer and grade prompts are fixed in this file; changing either makes runs
incomparable, so both carry a version string that is written into the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite <out>.partial.jsonl")
    a = ap.parse_args(argv)

    if a.summarize:
        d = json.loads(Path(a.summarize).read_text(encoding="utf-8"))
        print(json.dumps(summarize(d["rows"]), ensure_ascii=False, indent=1))
        return 0

    qs = load_jsonl(Path(a.set))
    if a.limit:
        qs = qs[: a.limit]
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    header = {"run": a.run, "model": a.model, "k": a.k,
              "answer_version": ANSWER_VERSION, "grade_version": GRADE_VERSION}
    ckpt = Path(a.out + ".partial.jsonl")
    done = load_checkpoint(ckpt, header) if not a.fresh else {}
    if a.fresh or not ckpt.exists():
        ckpt.write_text(json.dumps({"header": header}, ensure_ascii=False) + "\n", encoding="utf-8")
    todo = [q for q in qs if q["id"] in run["answers"] and q["id"] not in done]
    print(f"checkpoint: {len(done)} done, {len(todo)} to run", file=sys.stderr)

    lock = threading.Lock()
    stop = threading.Event()
    failed = {}
    quota_msg = []

    def work(q):
        if stop.is_set():
            return None
        try:
            return run_one(q, run["answers"][q["id"]], a.k, a.model)
        except llm.QuotaExhausted:
            stop.set()  # here, not in the collecting loop: idle workers check it next
            raise

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, q): q["id"] for q in todo}
        for f in as_completed(futs):
            qid = futs[f]
            try:
                row = f.result()
            except llm.QuotaExhausted as e:
                if not quota_msg:
                    quota_msg.append(str(e))
                    print(f"usage limit hit, stopping: {e}", file=sys.stderr)
                continue
            except Exception as e:
                failed[qid] = {"qid": qid, "error": f"{type(e).__name__}: {e}"}
                continue
            if row is None:
                continue
            done[qid] = row
            with lock, ckpt.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if stop.is_set():
        left = len([q for q in qs if q["id"] in run["answers"] and q["id"] not in done])
        print(f"stopped on usage limit: {len(done)} saved in {ckpt}, {left} left; "
              "re-run the same command to continue", file=sys.stderr)
        return 3
    order = {q["id"]: i for i, q in enumerate(qs)}
    rows = sorted(list(done.values()) + list(failed.values()),
                  key=lambda r: order.get(r["qid"], 10**9))
    out = {"run": a.run, "label": run.get("label"), "model": a.model, "k": a.k,
           "answer_version": ANSWER_VERSION, "grade_version": GRADE_VERSION,
           "rows": rows, "summary": summarize([r for r in rows if "grade" in r])}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"]["all"], ensure_ascii=False))
    if failed:
        print(f"{len(failed)} questions errored; they are not in the checkpoint, so re-running "
              "the same command retries only those", file=sys.stderr)
    else:
        ckpt.unlink()
    return 0


def load_checkpoint(path: Path, header: dict) -> dict:
    """Finished rows from an earlier interrupted run with the same settings."""
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return {}
    first = json.loads(lines[0])
    if first.get("header") != header:
        raise SystemExit(f"{path} was written with different settings "
                         f"({first.get('header')}); use --fresh or another --out")
    done = {}
    for ln in lines[1:]:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:  # a line cut off by a hard kill
            continue
        if "grade" in r:
            done[r["qid"]] = r
    return done


if __name__ == "__main__":
    sys.exit(main())
