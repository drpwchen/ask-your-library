"""
Judge results nobody has labelled yet (TREC-style pool completion).

    python eval/harness.py unjudged --set gold.jsonl --run runs/a.json runs/b.json --out unjudged.jsonl
    python eval/judge.py --set gold.jsonl --unjudged unjudged.jsonl --out judgments.jsonl

One model call per question: the question, what the answer should contain (if
the gold item says), and every unjudged passage for it. The judge returns a
grade per passage:

    2  directly answers the question
    1  relevant and on-topic, but not sufficient on its own
    0  not relevant
    conflict: true  — states something that CONTRADICTS the gold answer
                      (old edition, different dose); counted as contamination

Judgments are appended; an existing (qid, key) pair is never re-judged unless
--rejudge is given. Each line records the judge model, so a later reader knows
which verdicts came from a model and which from a person ("judge": "human").
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
from harness import load_jsonl  # noqa: E402

SYSTEM = (
    "You are a strict relevance assessor for a clinical retrieval benchmark. "
    "You see one question and several retrieved passages. Judge each passage ONLY "
    "on its own text. Grades: 2 = the passage itself contains the answer; "
    "1 = on-topic and useful but does not contain the answer; 0 = not relevant. "
    "Set conflict=true only when the passage states a fact that contradicts the "
    "expected answer (e.g. a different dose, an outdated recommendation). "
    "Reply with JSON only."
)


def _label(r: dict) -> str:
    if r.get("corpus") == "vault":
        return f"[note] {r.get('note')} › {r.get('section', '')}"
    pg = f" p.{r.get('page_start')}-{r.get('page_end')}" if r.get("page_start") else ""
    return f"[{r.get('corpus')}] {r.get('book')} › {r.get('file')}{pg}"


def build_prompt(q: dict, items: list[dict]) -> str:
    lines = [f"QUESTION: {q['query']}"]
    ans = (q.get("answer") or {}).get("text")
    if ans:
        lines.append(f"EXPECTED ANSWER (for your reference): {ans}")
    lines.append("")
    for i, it in enumerate(items, 1):
        text = (it["result"].get("text") or "").strip()[:2500] or "(no text available)"
        lines.append(f"### P{i} {_label(it['result'])}\n{text}\n")
    lines.append('Return: [{"p": 1, "rel": 0|1|2, "conflict": false, "why": "<10 words"}, ...] '
                 f"covering all {len(items)} passages.")
    return "\n".join(lines)


def judge_question(q: dict, items: list[dict], model: str) -> list[dict]:
    out = []
    for start in range(0, len(items), 12):  # keep prompts bounded
        chunk = items[start:start + 12]
        verdicts = llm.ask_json(build_prompt(q, chunk), SYSTEM, model=model)
        by_p = {int(v.get("p", 0)): v for v in verdicts if isinstance(v, dict)}
        for i, it in enumerate(chunk, 1):
            v = by_p.get(i)
            if v is None:
                continue
            out.append({"qid": q["id"], "key": it["key"], "rel": int(v.get("rel", 0)),
                        "conflict": bool(v.get("conflict", False)), "why": v.get("why", ""),
                        "judge": model, "from": it.get("from")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--unjudged", required=True)
    ap.add_argument("--out", required=True, help="judgments.jsonl (appended)")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rejudge", action="store_true")
    a = ap.parse_args(argv)

    qs = {q["id"]: q for q in load_jsonl(Path(a.set))}
    done = {(j["qid"], j["key"]) for j in load_jsonl(Path(a.out))} if not a.rejudge else set()
    todo = defaultdict(list)
    for it in load_jsonl(Path(a.unjudged)):
        if (it["qid"], it["key"]) not in done and it["qid"] in qs:
            todo[it["qid"]].append(it)
    print(f"{sum(len(v) for v in todo.values())} passages over {len(todo)} questions", file=sys.stderr)

    out_path = Path(a.out)
    n_written = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge_question, qs[qid], items, a.model): qid for qid, items in todo.items()}
        for f in futs:
            try:
                rows = f.result()
            except Exception as e:  # one failed call must not lose the rest
                print(f"[judge] {futs[f]}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            with out_path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_written += len(rows)
    print(f"wrote {n_written} judgments → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
