"""
vault-search eval harness.

    run       query every question through one system, save the raw result lists
    score     metrics for a saved run (needs the question set + judgments)
    compare   paired test of two runs on the same question set
    unjudged  list top-k results nobody has judged yet (feed them to judge.py)

Examples

    python eval/harness.py run --set examples/synthetic_gold.jsonl --system mcp \\
        --server-dir server --out runs/mcp.json
    python eval/harness.py score --set examples/synthetic_gold.jsonl --run runs/mcp.json
    python eval/harness.py compare --set silver.jsonl --a runs/base.json --b runs/fts.json

Question sets, judgments and runs are plain JSON/JSONL; keep real ones outside
the repo (they quote your notes). Only synthetic examples ship here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    if not p or not Path(p).exists():
        return []
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def target_corpus(q: dict, searched: list[str]) -> str | None:
    """First of the question's acceptable corpora that this system searched."""
    wanted = q["corpus"] if isinstance(q["corpus"], list) else [q["corpus"]]
    return next((c for c in wanted if c in searched), None)


# --- run -------------------------------------------------------------------------

def cmd_run(a) -> int:
    import adapters as A

    qs = load_jsonl(Path(a.set))
    if a.limit:
        qs = qs[: a.limit]
    roots = {}
    if a.system == "ripgrep":
        roots = {"textbook": a.textbook_root, "vault": a.vault_root, "lecture": a.lecture_root}
    if a.env_json:
        for k, v in json.loads(Path(a.env_json).expanduser().read_text(encoding="utf-8")).items():
            if not k.startswith("_"):
                os.environ[k] = os.path.expanduser(v) if v.startswith("~") else v
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        os.environ[k] = os.path.expanduser(v)
    if a.system in ("mcp", "chat") and not a.base_url:
        A.prepare_env(Path(a.server_dir), Path(a.scratch),
                      Path(a.env_file) if a.env_file else None)
    api_key = None
    if a.api_key_file:
        api_key = Path(a.api_key_file).expanduser().read_text(encoding="utf-8").strip()
    sysobj = A.make_system(a.system, server_dir=Path(a.server_dir) if a.server_dir else None,
                           base_url=a.base_url, api_key=api_key, n=a.n, roots=roots, rg=a.rg)
    tool_args = json.loads(a.tool_args) if a.tool_args else {}
    out = {"system": a.system, "label": a.label or a.system, "set": str(a.set),
           "tool_args": tool_args, "env": a.env or [], "env_json": a.env_json,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": a.n,
           "server_dir": a.server_dir, "base_url": a.base_url, "answers": {}}
    corp_all = ("vault", "textbook", "lecture")
    for i, q in enumerate(qs, 1):
        corpora = corp_all if a.all_corpora else tuple(
            q["corpus"] if isinstance(q["corpus"], list) else [q["corpus"]])
        extra = dict(tool_args)
        if a.system == "ripgrep" and q.get("grep_terms"):
            extra["grep_terms"] = q["grep_terms"]
        try:
            r = sysobj.search(q["query"], corpora=corpora, extra=extra)
        except Exception as e:  # one broken query must not lose the whole run
            r = {"results": [], "searched": [], "latency_ms": {}, "error": f"{type(e).__name__}: {e}"}
        if not a.keep_text:
            for x in r["results"]:
                x["text"] = x.get("text", "")[:1500]
        out["answers"][q["id"]] = r
        if i % 10 == 0 or i == len(qs):
            print(f"[{a.system}] {i}/{len(qs)}", file=sys.stderr, flush=True)
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    errs = sum(1 for v in out["answers"].values() if v.get("error"))
    print(f"wrote {a.out}: {len(out['answers'])} questions, {errs} with errors")
    return 0


# --- score -----------------------------------------------------------------------

def score_run(qs: list[dict], run: dict, judgments: list[dict], contaminants: set[str],
              searched: dict[str, list[str]] | None = None) -> tuple[list[dict], dict]:
    """`searched` overrides the corpora a question's target is picked from (compare
    passes the intersection of both runs, so both are scored on the same corpus)."""
    jidx = M.build_judgment_index(judgments)
    rows, lats = [], []
    for q in qs:
        ans = run["answers"].get(q["id"])
        if ans is None:
            continue
        srch = searched[q["id"]] if searched and q["id"] in searched else ans.get("searched", [])
        tgt = target_corpus(q, srch)
        if tgt is None:
            row = {"qid": q["id"], "type": q.get("type", ""), "cluster": q.get("cluster") or q["id"],
                   "zero": True, "n_results": 0, "unjudged_at_hit": 0, "no_corpus": True}
            if not q.get("expect_abstain"):
                row.update({"hit@1": 0, "hit@3": 0, "mrr": 0.0, "contamination@3": 0})
                if q.get("relevant"):
                    row["recall@10"] = 0.0
                if any(g.get("page_start") for g in q.get("relevant", [])):
                    row["page_hit@3"] = 0
            else:
                row["abstain_question"] = True
            rows.append(row)
            continue
        qq = dict(q, corpus=tgt)
        row = M.score_question(qq, ans["results"], jidx=jidx, global_contaminants=contaminants)
        row["target"] = tgt
        rows.append(row)
        lat = ans.get("latency_ms", {})
        if lat:
            lats.append(float(lat.get(tgt, lat.get("all", sum(lat.values())))))
    return rows, M.aggregate(rows, lats)


def _load_contaminants(p: str | None) -> set[str]:
    if not p or not Path(p).exists():
        return set()
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    if isinstance(d, list):
        return set(d)
    return set(d.get("books", [])) | {f"vault:{f}" for f in d.get("vault_folders", [])}


def cmd_score(a) -> int:
    qs = load_jsonl(Path(a.set))
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    rows, agg = score_run(qs, run, load_jsonl(Path(a.judgments)) if a.judgments else [],
                          _load_contaminants(a.contaminants))
    agg["label"] = run.get("label")
    if a.out:
        Path(a.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    return 0


def cmd_compare(a) -> int:
    qs = load_jsonl(Path(a.set))
    j = load_jsonl(Path(a.judgments)) if a.judgments else []
    c = _load_contaminants(a.contaminants)
    run_a = json.loads(Path(a.a).read_text(encoding="utf-8"))
    run_b = json.loads(Path(a.b).read_text(encoding="utf-8"))
    # Score both runs on a corpus both searched: a run that also searches a new corpus
    # would otherwise be scored on that corpus while the other run is scored on another.
    both = {}
    for qid, ans_a in run_a["answers"].items():
        ans_b = run_b["answers"].get(qid)
        if ans_b is not None:
            sb = set(ans_b.get("searched", []))
            both[qid] = [x for x in ans_a.get("searched", []) if x in sb]
    ra, _ = score_run(qs, run_a, j, c, both)
    rb, _ = score_run(qs, run_b, j, c, both)
    ia, ib = {r["qid"]: r for r in ra}, {r["qid"]: r for r in rb}
    common = [q for q in ia if q in ib and "hit@3" in ia[q] and "hit@3" in ib[q]]
    res = {"n": len(common)}
    res["hit@3"] = M.mcnemar_exact([ia[q]["hit@3"] for q in common], [ib[q]["hit@3"] for q in common])
    res["hit@3"]["a"] = round(sum(ia[q]["hit@3"] for q in common) / max(1, len(common)), 4)
    res["hit@3"]["b"] = round(sum(ib[q]["hit@3"] for q in common) / max(1, len(common)), 4)
    res["mrr"] = M.cluster_bootstrap_diff([ia[q]["mrr"] for q in common], [ib[q]["mrr"] for q in common],
                                          [ia[q]["cluster"] for q in common])
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


def cmd_unjudged(a) -> int:
    qs = {q["id"]: q for q in load_jsonl(Path(a.set))}
    jidx = M.build_judgment_index(load_jsonl(Path(a.judgments)) if a.judgments else [])
    seen, out = set(), []
    for rp in a.run:
        run = json.loads(Path(rp).read_text(encoding="utf-8"))
        for qid, ans in run["answers"].items():
            q = qs.get(qid)
            if not q:
                continue
            for corpus in (q["corpus"] if isinstance(q["corpus"], list) else [q["corpus"]]):
                lst = sorted((r for r in ans["results"] if r["corpus"] == corpus), key=lambda r: r["rank"])
                for r in lst[: a.k]:
                    if M.relevance(r, dict(q, corpus=corpus), jidx) is not None:
                        continue
                    key = (qid, M.result_key(r))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"qid": qid, "key": key[1], "query": q["query"], "result": r,
                                "from": run.get("label")})
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + ("\n" if out else ""),
                           encoding="utf-8")
    print(f"{len(out)} unjudged results → {a.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--system", required=True, choices=["mcp", "chat", "ripgrep"])
    r.add_argument("--out", required=True)
    r.add_argument("--label")
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--server-dir", default=str(Path(__file__).resolve().parent.parent / "server"))
    r.add_argument("--env-file", help="config for a server dir that has no .env of its own")
    r.add_argument("--scratch", default=str(Path.home() / ".vault-search" / "eval" / "_scratch"))
    r.add_argument("--base-url", help="send MCP calls to a running server instead of in-process")
    r.add_argument("--api-key-file")
    r.add_argument("--all-corpora", action="store_true", help="search every corpus, not just the target")
    r.add_argument("--keep-text", action="store_true")
    r.add_argument("--tool-args", help="JSON merged into every tool call, e.g. {\"retrieval\": \"hybrid\"}")
    r.add_argument("--env-json", help="JSON object of env vars set before the server modules load")
    r.add_argument("--env", action="append", help="KEY=VALUE set before the server modules load (repeatable)")
    r.add_argument("--textbook-root")
    r.add_argument("--vault-root")
    r.add_argument("--lecture-root")
    r.add_argument("--rg", default="rg", help="ripgrep binary for --system ripgrep")
    r.set_defaults(fn=cmd_run)

    s = sp.add_parser("score")
    s.add_argument("--set", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--judgments")
    s.add_argument("--contaminants")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)

    c = sp.add_parser("compare")
    c.add_argument("--set", required=True)
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--judgments")
    c.add_argument("--contaminants")
    c.set_defaults(fn=cmd_compare)

    u = sp.add_parser("unjudged")
    u.add_argument("--set", required=True)
    u.add_argument("--run", required=True, nargs="+")
    u.add_argument("--judgments")
    u.add_argument("--k", type=int, default=10)
    u.add_argument("--out", required=True)
    u.set_defaults(fn=cmd_unjudged)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
