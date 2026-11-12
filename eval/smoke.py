"""
vault-search smoke check — a few fixed queries against the RUNNING server.

Not a benchmark. It answers one question each night: does search still work the
way it did when the eval passed? It catches silent breakage that a statistical
regression run would only report a week later:

* a tool returns an error, nothing, or not the source it always returned;
* textbook_search stopped using hybrid retrieval or the reranker
  (read back from the server's own search log entry for the query).

A reranker skipped only because another job holds the GPU is retried once after
--retry-wait seconds, then reported as a warning, not a failure.

Queries live in a private JSON file (they name your own notes and courses):

    [{"tool": "textbook_search", "query": "...", "expect": "BookFolder"},
     {"tool": "lecture_search",  "query": "...", "expect": "course_folder"},
     {"tool": "vault_search",    "query": "...", "expect": "Note name"}]

`expect` is a substring that must appear in the tool's text output (optional).

    python eval/smoke.py --queries ~/.vault-search/eval/smoke.json \\
        --api-key-file ~/.vault-search/api_key.txt

Exit 0 = all pass (warnings allowed), 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

LEASE_REASON = "gpu claimed"


def call(base_url: str, key: str | None, tool: str, query: str, timeout: float) -> str:
    body = json.dumps({"name": tool, "arguments": {"query": query},
                       "context": {"caller": "smoke"}}).encode()
    h = {"Content-Type": "application/json"}
    if key:
        h["X-API-Key"] = key
    req = urllib.request.Request(base_url.rstrip("/") + "/api/mcp/call", data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r).get("text", "")


def last_log_entry(log: Path, query: str) -> dict | None:
    """Newest search-log line for this query from the smoke caller."""
    if not log.exists():
        return None
    with log.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 400_000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("query") == query and d.get("caller") == "smoke":
            return d
    return None


def check(q: dict, a: argparse.Namespace, key: str | None) -> tuple[str, str]:
    """Return (status, message); status is ok / warn / fail / lease."""
    tool, query = q["tool"], q["query"]
    try:
        text = call(a.base_url, key, tool, query, a.timeout)
    except Exception as e:  # noqa: BLE001 — any transport error is a failure
        return "fail", f"{tool}: request failed: {type(e).__name__}: {e}"
    head = text[:400]
    if not text.strip() or (text.lstrip().startswith("{") and '"error"' in head):
        return "fail", f"{tool}: empty or error output: {head[:160]!r}"
    if q.get("expect") and q["expect"] not in text:
        return "fail", f"{tool}: expected {q['expect']!r} not in output"
    if tool != "textbook_search":
        return "ok", f"{tool}: ok"
    entry = last_log_entry(Path(a.search_log).expanduser(), query)
    if entry is None:
        return "fail", "textbook_search: no search-log entry for the smoke query"
    if a.expect_retrieval and entry.get("retrieval") != a.expect_retrieval:
        return "fail", f"textbook_search: retrieval={entry.get('retrieval')!r}, want {a.expect_retrieval!r}"
    rr = entry.get("rerank") or {}
    if a.expect_rerank and not rr.get("applied"):
        reason = str(rr.get("reason", "rerank missing from log"))
        if LEASE_REASON in reason:
            return "lease", f"textbook_search: rerank skipped ({reason})"
        return "fail", f"textbook_search: rerank not applied ({reason})"
    return "ok", f"textbook_search: ok ({entry.get('retrieval')}, rerank {rr.get('start', '-')})"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--queries", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:3789")
    p.add_argument("--api-key-file")
    p.add_argument("--search-log", default="~/.vault-search/search_log.jsonl",
                   help="the server's textbook search log (VAULT_SEARCH_LOG_PATH)")
    p.add_argument("--expect-retrieval", default="hybrid", help="'' to skip the check")
    p.add_argument("--no-expect-rerank", dest="expect_rerank", action="store_false")
    p.add_argument("--timeout", type=float, default=120.0, help="per call; reranker cold start is 10-35 s")
    p.add_argument("--retry-wait", type=float, default=300.0)
    a = p.parse_args(argv)

    key = Path(a.api_key_file).expanduser().read_text().strip() if a.api_key_file else None
    queries = json.loads(Path(a.queries).expanduser().read_text(encoding="utf-8"))
    results = [check(q, a, key) for q in queries]
    if any(s == "lease" for s, _ in results):
        time.sleep(a.retry_wait)
        results = [check(q, a, key) if s == "lease" else (s, m)
                   for q, (s, m) in zip(queries, results)]

    fails = 0
    for status, msg in results:
        if status == "lease":
            status = "warn"
        fails += status == "fail"
        print(f"[smoke] {status.upper():4} {msg}")
    print(f"[smoke] {len(results) - fails}/{len(results)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
