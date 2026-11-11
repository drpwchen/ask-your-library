# eval — retrieval and end-to-end evaluation

Measures what the search actually returns, so changes (a new embedding model,
FTS, a re-chunked corpus) are decided by numbers instead of impressions.

Only the harness and a synthetic example ship here. Real question sets,
judgments and runs quote your notes and books — keep them outside the repo
(e.g. `~/.vault-search/eval/`).

## Files

| file | what it does |
|---|---|
| `harness.py` | `run` a question set through a system, `score` a run, `compare` two runs, list `unjudged` results |
| `adapters.py` | systems: `mcp` (the tool path agents use), `chat` (the context the plugin chat builds), `ripgrep` (pooling only) |
| `metrics.py` | relevance matching, hit@k / MRR / recall@10 / page-window hit / contamination@3, McNemar, cluster bootstrap, cluster permutation, Holm |
| `judge.py` | LLM judgments for pooled results nobody labelled yet |
| `e2e.py` | answer from the retrieved sources with a fixed prompt, then grade every claim |
| `examples/synthetic_gold.jsonl` | the question format, with made-up books |

## Question format

One JSON object per line:

```json
{"id": "T1-001", "type": "T1", "tier": "gold", "corpus": "textbook",
 "query": "starting dose of drug X for spasticity", "cluster": "Book_2e/ch12",
 "relevant": [{"corpus": "textbook", "book": "Book_2e", "file": "Book_2e/ch12.md",
               "page_start": 341, "page_end": 342, "span": "start at 5 mg tid", "grade": 2}],
 "contaminants": [{"corpus": "textbook", "book": "Book_1e"}],
 "answer": {"text": "5 mg PO tid", "critical": ["dose", "unit", "route"]},
 "expect_abstain": false}
```

- `corpus` is the one result list the question is scored on (`vault`, `textbook`,
  `lecture`), or a priority list such as `["lecture", "vault"]` — the first one the
  system searched is used. The corpora are never merged into one ranking.
- `relevant` items match by note title (vault) or book + file (textbook/lecture);
  `files` may list several files (a whole chapter). Page-window hit = same file and
  the result's pages overlap the gold pages ±2.
- `cluster` groups questions that are not independent (same chapter / course); the
  bootstrap resamples whole clusters.
- `expect_abstain: true` marks a question whose answer is not in the corpus; it is
  excluded from hit metrics and checked in the e2e step.

## Typical session

```bash
# retrieval runs (in-process against a server checkout; --base-url for a running server)
python eval/harness.py run --set gold.jsonl --system mcp  --server-dir server --keep-text --out runs/mcp.json
python eval/harness.py run --set gold.jsonl --system chat --server-dir server --keep-text --out runs/chat.json

# complete the pool, then score
python eval/harness.py unjudged --set gold.jsonl --run runs/mcp.json runs/chat.json --out unjudged.jsonl
python eval/judge.py --set gold.jsonl --unjudged unjudged.jsonl --out judgments.jsonl
python eval/harness.py score --set gold.jsonl --run runs/mcp.json --judgments judgments.jsonl --contaminants contaminants.json

# paired comparison (cluster permutation on hit@3 and MRR; McNemar / bootstrap as context)
python eval/harness.py compare --set silver.jsonl --a runs/base.json --b runs/fts.json

# claim-level end-to-end from a saved run
python eval/e2e.py --set gold.jsonl --run runs/mcp.json --out runs/mcp.e2e.json
```

`--env-json` sets server configuration before the modules load (useful for a dev
checkout that has no `.env`); `--tool-args '{"retrieval": "hybrid"}'` passes
experiment switches to every tool call. In-process runs write their search logs
under `--scratch`, never into the live logs.

`contaminants.json`: `{"books": ["Old_Edition_Folder"], "vault_folders": ["Shared Notes"]}` —
a source in the top 3 from this list counts toward contamination@3.

## Decision rules (fix these before looking at results)

A/B decisions are made on rules written down before the runs, not on whichever
comparison turns out significant afterwards. The defaults `compare` supports:

- **Primary test: cluster sign-flip permutation** (`perm` in the `compare`
  output), two-sided, alpha = 0.05. Each cluster's summed difference keeps or
  flips its sign as a whole. With ~20 clusters the bootstrap percentile p is
  optimistic, so McNemar (hit@3) and the cluster bootstrap CI (MRR) are
  reported as context only.
- **Primary metric: MRR**; hit@3 is secondary. With pooled judging (rel >= 1)
  hit@3 sits near its ceiling and separates systems poorly.
- **Several pre-registered contrasts in one decision**: adjust their primary
  p-values with Holm (`metrics.holm`).
- **Name the judgment basis** next to every number: gold files only, or gold
  plus pooled LLM judgments (rel >= 1). A result that holds on one basis and not
  the other is reported as such.
- **Run with the `n` the real callers use**: ranking depends on `n` (the
  candidate pool and the diversity backfill), so a top-5 measured with `--n 50`
  is not the top-5 a caller asking for 10 sees.

## Tests

`python eval/tests/test_eval_metrics.py` (stdlib only).
