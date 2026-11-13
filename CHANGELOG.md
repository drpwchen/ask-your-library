# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 2.6.0 were not tagged. `plugin/manifest.json` carries the Obsidian
plugin's own version (2.5.1) and moves only when something inside `plugin/` changes,
so it deliberately does not track these tags.

## [3.3.0] — 2026-11-14 — ask-your-library: corpus folders with one command; reference chapters stay out of the v3 index

First public release since 2.9.1; it also carries 2.10.0–3.2.0 (sections below), which were
tagged locally but never published separately.

vault-search is now **ask-your-library** (`drpwchen/ask-your-library`); README, AGENTS.md, docs and the plugin
description use the new name. Env vars (`VAULT_SEARCH_*`), MCP server/tool names, the plugin id and
`~/.vault-search` are unchanged.

### Added
- **Corpus folder format `t2n-corpus/1`** (`docs/corpus-format.md`): the one specification of
  how textbooks, guidelines and lectures reach vault-search — a `corpus.json` marker
  (`{"format": "t2n-corpus/1", "kind": "textbook" | "lecture"}`), the textbook layout
  (source folders, `_Guidelines/<Name>/`, page markers, the `_manifest.json` fields the
  indexer reads, `books.json`) and the lecture layout (course folders, `_meta.md`, entries
  that open with an H1 and a `Source:` line). It writes down what the v3 indexer already
  reads; nothing new is required of existing corpora (a missing marker is a warning).
- **`server/corpus.py`**: `check <root>` validates a corpus folder (errors exit 1,
  `--strict` fails on warnings too, `--json` for tools); `add <root>` checks, runs
  `textbook_indexer_v3.py` (changed files only on re-runs) and its `--audit`, and prints the
  `.env` line the server needs. Default database: the one already configured for that kind
  when it is a v3 database, else `~/.vault-search/textbook_db` / `lecture_db_v3`. Besides
  the indexer's own live-database guard, `add` refuses a v3 database built from a
  different corpus root (a run deletes rows of files outside its root).
- Fixtures `examples/corpus-fixture/{textbook,lecture}` and `server/tests/test_corpus.py`
  (22 tests: both fixtures pass, broken copies fail, `add --dry-run` commands and refusals,
  `add` end to end with a fake embedder).
- `AGENTS.md` (with a `CLAUDE.md` bridge): setup notes for coding agents — the `python-dotenv`
  requirement and the separate-deployment caveat.

### Changed
- README (EN + zh-TW): new "Add a corpus" section built on `corpus.py add`, with the
  `_Guidelines/` layout and `books.json`; a "Vault index v3" section (build with
  `--profile v3 --db`, set `VAULT_SEARCH_DB_PATH`, create the flag file, restart); the
  repository layout lists the current files.

### Fixed
- **`lecture_search` shows date, speaker and course again on a v3 lecture index.** The v3
  chunker drops `Source:` lines as converter bookkeeping, so a v3 lecture database holds
  none (measured on the author's index: 0 of 16,877 parents) and every hit came back with
  those fields empty. Entries the parents table cannot answer are now read from the corpus
  root the indexer recorded (`_v3_state/index_meta.json`). No reindex needed.
- **Guideline chapters get their manifest.** textbook-to-note `--resplit` writes one
  `_Guidelines/_manifest.json` (entries `<Name>/<file>`), but the v3 indexer only looked
  for a manifest inside each `_Guidelines/<Name>/` folder, so split guideline chapters
  were indexed with no chapter number, title, printed pages or `split_method`. A
  guideline folder without its own manifest now uses the shared one. Rows already in an
  index change only when that guideline is re-indexed (`--book <Name>`).
- README's lecture example used `--db ~/.vault-search/lecture_db`, which the indexer
  refuses (it is the default live lecture path); it is `lecture_db_v3` now. `env.example`
  no longer says the lecture index is built with `textbook_indexer.py`, and marks which
  textbook settings only the older indexer reads.
- **v3 textbook indexer: a chapter file named like a reference list is no longer indexed
  past its heading.** A file such as `ch74_REFERENCES.md` with no manifest kind (every
  `_Guidelines/` source) slipped through all three filters: the heading is a plain text
  line rather than `#`, and wrapped citations keep the per-child density under 0.6. On the
  live corpus that was 1,119 rows across 31 files, three of them in a single top-5. Now,
  when the file name reads References / Bibliography / Suggested or Further Reading /
  Select(ed) References, the text is cut at the first line that is only that heading;
  body text before it (a split that landed mid-page) is kept, and a file with no such
  line is kept whole. `--keep-references` disables the cut. Rows already in an index
  change only when that source is re-indexed (`--book`).
- Reference headings are recognised through zero-width joiners inside the word and in
  the "Select References" spelling.

## [3.2.0] — 2026-11-13 — opt-in hybrid retrieval and multi-section excerpts for the vault

Both default OFF; with neither set, results are unchanged.

### Added
- **`VAULT_SEARCH_VAULT_RETRIEVAL=hybrid`**: v2 vault search fuses the vector
  ranking with a full-text ranking (RRF), as `textbook_search` does, so an
  exact term (an abbreviation, a test or drug name) can surface a note the
  embedding ranks low. Full-text side reads `VAULT_SEARCH_VAULT_FTS_COLUMN`
  (default `text`). `similarity` stays the cosine; the order follows the
  fused score under the same folder / recency weights. Without an FTS index
  LanceDB scans the text (slower, same answer); a search error falls back to
  vector. The search log records `retrieval`, `retrieval_fallback` and
  `sections_per_note`.
- **`VAULT_SEARCH_VAULT_SECTIONS_PER_NOTE=k`**: a result quotes up to `k`
  distinct sections of its note (best-matching first, the next ones in rank
  order) instead of one, so an answer spread over a long note's sections
  reaches the reader. Ranking is unchanged.
- Both can also be set per call (`retrieval`, `sections_per_note` arguments;
  not in the tool schema) for evaluation without a restart.

## [3.1.0] — 2026-11-13 — vault excerpts quote the whole matched section

### Changed
- **v2 vault excerpts are the whole matched section, not its first 800
  characters** (`vault_search` with `include_excerpt`, and the chat context).
  Search matches a small child chunk but quotes its parent section, and the
  child can sit anywhere in that section. Measured on 60 logged queries before
  the change: in the chat context (3 vault notes per question) 72% of excerpts
  were cut, the matched passage fell entirely outside the excerpt for 8% and
  was cut part-way for another 18%. Over 387 `vault_search` results (n=10) the
  800-char head kept the whole matched passage 64% of the time; the whole
  section keeps it every time, at a median of 7.4k characters per query
  (p90 11.8k) instead of 4.0k. Textbook and lecture passages already quoted
  the whole section; vault now matches them.
- New setting `VAULT_SEARCH_VAULT_EXCERPT_CHARS` (default `0` = whole
  section). Set `800` to restore the previous behaviour.
- The v1 path (`vault_similar`, and the v1 fallback of `vault_search`) still
  quotes 800 characters of chunk text.

## [3.0.0] — 2026-11-13 — the v3 index generation

No code changes over 2.12.3. 3.0.0 marks the end of the retrieval rework
released piecemeal as 2.10.0–2.12.3, so it can be adopted as one step. Nothing
in the HTTP API or the MCP tool signatures was removed; an existing v2 setup
keeps working unchanged. What a new install gets, in one place:

- **A v3 textbook index in its own database** (`textbook_indexer_v3.py`,
  `VAULT_SEARCH_TEXTBOOK_DB_PATH`): real page provenance, table units,
  reference lists filtered, one source per guideline, a full-text index;
  incremental re-runs, `--book`, `--audit`. (2.11.0, 2.12.0)
- **Hybrid retrieval by default** for `textbook_search` (vector + full-text,
  RRF), with real similarity values. (2.11.2, 2.12.0)
- **Optional cross-encoder reranker** (`BAAI/bge-reranker-v2-m3`, separate
  process, yields the GPU on demand), then a small guideline / recency prior.
  (2.12.0, 2.12.1)
- **Source precedence** stated in the `textbook_search` description, plus the
  corpus root the result paths are relative to. (2.12.2)
- **`lecture_search`** over a separate lecture corpus with course metadata;
  **search scopes**; the **book table** joined at query time; chat on the same
  search path as MCP. (2.10.0, 2.11.1)
- **`eval/`**: retrieval and claim-level evaluation with pre-registered
  decision rules, and `eval/smoke.py` for a nightly check of the running
  server. (2.10.0–2.12.3)

### Changed
- **README** (both languages): the reference-corpus setup now uses
  `textbook_indexer_v3.py` and its own database, documents source precedence
  and the source prior, and shows the `--corpus` level for a lecture corpus.

## [2.12.3] — 2026-11-12 — nightly smoke check

### Added
- **`eval/smoke.py`**: a few fixed queries against the running server. Each
  tool must answer with its expected source, and `textbook_search` must have
  used hybrid retrieval and the reranker (read back from the server's search
  log). A reranker skipped only because another job holds the GPU is retried
  once, then reported as a warning. Exit 1 on any failure, so a scheduler can
  alert on it. Queries live in a private JSON file; the format is in the
  module docstring. This replaces the planned weekly 60-question regression:
  retrieval only changes when code, model or index change, and the full eval
  runs before each release.

## [2.12.2] — 2026-11-12 — textbook_search names its corpus root and source precedence

### Added
- **`textbook_search` output says what `file` is relative to.** A v3 textbook
  DB records the corpus it was built from (`_v3_state/index_meta.json` →
  `corpus`); the second line of the output is now
  ``files are relative to `<corpus>` ``. Result paths point into the re-split
  corpus, whose chapter files differ from the original conversion, so opening
  them against the old tree silently failed. A v2 DB adds nothing.
- **Source precedence when passages disagree.** The tool description now
  states the rule (newer edition of the same work wins; a guideline not older
  than a textbook edition takes precedence over it; a single study never
  overrides a guideline or textbook consensus; name the disagreement and cite
  both). When a guideline appears next to an older textbook edition, the
  output ends with a one-line `> precedence:` reminder. Answers in the Phase 0
  end-to-end check followed older textbooks against newer guidelines (T3).

## [2.12.1] — 2026-11-12 — guideline / recency prior after reranking

### Changed
- **`textbook_search` adds a source prior to the cross-encoder score** before
  the final cut: `VAULT_SEARCH_RERANK_GUIDELINE_PRIOR` (default 3, bge logit
  scale) for guidelines, plus `VAULT_SEARCH_RERANK_YEAR_PRIOR` (default 0.5) ×
  `clip((year − 2015) / 10, 0, 1)` for every source with a year. Guideline
  questions were losing to older textbooks that score about the same. Results
  carry `rerank_prior` when it moved them; `rerank_score` stays the raw
  cross-encoder score. Set both to 0 for the 2.12.0 ordering. Only applies when
  the reranker ran.

## [2.12.0] — 2026-11-12 — hybrid by default; optional cross-encoder reranker; separate textbook DB

### Changed
- **`textbook_search` defaults to hybrid retrieval** (vector + full-text, RRF).
  `VAULT_SEARCH_TEXTBOOK_RETRIEVAL` now defaults to `hybrid`; set it to
  `vector` for the old behaviour. An index without a full-text index still
  falls back to vector (and logs `retrieval_fallback`).

### Added
- **Optional cross-encoder reranker for `textbook_search`**
  (`server/reranker.py` client, `server/reranker_service.py` service). Off
  unless `VAULT_SEARCH_RERANK_PYTHON` points at a Python with CUDA torch +
  transformers. The search builds its usual result list at a pool of
  `VAULT_SEARCH_RERANK_POOL` (50), scores each passage as the caller would see
  it (parent section, 800 tokens; table units whole) with
  `BAAI/bge-reranker-v2-m3`, and keeps the top n. Results gain `rerank_score`
  and `rank_before_rerank`; `similarity` stays the cosine. The service starts
  on the first query, exits after `VAULT_SEARCH_RERANK_IDLE_S` (600 s) idle,
  and exits immediately when another job presses the GPU lease's PAUSE flag or
  queues for it — it never acquires the lease and never holds the card
  against a queued job. Every failure (not configured, GPU claimed, start
  timeout, HTTP error) is soft: the un-reranked list is returned and the
  reason is logged under `rerank` in the textbook search log. Per-call
  override: `rerank: true|false`. Lecture search is not reranked.
- **`VAULT_SEARCH_TEXTBOOK_DB_PATH`**: the textbook tables can live in their
  own database (e.g. a rebuilt v3 index) while the vault stays on
  `VAULT_SEARCH_DB_PATH`. Default: the same database. Search side and
  `textbook_indexer_v3.py` follow it; the legacy `textbook_indexer.py` keeps
  `VAULT_SEARCH_DB_PATH` (lecture indexing sets that one).
- **Guards between v2 and v3 databases.** `textbook_indexer_v3.py` may now
  update a live database that is itself a v3 database (the normal incremental
  run after a switch-over) — a live v2 database is still refused. The v2
  `textbook_indexer.py` refuses a v3 database, and a default-profile
  `vault_indexer_v2.py` run refuses a database holding a v3-profile vault index.
- Evidence (author's corpus, 435 silver questions / 163 clusters):
  0.6B + FTS → + bge MRR +0.079 (cluster CI 0.047–0.111, permutation
  p < 0.001); ~0.5 s per query on an RTX 3070 Ti.

## [2.11.2] — 2026-11-11 — hybrid search reports real similarity; permutation test in eval

### Fixed
- **Hybrid and FTS textbook results reported `similarity` 0.0 for every row.**
  LanceDB's RRF reranker drops `_distance` (it returns `_relevance_score`
  instead), so `_row_similarity` fell through to 0.0 — 660 of 660 results in a
  66-question pilot. The cosine is now computed from the row's own vector
  against the query vector (both L2-normalised), which equals vector search's
  `1 - _distance`. Ranking is unchanged; only the reported number was wrong.
  Anything that reads `similarity` (the result display, confidence or "nothing
  found" logic) now works with `VAULT_SEARCH_TEXTBOOK_RETRIEVAL=hybrid`.

### Added
- **eval: cluster sign-flip permutation test** (`metrics.cluster_permutation_diff`)
  on hit@3 and MRR in `harness.py compare` (`perm`), and `metrics.holm` for a
  family of contrasts. With ~20 clusters the cluster bootstrap's p runs
  optimistic (0.008 vs 0.022 on the same data).
- **eval/README.md: "Decision rules"** — primary test, primary metric, alpha,
  multiple-comparison handling, judgment basis and `n` fixed before the runs.

## [2.11.1] — 2026-11-09 — lecture_search: course metadata, no _meta.md hits

### Changed
- **`lecture_search` no longer returns a course's `_meta.md`.** That file is the
  course's bookkeeping (clip table, material paths, speaker list), not lecture
  content, yet it made up about one chunk in eleven of a real lecture index. It is
  now excluded with a prefilter, so it does not use up result slots.

### Added
- **`lecture_search` results carry `date`, `speaker` and `course_title`**, parsed
  from the entry's `Source:` line (read from the entry's H1 parent, one query per
  search) and shown as a `source:` line in the tool output and the chat context.
  Fields the line does not state (`講者未載明`, no date) are returned empty.

## [2.11.0] — 2026-11-09 — v3 textbook index with real provenance; table units; hybrid search switch

Nothing in an existing index changes until you build a new one: the v3 indexer
writes only to a database you name, and the default vault profile is untouched.
The query-side changes read a v2 index as before.

### Added
- **`server/textbook_indexer_v3.py`** builds a new textbook index (`--corpus`, `--db`;
  the configured live database is refused):
  - **Pages per chunk, not per file.** Every chunk records the physical PDF pages of
    its own lines (`pdf_page_start/end`), the printed page labels when the book's
    `_manifest.json` has them (`printed_page_start/end`, `page_confidence` =
    `printed` | `pdf_only`), character offsets into the source markdown, the source
    file's sha256, `chapter_id` / `chapter_number`, and edition / year / tier /
    status / corpus_type from `books.json`. On a test book 34 of 35 chunks now span
    one page; before, a chunk inherited its whole chapter's range.
  - **Tables are their own passages.** A table is never cut at a page marker or at
    the chunk length limit; one longer than the embedder's context embeds its
    caption, header and first rows, and the search still returns the whole table
    with its review flag.
  - **Filtered at index time:** References sections (by manifest kind, heading, or
    a conservative citation-density test) and download-site watermarks. Sources
    marked superseded / derivative / low_trust in `books.json` are not indexed.
  - **One source per guideline**, and per folder either `full_text.md` or its
    chapter files, never both.
  - `--book` always re-embeds; a file that became empty loses its rows; `--audit`
    checks coverage chapter by chapter; `--dry-run` chunks without a GPU;
    `--holding-report` lists sources to retire (moves nothing).
  - A non-default `--model` writes its own tables; `indexing_signature` pins the
    model digest, vector width, document instruction and normalisation.
  - Resumable: the hash cache under `<db>/_v3_state/` is saved every 10 files.
  - Builds the native full-text index used by hybrid search.
- **`vault_indexer_v2.py --profile v3 --db <new db>`**: real YAML frontmatter, no
  backup copies / template folders / `VAULT_SEARCH_V3_EXCLUDE_FOLDERS`, hard length
  caps (a note with no blank lines no longer becomes one 50k-token chunk), and a
  `scope` column. The default profile is unchanged.
- **`textbook_search`**: optional `query_en` — an English rewrite used by keyword
  matching. `VAULT_SEARCH_TEXTBOOK_RETRIEVAL` picks `vector` (default) / `hybrid` /
  `fts`; without a full-text index it falls back to vector and logs why.
  `VAULT_SEARCH_TEXTBOOK_TABLE_SUFFIX` selects a non-default model's tables.
- Results carry `page_confidence`, `table_review_flag`, `tier`, `year`, `status`,
  `corpus_type` (omitted when unknown). On a v2 index the table flag is read from
  the converter's flag comments.

### Changed
- **Page labels in `textbook_search` output**: physical pages print as `PDF p.N`;
  printed pages, when known, print first as `p.N`. A table that needs checking
  against the PDF says so on its own line. `eval/adapters.py` reads both forms.
- **Book diversity counts editions as one book** (`Braddom_6e` and `_7e` share the
  cap) and each guideline as its own source; when the cap leaves fewer than
  `n_results`, the capped passages are appended back in rank order.

### Fixed
- **`textbook_indexer.py --book` emptied the book.** It deleted the book's rows and
  then skipped every file whose hash was unchanged, leaving nothing indexed.

Gold set, tool path, against 2.10.0 on the same v2 index: hit@3 0.921 → 0.936,
MRR 0.871 → 0.880, contamination@3 0.129 → 0.136 (one question).

## [2.10.2] — 2026-11-09 — Eval e2e: resumable runs, stop on usage limit

### Fixed
- **An interrupted `eval/e2e.py` run lost every finished question.** Results were
  written only when the whole batch ended, so a stopped task or a usage limit threw
  away up to ~150 paid model calls. Each finished question is now appended to
  `<out>.partial.jsonl`; re-running the same command skips those and continues.
  The checkpoint records run, model, k and prompt versions and refuses to mix runs
  with different settings (`--fresh` starts over). It is removed once a run finishes
  with no errors.
- **A usage-limit reply was graded as a wrong answer.** `claude -p` returns the limit
  notice as an ordinary result, so one run recorded 48 of 150 questions as errors.
  `eval/llm.py` now raises `QuotaExhausted` on that notice; `e2e.py` stops sending
  new questions, keeps what finished, and exits with code 3.

## [2.10.1] — 2026-11-08 — Eval harness: isolated judge calls, same-corpus compare

### Fixed
- **Every judge call loaded the user's hooks and MCP servers.** `eval/llm.py` runs
  `claude -p` once per question; each run fired the user's Stop and session-start
  hooks (a notification per call, secret reads that tripped an access monitor) and
  started every configured MCP server. Judge calls now pass
  `--settings '{"disableAllHooks": true}'` and `--strict-mcp-config`: the judge only
  reads a prompt and returns JSON.
- **`harness.py compare` could score the two runs on different corpora.** A question
  that accepts several corpora is scored on the first one the run searched; when
  only run B searched a new corpus (v2.10.0 added `lecture`), A was scored on `vault`
  and B on `lecture`, and identical top-3 lists showed up as wins and losses.
  `compare` now picks each question's target from the corpora both runs searched.
  `score` on a single run is unchanged.

## [2.10.0] — 2026-11-06 — Chat on the shared search path, search scopes, book table, eval harness

### Fixed
- **The plugin chat searched a stale index and quoted the wrong pages.** Its context
  builder ran its own retrieval: the v1 `vault` table (frozen since the v2 switch)
  and a textbook query that read a `page` column the v2 schema does not have. Every
  passage came back as "page 0", so the "expanded context" sent to the model was
  the first pages of the file, not the passage that matched. It also skipped the
  derivative-source exclusion: on a 436-question benchmark, 25% of chat answers had
  an excluded source in the top 3, against 2.3% on the tool path. Chat now calls
  the same functions as `vault_search` / `textbook_search` / `lecture_search`
  (3 / 4 / 2 results), with the real page range on every passage.
- **The vault v2 indexer never closed its error log.** A file that failed once and
  was fully re-indexed later kept its entries `pending` forever (852 on the author's
  vault, all long since indexed). Entries are now closed when their file completes,
  and `vault_indexer_v2.py --reconcile-errors` closes old ones without embedding
  anything: fixed when the file's current hash is cached and it has rows in the
  index, obsolete when the file is gone, open otherwise.

### Added
- **Search scopes.** A JSON map of scope → top-level folders (`VAULT_SEARCH_SCOPE_MAP`,
  default `~/.vault-search/scope_map.json`) filters `vault_search`, `vault_similar`,
  `vault_related` and the chat context with a LanceDB prefilter — no reindex.
  Fail-closed: unlisted folders are outside every named scope, `personal` is the
  complement, `all` disables it, an unknown scope or unreadable map refuses instead
  of searching everything. The default scope comes from the server (map `_default`
  or `VAULT_SEARCH_DEFAULT_SCOPE`), never from the prompt. Graph-linked suggestions
  are filtered too. No map = no change.
- **Book table joined at query time** (`VAULT_SEARCH_BOOKS_META`, default
  `~/.vault-search/books.json`): edition / year / tier / corpus_type / status /
  superseded_by per source folder. `textbook_search` hides `superseded` and
  `derivative` sources unless `include_superseded=true`, accepts `corpus_type`
  (e.g. `guideline`), breaks similarity ties by tier then year, and prints a
  one-line `source:` provenance under each passage.
- **`lecture_search`** — a third corpus in its own DB (`VAULT_SEARCH_LECTURE_DB_PATH`,
  default `~/.vault-search/lecture_db`), same schema and model as the textbook
  index, ranked separately. The tool is listed only when that DB exists.
- **Who asked, in the search logs.** Every log line now carries `caller`
  (`mcp` / `chat`) and `session_id` (the thin client forwards Claude Code's session
  id; chat uses the request id), and a zero-result search is logged as `warn`.
- **`eval/`** — retrieval and claim-level evaluation harness: adapters for the tool
  path and the chat path, hit@k / MRR / recall@10 / page-window hit /
  contamination@3, McNemar and cluster-bootstrap paired tests, LLM pool
  completion, and an answer-then-grade end-to-end step. Stdlib-only metrics, tested
  in CI. Real question sets stay outside the repo.

### Changed
- `textbook_search` accepts an experimental `retrieval` switch (`vector` default,
  `fts`, `hybrid` with RRF) that the eval harness uses; it is not in the tool schema
  and needs a native FTS index you build yourself.

## [2.9.1] — 2026-11-03 — Model unloaded after indexing; faster Ollama calls on Windows

### Fixed
- **The v2 indexer left the embedding model loaded for two hours after it finished.**
  Embedding calls carry `keep_alive="2h"` so the runner survives idle gaps between
  batches, but nothing ever shortened it at the end, so a nightly run kept about
  4 GB of memory commit pinned until the timer expired. `index_vault_v2` now sends
  one `keep_alive=0` unload when the run ends, on failure as well as success; an
  unload that itself fails is logged and does not fail the run.
- **Every Ollama call on Windows waited about 2 seconds first.** The default `OLLAMA_HOST`
  was `http://localhost:11434`; Windows tries IPv6 (`::1`) first, Ollama listens on IPv4,
  and each call paid for the failed attempt. The default is now `http://127.0.0.1:11434`.
  An `OLLAMA_HOST` you set yourself is used as before.

## [2.9.0] — 2026-09-06 — Parent-child vault index, one GPU lease, no hardcoded folders

Vault search gets the retrieval architecture the textbook corpus has had for
months: every note is split into small child chunks for a precise vector hit,
plus subsection-sized parents that are what you actually get back. It ships as a
**second pair of tables**, so the existing index keeps serving until you decide
to switch — create `~/.vault-search/vault_v2.enabled` to read v2, delete it to go
back, no restart either way.

This release also removes the last folder names baked into the code. Ranking
weights, the archive folder, derivative-note exclusion, entity-extraction
priority and both query templates now come from the environment, so the defaults
suit any vault rather than the author's.

### Added

- **`server/vault_indexer_v2.py`** — parent-child indexing for the vault, with
  incremental hashing, a generation marker and error accounting. Its embedding
  model is its own knob (`VAULT_SEARCH_V2_MODEL`): a vault-only install never has
  to configure a textbook corpus to use v2.
- **Dual-track retrieval.** `vault_search`, `/api/search` and the chat context
  picker read v2 when the flag file is present and fall back to v1 when the v2
  index is missing or broken — and the fallback is RECORDED
  (`index_version`, `v2_fallback_reason`) in `vault_search_log.jsonl`, because a
  silent downgrade to the older index looks exactly like "search got worse for no
  reason".
- **`server/gpu_lease_client.py`** — one `GpuLease` class shared by both
  indexers: acquire before any GPU work, heartbeat while a long run makes
  progress, and yield the card between files when another job is queued.
  Unset `VAULT_SEARCH_GPU_LEASE` (the default) makes every call a no-op — no
  subprocess is ever spawned on the open-source path.

### Changed

- **Ranking is configuration, not code.** `VAULT_SEARCH_PATH_WEIGHTS`,
  `VAULT_SEARCH_ARCHIVE_FOLDER`, `VAULT_SEARCH_EXCLUDE_PATTERNS` and
  `VAULT_SEARCH_ENTITY_PRIORITY_FOLDERS` replace folder names that were written
  into `scoring.py`, `mcp_server.py` and `graph_builder.py`. All default to
  empty: every folder weighted equally, nothing excluded, no folder favoured.
- **Query templates are configurable and self-identifying.** The logged
  `query_template_version` is now the template's name plus a short hash of the
  prefix itself, so a log line can never name one template while a different
  prefix is in use. The template is deliberately NOT part of the indexing
  signature — it only shapes queries, never the stored vectors, so editing it
  no longer forces a full re-index.
- **One excerpt budget.** `vault_similar` quoted 500 characters of a note where
  `vault_search` quoted 800, because one path built its rows through
  `format_results()` and the other inline. Both now use `EXCERPT_CHARS = 800`.

### Fixed

- **`_search_vault_context` embedded its query with the wrong model.** The chat
  context picker built a bge-m3 vector and searched the qwen3 textbook table with
  it. Both are 1024-dimensional, so nothing raised — the results were just
  quietly worse.
- **The entities block was ordered by `PYTHONHASHSEED`.** It was built by
  iterating a `set` of note names, and with its 15-entry cap that meant two
  processes on the same index answered one query with a different SUBSET of
  entities. It now follows result rank.
- **`/api/similar` reported `similarity: 0.0` for some graph-only neighbours.**
  Their cosine is fetched with one prefiltered vector query, and that query ranks
  chunks, not notes: with parent-child chunking the row budget ran out before the
  least similar notes were reached — exactly the ones a wiki-link candidate tends
  to be. They reported 0.0 and fell under every client threshold, so the Obsidian
  panel hid rows the endpoint had returned. Whatever the budget cuts is now asked
  for again. On a 50,700-chunk vault this took a 40-candidate lookup from 36
  answers to 40.

### Notes

- Tests never touch a real data directory any more. `config.py` resolves
  `DATA_DIR` at import time, so a module setting the env var in its own header
  loses to whichever test file imported config first — and a test that deletes
  the v2 flag file then deletes the one in `~/.vault-search`. The new
  `server/tests/conftest.py` sets it before any test module is imported, and the
  flag-file tests override the module attribute outright.
- 57 tests, no network and no lancedb required for most of them.

## [2.8.1] — 2026-09-05 — Related Notes showed nothing: /api/similar reported a fusion rank, not a similarity

If the Obsidian plugin's Related Notes panel (🔗) has been answering "No results
with similarity ≥ 0.50" for every note in your vault, this release is the fix.
Nothing was wrong with your index; the number the panel filtered on was never a
similarity in the first place. Reported by @tony860616-sudo in #5, with the root
cause already located — thank you.

### Fixed

- **`/api/similar` and the `vault_similar` MCP tool now report a real cosine
  similarity.** Both rank their candidates by Reciprocal Rank Fusion over the
  semantic and wiki-link-graph rankings, then wrote that fusion score into the
  `similarity` field. An RRF score is bounded by `2/RRF_K` — about 0.033, with a
  single-ranking top hit at 0.0167 — so it could never approach the 0.5 and 0.7
  thresholds the plugin's panel, and its chat context picker, are calibrated for.
  Every result was filtered out no matter how close the content actually was.
  The real cosine the LanceDB search already computed was discarded before the
  overwrite. `similarity` now means on these two paths exactly what it means on
  `/api/search`: cosine under the same path and recency weighting.
- **The fusion ranking is unchanged.** `rerank()` takes a new optional
  `rank_key` argument: it orders by that field under the same weights and
  consumes it, leaving `similarity` to report the similarity. Graph-led results
  still outrank closer-but-unlinked notes exactly as before — only the number
  shown next to them changed.
- **Graph-only candidates carry a real number too.** A note that reaches the
  result set through the wiki-link graph never went through the vector search,
  so it had no similarity at all. One prefiltered vector query now fetches
  theirs. Notes with no indexed content (an attachment, an empty stub) honestly
  report 0.00 and fall below the panel's threshold, where they belong.

### Notes

- `server/tests/test_similarity_reporting.py` pins all of it: the reported value
  is the cosine, the fused ordering survives, the weights reach both numbers,
  and the ranking field never leaks into a result. Plain stdlib, no lancedb.
- Restart the API server to pick this up — the plugin talks to the running
  process, not to the files on disk.

## [2.8.0] — 2026-08-21 — Unbounded disk growth fixed: LanceDB old versions are now purged

If your `lance_db/` directory keeps growing even though your vault does not, this
release is the fix — and the growth it stops is not small. LanceDB snapshots the
whole table state on **every** write and never deletes those snapshots on its own.
Neither indexer ever asked it to, so every incremental run left one more full
version on disk. On the author's machine the textbook index reached 51 versions
and 13 GB on disk for what compacts to under 4 GB, and eventually filled the drive
to the point that indexing itself crashed with `LanceError(IO): os error 112`
(compaction needs scratch space, so the failure shows up exactly when you can least
afford it). A friend running this project reported the same mystery shrinkage.

Two compounding causes, both fixed:

### Fixed

- **Both indexers now purge old table versions at the end of every run**
  (`table.optimize(cleanup_older_than=0)`, with a `compact_files()` +
  `cleanup_old_versions()` fallback for older lancedb). Cleanup never touches the
  current version, so live queries are unaffected. Note that `compact_files()`
  alone — what `textbook_indexer.py` did before — actually made things *worse*:
  it wrote the merged fragments as yet another version while all previous
  versions stayed on disk.
- **`indexer.py` incremental mode batches its deletes.** It used to issue one
  `table.delete()` per modified file, and each of those calls is a full version
  snapshot; re-indexing after touching 200 notes meant 200 snapshots before this
  release. Deletes now go through one `file IN (...)` predicate per 400 files
  (matching what `textbook_indexer.py` already did), so the same run writes a
  handful of versions instead.

### Added

- `server/tests/test_version_cleanup.py`: three tests covering version purge,
  delete batching, and quote escaping in the batched predicate. They need
  `lancedb` installed and skip themselves cleanly on the hermetic CI runners.
  All three were mutation-verified (cleanup made a no-op, batching reverted to
  per-file, escaping removed — each mutation is caught).

### For existing installs

The fix prevents future growth but does not shrink what is already on disk —
old versions are purged by the *next* index run. Run either indexer once (even a
no-op incremental run cleans up), or reclaim immediately without re-indexing:

```python
import lancedb
from datetime import timedelta
db = lancedb.connect("path/to/lance_db")
for name in db.table_names():
    db.open_table(name).optimize(cleanup_older_than=timedelta(0))
```

## [2.7.0] — 2026-08-19 — Low-memory server and MCP client, OpenBLAS cap, tests in CI

Nothing here changes what a search returns. It is all about what the tooling costs
to keep running, plus the first CI that actually executes the test suite.

One thread runs through it, and it is worth stating up front because it nearly sent
this release out with the wrong conclusion. The resident `api_server` appeared to
hold the whole index in 116 MB while a standalone `mcp_server.py` needed 2.2 GB for
the same `textbook_search`. It did not. `uvicorn(reload=True)` forks a supervisor
that owns the port and a child that owns the app; 116 MB was the supervisor, and the
child beside it held 2,211 MB. Both paths ran the same code and paid the same price.
The real cost was two resident structures neither path needed to keep.

### Added

- **`server/mcp_thin.py`, a thin MCP client, and the two API endpoints it talks to
  (`GET /api/mcp/tools`, `POST /api/mcp/call`).** Registering `mcp_server.py`
  directly gives every MCP client session its own copy of the server, and each copy
  opens lancedb plus, on first use, the textbook tokenizer and the graph cache.
  Measured on the author's vault after the memory fixes below, one such session holds
  103 MB at import, 293 MB after a single `vault_search`, and 338 MB once a
  `textbook_search` has run — before those fixes that last step reached 2.2 GB.
  `mcp_thin.py` uses only the standard library, holds no index, and forwards each
  call to a running `api_server.py`; the same three steps leave it at 14 MB.

  Behaviour is identical by construction: `/api/mcp/call` dispatches to the same
  `handle_tool_call()` the standalone server uses, and `/api/mcp/tools` returns the
  same `TOOLS` list. The trade-off is a dependency on the API server being up, so
  the client degrades rather than failing: `tools/list` falls back to the last
  cached schema (`<data dir>/mcp_tools_cache.json`) so the session still starts, and
  a tool call returns a readable error naming the URL it could not reach.

  This is opt-in. `mcp_server.py` still works exactly as before, and remains the
  right choice for a single session with no API server running.

- **A `Tests` workflow.** `.github/workflows/tests.yml` byte-compiles every module
  under `server/` and runs `server/tests/test_file_selection.py` on Python 3.10,
  3.11 and 3.12. The repo has had those 17 tests since 2.6.0 but nothing ran them.
  They are plain stdlib and never touch lancedb or ollama, so CI needs no index and
  installs no dependencies.

  Its first run earned its keep: `test_whole_book_match_is_case_insensitive` used a
  `FULL_TEXT.MD` fixture, which `rglob("*.md")` never sees on Linux, so the test had
  been passing on Windows for a reason that does not hold elsewhere. The fixture is
  now `FULL_TEXT.md` — uppercase basename, lowercase extension — which is the case
  rule `file_selection.py` actually owns. Worth knowing separately: on Linux a
  genuinely `.MD`-suffixed file is not indexed at all, because the indexer's glob is
  case-sensitive there. That behaviour is unchanged here.

### Changed

- **Textbook parents are read per search instead of being held in a dict.**
  `_load_parent_map()` pulled every parent row into memory at first use: 2,120 MB
  of private bytes for a 162,864-parent corpus. A search only ever needs the ~20
  parents it is about to return, so `fetch_textbook_parents()` asks LanceDB for
  exactly those with one `parent_id IN (...)` query — 13–17 ms warm, against the
  3–4 s an end-to-end search already spends embedding the query. The vault-side
  parent map went the same way in the author's copy.

  This also retires the mtime + `READY.*` marker reload dance and the
  `POST /api/admin/reload-parents` endpoint it existed for: every search now reads
  the current index, so a reindex has no stale window. The route still answers, so
  old callers do not 404, but it reports that it did nothing.

- **The Qwen3 tokenizer loads through `tokenizers`, not `transformers`.**
  `AutoTokenizer` imports torch, which cost 371 MB of private bytes for what is
  pure Rust tokenization; the `tokenizers` backend costs ~58 MB and never touches
  torch. `get_tokenizer()` returns a thin adapter with the same `encode`/`decode`
  shape, and prefers the local Hugging Face cache so an offline machine still
  starts. Verified identical before the swap on 300 real parent texts plus empty,
  whitespace, CJK, emoji and 5000-character inputs: same ids, same decode output,
  zero mismatches. `requirements.txt` swaps `transformers` for `tokenizers` +
  `huggingface-hub`.

- **`api_server.py` no longer starts uvicorn's reloader by default.** Pass
  `--reload` while editing the server; `--no-reload` is still accepted. Besides the
  supervisor's own ~116 MB, the reloader's child process has no readable command
  line and answers no HTTP, which is exactly the shape a stray-process sweep flags
  as suspect.

- **`api_server.py` and `mcp_server.py` cap OpenBLAS at one thread on import.**
  numpy bundles its own OpenBLAS, which pre-commits one scratch buffer per CPU core
  the moment numpy is imported: 373 MB of private bytes against 19 MB capped,
  measured on a 12-thread machine with numpy 2.5.1. Neither process does linear
  algebra of its own — Ollama computes embeddings, LanceDB does the vector math — so
  the buffers were pure overhead. `os.environ.setdefault` is used, so exporting your
  own `OPENBLAS_NUM_THREADS` still wins.

- **`_textbook_search_v2()` split into a thin Markdown wrapper over a new
  `_textbook_search_v2_raw()` that returns the structured result list.** Since the
  MCP output moved to compact Markdown, the only way to consume results
  programmatically was to parse that Markdown back — which is why the author's own
  smoke benchmark had been broken since the change. The missing-index case now
  raises `TextbookIndexMissing` and the wrapper converts it to the same error JSON
  as before, so the tool's output is byte-for-byte unchanged.

### Measured

One MCP session, on the author's vault (private bytes, 162,864-parent textbook
corpus):

| step | 2.6.0 | 2.7.0 |
|---|---|---|
| import | 105 MB | 103 MB |
| after one `vault_search` | 367 MB | 293 MB |
| after one `textbook_search` | 2,225 MB | 338 MB |
| the same three steps via `mcp_thin.py` | — | 14 MB |

The 20-query textbook smoke suite passes 20/20 before and after, and the 17 stdlib
tests pass on 3.10–3.12.

## [2.6.0] — 2026-08-16 — Clean-install fixes, textbook-only installs, whole-book deduplication

Three community reports, all reproducible, all fixed. Thanks to
[@drivysu](https://github.com/drivysu) (PRs #1, #2, #3) and
[@tony860616-sudo](https://github.com/tony860616-sudo) (issue #4).

### Fixed

- **`pandas` was missing from `requirements.txt`, so search was broken on every
  clean install** (#1, #4). Every LanceDB read goes through `.to_pandas()` — 8 call
  sites in `api_server.py`, 7 in `mcp_server.py`, 3 in `textbook_indexer.py` — but
  `pyarrow` does not pull pandas in, it only shims it if already present. Indexing
  appeared to succeed and then every query died with
  `ModuleNotFoundError: No module named 'pandas'`. The failure surfaced late and
  read like an index problem: the imports sit inside call paths, so
  `claude mcp list` still reported the server **✔ Connected** and the error only
  appeared on the first tool call — in one report, after a multi-hour index had
  already been built. Reported independently on macOS 15 and Windows 11.
- **Textbook-only installs could not call any tool** (#3). `handle_tool_call()`
  opened the `vault` notes table before dispatching, so an install that indexes
  only a textbook corpus — `VAULT_SEARCH_TEXTBOOK_PATH` set, the vault itself never
  indexed — failed every `textbook_search` with
  `ValueError: Table 'vault' was not found`, even though that tool never reads the
  notes table. The notes table is now opened only by the three tools that use it
  (`vault_search`, `vault_related`, `vault_similar`), and `vault_stats` degrades to
  reporting the textbook half instead of failing the whole call.

### Added

- **`VAULT_SEARCH_TEXTBOOK_SKIP_FILES`** (#2) — comma-separated basenames never
  indexed from the textbook corpus, default `INDEX.md`. Previously hardcoded.
- **`VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES`** — whole-book files that duplicate
  per-chapter siblings, default `full_text.md`. See *Changed* below. Set it to an
  empty value to restore the old behaviour.
- **`VAULT_SEARCH_TEXTBOOK_CHAPTER_PATTERN`** — what a chapter filename looks like,
  and therefore the evidence that the whole-book file beside it is redundant. The
  default matches `ch01_x.md`, `01_x.md` and `pages_0001-0030.md`. A corpus naming
  chapters some other way matches nothing and keeps both files — duplicates, never
  data loss.
- **`server/tests/test_file_selection.py`** — the repo's first tests. 17 cases,
  stdlib only, no lancedb/ollama import, runnable as
  `python server/tests/test_file_selection.py` or under pytest.

### Changed

- **Whole-book files are no longer indexed alongside their chapter files.**
  Converters such as [textbook-to-note](https://github.com/drpwchen/textbook-to-note)
  emit both `full_text.md` and `chNN_*.md` for any PDF with bookmarks. Indexing both
  embedded every page twice: index time doubled, and each query returned the same
  passage twice — once attributed to the chapter, once to the whole book — burning
  the `limit` budget. On the 399-book corpus this was developed against, 340 books
  were affected.

  The rule is **per book, not global**: `full_text.md` is dropped only where a
  sibling file whose name looks like a chapter exists. A book whose whole-book file
  is its only copy (no PDF bookmarks to split on) keeps it and stays searchable —
  59 books on that same corpus. This is deliberately not what
  `VAULT_SEARCH_TEXTBOOK_SKIP_FILES` does; see the footgun below.

  "Looks like a chapter" is a name test, not "any other markdown in the folder".
  The looser rule would let a stray `README.md` stand in for a chapter split, drop
  the only real copy of the book, and have `orphan_cleanup()` delete its rows.

- **Basename matching is now case-insensitive.** This runs on Windows and macOS,
  where `Index.md` and `INDEX.md` are the same file. Comparing case-sensitively
  there let a differently-cased index page pass as book content — with the looser
  chapter rule above, that was enough to displace the book itself.

- **`vault_related` no longer opens the notes table.** It answers from the
  wiki-link graph alone and never reads that table, so a textbook-only install can
  use it too. This completes the boundary PR #3 drew.

  **Upgrade note:** the first indexer run after upgrading drops the duplicate rows
  via `orphan_cleanup()` — any run that is not `--book <name>`, incremental or full
  rebuild alike. This deletes rows, it does not re-embed; the surviving chapter rows
  are untouched. Set `VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES=` to keep the old
  behaviour.

- **`VAULT_SEARCH_TEXTBOOK_SKIP_FILES` can no longer silently erase a book.** The
  skip list is global, but "is the whole-book file redundant?" is a per-book
  question — the footgun @drivysu flagged on PR #2 after deploying it. On a mixed
  corpus, `VAULT_SEARCH_TEXTBOOK_SKIP_FILES=INDEX.md,full_text.md` dropped every
  unsplit book, and the next `--incremental` run's `orphan_cleanup()` then deleted
  its already-indexed rows: no error, the book just stopped being retrievable. If
  the configured skip list would leave a directory with nothing to index, its files
  are now kept and a `[WARN]` names the directory. `INDEX.md` is exempt from the
  guard — it is never a book's own content.

- File-selection rules moved to `server/file_selection.py` so they can be tested
  without importing lancedb or ollama. No behaviour change from the move itself.
