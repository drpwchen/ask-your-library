"""
Vault Semantic Indexer v2 — parent-child chunking.

Ports the parent-child architecture from textbook_indexer.py to your notes.
The idea: embed SMALL child chunks so the vector hit is precise, but return the
surrounding subsection (the parent) so the answer has context. A one-tier index
has to choose between the two; this one does not.

Writes to SEPARATE tables (vault_chunks_v2 / vault_parents_v2). The v1 `vault`
table and indexer.py are left untouched, so you can build v2, compare the two,
and only then switch. Retrieval reads v2 only when the flag file exists:

    touch ~/.vault-search/vault_v2.enabled     # switch to v2, no restart needed
    rm    ~/.vault-search/vault_v2.enabled     # back to v1

Vault-specific adaptations vs textbook_indexer:
  - frontmatter stripped; tags/aliases carried into both schemas
  - heading levels remapped per note (shallowest level found -> H2, next -> H3).
    Notes are inconsistent about whether they start at # or ##, while
    build_parents keys on H2/H3.
  - no page markers (page_start/end always 0)
  - the textbook "book" slot holds the top-level folder and "chapter" holds the
    note name, so a child's embed prefix reads "[Folder — Note > Section]"
  - whole-note fallback parent for stub notes build_parents would drop: v1
    indexed anything over 20 characters, and a title plus one line still
    deserves to be findable

Usage:
    python vault_indexer_v2.py                   # full rebuild (drops v2 tables)
    python vault_indexer_v2.py --incremental     # only changed files
    python vault_indexer_v2.py --limit 50        # smoke test on first N files
    python vault_indexer_v2.py --profile v3 --db ~/.vault-search/lance_db_v3
                                                 # v3 profile into a NEW database
                                                 # (see vault_v3.py)
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa

from textbook_indexer import (
    # chunking machinery (module constants PARENT_MAX_TOKENS etc. apply as-is)
    parse_md_pages_and_headings,
    build_parents,
    build_children_for_parent,
    tok_len,
    # embedding + resilience
    get_tokenizer,
    LatencyWatchdog,
    upsert_error_log,
    _l2_normalize,
    compact_and_cleanup,
    PRUNE_WINDOW_FINAL,
    # constants reused verbatim
    EMBEDDING_DIM,
    CHUNKING_VERSION,
    ALGORITHM_VERSION,
    CHILD_TARGET_TOKENS,
    CHILD_OVERLAP_TOKENS,
    PARENT_MAX_TOKENS,
    PARENT_MIN_TOKENS,
    TABLE_HARD_MAX_TOKENS,
    OLLAMA_NUM_CTX,
    WATCHDOG_BASELINE_BATCHES,
    WATCHDOG_MULTIPLIER,
    WATCHDOG_TRIGGER_CONSECUTIVE,
    BATCH_SIZE,
    PARENT_BATCH_SIZE,
    now_iso,
)
from indexer import (
    VAULT_PATH,
    collect_md_files,
    parse_frontmatter,
    file_hash,
    get_folder,
    _escape_sql,
)

# --- Config (resolved from environment / .env via config.py) ---
from config import (
    DB_PATH,
    OLLAMA_HOST,
    VAULT_V2_EMBEDDING_MODEL as EMBEDDING_MODEL,
    VAULT_V2_HASH_CACHE as HASH_CACHE_PATH,
    VAULT_V2_ERROR_LOG as ERROR_LOG_PATH,
    VAULT_QUERY_PREFIX,
    VAULT_QUERY_TEMPLATE_NAME,
    GPU_LEASE_PATH,
    GPU_LEASE_NAME_VAULT,
    GPU_LEASE_ACQUIRE_TIMEOUT,
    GPU_LEASE_MIN_HOLD,
    GPU_LEASE_QUEUE_DIR,
    template_version,
)
from gpu_lease_client import GpuLease, acquire_or_exit
import vault_v3 as V3

# This run's turn at the GPU. Registers under its own name: the vault and
# textbook indexers are separate jobs and must never queue behind each other.
# A no-op unless VAULT_SEARCH_GPU_LEASE points at a lease script.
_LEASE = GpuLease(
    GPU_LEASE_NAME_VAULT,
    script=GPU_LEASE_PATH,
    queue_dir=GPU_LEASE_QUEUE_DIR,
    acquire_timeout=GPU_LEASE_ACQUIRE_TIMEOUT,
    min_hold_s=GPU_LEASE_MIN_HOLD,
)

# =============================================================================
# Vault v2 constants
# =============================================================================

READY_MARKER_DIR = DB_PATH  # READY_VAULT.{generation_id} marker lives beside the index

VCHUNKS_TABLE = "vault_chunks_v2"
VPARENTS_TABLE = "vault_parents_v2"

# "v2" (default: what the nightly incremental run of an existing index uses,
# unchanged) or "v3" (vault_v3.py: YAML frontmatter, exclusions, hard caps,
# scope column). Set by main(); v3 only ever writes to a database named with --db.
PROFILE = "v2"

# Name + a hash of the prefix itself, so a log line can never name one template
# while a different prefix is actually in use. Recorded per row and per search;
# deliberately NOT part of the indexing signature (see compute_indexing_signature).
VAULT_QUERY_TEMPLATE_VERSION = template_version(VAULT_QUERY_TEMPLATE_NAME, VAULT_QUERY_PREFIX)

# Vault stubs are meaningful (a title + one line still deserves retrieval);
# textbook's MIN_EMBED_TOKENS=20 would drop them.
VAULT_MIN_CHILD_TOKENS = 10
VAULT_MIN_PARENT_TOKENS = 5

# Embed-input hard cap. Some notes (e.g. exports from other apps that arrive as
# one unbroken wall of text) produce a parent whose embed text exceeds num_ctx
# 4096, and Ollama then returns 400 for the WHOLE batch, deterministically, with
# every retry burning minutes. Stored text stays full; only the embedding input
# is truncated.
MAX_EMBED_INPUT_TOKENS = 3800
# Prevent ollama idle-unload during long CPU chunking gaps between batches —
# model reload was measured at 105-195s under memory pressure.
EMBED_KEEP_ALIVE = "2h"

# Token-budgeted batching. Counting items alone is not enough: 64 near-cap
# parents is roughly 240K tokens in one request, which crashes the runner and
# loses the whole batch. The budget caps the request, whichever limit hits first.
BATCH_TOKEN_BUDGET = 48_000
# Runner reload under memory pressure measured 100-360s; wait this long for
# health before the one-by-one salvage pass.
OLLAMA_HEALTH_WAIT_S = 420

FLUSH_EVERY_FILES = 200
HASH_SAVE_EVERY_FILES = 400

VCHUNKS_SCHEMA = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("parent_id", pa.utf8()),
    pa.field("file", pa.utf8()),
    pa.field("note", pa.utf8()),
    pa.field("folder", pa.utf8()),
    pa.field("section_path", pa.utf8()),
    pa.field("section_path_raw", pa.utf8()),
    pa.field("tags", pa.utf8()),
    pa.field("aliases", pa.utf8()),
    pa.field("chunk_kind", pa.utf8()),
    pa.field("chunk_idx", pa.int32()),
    pa.field("n_siblings", pa.int32()),
    pa.field("token_count", pa.int32()),
    pa.field("heading_origin", pa.utf8()),
    pa.field("chunking_version", pa.utf8()),
    pa.field("embedding_model", pa.utf8()),
    pa.field("query_template_version", pa.utf8()),
    pa.field("indexing_signature", pa.utf8()),
    pa.field("mtime", pa.float64()),
])

VPARENTS_SCHEMA = pa.schema([
    pa.field("parent_id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("file", pa.utf8()),
    pa.field("note", pa.utf8()),
    pa.field("folder", pa.utf8()),
    pa.field("section_path", pa.utf8()),
    pa.field("section_path_raw", pa.utf8()),
    pa.field("tags", pa.utf8()),
    pa.field("aliases", pa.utf8()),
    pa.field("token_count", pa.int32()),
    pa.field("heading_origin", pa.utf8()),
    pa.field("chunking_version", pa.utf8()),
    pa.field("embedding_model", pa.utf8()),
    pa.field("query_template_version", pa.utf8()),
    pa.field("indexing_signature", pa.utf8()),
    pa.field("mtime", pa.float64()),
])


def chunks_schema() -> pa.Schema:
    return VCHUNKS_SCHEMA.append(pa.field("scope", pa.utf8())) if PROFILE == "v3" else VCHUNKS_SCHEMA


def parents_schema() -> pa.Schema:
    return VPARENTS_SCHEMA.append(pa.field("scope", pa.utf8())) if PROFILE == "v3" else VPARENTS_SCHEMA


def _signature_parts() -> list[str]:
    """The settings that actually determine what is stored in the tables."""
    parts = [
        ALGORITHM_VERSION,
        CHUNKING_VERSION,
        EMBEDDING_MODEL,
        f"CHILD_TARGET={CHILD_TARGET_TOKENS}",
        f"CHILD_OVERLAP={CHILD_OVERLAP_TOKENS}",
        f"PARENT_MAX={PARENT_MAX_TOKENS}",
        f"PARENT_MIN={PARENT_MIN_TOKENS}",
        f"TABLE_MAX={TABLE_HARD_MAX_TOKENS}",
        f"VAULT_MIN_CHILD={VAULT_MIN_CHILD_TOKENS}",
    ]
    if PROFILE == "v3":
        parts += ["profile=v3", f"CHILD_HARD={V3.CHILD_HARD_MAX}",
                  f"PARENT_HARD={V3.PARENT_HARD_MAX}", "frontmatter=yaml"]
    return parts


def compute_indexing_signature() -> str:
    """Fingerprint of the settings the stored vectors depend on.

    The query prefix is deliberately absent: documents are embedded without it
    (see vault_embed_texts), so changing it changes queries only and a "rebuild"
    would re-emit byte-identical vectors. It is recorded in the search log
    instead. See textbook_indexer.compute_indexing_signature for the same rule.
    """
    return hashlib.sha256("|".join(_signature_parts()).encode()).hexdigest()[:16]


def legacy_indexing_signature() -> str:
    """The pre-2.9.0 signature, which included the query template name.

    Kept so a corpus built by an older version is recognized as compatible
    instead of demanding a pointless full rebuild after upgrading.
    """
    parts = _signature_parts()
    parts.insert(3, VAULT_QUERY_TEMPLATE_NAME)  # sat right after the model name
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def compatible_signatures() -> set[str]:
    """Signatures an existing index may legitimately carry."""
    return {compute_indexing_signature(), legacy_indexing_signature()}


def check_signature(chunks_table, sig: str) -> None:
    """Refuse to add rows to an index built under different chunking rules.

    Mixing two chunkings in one table is silent corruption: the rows coexist,
    every search reads both, and nothing ever reports an error. Better to stop
    and make the rebuild explicit.
    """
    try:
        sample = chunks_table.search().select(["indexing_signature"]).limit(1).to_pandas()
    except Exception:
        return  # empty or unreadable table — nothing to compare against
    if sample.empty:
        return
    old_sig = sample["indexing_signature"].iloc[0]
    if old_sig == sig:
        return
    if old_sig == legacy_indexing_signature():
        print(
            f"[migrate] index carries the pre-2.9.0 signature ({old_sig}); the "
            f"settings behind it are unchanged, so it stays valid. New rows are "
            f"stamped {sig}.",
            flush=True,
        )
        return
    print(
        f"[WARN] indexing_signature mismatch (old={old_sig}, new={sig}). "
        f"The existing index was built with different chunking/model settings. "
        f"Re-run without --incremental for a full rebuild.",
        flush=True,
    )
    sys.exit(2)


def _truncate_for_embed(text: str) -> str:
    if tok_len(text) <= MAX_EMBED_INPUT_TOKENS:
        return text
    tk = get_tokenizer()
    ids = tk.encode(text, add_special_tokens=False)[:MAX_EMBED_INPUT_TOKENS]
    return tk.decode(ids, skip_special_tokens=True)


def _wait_for_ollama(client) -> bool:
    """Poll until the embedding runner answers (it may be reloading for minutes)."""
    deadline = time.time() + OLLAMA_HEALTH_WAIT_S
    while time.time() < deadline:
        try:
            client.embed(model=EMBEDDING_MODEL, input=["ok"], keep_alive=EMBED_KEEP_ALIVE)
            return True
        except Exception:
            time.sleep(10)
    return False


def _validate(emb, meta, error_log):
    import math
    if any(math.isnan(v) for v in emb):
        upsert_error_log(error_log, meta, "NaN", "embedding contained NaN values")
        return None
    return _l2_normalize(emb)


def vault_embed_texts(texts, client, error_log, chunk_meta, watchdog):
    """textbook_indexer.embed_texts with vault fixes: input truncation (400-proof),
    keep_alive (no idle-unload), and a one-by-one salvage pass after batch failure
    (a runner crash costs at most the poison item, not the whole batch).

    Note there is no query prefix here: documents are embedded as-is. Only
    queries get VAULT_QUERY_PREFIX, at search time.
    """
    texts = [_truncate_for_embed(t) for t in texts]
    t0 = time.time()
    last_err = None
    embs = None
    for attempt in range(4):
        try:
            r = client.embed(
                model=EMBEDDING_MODEL,
                input=texts,
                options={"num_ctx": OLLAMA_NUM_CTX},
                keep_alive=EMBED_KEEP_ALIVE,
            )
            embs = r["embeddings"]
            watchdog.observe(time.time() - t0)
            break
        except Exception as e:
            last_err = e
            # deterministic client error → retrying the same batch is pure waste
            if "400" in str(e):
                break
            if attempt < 3:
                time.sleep(3 * (attempt + 1))

    if embs is not None:
        return [_validate(emb, meta, error_log) for emb, meta in zip(embs, chunk_meta)]

    # Batch failed — runner likely crashed and is reloading (measured 100-360s).
    # Wait for health, then salvage item by item.
    print(f"  [salvage] batch of {len(texts)} failed ({str(last_err)[:80]}); "
          f"waiting for runner + retrying one-by-one", flush=True)
    _wait_for_ollama(client)
    out = []
    for text, meta in zip(texts, chunk_meta):
        try:
            r = client.embed(
                model=EMBEDDING_MODEL,
                input=[text],
                options={"num_ctx": OLLAMA_NUM_CTX},
                keep_alive=EMBED_KEEP_ALIVE,
            )
            out.append(_validate(r["embeddings"][0], meta, error_log))
        except Exception as e:
            upsert_error_log(error_log, meta, "item_failure", str(e))
            out.append(None)
            _wait_for_ollama(client)  # poison item may have crashed the runner again
    return out


# =============================================================================
# Vault-specific chunk preparation
# =============================================================================

def remap_heading_levels(headings: list[dict]) -> list[dict]:
    """Notes use inconsistent top heading levels (# vs ##). build_parents keys on
    H2 (parent boundary) / H3 (sub-split), so remap: shallowest level in this
    note → 2, next → 3, deeper dropped."""
    levels = sorted({h["level"] for h in headings})
    if not levels:
        return []
    mapping = {levels[0]: 2}
    if len(levels) >= 2:
        mapping[levels[1]] = 3
    return [{**h, "level": mapping[h["level"]]} for h in headings if h["level"] in mapping]


def whole_note_parent(body: str, folder: str, rel_path: str, note: str, mtime: float) -> dict:
    pid_raw = f"{folder}|{rel_path}|whole_note|0"
    return {
        "parent_id": hashlib.md5(pid_raw.encode("utf-8", errors="replace")).hexdigest(),
        "text": body.strip(),
        "file": rel_path,
        "book": folder,       # textbook-slot naming; mapped to `folder` in records
        "chapter": note,      # mapped to `note` in records
        "section_path": "",
        "section_path_raw": "",
        "page_start": 0,
        "page_end": 0,
        "token_count": tok_len(body),
        "heading_origin": "whole_note",
        "figure_ids": [],
        "has_figure_ref": False,
        "mtime": mtime,
    }


def chunk_note(body: str, folder: str, rel_path: str, note: str, mtime: float):
    """Returns (parents, children, parent_embed_texts)."""
    parsed = parse_md_pages_and_headings(body)
    headings = remap_heading_levels(parsed["headings"])
    parents = build_parents(parsed["lines"], headings, folder, rel_path, note, mtime)

    # Stub fallback: v1 indexed anything >20 chars; keep stubs searchable
    if not parents and body.strip() and tok_len(body) >= VAULT_MIN_PARENT_TOKENS:
        parents = [whole_note_parent(body, folder, rel_path, note, mtime)]

    children = []
    for p in parents:
        children.extend(build_children_for_parent(p))
    children = [c for c in children if c["token_count"] >= VAULT_MIN_CHILD_TOKENS]

    embed_texts_p = []
    for p in parents:
        prefix = f"[{p['book']} — {p['chapter']}"
        if p["section_path"]:
            prefix += f" > {p['section_path']}"
        prefix += "]\n"
        embed_texts_p.append(prefix + p["text"])
    return parents, children, embed_texts_p


# =============================================================================
# Error log / hash cache (separate files from textbook's)
# =============================================================================

def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_json(path: Path, data: dict, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def write_ready_marker(generation_id: str):
    READY_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    for old in READY_MARKER_DIR.glob("READY_VAULT.*"):
        try:
            old.unlink()
        except Exception:
            pass
    marker = READY_MARKER_DIR / f"READY_VAULT.{generation_id}"
    marker.write_text(now_iso(), encoding="utf-8")
    print(f"[ok] READY marker written: {marker.name}", flush=True)


# =============================================================================
# Main indexer
# =============================================================================

def get_or_create_table(db, name: str, schema: pa.Schema):
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema, mode="overwrite")


def _unload_embedder(client) -> None:
    """Drop the embedding model from Ollama now instead of after EMBED_KEEP_ALIVE.

    The 2h keep_alive only exists to bridge idle gaps INSIDE a run. Left in place
    after the run it pins the runner (~4 GB commit) for two more hours. An empty
    input with keep_alive=0 is Ollama's documented unload call. Best-effort: a
    failure is logged and never fails the index run.
    """
    try:
        client.embed(model=EMBEDDING_MODEL, input=[], keep_alive=0)
        print(f"[unload] {EMBEDDING_MODEL} released from Ollama", flush=True)
    except Exception as e:
        print(f"[unload] could not unload {EMBEDDING_MODEL}: {e}", flush=True)


def index_vault_v2(incremental: bool = False, limit: int | None = None):
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        _index_vault_v2(client, incremental, limit)
    finally:
        _unload_embedder(client)


def _index_vault_v2(client, incremental: bool, limit: int | None):
    print(f"[start] vault_indexer_v2 — {now_iso()}", flush=True)
    print(f"  incremental={incremental} limit={limit}", flush=True)

    db = lancedb.connect(str(DB_PATH))
    if not incremental and limit is None:
        for t in (VCHUNKS_TABLE, VPARENTS_TABLE):
            try:
                db.drop_table(t)
                print(f"  full rebuild: dropped {t}", flush=True)
            except Exception:
                pass
    chunks_table = get_or_create_table(db, VCHUNKS_TABLE, chunks_schema())
    parents_table = get_or_create_table(db, VPARENTS_TABLE, parents_schema())

    sig = compute_indexing_signature()
    print(f"  indexing_signature={sig}", flush=True)
    if incremental:
        check_signature(chunks_table, sig)

    md_files = collect_md_files(VAULT_PATH)
    scope_map = {}
    if PROFILE == "v3":
        md_files, why = _v3_select(md_files)
        print(f"  [v3] excluded {sum(why.values())} files: {dict(why)}", flush=True)
        try:
            import scope as S
            from config import SCOPE_MAP_PATH
            scope_map = S.load_scope_map(SCOPE_MAP_PATH)
        except Exception as e:
            print(f"  [v3] scope map unreadable, scope column left empty: {e}", flush=True)
    if limit:
        md_files = md_files[:limit]
    print(f"  found {len(md_files)} md files", flush=True)

    hash_cache = load_json(HASH_CACHE_PATH) if incremental else {}
    error_log = load_json(ERROR_LOG_PATH)

    watchdog = LatencyWatchdog(
        WATCHDOG_BASELINE_BATCHES, WATCHDOG_MULTIPLIER, WATCHDOG_TRIGGER_CONSECUTIVE
    )

    pending_parents: list[tuple[str, dict, str]] = []   # (rel_path, parent, embed_text)
    pending_children: list[tuple[str, dict]] = []       # (rel_path, child)
    pending_chunk_records: list[dict] = []
    pending_parent_records: list[dict] = []
    file_state: dict[str, dict] = {}
    file_meta: dict[str, dict] = {}                     # rel_path → {tags, aliases}
    batch_tokens = {"p": 0, "c": 0}                     # token-budget batching

    stats = {"files": 0, "skipped": 0, "chunks": 0, "parents": 0, "failed": 0, "dropped_short": 0}
    t_start = time.time()

    def _flush_parents():
        if not pending_parents:
            return
        batch_meta = [
            {"file": rp, "parent_id": p["parent_id"], "chunk_idx": -1, "text": p["text"]}
            for rp, p, _ in pending_parents
        ]
        texts = [t for _, _, t in pending_parents]
        embs = vault_embed_texts(texts, client, error_log, batch_meta, watchdog)
        for (rp, p, _), emb in zip(pending_parents, embs):
            st = file_state[rp]
            st["done_p"] += 1
            if emb is None:
                stats["failed"] += 1
                st["failed"] = True
                continue
            fm = file_meta.get(rp, {})
            rec = {
                "parent_id": p["parent_id"], "vector": emb, "text": p["text"],
                "file": p["file"], "note": p["chapter"], "folder": p["book"],
                "section_path": p["section_path"], "section_path_raw": p["section_path_raw"],
                "tags": fm.get("tags", ""), "aliases": fm.get("aliases", ""),
                "token_count": p["token_count"], "heading_origin": p["heading_origin"],
                "chunking_version": CHUNKING_VERSION, "embedding_model": EMBEDDING_MODEL,
                "query_template_version": VAULT_QUERY_TEMPLATE_VERSION,
                "indexing_signature": sig, "mtime": p["mtime"],
            }
            if PROFILE == "v3":
                rec["scope"] = fm.get("scope", "")
            pending_parent_records.append(rec)
            st["ok_p"] += 1
            stats["parents"] += 1
        pending_parents.clear()

    def _flush_children():
        if not pending_children:
            return
        batch_meta = [
            {"file": rp, "parent_id": c["parent_id"], "chunk_idx": c["chunk_idx"], "text": c["text"]}
            for rp, c in pending_children
        ]
        texts = [c["text"] for _, c in pending_children]
        embs = vault_embed_texts(texts, client, error_log, batch_meta, watchdog)
        for (rp, c), emb in zip(pending_children, embs):
            st = file_state[rp]
            st["done_c"] += 1
            if emb is None:
                stats["failed"] += 1
                st["failed"] = True
                continue
            cid = hashlib.md5(
                f"{c['file']}|{c['parent_id']}|{c['chunk_idx']}".encode("utf-8")
            ).hexdigest()
            fm = file_meta.get(rp, {})
            rec = {
                "id": cid, "vector": emb, "text": c["text"],
                "parent_id": c["parent_id"], "file": c["file"],
                "note": c["chapter"], "folder": c["book"],
                "section_path": c["section_path"], "section_path_raw": c["section_path_raw"],
                "tags": fm.get("tags", ""), "aliases": fm.get("aliases", ""),
                "chunk_kind": c["chunk_kind"], "chunk_idx": c["chunk_idx"],
                "n_siblings": c["n_siblings"], "token_count": c["token_count"],
                "heading_origin": c["heading_origin"],
                "chunking_version": CHUNKING_VERSION, "embedding_model": EMBEDDING_MODEL,
                "query_template_version": VAULT_QUERY_TEMPLATE_VERSION,
                "indexing_signature": sig, "mtime": c["mtime"],
            }
            if PROFILE == "v3":
                rec["scope"] = fm.get("scope", "")
            pending_chunk_records.append(rec)
            st["ok_c"] += 1
            stats["chunks"] += 1
        pending_children.clear()

    def _finalize_completed_files():
        done = []
        for rp, st in file_state.items():
            if st["done_p"] >= st["expected_p"] and st["done_c"] >= st["expected_c"]:
                if not st["failed"]:
                    hash_cache[rp] = st["h"]
                    # The whole file is in the index now, so every earlier
                    # failure logged for it is resolved. Without this the log
                    # kept 852 long-fixed entries "pending" for months.
                    resolve_file_errors(error_log, rp)
                done.append(rp)
        for rp in done:
            del file_state[rp]
            file_meta.pop(rp, None)

    def _db_flush():
        nonlocal pending_chunk_records, pending_parent_records
        _flush_parents()
        _flush_children()
        _finalize_completed_files()
        if pending_chunk_records:
            chunks_table.add(pending_chunk_records)
            pending_chunk_records = []
        if pending_parent_records:
            parents_table.add(pending_parent_records)
            pending_parent_records = []

    for i, fpath in enumerate(md_files):
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [SKIP] read error {fpath.name}: {e}", flush=True)
            continue
        rel_path = str(fpath.relative_to(VAULT_PATH)).replace("\\", "/")
        h = file_hash(content)
        if incremental and hash_cache.get(rel_path) == h:
            stats["skipped"] += 1
            continue

        if PROFILE == "v3":
            meta, body = V3.parse_frontmatter_yaml(content, parse_frontmatter)
        else:
            meta, body = parse_frontmatter(content)
        if not body.strip():
            hash_cache[rel_path] = h
            continue
        if PROFILE == "v3":
            tags, aliases = V3.as_list(meta.get("tags")), V3.as_list(meta.get("aliases"))
        else:
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            aliases = meta.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]

        note = fpath.stem
        folder = get_folder(rel_path)
        try:
            parents, children, embed_texts_p = chunk_note(
                body, folder, rel_path, note, fpath.stat().st_mtime
            )
            if PROFILE == "v3":
                parents, children, embed_texts_p = _v3_cap(parents, children)
        except Exception as e:
            print(f"  [SKIP] chunk error {rel_path}: {e}", flush=True)
            continue
        if not parents:
            hash_cache[rel_path] = h
            continue

        # Idempotent re-index: clear prior rows for this file
        if incremental:
            esc = _escape_sql(rel_path)
            for t in (chunks_table, parents_table):
                try:
                    t.delete(f"file = '{esc}'")
                except Exception:
                    pass

        file_state[rel_path] = {
            "h": h, "expected_p": len(parents), "expected_c": len(children),
            "ok_p": 0, "ok_c": 0, "done_p": 0, "done_c": 0, "failed": False,
        }
        file_meta[rel_path] = {
            "tags": ", ".join(tags) if tags else "",
            "aliases": ", ".join(aliases) if aliases else "",
        }
        if PROFILE == "v3":
            import scope as S
            file_meta[rel_path]["scope"] = S.scope_of(folder, scope_map)

        for p, et in zip(parents, embed_texts_p):
            pending_parents.append((rel_path, p, et))
            batch_tokens["p"] += min(p["token_count"], MAX_EMBED_INPUT_TOKENS)
            if len(pending_parents) >= PARENT_BATCH_SIZE or batch_tokens["p"] >= BATCH_TOKEN_BUDGET:
                _flush_parents()
                batch_tokens["p"] = 0
        for c in children:
            pending_children.append((rel_path, c))
            batch_tokens["c"] += min(c["token_count"], MAX_EMBED_INPUT_TOKENS)
            if len(pending_children) >= BATCH_SIZE or batch_tokens["c"] >= BATCH_TOKEN_BUDGET:
                _flush_children()
                batch_tokens["c"] = 0

        stats["files"] += 1
        if stats["files"] % FLUSH_EVERY_FILES == 0:
            _db_flush()
            elapsed = time.time() - t_start
            rate = stats["files"] / max(elapsed, 0.01)
            eta = (len(md_files) - i - 1) / max(rate, 0.001)
            print(
                f"  [{i + 1}/{len(md_files)}] chunks={stats['chunks']} "
                f"parents={stats['parents']} failed={stats['failed']} "
                f"rate={rate:.2f} files/s eta={eta / 60:.0f} min",
                flush=True,
            )
            # A full run holds the lease for hours — outlive the stale-lease reaper.
            _LEASE.heartbeat()
        if stats["files"] % HASH_SAVE_EVERY_FILES == 0:
            save_json(HASH_CACHE_PATH, hash_cache)
            save_json(ERROR_LOG_PATH, error_log, indent=2)

        # Cooperative yield — see textbook_indexer for the reasoning. _db_flush()
        # writes out everything pending; after the yield another process owns
        # the GPU, so nothing may be left in flight.
        if _LEASE.yield_wanted():
            _db_flush()
            save_json(HASH_CACHE_PATH, hash_cache)
            save_json(ERROR_LOG_PATH, error_log, indent=2)
            _LEASE.yield_now(unload_model=EMBEDDING_MODEL)

    # Final flush
    _db_flush()
    save_json(HASH_CACHE_PATH, hash_cache)
    save_json(ERROR_LOG_PATH, error_log, indent=2)

    # Orphan cleanup (incremental only — full rebuild starts clean)
    if incremental:
        current = {str(p.relative_to(VAULT_PATH)).replace("\\", "/") for p in collect_md_files(VAULT_PATH)}
        for table, label in [(chunks_table, VCHUNKS_TABLE), (parents_table, VPARENTS_TABLE)]:
            try:
                indexed = set(
                    table.search().select(["file"]).limit(10**9).to_pandas()["file"].unique()
                )
                gone = indexed - current
                if gone:
                    quoted = ", ".join("'" + _escape_sql(f) + "'" for f in gone)
                    table.delete(f"file IN ({quoted})")
                    print(f"[orphan] deleted {len(gone)} orphan files from {label}", flush=True)
            except Exception as e:
                print(f"[orphan] {label} cleanup failed: {e}", flush=True)

    compact_and_cleanup(chunks_table, VCHUNKS_TABLE, PRUNE_WINDOW_FINAL)
    compact_and_cleanup(parents_table, VPARENTS_TABLE, PRUNE_WINDOW_FINAL)

    if limit is None:  # smoke tests don't flip the READY marker
        write_ready_marker(sig + "_" + str(int(time.time())))

    elapsed = time.time() - t_start
    print(f"\n[done] in {elapsed / 60:.1f} min", flush=True)
    print(f"  Files: {stats['files']} processed, {stats['skipped']} skipped", flush=True)
    print(f"  Chunks: {stats['chunks']} | Parents: {stats['parents']}", flush=True)
    print(f"  Embed failures: {stats['failed']} (see {ERROR_LOG_PATH})", flush=True)
    if stats["failed"] > 0:
        print("  ⚠ dropped chunks — re-run with --incremental once GPU is free.", flush=True)


def _v3_select(md_files):
    from collections import Counter
    keep, why = [], Counter()
    extra = V3.extra_excluded_folders()
    for f in md_files:
        rel = str(f.relative_to(VAULT_PATH)).replace("\\", "/")
        reason = V3.excluded(rel, extra)
        if reason:
            why[reason] += 1
        else:
            keep.append(f)
    return keep, why


def _v3_cap(parents, children):
    """Hard length caps (vault_v3.py). Children are re-cut from capped parents."""
    def window(text, n):
        tk = get_tokenizer()
        ids = tk.encode(text, add_special_tokens=False)
        return [tk.decode(ids[i:i + n], skip_special_tokens=True) for i in range(0, len(ids), n)]

    capped = V3.cap_parents(parents, V3.PARENT_HARD_MAX, tok_len, window)
    if len(capped) != len(parents):
        children = []
        for p in capped:
            children.extend(build_children_for_parent(p))
        children = [c for c in children if c["token_count"] >= VAULT_MIN_CHILD_TOKENS]
    by_parent: dict = {}
    for c in children:
        by_parent.setdefault(c["parent_id"], []).append(c)
    children = []
    for pid, cs in by_parent.items():
        children.extend(V3.cap_children(cs, V3.CHILD_HARD_MAX, tok_len, window))
    embed_texts_p = []
    for p in capped:
        prefix = f"[{p['book']} — {p['chapter']}"
        if p["section_path"]:
            prefix += f" > {p['section_path']}"
        embed_texts_p.append(prefix + "]\n" + p["text"])
    return capped, children, embed_texts_p


def configure(db: str | None, profile: str) -> None:
    """Point this run at another database (state files move beside it) and/or
    the v3 profile. v3 refuses the live database: its rows would mix with v2's."""
    global DB_PATH, HASH_CACHE_PATH, ERROR_LOG_PATH, READY_MARKER_DIR, PROFILE
    PROFILE = profile
    if db:
        from textbook_indexer_v3 import check_target, live_paths
        target = Path(db).expanduser()
        check_target(target, None, live_paths())
        DB_PATH = target
        READY_MARKER_DIR = target
        state = target / "_vault_state"
        state.mkdir(parents=True, exist_ok=True)
        HASH_CACHE_PATH = state / "vault_hashes.json"
        ERROR_LOG_PATH = state / "vault_index_errors.json"
    elif profile == "v3":
        raise SystemExit("[refuse] --profile v3 needs --db pointing at a NEW database "
                         "(e.g. ~/.vault-search/lance_db_v3)")
    if profile == "v2" and (Path(DB_PATH) / "_vault_state").is_dir():
        # A v3-profile vault index (scope column, YAML frontmatter): default-
        # profile rows would mix into it. Pass --profile v3 --db <it> instead.
        raise SystemExit(f"[refuse] {DB_PATH} holds a v3-profile vault index; "
                         f"run with --profile v3 --db {DB_PATH}")


def resolve_file_errors(error_log: dict, rel_path: str, status: str = "fixed") -> int:
    """Mark every open error entry of one file resolved. Returns how many.

    File-level on purpose: when a file is re-indexed its parent ids may change,
    so an entry keyed by an old parent id can never be matched chunk by chunk.
    """
    n = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for e in error_log.values():
        if e.get("file") == rel_path and e.get("status") not in ("fixed", "obsolete"):
            e["status"] = status
            e["last_seen"] = now
            n += 1
    return n


def reconcile_error_log(error_log: dict, hash_cache: dict, indexed_files: set,
                        vault_path: Path) -> dict:
    """Close error entries that later runs already fixed — no embedding needed.

    fixed     the file's current content hash is in the cache (a cache entry is
              written only when every chunk of the file embedded) and the file
              has rows in the index
    obsolete  the file no longer exists
    open      anything else — it really still needs a re-run
    """
    out = {"fixed": 0, "obsolete": 0, "open": 0}
    for rp in sorted({e.get("file") for e in error_log.values()
                      if e.get("status") not in ("fixed", "obsolete")}):
        fp = vault_path / rp
        if not fp.exists():
            out["obsolete"] += resolve_file_errors(error_log, rp, "obsolete")
            continue
        h = file_hash(fp.read_text(encoding="utf-8", errors="replace"))
        if hash_cache.get(rp) == h and rp in indexed_files:
            out["fixed"] += resolve_file_errors(error_log, rp)
        else:
            out["open"] += sum(1 for e in error_log.values()
                               if e.get("file") == rp and e.get("status") not in ("fixed", "obsolete"))
    return out


def main():
    from config import require_vault_path
    require_vault_path()
    p = argparse.ArgumentParser(description="Vault indexer v2 (parent-child chunking)")
    p.add_argument("--reconcile-errors", action="store_true",
                   help="Close error-log entries for files that are already fully indexed "
                        "(no GPU, no embedding), then exit")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--limit", type=int, help="Smoke test: only first N files, no table drop")
    p.add_argument("--db", help="Write to this database instead of VAULT_SEARCH_DB_PATH "
                                "(state files go to <db>/_vault_state). The live DB is refused.")
    p.add_argument("--profile", choices=["v2", "v3"], default="v2",
                   help="v3 = YAML frontmatter, exclusions, hard caps, scope column (vault_v3.py)")
    p.add_argument("--no-gpu-lease", action="store_true",
                   help="Skip the GPU lease entirely (DANGEROUS: another GPU job can then "
                        "collide with the embedder and silently drop chunks; only for "
                        "machines with no lease script configured)")
    args = p.parse_args()
    configure(args.db, args.profile)

    if args.reconcile_errors:
        log = load_json(ERROR_LOG_PATH)
        db = lancedb.connect(str(DB_PATH))
        indexed = set()
        if VCHUNKS_TABLE in db.table_names():
            indexed = set(db.open_table(VCHUNKS_TABLE).search().select(["file"])
                          .limit(10**9).to_pandas()["file"].unique())
        res = reconcile_error_log(log, load_json(HASH_CACHE_PATH), indexed, VAULT_PATH)
        save_json(ERROR_LOG_PATH, log, indent=2)
        print(f"[reconcile] {res}", flush=True)
        return

    # Ordinary lease taker: always acquire before touching the GPU, refuse to
    # run without it. No lease script configured (the default) = no-op.
    if not args.no_gpu_lease:
        acquire_or_exit(_LEASE)
    try:
        index_vault_v2(incremental=args.incremental, limit=args.limit)
    finally:
        if _LEASE.held:
            ok, out = _LEASE.release()
            print(f"[gpu-lease] release: {'ok' if ok else 'FAILED — check the lease script'} "
                  f"{out.strip()[-160:]}", flush=True)


if __name__ == "__main__":
    main()
