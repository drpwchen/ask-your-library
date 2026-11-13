"""
vault-search — local-first semantic search for Obsidian.
Author: P.W. Chen · https://drpwchen.com · https://github.com/drpwchen

Central configuration for vault-search.

Every path, model name, and tunable is resolved here from environment variables,
with sensible defaults. Set them in a `.env` file (auto-loaded if python-dotenv is
installed) or export them in your shell. The ONLY value you must set is VAULT_PATH.

See `.env.example` in the repo root for the full list.
"""

import hashlib
import json
import os
from pathlib import Path

# Optionally load a .env file from the repo root (no hard dependency).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


def _path(env: str, default: str | None) -> Path | None:
    val = os.environ.get(env)
    if val:
        return Path(val).expanduser()
    return Path(default).expanduser() if default else None


def _csv(env: str, default: str) -> list[str]:
    raw = os.environ.get(env, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def template_version(name: str, prefix: str) -> str:
    """Build the query-template identifier written to the search logs.

    The name alone is a promise nothing enforces: rename the template without
    editing the prefix (or edit the prefix and forget the name) and every log
    line afterwards attributes results to a template that was never used.
    Appending a short hash of the prefix itself makes that impossible — the
    name stays readable, the hash is what actually identifies the text.
    """
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:8]
    return f"{name}-{digest}"


def _json_file(env: str, default: str | None) -> dict:
    p = _path(env, default)
    if p and p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception:
            return {}
    return {}


# --- Data directory (all derived state lives here) ---------------------------
DATA_DIR = _path("VAULT_SEARCH_DATA_DIR", "~/.vault-search")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Vault ------------------------------------------------------------------
# REQUIRED: absolute path to your Obsidian vault (the folder holding your .md notes).
VAULT_PATH = _path("VAULT_PATH", None)
if VAULT_PATH is None:
    raise RuntimeError(
        "VAULT_PATH is not set. Point it at your Obsidian vault, e.g.\n"
        "  export VAULT_PATH=/home/you/Documents/MyVault   (Linux/macOS)\n"
        '  setx VAULT_PATH "C:\\Users\\you\\Documents\\MyVault"  (Windows)\n'
        "or put it in a .env file in the repo root."
    )

# Top-level folders never indexed (attachments, templates, system folders, etc.)
SKIP_FOLDERS = set(_csv(
    "VAULT_SEARCH_SKIP_FOLDERS",
    ".obsidian,.trash,.git,node_modules",
))

# --- LanceDB / Ollama -------------------------------------------------------
DB_PATH = _path("VAULT_SEARCH_DB_PATH", str(DATA_DIR / "lance_db"))
# The textbook tables can live in their own database (e.g. a rebuilt v3 index)
# while the vault keeps DB_PATH. Default: the same database as the vault.
# Only the SEARCH side and textbook_indexer_v3.py follow this; the legacy
# textbook_indexer.py keeps writing to DB_PATH (lecture indexing relies on it).
TEXTBOOK_DB_PATH = _path("VAULT_SEARCH_TEXTBOOK_DB_PATH", None) or DB_PATH
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Embedding model for vault notes. bge-m3 is bilingual (CN/EN), 1024-dim, runs on CPU.
EMBEDDING_MODEL = os.environ.get("VAULT_SEARCH_EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = int(os.environ.get("VAULT_SEARCH_EMBEDDING_DIM", "1024"))

# Incremental-index hash cache.
HASH_CACHE_PATH = _path("VAULT_SEARCH_HASH_CACHE", str(DATA_DIR / "file_hashes.json"))

# Shared observability log.
SEARCH_LOG_PATH = _path("VAULT_SEARCH_LOG_PATH", str(DATA_DIR / "search_log.jsonl"))

# --- Vault v2 index (parent-child chunking) ----------------------------------
# v2 splits every note into small child chunks (precise vector hit) plus
# subsection-sized parents (the context actually returned). It is a separate pair
# of tables, so v1 and v2 can coexist and be compared before you commit to one.
#
# Retrieval reads v2 only when this flag FILE exists — create or delete it to
# switch paths without restarting the server:
#   touch ~/.vault-search/vault_v2.enabled
VAULT_V2_FLAG = _path("VAULT_SEARCH_V2_FLAG", str(DATA_DIR / "vault_v2.enabled"))

# v2 embedding model. Defaults to the textbook model because the parent-child
# architecture came from there, but it is its own knob: a vault-only install
# never has to configure a textbook corpus to use v2.
VAULT_V2_EMBEDDING_MODEL = os.environ.get(
    "VAULT_SEARCH_V2_MODEL",
    os.environ.get("VAULT_SEARCH_TEXTBOOK_MODEL", "qwen3-embedding:0.6b"),
)

VAULT_V2_HASH_CACHE = _path("VAULT_SEARCH_V2_HASH", str(DATA_DIR / "vault_v2_hashes.json"))
VAULT_V2_ERROR_LOG = _path("VAULT_SEARCH_V2_ERRORS", str(DATA_DIR / "vault_v2_index_errors.json"))

# Observability log for vault searches (the textbook side has its own).
VAULT_SEARCH_LOG_PATH = _path(
    "VAULT_SEARCH_VAULT_LOG_PATH", str(DATA_DIR / "vault_search_log.jsonl")
)

# Characters of the matched section quoted as a v2 vault excerpt (vault_search
# with include_excerpt, and the chat context). 0 = the whole section, which is
# what the textbook and lecture passages already get. Measured before the
# change: an 800-char head left the matched passage entirely outside the
# excerpt for 8% of chat results and cut it for another 18%. Set 800 to go back.
VAULT_EXCERPT_CHARS = int(os.environ.get("VAULT_SEARCH_VAULT_EXCERPT_CHARS", "0"))

# Instruction prefix prepended to every vault QUERY before embedding (Qwen3
# instruct format). Documents are embedded WITHOUT it, so changing this changes
# only how queries are phrased — never the stored vectors.
VAULT_QUERY_PREFIX = os.environ.get(
    "VAULT_SEARCH_V2_QUERY_PREFIX",
    "Instruct: Given a query, retrieve the most relevant passages "
    "from the user's personal notes.\nQuery: ",
)
# Human-readable name for the query template above, recorded in the search log.
# See `query_template_version()` in vault_indexer_v2.py: the logged value is this
# name plus a short hash of the prefix, so the name can never claim one template
# while a different prefix is actually in use.
VAULT_QUERY_TEMPLATE_NAME = os.environ.get(
    "VAULT_SEARCH_V2_TEMPLATE_VERSION", "qwen3-vault-v1"
)

# --- API server -------------------------------------------------------------
API_HOST = os.environ.get("VAULT_SEARCH_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("VAULT_SEARCH_API_PORT", "3789"))

# Path to the `claude` CLI used for Hybrid/Free chat (Claude subscription).
# Leave empty to disable Claude-backed chat (Vault mode still works via Ollama).
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# Local Ollama model used for zero-cost Vault-mode chat. Any chat model you've pulled.
OLLAMA_VAULT_MODEL = os.environ.get("VAULT_SEARCH_CHAT_MODEL", "gemma2:9b")

# Assistant persona used in chat system prompts. Customize for your domain.
ASSISTANT_PERSONA = os.environ.get(
    "VAULT_SEARCH_PERSONA",
    "You are a knowledgeable assistant helping the user reason over their personal notes.",
)
# Reply language instruction appended to chat prompts.
ASSISTANT_LANGUAGE = os.environ.get(
    "VAULT_SEARCH_LANGUAGE",
    "Answer in the same language as the user's question.",
)

# Folder (relative to the vault) where the plugin saves chat transcripts.
CHAT_HISTORY_FOLDER = os.environ.get("VAULT_SEARCH_CHAT_HISTORY_FOLDER", "VaultChatHistory")

# --- Reranking weights ------------------------------------------------------
# Folder -> score multiplier. Boost the folders you trust most; demote archives.
# Default is empty (every folder weighted equally). Example:
#   VAULT_SEARCH_PATH_WEIGHTS='{"Reference":1.2,"Archive":0.8}'
PATH_WEIGHTS = (
    json.loads(os.environ["VAULT_SEARCH_PATH_WEIGHTS"])
    if os.environ.get("VAULT_SEARCH_PATH_WEIGHTS")
    else {}
)

# Folder name treated as "archived" (deprioritized unless include_archived=True).
ARCHIVE_FOLDER = os.environ.get("VAULT_SEARCH_ARCHIVE_FOLDER", "")

# --- Search scopes ----------------------------------------------------------
# Optional JSON map of scope name -> top-level folders, e.g.
#   {"_default": "work", "work": ["Projects", "Meetings"]}
# Folders not listed under a scope are OUT of it (fail-closed); the implicit
# scope "personal" is everything not listed anywhere. See server/scope.py.
# No map file = scopes disabled, every search sees the whole vault.
SCOPE_MAP_PATH = _path("VAULT_SEARCH_SCOPE_MAP", str(DATA_DIR / "scope_map.json"))
# Scope used when a caller does not name one. Precedence: this env var, then the
# map's "_default" key, then "all". It lives on the server, not in a prompt, so
# a caller that forgets to ask for a scope still gets the safe one.
DEFAULT_SCOPE_ENV = os.environ.get("VAULT_SEARCH_DEFAULT_SCOPE", "")

# --- Cowork / derivative exclusion ------------------------------------------
# Substrings (case-insensitive) marking derivative notes to filter out of search
# by default (e.g. shared exam-prep notes you should not cite as primary sources).
# Empty by default. Example: VAULT_SEARCH_EXCLUDE_PATTERNS="cowork,draft,scratch"
COWORK_PATTERNS = _csv("VAULT_SEARCH_EXCLUDE_PATTERNS", "")

# --- Knowledge graph (optional) ---------------------------------------------
GRAPH_PATH = _path("VAULT_SEARCH_GRAPH_PATH", str(DATA_DIR / "knowledge_graph.json"))
EXTRACT_PROGRESS_PATH = _path("VAULT_SEARCH_EXTRACT_PROGRESS", str(DATA_DIR / "extract_progress.json"))
# Optional dir of extra markdown (e.g. a glossary) used to canonicalize NER entities.
ENTITY_CANON_DIR = _path("VAULT_SEARCH_ENTITY_CANON_DIR", None)
# Folders whose notes get entity extraction FIRST. Extraction is slow and often
# time-boxed, so this decides what gets covered when a run cannot finish
# everything. Empty (the default) = no folder is favoured.
# Example: VAULT_SEARCH_ENTITY_PRIORITY_FOLDERS="Reference,Projects"
ENTITY_PRIORITY_FOLDERS = set(_csv("VAULT_SEARCH_ENTITY_PRIORITY_FOLDERS", ""))

# --- Textbook corpus (optional add-on) --------------------------------------
# A second corpus of long-form reference docs, indexed separately with its own model.
# Leave TEXTBOOK_PATH unset to disable textbook search entirely.
TEXTBOOK_PATH = _path("VAULT_SEARCH_TEXTBOOK_PATH", None)
TEXTBOOK_EMBEDDING_MODEL = os.environ.get("VAULT_SEARCH_TEXTBOOK_MODEL", "qwen3-embedding:0.6b")
TEXTBOOK_HASH_CACHE = _path("VAULT_SEARCH_TEXTBOOK_HASH", str(DATA_DIR / "textbook_hashes.json"))
TEXTBOOK_ERROR_LOG = _path("VAULT_SEARCH_TEXTBOOK_ERRORS", str(DATA_DIR / "textbook_index_errors.json"))
TEXTBOOK_ERROR_ARCHIVE = _path("VAULT_SEARCH_TEXTBOOK_ERRORS_ARCHIVE", str(DATA_DIR / "textbook_index_errors.archive.json"))
# Optional book metadata table, joined onto textbook results at query time
# (edition / year / tier / corpus_type / status / superseded_by). Keyed by the
# book folder name. See examples/books.example.json and server/books_meta.py.
BOOKS_META_PATH = _path("VAULT_SEARCH_BOOKS_META", str(DATA_DIR / "books.json"))
# Retrieval used by textbook_search when the caller does not ask: "hybrid"
# (default: vector + full-text fused with RRF), "vector" or "fts". Hybrid and
# fts need a native full-text index on the chunks table (textbook_indexer_v3.py
# builds one); without it the search falls back to vector and logs why.
# (Default was "vector" before 2.12: hybrid beat it on every eval set.)
TEXTBOOK_RETRIEVAL = os.environ.get("VAULT_SEARCH_TEXTBOOK_RETRIEVAL", "hybrid").strip().lower()

# --- Cross-encoder reranker (optional, needs a CUDA GPU) --------------------
# A Python interpreter that has CUDA torch + transformers. Set it to turn the
# reranker on: textbook_search then takes a pool of RERANK_POOL results, scores
# them with the cross-encoder and keeps the top n. The service runs in that
# interpreter (server/reranker_service.py), starts on the first query, exits
# after RERANK_IDLE_S idle seconds or as soon as another job claims the GPU
# lease. Unset (default) = no reranking. Any reranker failure falls back to
# the un-reranked list.
RERANK_PYTHON = _path("VAULT_SEARCH_RERANK_PYTHON", None)
# "on" (default when RERANK_PYTHON is set) or "off" to keep the config but stop
# reranking. A caller's `rerank` argument overrides it per query.
TEXTBOOK_RERANK = os.environ.get("VAULT_SEARCH_TEXTBOOK_RERANK", "on").strip().lower() != "off"
RERANK_MODEL = os.environ.get("VAULT_SEARCH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_PORT = int(os.environ.get("VAULT_SEARCH_RERANK_PORT", "3791"))
RERANK_POOL = int(os.environ.get("VAULT_SEARCH_RERANK_POOL", "50"))
RERANK_MAX_LENGTH = int(os.environ.get("VAULT_SEARCH_RERANK_MAX_LENGTH", "512"))
RERANK_IDLE_S = float(os.environ.get("VAULT_SEARCH_RERANK_IDLE_S", "600"))
RERANK_START_TIMEOUT = float(os.environ.get("VAULT_SEARCH_RERANK_START_TIMEOUT", "90"))
# Source prior added to the cross-encoder score before the final cut (bge logit
# scale). Guidelines get RERANK_GUIDELINE_PRIOR; every source gets
# RERANK_YEAR_PRIOR * clip((year - 2015) / 10, 0, 1). Chosen 2026-11-12 on the
# gold T3 set (guideline questions lost to older textbooks); 0 turns either off.
RERANK_GUIDELINE_PRIOR = float(os.environ.get("VAULT_SEARCH_RERANK_GUIDELINE_PRIOR", "3"))
RERANK_YEAR_PRIOR = float(os.environ.get("VAULT_SEARCH_RERANK_YEAR_PRIOR", "0.5"))
# Table-name suffix for a textbook index built with a non-default embedding
# model (textbook_indexer_v3.py --model ...), e.g. "__qwen3_embedding_4b".
# Set VAULT_SEARCH_TEXTBOOK_MODEL to the same model. Empty = the default tables.
TEXTBOOK_TABLE_SUFFIX = os.environ.get("VAULT_SEARCH_TEXTBOOK_TABLE_SUFFIX", "").strip()

# --- Lecture corpus (optional, third corpus) --------------------------------
# A second textbook-format LanceDB (same indexer, same model) holding lecture
# notes. Searched by `lecture_search`; ranked separately, never merged with the
# textbook list. Disabled when the directory does not exist.
LECTURE_DB_PATH = _path("VAULT_SEARCH_LECTURE_DB_PATH", str(DATA_DIR / "lecture_db"))
LECTURE_SEARCH_LOG_PATH = _path(
    "VAULT_SEARCH_LECTURE_LOG_PATH", str(DATA_DIR / "lecture_search_log.jsonl")
)

# Per-source ranking boost for the textbook corpus (see examples/source_boost.example.json).
SOURCE_BOOST_PATH = _path("VAULT_SEARCH_SOURCE_BOOST", None)
# --- GPU lease (optional) ----------------------------------------------------
# Path to a machine-wide lease script that serializes GPU jobs. Unset (the
# default) means indexing just runs — right on a machine that has the GPU to
# itself. See server/gpu_lease_client.py for the interface it must expose.
GPU_LEASE_PATH = _path("VAULT_SEARCH_GPU_LEASE", None)

# Names this project registers under, so `<script> status` shows which indexer
# holds the card. One per indexer: they are separate jobs and must not be able
# to queue behind themselves.
GPU_LEASE_NAME_TEXTBOOK = os.environ.get(
    "VAULT_SEARCH_GPU_LEASE_NAME_TEXTBOOK", "vault_search_textbook_index")
GPU_LEASE_NAME_VAULT = os.environ.get(
    "VAULT_SEARCH_GPU_LEASE_NAME_VAULT", "vault_search_vault_index")

# How long to wait for a turn before giving up (default 6 h: a full index run
# ahead of us legitimately takes hours).
GPU_LEASE_ACQUIRE_TIMEOUT = float(
    os.environ.get("VAULT_SEARCH_GPU_LEASE_TIMEOUT", "21600"))
# Minimum time to hold the lease before yielding, so a burst of queued jobs
# cannot make the indexer thrash release/re-acquire without making progress.
GPU_LEASE_MIN_HOLD = float(os.environ.get("VAULT_SEARCH_GPU_LEASE_MIN_HOLD", "180"))
# Optional: the lease script's state directory. Knowing it lets the yield check
# peek at the queue with a directory listing instead of spawning a subprocess
# once per file. Without it the check still works, just time-throttled.
GPU_LEASE_STATE_DIR = _path("VAULT_SEARCH_GPU_LEASE_STATE", None)
GPU_LEASE_QUEUE_DIR = (GPU_LEASE_STATE_DIR / "queue") if GPU_LEASE_STATE_DIR else None
