"""
Vault Semantic Search MCP Server
Provides semantic search over the Obsidian vault via LanceDB + Ollama embeddings.

Tools:
  - vault_search: Semantic search across all vault notes
  - vault_similar: Find notes similar to a given note
  - vault_stats: Show index statistics

Run as: python mcp_server.py
Register: claude mcp add vault-search -- python "/abs/path/to/server/mcp_server.py"
"""

import os

# numpy ships its own OpenBLAS, which pre-commits one thread scratch buffer per CPU
# core the moment numpy is imported: 373 MB of private bytes uncapped versus 19 MB
# capped, measured on a 12-thread machine with numpy 2.5.1. This process runs no
# linear algebra of its own — embeddings come from Ollama, vector math from LanceDB
# — so cap it before anything below pulls numpy in. Export your own value to override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import logging
import logging.handlers
import sys
import time
import traceback
from pathlib import Path

import lancedb
import ollama

from scoring import rerank
from indexer import open_db, TABLE_NAME, EMBEDDING_MODEL, OLLAMA_HOST, QUERY_PREFIX, _escape_sql
from graph_builder import load_graph, get_neighbors
from ppr import personalized_pagerank, related_notes

# Textbook v2 constants. The vault v1 index uses its own model (bge-m3);
# vault v2 uses the qwen3-family model imported further below.
from textbook_indexer import (
    TEXTBOOK_EMBEDDING_MODEL,
    TEXTBOOK_QUERY_PREFIX,
    QUERY_TEMPLATE_VERSION,
    CHUNKING_VERSION,
    CHUNKS_TABLE,
    PARENTS_TABLE,
    OLLAMA_NUM_CTX,
    DB_PATH as TEXTBOOK_DB_PATH,
    CHILD_TARGET_TOKENS,
    CHILD_OVERLAP_TOKENS,
    get_tokenizer,
    tok_len,
)

# Vault v2 (parent-child chunking) — dual-track retrieval. The flag FILE decides
# which index vault_search reads; absent flag = legacy v1 (`vault` table).
# Toggle with `touch $VAULT_SEARCH_DATA_DIR/vault_v2.enabled` — no restart needed.
from vault_indexer_v2 import (
    VCHUNKS_TABLE,
    VPARENTS_TABLE,
    VAULT_QUERY_TEMPLATE_VERSION,
    EMBEDDING_MODEL as VAULT_V2_MODEL,
)

# Lazy-loaded graph
_graph_cache = None

# Parent lookup (textbook v2) - on-demand, NOT a resident cache.
# Holding every parent in a dict cost 2,120 MB of private bytes on a 162k-parent
# corpus; one `parent_id IN (...)` query for the ~20 parents a search really
# returns costs ~13 ms warm and leaves the process near 115 MB. Reading per query
# also removes the mtime + READY-marker reload dance: a search always sees the
# current index, with no stale window after a reindex.
PARENT_CACHE_COLUMNS = [
    "parent_id", "text", "file", "book", "chapter",
    "section_path", "section_path_raw", "page_start", "page_end",
    "token_count", "heading_origin", "figure_ids", "has_figure_ref",
    "chunking_version",
]

# Search-side parameters
MAX_PARENT_RETURN_TOKENS = 800
ABSOLUTE_PARENT_MAX_RETURN_TOKENS = 4000
MAX_TOTAL_PARENT_RETURN_TOKENS = 12000
MAX_CHILDREN_PER_PARENT = 2
SOFT_BOOK_DIVERSITY_THRESHOLD = 0.5  # apply max_per_book only if >50% from one book
DEFAULT_MAX_PER_BOOK = 3

# Cowork exclusion patterns — substrings marking DERIVATIVE notes (drafts, shared
# co-notes, scratch material) that should be filtered out of search by default.
# Configure via VAULT_SEARCH_EXCLUDE_PATTERNS. Empty by default.
from config import (
    COWORK_PATTERNS, SEARCH_LOG_PATH,
    VAULT_V2_FLAG, VAULT_QUERY_PREFIX, VAULT_SEARCH_LOG_PATH,
    SCOPE_MAP_PATH, DEFAULT_SCOPE_ENV,
    LECTURE_DB_PATH, LECTURE_SEARCH_LOG_PATH, BOOKS_META_PATH,
)
import scope as S
import books_meta
import lecture_meta

def _is_cowork_path(s: str) -> bool:
    """Check if a book name, folder, or file path matches cowork exclusion patterns."""
    if not s:
        return False
    s_lower = s.lower()
    for pat in COWORK_PATTERNS:
        if pat.lower() in s_lower:
            return True
    return False

# Retrieval observability log (thread-safe via RotatingFileHandler)
_search_log_path = SEARCH_LOG_PATH
_search_log_path.parent.mkdir(parents=True, exist_ok=True)
_search_logger = logging.getLogger("textbook_search")
_search_logger.setLevel(logging.INFO)
if not _search_logger.handlers:
    _h = logging.handlers.RotatingFileHandler(
        str(_search_log_path), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _h.setFormatter(logging.Formatter("%(message)s"))
    _search_logger.addHandler(_h)
    _search_logger.propagate = False

# Vault search observability log — records BOTH the v1 and v2 paths with the
# index_version that served each query, so a v1/v2 comparison has evidence from
# the first search rather than only after you decide to look.
_vault_log_path = VAULT_SEARCH_LOG_PATH
_vault_log_path.parent.mkdir(parents=True, exist_ok=True)
_vault_logger = logging.getLogger("vault_search")
_vault_logger.setLevel(logging.INFO)
if not _vault_logger.handlers:
    _vh = logging.handlers.RotatingFileHandler(
        str(_vault_log_path), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _vh.setFormatter(logging.Formatter("%(message)s"))
    _vault_logger.addHandler(_vh)
    _vault_logger.propagate = False

_lecture_logger = logging.getLogger("lecture_search")
_lecture_logger.setLevel(logging.INFO)
if not _lecture_logger.handlers:
    LECTURE_SEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _lh = logging.handlers.RotatingFileHandler(
        str(LECTURE_SEARCH_LOG_PATH), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _lh.setFormatter(logging.Formatter("%(message)s"))
    _lecture_logger.addHandler(_lh)
    _lecture_logger.propagate = False


def _log(logger, entry: dict, ctx: dict | None) -> None:
    """Write one observability line. Adds who asked (caller, session_id) and
    marks a zero-result search as a warning so it stands out when grepping."""
    ctx = ctx or {}
    entry["caller"] = ctx.get("caller") or "mcp"
    entry["session_id"] = ctx.get("session_id")
    n = entry.get("n_returned")
    entry["level"] = "warn" if n == 0 else "info"
    if n == 0:
        entry["warn"] = "zero_results"
    try:
        logger.info(_safe_json_dumps(entry))
    except Exception:
        pass  # never let logging break search


def find_relations_for_notes(graph_relations: list[dict], note_names: set[str]) -> list[dict]:
    """Find extracted relations where source_note is in the result set.

    Returns compact relation dicts: {from, to, type, source_note}.
    Limits to 15 relations to avoid output bloat.
    """
    if not graph_relations:
        return []
    matched = []
    for rel in graph_relations:
        if rel.get("source_note") in note_names:
            matched.append({
                "from": rel.get("from", ""),
                "to": rel.get("to", ""),
                "type": rel.get("type", ""),
                "source_note": rel.get("source_note", ""),
            })
    return matched[:15]


def find_entities_for_notes(graph_entities: dict, note_names) -> list[dict]:
    """Find extracted entities for result notes, in the order given.

    Returns compact list: [{note, entities: [{name, type, canonical}]}].
    Limits to 15 entries to avoid output bloat.

    `note_names` must be an ORDERED sequence. Passing a set made the block's
    order depend on PYTHONHASHSEED, so two servers on the same index answered
    the same query with the entity list shuffled — and the 15-entry cap then
    kept a different subset on each. Result rank is the order that means
    something, so callers pass the results in rank order.
    """
    if not graph_entities:
        return []
    matched = []
    for note in note_names:
        ents = graph_entities.get(note, [])
        if ents:
            matched.append({"note": note, "entities": ents[:10]})
    return matched[:15]


def get_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = load_graph()
    return _graph_cache


def get_table():
    db = open_db()
    return db.open_table(TABLE_NAME)


def get_textbook_chunks_table():
    db = open_db()
    if CHUNKS_TABLE not in db.table_names():
        return None
    return db.open_table(CHUNKS_TABLE)


def get_textbook_parents_table():
    db = open_db()
    if PARENTS_TABLE not in db.table_names():
        return None
    return db.open_table(PARENTS_TABLE)


# --- Lecture corpus: a second textbook-format DB, opened by path -------------

def lecture_enabled() -> bool:
    return bool(LECTURE_DB_PATH) and LECTURE_DB_PATH.exists()


def _lecture_db():
    return lancedb.connect(str(LECTURE_DB_PATH))


def get_corpus_tables(corpus: str):
    """(chunks, parents) for 'textbook' or 'lecture'; (None, None) when absent."""
    if corpus == "lecture":
        if not lecture_enabled():
            return None, None
        db = _lecture_db()
    else:
        db = open_db()
    names = db.table_names()
    chunks = db.open_table(CHUNKS_TABLE) if CHUNKS_TABLE in names else None
    parents = db.open_table(PARENTS_TABLE) if PARENTS_TABLE in names else None
    return chunks, parents


# IN-list size per query. LanceDB scans the parents table for this filter, so one
# round trip per search is the win; splitting huge id lists keeps the SQL bounded.
PARENT_FETCH_BATCH = 200


def fetch_parents_by_id(table, parent_ids, columns) -> dict[str, dict]:
    """Read just the named parents, keyed by parent_id. Empty dict on any miss."""
    if table is None:
        return {}
    uniq = list(dict.fromkeys(pid for pid in parent_ids if pid))
    if not uniq:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(uniq), PARENT_FETCH_BATCH):
        batch = uniq[i:i + PARENT_FETCH_BATCH]
        quoted = ",".join("'" + _escape_sql(pid) + "'" for pid in batch)
        df = (
            table.search()
            .where(f"parent_id IN ({quoted})")
            .select(columns)
            .limit(len(batch))
            .to_pandas()
        )
        for _, row in df.iterrows():
            out[row["parent_id"]] = row.to_dict()
    return out


def fetch_textbook_parents(parent_ids) -> dict[str, dict]:
    """Read the textbook parents a search actually returns. Metadata columns only -
    pulling the vector column here would multiply the transfer for no gain."""
    return fetch_parents_by_id(
        get_textbook_parents_table(), parent_ids, PARENT_CACHE_COLUMNS
    )


# --- Vault v2 tables --------------------------------------------------------

VAULT_PARENT_CACHE_COLUMNS = ["parent_id", "text", "token_count"]


def vault_v2_active() -> bool:
    """True when the v2 index should serve vault searches (flag file present)."""
    return bool(VAULT_V2_FLAG) and VAULT_V2_FLAG.exists()


def get_vault_chunks_table():
    db = open_db()
    if VCHUNKS_TABLE not in db.table_names():
        return None
    return db.open_table(VCHUNKS_TABLE)


def fetch_vault_parents(parent_ids) -> dict[str, dict]:
    """Read the v2 vault parents a search returns — same on-demand rule as the
    textbook side above: never hold the whole parent table in memory."""
    db = open_db()
    if VPARENTS_TABLE not in db.table_names():
        return {}
    return fetch_parents_by_id(
        db.open_table(VPARENTS_TABLE), parent_ids, VAULT_PARENT_CACHE_COLUMNS
    )


def _truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, bool, int]:
    """Truncate text at sentence boundary within max_tokens. Returns (text, was_truncated, tokens_returned)."""
    tokenizer = get_tokenizer()
    full_tokens = tok_len(text)
    if full_tokens <= max_tokens:
        return (text, False, full_tokens)
    # Token-based truncation: encode + decode at limit
    ids = tokenizer.encode(text, add_special_tokens=False)
    truncated_ids = ids[:max_tokens]
    truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    # Snap to last sentence boundary if reasonable
    for sep in ["。", "！", "？", ".", "!", "?", "\n\n", "\n"]:
        idx = truncated.rfind(sep)
        if idx > len(truncated) * 0.7:  # only snap if we don't lose too much
            truncated = truncated[: idx + len(sep)]
            break
    return (truncated, True, max_tokens)


def get_ollama():
    return ollama.Client(host=OLLAMA_HOST)


def embed_query(text: str) -> list[float]:
    """Embed a search query with instruction prefix."""
    client = get_ollama()
    response = client.embed(model=EMBEDDING_MODEL, input=[QUERY_PREFIX + text])
    return response["embeddings"][0]


def _clean(text: str) -> str:
    """Remove surrogate characters that break JSON serialization on Windows."""
    if not isinstance(text, str):
        return str(text)
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def _safe_json_dumps(obj, **kwargs) -> str:
    """json.dumps that guarantees no surrogate characters in output."""
    raw = json.dumps(obj, ensure_ascii=False, **kwargs)
    return _clean(raw)


def _cosine_for_notes(table, query_embedding, notes: list[str]) -> dict[str, float]:
    """Best cosine similarity per note, for notes outside the top-N vector hits.

    Graph candidates reach vault_similar through the wiki-link graph, not through
    the vector search, so they arrive carrying no similarity at all. A field the
    caller is expected to threshold on has to hold a real number for every row,
    so fetch theirs with one prefiltered vector query.
    """
    if not notes:
        return {}
    quoted = ", ".join("'" + _escape_sql(n) + "'" for n in notes)
    try:
        df = (table.search(query_embedding).metric("cosine")
              .where(f"note IN ({quoted})", prefilter=True)
              .limit(min(len(notes) * 20, 2000)).to_pandas())
    except Exception:
        # Older lancedb without prefilter support: better a missing number than
        # a broken tool call. These notes fall back to 0.0 at the call site.
        return {}
    best: dict[str, float] = {}
    for _, row in df.iterrows():
        note = _clean(str(row.get("note", "")))
        sim = round(max(0.0, 1 - float(row.get("_distance", 0))), 4)
        if sim > best.get(note, -1.0):
            best[note] = sim
    return best


# Characters of chunk text returned as an excerpt. The v2 retrieval path uses
# the same budget, so vault_search and vault_similar quote a note to the same
# depth — they used to differ (500 vs 800), which read as vault_similar
# truncating notes for no stated reason.
EXCERPT_CHARS = 800


def format_results(results_df, include_excerpt: bool = False) -> list[dict]:
    """Format LanceDB DataFrame results into readable output.

    Reads the section from whichever column the table has: v1 stores one heading
    in `section`, v2 stores the full heading path in `section_path`.
    """
    formatted = []
    for _, row in results_df.iterrows():
        entry = {
            "file": _clean(str(row.get("file", ""))),
            "note": _clean(str(row.get("note", ""))),
            "section": _clean(str(row.get("section", "") or row.get("section_path", ""))),
            "folder": _clean(str(row.get("folder", ""))),
            "tags": _clean(str(row.get("tags", ""))),
            "similarity": round(max(0, 1 - float(row.get("_distance", 0))), 4) if "_distance" in row.index else None,
            "mtime": float(row["mtime"]) if row.get("mtime") is not None else None,
        }
        if include_excerpt:
            text = str(row.get("text", ""))
            entry["excerpt"] = _clean(text[:EXCERPT_CHARS]) if text else ""
        formatted.append(entry)
    return formatted


# --- Compact Markdown output (token-efficient; replaces raw nested JSON) ---
# Tool results fed to the model as raw indent=2 JSON spend ~50-60% of tokens on
# braces, quotes, key names, and duplicate/internal fields. Markdown drops that
# overhead with zero loss of the information the model actually uses. The
# observability JSON still goes to the .jsonl search log.

def _fmt_sim(s) -> str:
    return f"{s:.2f}" if isinstance(s, (int, float)) else "?"


def format_vault_markdown(results: list[dict], query: str, *,
                          include_excerpt: bool = False,
                          graph_linked: list[str] | None = None,
                          entities: list[dict] | None = None,
                          relations: list[dict] | None = None,
                          header: str = "vault_search") -> str:
    """Render vault_search / vault_similar results as compact Markdown."""
    if not results:
        return f'No vault notes found for "{query}".'
    lines = [f'# {header}: "{query}" — {len(results)} notes', ""]
    for i, r in enumerate(results, 1):
        meta = []
        sec = r.get("section", "")
        if sec:
            meta.append(sec)
        folder = r.get("folder", "")
        if folder:
            meta.append(folder)
        meta.append(f"sim {_fmt_sim(r.get('similarity'))}")
        lines.append(f"{i}. [[{r.get('note','')}]] · " + " · ".join(meta))
        tags = r.get("tags", "")
        if tags and tags not in ("", "[]", "nan"):
            lines.append(f"   tags: {tags}")
        if include_excerpt and r.get("excerpt"):
            lines.append(f"   {r['excerpt']}")
    if graph_linked:
        lines += ["", "graph-linked: " + ", ".join(f"[[{n}]]" for n in graph_linked)]
    if entities:
        lines += ["", "entities:"]
        for e in entities:
            parts = []
            for x in e.get("entities", []):
                nm = " ".join(str(x.get("name", "")).split())[:40]
                if nm:
                    parts.append(f"{nm}({x.get('type','')})")
            if parts:
                lines.append(f"- [[{e.get('note','')}]]: {', '.join(parts)}")
    if relations:
        rel_str = ", ".join(
            f"{r.get('from','')} —{r.get('type','')}→ {r.get('to','')}" for r in relations
        )
        lines += ["", "relations: " + rel_str]
    return "\n".join(lines)


def format_related_markdown(related: list[dict], note: str) -> str:
    """Render vault_related results as compact Markdown."""
    if not related:
        return f'No graph-related notes found for "{note}".'
    lines = [f'# vault_related: [[{note}]] — {len(related)} related', ""]
    for i, r in enumerate(related, 1):
        meta = [m for m in (r.get("folder", ""),) if m]
        meta.append(f"score {_fmt_sim(r.get('score'))}")
        lines.append(f"{i}. [[{r.get('note','')}]] · " + " · ".join(meta))
    return "\n".join(lines)


def format_textbook_markdown(results: list[dict], query: str, header: str = "textbook_search") -> str:
    """Render textbook_search / lecture_search results as compact Markdown.

    Drops internal/duplicate fields (parent_id, section==section_path,
    page==page_start, parent_truncated, *_tokens, chunk_*, n_siblings,
    heading_origin, and the child `text` excerpt already contained in parent_text)."""
    if not results:
        what = "lecture" if header == "lecture_search" else "textbook"
        return f'No {what} passages found for "{query}".'
    lines = [f'# {header}: "{query}" — {len(results)} passages', ""]
    for i, r in enumerate(results, 1):
        loc = []
        ps, pe = r.get("page_start", 0), r.get("page_end", 0)
        if ps:
            loc.append(f"p.{ps}-{pe}" if pe and pe != ps else f"p.{ps}")
        loc.append(f"sim {_fmt_sim(r.get('similarity'))}")
        sec = r.get("section") or r.get("section_path") or ""
        head = f"{i}. **{r.get('book','')}**"
        if sec:
            head += f" · {sec}"
        head += " · " + " · ".join(loc)
        lines.append(head)
        lines.append(f"   ({r.get('file','')})")
        prov = books_meta.describe(r.get("meta") or {}) or lecture_meta.describe(r)
        if prov:
            lines.append(f"   source: {prov}")
        ptext = r.get("parent_text") or r.get("text") or ""
        if ptext:
            lines.append(ptext)
        if r.get("has_figure_ref") and r.get("figure_ids"):
            lines.append(f"   figures: {', '.join(r['figure_ids'])}")
        lines.append("")
    return "\n".join(lines).rstrip()


# --- MCP Protocol ---

TOOLS = [
    {
        "name": "vault_search",
        "description": "Semantic search across the Obsidian vault. Returns the most relevant note chunks for a natural language query. Supports Chinese, English, and mixed queries. Use this to find notes related to a topic, symptom, condition, or concept.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (Chinese, English, or mixed). E.g. '脊髓損傷的膀胱管理' or 'stroke rehabilitation upper limb'"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10, max: 30)",
                    "default": 10
                },
                "scope": {
                    "type": "string",
                    "description": "Optional: which part of the vault to search — a scope name from the server's scope map, 'personal' (folders not in any scope) or 'all'. Omit to use the server's default scope."
                },
                "folder": {
                    "type": "string",
                    "description": "Optional: filter to a specific top-level folder of your vault"
                },
                "include_excerpt": {
                    "type": "boolean",
                    "description": "Include text excerpt in results (default: false). Set true when you need content for summarization.",
                    "default": False
                },
                "boost_recent": {
                    "type": "boolean",
                    "description": "Apply recency weighting — recently modified notes score higher (default: true)",
                    "default": True
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "When false (default), notes in the configured archive folder are deprioritized. Set true to treat archived notes equally.",
                    "default": False
                },
                "exclude_cowork": {
                    "type": "boolean",
                    "description": "When true (default), filter out derivative notes matching the configured exclude patterns (e.g. drafts, shared co-notes, scratch material). Set false only if you explicitly need to include them (rare — e.g., for cleanup). No-op when no exclude patterns are configured.",
                    "default": True
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "vault_related",
        "description": "Find notes RELATED to a given note via the wiki-link graph (personalized PageRank / LinearRAG-style propagation). Unlike vault_search (semantic), this seeds from a note's own links and surfaces multi-hop connected notes — including ones that share no text but are conceptually linked. Use to discover related notes, expand context around a note, or suggest links. Give the note's title (filename without .md).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note title (filename without .md) to find related notes for. E.g. 'Achilles tendinopathy' or 'CVA'."
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of related notes to return (default: 12, max: 30)",
                    "default": 12
                },
                "scope": {
                    "type": "string",
                    "description": "Optional: which part of the vault to search — a scope name from the server's scope map, 'personal' (folders not in any scope) or 'all'. Omit to use the server's default scope."
                }
            },
            "required": ["note"]
        }
    },
    {
        "name": "vault_similar",
        "description": "Find notes similar to a given note. Provide a note filename (without .md) to find semantically related notes in the vault.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_name": {
                    "type": "string",
                    "description": "The note name (without .md extension) to find similar notes for"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of similar notes to return (default: 10)",
                    "default": 10
                },
                "scope": {
                    "type": "string",
                    "description": "Optional: which part of the vault to search — a scope name from the server's scope map, 'personal' (folders not in any scope) or 'all'. Omit to use the server's default scope."
                }
            },
            "required": ["note_name"]
        }
    },
    {
        "name": "vault_stats",
        "description": "Show statistics about the vault semantic index: total chunks, files indexed, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "textbook_search",
        "description": "Semantic search across an optional secondary corpus of long-form reference documents (e.g. converted textbooks / manuals), indexed separately from your notes. Parent-child two-tier index — children for the precise hit, parent_text for the surrounding subsection context. Set full_parent=true to retrieve untruncated parent text (capped at 4K tokens per parent and 12K total).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (Chinese, English, or mixed)"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default: 10, max: 50). 10 is usually sufficient — avoid 20+ to save tokens.",
                    "default": 10
                },
                "book": {
                    "type": "string",
                    "description": "Optional: filter by source/book folder name within the reference corpus"
                },
                "full_parent": {
                    "type": "boolean",
                    "description": "If true, return untruncated parent_text (per-parent cap 4K tokens, total budget 12K). Default false (parent_text truncated to 800 tokens).",
                    "default": False
                },
                "exclude_cowork": {
                    "type": "boolean",
                    "description": "When true (default), filter out derivative sources matching the configured exclude patterns. Set false only if you explicitly need to inspect them (rare — e.g., for cleanup or comparison). No-op when no exclude patterns are configured.",
                    "default": True
                },
                "corpus_type": {
                    "type": "string",
                    "description": "Optional: restrict to one source type from the book table, e.g. 'guideline' or 'textbook'. Use this to search guidelines, rather than guessing a book folder name."
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": "When false (default), sources marked superseded or derivative in the book table are hidden. Set true to compare an old edition against its replacement.",
                    "default": False
                }
            },
            "required": ["query"]
        }
    }
]

LECTURE_TOOL = {
    "name": "lecture_search",
    "description": "Semantic search across the lecture-notes corpus (talks and courses the user attended: what a specific speaker taught and how they do it). Separate index and separate ranking from textbook_search — use it for 'what did the lecture say' questions, and textbook_search for reference-text answers.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query (Chinese, English, or mixed)"},
            "n_results": {"type": "integer", "description": "Number of results (default: 10, max: 50)", "default": 10},
            "course": {"type": "string", "description": "Optional: restrict to one course folder"},
            "full_parent": {"type": "boolean", "description": "Return untruncated parent text (same caps as textbook_search)", "default": False}
        },
        "required": ["query"]
    }
}


if lecture_enabled():
    TOOLS.append(LECTURE_TOOL)


def _row_similarity(row) -> float:
    """Cosine similarity from `_distance`; 0.0 for keyword-only rows (FTS/hybrid
    hits the vector side never scored carry no distance, or NaN)."""
    d = row.get("_distance")
    try:
        d = float(d)
    except (TypeError, ValueError):
        return 0.0
    if d != d:  # NaN
        return 0.0
    return round(max(0.0, 1.0 - d), 4)


class TextbookIndexMissing(RuntimeError):
    """The textbook v2 index has not been built yet."""


def _textbook_search_v2(arguments: dict, ctx: dict | None = None, corpus: str = "textbook") -> str:
    """MCP entry point: run the v2 search and render it as compact Markdown.

    Thin wrapper over `_textbook_search_v2_raw()` so tests and benchmarks can
    consume structured results instead of parsing Markdown back into dicts.
    """
    try:
        results = _textbook_search_v2_raw(arguments, ctx=ctx, corpus=corpus)
    except TextbookIndexMissing as exc:
        return json.dumps({"error": str(exc)})
    header = "lecture_search" if corpus == "lecture" else "textbook_search"
    return _clean(format_textbook_markdown(results, _clean(arguments["query"]), header=header))


def _textbook_search_v2_raw(arguments: dict, ctx: dict | None = None,
                            corpus: str = "textbook") -> list[dict]:
    """v2 parent-child semantic search with dedup, parent context, observability.

    `corpus` picks the database: "textbook" (the main index) or "lecture" (the
    lecture-notes DB, same schema and model). Returns the structured result
    list; the Markdown rendering lives in the `_textbook_search_v2()` wrapper."""
    t0 = time.time()
    query = _clean(arguments["query"])
    n = min(arguments.get("n_results", 10), 50)
    book_filter = arguments.get("book") or arguments.get("course")
    full_parent = bool(arguments.get("full_parent", False))
    exclude_cowork = bool(arguments.get("exclude_cowork", True))
    include_superseded = bool(arguments.get("include_superseded", False))
    corpus_type = arguments.get("corpus_type") or None
    meta = books_meta.load(BOOKS_META_PATH) if corpus == "textbook" else {}

    chunks_table, parents_table = get_corpus_tables(corpus)
    if chunks_table is None:
        if corpus == "lecture":
            raise TextbookIndexMissing(f"Lecture index not found at {LECTURE_DB_PATH}.")
        raise TextbookIndexMissing("Textbook v2 index not built. Run textbook_indexer.py first.")

    # Embed query with Qwen3 instruction prefix; L2 normalize
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.embed(
        model=TEXTBOOK_EMBEDDING_MODEL,
        input=[TEXTBOOK_QUERY_PREFIX + query],
        options={"num_ctx": OLLAMA_NUM_CTX},
    )
    query_embedding = response["embeddings"][0]
    # L2 normalize
    import math
    s = math.sqrt(sum(v * v for v in query_embedding))
    if s > 0:
        query_embedding = [v / s for v in query_embedding]

    # Fetch n*10 to give dedup room (Gemini r5 #3)
    fetch_limit = n * 10
    # `retrieval` is an EXPERIMENT switch for the eval harness, not part of the
    # tool schema: "vector" (default, what the tool does), "fts" (keyword only)
    # or "hybrid" (vector + FTS fused by RRF). The non-vector modes need a
    # native FTS index on the chunks table's `text` column.
    retrieval = arguments.get("retrieval", "vector")
    fts_text = arguments.get("query_en") or query
    if retrieval == "fts":
        search_builder = chunks_table.search(fts_text, query_type="fts").limit(fetch_limit)
    elif retrieval == "hybrid":
        from lancedb.rerankers import RRFReranker
        search_builder = (chunks_table.search(query_type="hybrid")
                          .vector(query_embedding).text(fts_text)
                          .distance_type("cosine")
                          .rerank(RRFReranker()).limit(fetch_limit))
    else:
        search_builder = chunks_table.search(query_embedding).metric("cosine").limit(fetch_limit)
    where = S.combine(
        f"book = '{_escape_sql(book_filter)}'" if book_filter else None,
        books_meta.prefilter(corpus_type, meta) if corpus == "textbook" else None,
        # A course's _meta.md is bookkeeping (clip table, material paths), not
        # lecture content: never a lecture_search hit.
        lecture_meta.META_EXCLUDE_WHERE if corpus == "lecture" else None,
    )
    if where:
        search_builder = search_builder.where(where, prefilter=True)
    df = search_builder.to_pandas()

    # Score: cosine similarity = 1 - distance (since vectors L2-normalized)
    raw_results = []
    for _, row in df.iterrows():
        raw_results.append({
            "id": _clean(str(row.get("id", ""))),
            "parent_id": _clean(str(row.get("parent_id", ""))),
            "file": _clean(str(row.get("file", ""))),
            "book": _clean(str(row.get("book", ""))),
            "chapter": _clean(str(row.get("chapter", ""))),
            "section_path": _clean(str(row.get("section_path", ""))),
            "page_start": int(row.get("page_start", 0) or 0),
            "page_end": int(row.get("page_end", 0) or 0),
            "chunk_kind": _clean(str(row.get("chunk_kind", ""))),
            "chunk_idx": int(row.get("chunk_idx", 0) or 0),
            "n_siblings": int(row.get("n_siblings", 0) or 0),
            "token_count": int(row.get("token_count", 0) or 0),
            "heading_origin": _clean(str(row.get("heading_origin", ""))),
            "figure_ids": [_clean(str(s)) for s in (row.get("figure_ids") if row.get("figure_ids") is not None else [])],
            "has_figure_ref": bool(row.get("has_figure_ref", False)),
            "text": _clean(str(row.get("text", ""))),
            "similarity": _row_similarity(row),
        })

    # Cowork exclusion (default true) — drop derivative notes before dedup
    if exclude_cowork:
        raw_results = [
            r for r in raw_results
            if not _is_cowork_path(r.get("book", "")) and not _is_cowork_path(r.get("file", ""))
        ]

    # Book metadata (books.json), joined here at query time: hide superseded /
    # derivative sources unless asked, filter by corpus_type, and let tier /
    # year break similarity ties. The lecture DB has no book table.
    n_before_meta = len(raw_results)
    if corpus == "textbook":
        books_meta.annotate(raw_results, meta)
        raw_results = [r for r in raw_results if books_meta.visible(r, include_superseded, corpus_type)]
        if retrieval == "vector":  # fused rankings are not ordered by similarity
            raw_results = books_meta.tie_break(raw_results)
    hidden_by_meta = n_before_meta - len(raw_results)

    # Dedup: max children per parent
    parent_count: dict[str, int] = {}
    deduped: list[dict] = []
    for r in raw_results:
        pid = r["parent_id"]
        if parent_count.get(pid, 0) >= MAX_CHILDREN_PER_PARENT:
            continue
        parent_count[pid] = parent_count.get(pid, 0) + 1
        deduped.append(r)

    # Soft book diversity: only enforce max_per_book if >50% from one book
    if not book_filter and deduped:
        from collections import Counter
        bk_counter = Counter(r["book"] for r in deduped)
        top_book, top_count = bk_counter.most_common(1)[0]
        if top_count / len(deduped) > SOFT_BOOK_DIVERSITY_THRESHOLD:
            book_count: dict[str, int] = {}
            soft_filtered = []
            for r in deduped:
                if book_count.get(r["book"], 0) >= DEFAULT_MAX_PER_BOOK:
                    continue
                book_count[r["book"]] = book_count.get(r["book"], 0) + 1
                soft_filtered.append(r)
            deduped = soft_filtered

    deduped = deduped[:n]

    # Group by unique parent for budget allocation (Gemini r6 #2)
    # Top-1 unique parent gets full (up to ABSOLUTE_PARENT_MAX_RETURN_TOKENS),
    # subsequent shrink progressively.
    seen_parent_ids: list[str] = []
    parent_score: dict[str, float] = {}
    for r in deduped:
        pid = r["parent_id"]
        if pid not in parent_score:
            parent_score[pid] = r["similarity"]
            seen_parent_ids.append(pid)
        # else keep first (highest) score — already sorted

    # Budget allocation per unique parent
    budget_per_parent: dict[str, int] = {}
    remaining_budget = MAX_TOTAL_PARENT_RETURN_TOKENS if full_parent else (n * MAX_PARENT_RETURN_TOKENS)
    for rank, pid in enumerate(seen_parent_ids):
        if not full_parent:
            budget_per_parent[pid] = MAX_PARENT_RETURN_TOKENS
            continue
        if rank == 0:
            quota = ABSOLUTE_PARENT_MAX_RETURN_TOKENS
        elif rank == 1:
            quota = 2000
        else:
            # Progressively shrink for top-3..N
            quota = max(400, remaining_budget // max(1, len(seen_parent_ids) - rank))
        quota = min(quota, ABSOLUTE_PARENT_MAX_RETURN_TOKENS, remaining_budget)
        budget_per_parent[pid] = quota
        remaining_budget = max(0, remaining_budget - quota)

    # One round trip for exactly the parents this search returns
    parent_rows = fetch_parents_by_id(parents_table, seen_parent_ids, PARENT_CACHE_COLUMNS)

    # Apply budget to enrich results with parent_text
    formatted_results = []
    for r in deduped:
        pid = r["parent_id"]
        parent_data = parent_rows.get(pid)
        parent_text_full = ""
        parent_token_count = 0
        if parent_data:
            parent_text_full = str(parent_data.get("text", ""))
            parent_token_count = int(parent_data.get("token_count", 0) or 0)
        budget = budget_per_parent.get(pid, MAX_PARENT_RETURN_TOKENS)
        truncated_text, was_truncated, returned_tokens = _truncate_text_to_tokens(parent_text_full, budget)

        # API surface: keep legacy field 'page' as page_start; add new fields
        formatted_results.append({
            "file": r["file"],
            "book": r["book"],
            "chapter": r["chapter"],
            "section": r["section_path"],  # legacy field name
            "section_path": r["section_path"],
            "page": r["page_start"],  # legacy
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "text": r["text"][:300],  # legacy 300-char excerpt
            "similarity": r["similarity"],
            "parent_id": r["parent_id"],
            "parent_text": _clean(truncated_text),
            "parent_truncated": was_truncated,
            "parent_return_tokens": returned_tokens,
            "parent_original_tokens": parent_token_count,
            "chunk_kind": r["chunk_kind"],
            "chunk_idx": r["chunk_idx"],
            "n_siblings": r["n_siblings"],
            "token_count": r["token_count"],
            "heading_origin": r["heading_origin"],
            "figure_ids": r["figure_ids"],
            "has_figure_ref": r["has_figure_ref"],
            "corpus": corpus,
            "meta": r.get("meta", {}),
        })

    # Lecture entries carry speaker / date / course in their `Source:` line,
    # which sits in the entry's H1 parent rather than in every hit.
    if corpus == "lecture" and formatted_results:
        src = lecture_meta.source_by_file(
            parents_table, [r["file"] for r in formatted_results], _escape_sql)
        for r in formatted_results:
            r.update(src.get(r["file"], {"date": "", "speaker": "", "course_title": "", "source_line": ""}))

    # Compute observability metrics
    latency_ms = int((time.time() - t0) * 1000)
    top_score = formatted_results[0]["similarity"] if formatted_results else 0
    score_gap = (
        formatted_results[0]["similarity"] - formatted_results[1]["similarity"]
        if len(formatted_results) >= 2 else 0
    )
    similarities = [r["similarity"] for r in formatted_results]
    if len(similarities) >= 2:
        mean_s = sum(similarities) / len(similarities)
        score_std = (sum((s - mean_s) ** 2 for s in similarities) / len(similarities)) ** 0.5
    else:
        score_std = 0.0
    raw_unique = len({r["parent_id"] for r in raw_results}) / max(1, len(raw_results))
    deduped_unique = len({r["parent_id"] for r in formatted_results}) / max(1, len(formatted_results)) if formatted_results else 0

    log_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query": query,
        "n_requested": n,
        "book_filter": book_filter,
        "full_parent": full_parent,
        "embedding_model": TEXTBOOK_EMBEDDING_MODEL,
        "chunking_version": CHUNKING_VERSION,
        "tokenizer_version": "Qwen3-Embedding-0.6B",
        "query_template_version": QUERY_TEMPLATE_VERSION,
        "embedding_prompt_variant": "no_doc_prefix",
        "overlap_tokens": CHILD_OVERLAP_TOKENS,
        "child_target_tokens": CHILD_TARGET_TOKENS,
        "top_score": round(top_score, 4),
        "score_gap_top1_top2": round(score_gap, 4),
        "score_std_topN": round(score_std, 4),
        "raw_parent_hit_rate": round(raw_unique, 3),
        "post_dedup_parent_hit_rate": round(deduped_unique, 3),
        "top_chunk_ids": [r["parent_id"] for r in formatted_results[:5]],
        # Rank order, not a set: a set's order follows PYTHONHASHSEED.
        "top_books": list(dict.fromkeys(r["book"] for r in formatted_results[:5])),
        "latency_ms": latency_ms,
        "corpus": corpus,
        "n_returned": len(formatted_results),
        "corpus_type_filter": corpus_type,
        "retrieval": retrieval,
        "hidden_by_books_meta": hidden_by_meta,
    }
    _log(_lecture_logger if corpus == "lecture" else _search_logger, log_entry, ctx)

    return formatted_results


class NoVaultIndex(Exception):
    """Neither index is built — a setup problem, not a search failure."""


def _folder_where(folder: str | None, scope_where: str | None) -> str | None:
    return S.combine(f"folder = '{_escape_sql(folder)}'" if folder else None, scope_where)


def _vault_retrieve_v1(query: str, folder: str | None, fetch_n: int,
                       include_excerpt: bool, scope_where: str | None = None) -> list[dict]:
    """Legacy retrieval: the `vault` table, one tier, heading-sized chunks."""
    db = open_db()
    if TABLE_NAME not in db.table_names():
        # A v2-only install with the flag off lands here. Say which index is
        # missing and how to reach the one that does exist, rather than
        # surfacing LanceDB's "Table 'vault' was not found".
        if VCHUNKS_TABLE in db.table_names():
            raise NoVaultIndex(
                f"The v1 '{TABLE_NAME}' table does not exist, but the v2 index does. "
                f"Enable v2 by creating the flag file: {VAULT_V2_FLAG}"
            )
        raise NoVaultIndex(
            "No vault index found. Build one with `python server/indexer.py` (v1) "
            "or `python server/vault_indexer_v2.py` (v2)."
        )
    table = get_table()
    query_embedding = embed_query(query)
    search_builder = table.search(query_embedding).metric("cosine").limit(fetch_n)
    where = _folder_where(folder, scope_where)
    if where:
        search_builder = search_builder.where(where, prefilter=True)
    results_df = search_builder.to_pandas()
    return format_results(results_df, include_excerpt=include_excerpt)


def _vault_retrieve_v2(query: str, folder: str | None, fetch_n: int,
                       include_excerpt: bool, scope_where: str | None = None) -> list[dict]:
    """v2 retrieval: vault_chunks_v2, child hit → parent text as the excerpt."""
    chunks_table = get_vault_chunks_table()
    if chunks_table is None:
        raise RuntimeError(f"v2 index enabled but table '{VCHUNKS_TABLE}' is missing")

    client = get_ollama()
    response = client.embed(
        model=VAULT_V2_MODEL,
        input=[VAULT_QUERY_PREFIX + query],
        options={"num_ctx": OLLAMA_NUM_CTX},
    )
    import math
    qe = response["embeddings"][0]
    s = math.sqrt(sum(v * v for v in qe))
    if s > 0:
        qe = [v / s for v in qe]

    search_builder = chunks_table.search(qe).metric("cosine").limit(fetch_n)
    where = _folder_where(folder, scope_where)
    if where:
        search_builder = search_builder.where(where, prefilter=True)
    df = search_builder.to_pandas()

    formatted = []
    parent_seen: dict[str, int] = {}
    pending: list[tuple[dict, str, str]] = []  # (entry, parent_id, child-text fallback)
    for _, row in df.iterrows():
        pid = str(row.get("parent_id", ""))
        # Same cap the textbook side uses: several children of one parent say
        # the same thing, and each one costs a result slot.
        if parent_seen.get(pid, 0) >= MAX_CHILDREN_PER_PARENT:
            continue
        parent_seen[pid] = parent_seen.get(pid, 0) + 1
        entry = {
            "file": _clean(str(row.get("file", ""))),
            "note": _clean(str(row.get("note", ""))),
            "section": _clean(str(row.get("section_path", ""))),
            "folder": _clean(str(row.get("folder", ""))),
            "tags": _clean(str(row.get("tags", ""))),
            "similarity": round(max(0, 1 - float(row.get("_distance", 0))), 4),
            "mtime": float(row["mtime"]) if row.get("mtime") is not None else None,
        }
        if include_excerpt:
            pending.append((entry, pid, str(row.get("text", ""))))
        formatted.append(entry)

    if pending:
        parent_rows = fetch_vault_parents([pid for _, pid, _ in pending])
        for entry, pid, child_text in pending:
            parent = parent_rows.get(pid)
            text = str(parent.get("text", "")) if parent else child_text
            entry["excerpt"] = _clean(text[:EXCERPT_CHARS])
    return formatted


class ScopeRefused(Exception):
    """The requested/default scope cannot be applied — refuse, never search wide."""


def resolve_scope(arguments: dict) -> tuple[str, dict, str | None]:
    """(scope name, scope map, SQL clause) for a vault call.

    The default comes from the server config, so a caller that never mentions
    scope gets the configured default — not the whole vault.
    """
    try:
        smap, default = S.load_scope_config(SCOPE_MAP_PATH, DEFAULT_SCOPE_ENV)
        sc = S.resolve(arguments.get("scope"), smap, default)
    except S.ScopeError as e:
        raise ScopeRefused(str(e)) from e
    return sc, smap, S.where_clause(sc, smap)


def _graph_folder(meta: dict, note: str) -> str:
    return (meta.get(note) or {}).get("folder", "")


def _vault_search_raw(arguments: dict, ctx: dict | None = None) -> dict:
    """vault_search without the Markdown: v1/v2 retrieval, scope, rerank, graph.

    Returns {"results", "graph_linked", "entities", "relations", "scope"}.
    Shared by the MCP tool and the chat context builder so both see the same
    ranking. Raises NoVaultIndex / ScopeRefused.
    """
    t0 = time.time()
    query = _clean(arguments["query"])
    n = min(arguments.get("n_results", 10), 30)
    folder = arguments.get("folder")
    include_excerpt = arguments.get("include_excerpt", False)
    exclude_cowork = bool(arguments.get("exclude_cowork", True))
    sc, smap, scope_where = resolve_scope(arguments)

    use_v2 = vault_v2_active()
    fallback_reason = None
    if use_v2:
        try:
            # v2 needs more dedup room: child chunks are denser, so a given
            # number of hits covers fewer distinct notes.
            formatted = _vault_retrieve_v2(query, folder, min(n * 10, 150), include_excerpt, scope_where)
        except Exception as e:
            # v2 index missing or broken → serve v1 rather than fail the
            # call. Recorded in the log so a silent downgrade stays visible.
            use_v2 = False
            fallback_reason = f"{type(e).__name__}: {e}"
            formatted = _vault_retrieve_v1(query, folder, min(n * 5, 100), include_excerpt, scope_where)
    else:
        formatted = _vault_retrieve_v1(query, folder, min(n * 5, 100), include_excerpt, scope_where)

    # Belt and braces: the prefilter already applied the scope; this re-check
    # costs nothing and keeps a backend that ignored the WHERE from leaking.
    formatted = [r for r in formatted if S.folder_in_scope(r.get("folder"), sc, smap)]

    raw_count = len(formatted)
    raw_unique_notes = len({r["note"] for r in formatted}) / max(1, raw_count)

    # Cowork exclusion (default true) — drop derivative notes
    if exclude_cowork:
        formatted = [
            r for r in formatted
            if not _is_cowork_path(r.get("file", "") or r.get("note", ""))
            and not _is_cowork_path(r.get("folder", ""))
        ]

    # Deduplicate by note name, keeping highest similarity
    seen_notes = {}
    deduped = []
    for r in formatted:
        note = r["note"]
        if note not in seen_notes:
            seen_notes[note] = True
            deduped.append(r)

    # Rerank with path + recency + relation weighting
    boost_recent = arguments.get("boost_recent", True)
    include_archived = arguments.get("include_archived", False)
    graph = get_graph()
    graph_relations = graph.get("extracted_relations", [])
    deduped = rerank(deduped, boost_recent=boost_recent, include_archived=include_archived,
                     query=query, graph_relations=graph_relations)

    # Trim to requested n after dedup + rerank
    deduped = deduped[:n]

    # Graph expansion: PPR (personalized PageRank) seeded on the top semantic
    # hits ranks multi-hop linked notes by graph proximity — replaces the old
    # naive 1-hop neighbor expansion (sparse-bridge recall@10 ~60% -> ~84%).
    # Linked notes outside the scope are dropped: a scoped search must not
    # surface out-of-scope titles through the side door.
    result_notes = {r["note"] for r in deduped}
    adjacency = graph.get("adjacency", {})
    gmeta = graph.get("metadata", {})
    ppr_ranked = personalized_pagerank(
        adjacency,
        [r["note"] for r in deduped[:5]],
        top_k=15 if sc == S.ALL else 40,
        exclude=result_notes,
    )
    graph_suggested = [nn for nn, _ in ppr_ranked
                       if S.folder_in_scope(_graph_folder(gmeta, nn), sc, smap)][:15]

    # Remove internal mtime field from output
    for r in deduped:
        r.pop("mtime", None)
        r.pop("raw_similarity", None)

    # Attach extracted entities + relations for result notes
    result_note_names = list(dict.fromkeys(r["note"] for r in deduped))
    entities_data = find_entities_for_notes(
        graph.get("extracted_entities", {}), result_note_names
    )
    relations = find_relations_for_notes(
        graph_relations, set(result_note_names)
    )

    # Observability — mirrors the textbook search log's fields
    sims = [r.get("similarity") for r in deduped if isinstance(r.get("similarity"), (int, float))]
    top_score = sims[0] if sims else 0.0
    score_gap = (sims[0] - sims[1]) if len(sims) >= 2 else 0.0
    if len(sims) >= 2:
        mean_s = sum(sims) / len(sims)
        score_std = (sum((x - mean_s) ** 2 for x in sims) / len(sims)) ** 0.5
    else:
        score_std = 0.0
    log_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query": query,
        "n_requested": n,
        "folder_filter": folder,
        "scope": sc,
        "index_version": "v2" if use_v2 else "v1",
        "embedding_model": VAULT_V2_MODEL if use_v2 else EMBEDDING_MODEL,
        "query_template_version": VAULT_QUERY_TEMPLATE_VERSION if use_v2 else "v1-noprefix",
        "v2_fallback_reason": fallback_reason,
        "top_score": round(top_score, 4),
        "score_gap_top1_top2": round(score_gap, 4),
        "score_std_topN": round(score_std, 4),
        "raw_note_hit_rate": round(raw_unique_notes, 3),
        "n_returned": len(deduped),
        "top_notes": [r.get("note", "") for r in deduped[:5]],
        "latency_ms": int((time.time() - t0) * 1000),
    }
    _log(_vault_logger, log_entry, ctx)

    return {"results": deduped, "graph_linked": graph_suggested[:10],
            "entities": entities_data, "relations": relations, "scope": sc}


def _vault_search(arguments: dict, ctx: dict | None = None) -> str:
    """vault_search: structured search, rendered as compact Markdown."""
    query = _clean(arguments["query"])
    try:
        out = _vault_search_raw(arguments, ctx)
    except (NoVaultIndex, ScopeRefused) as e:
        return str(e)
    return _clean(format_vault_markdown(
        out["results"], query,
        include_excerpt=arguments.get("include_excerpt", False),
        graph_linked=out["graph_linked"] or None,
        entities=out["entities"] or None,
        relations=out["relations"] or None,
    ))


def handle_tool_call(name: str, arguments: dict, context: dict | None = None) -> str:
    """Execute a tool and return the result as text.

    `context` says who is asking ({"caller": "mcp"|"chat"|..., "session_id": ...});
    it only goes into the search logs and never changes the results.
    """
    ctx = context or {}
    try:
        # Opened lazily, per tool. The textbook corpus lives in its own tables
        # (CHUNKS_TABLE / PARENTS_TABLE), so an install that indexes only
        # textbooks has no 'vault' table at all — opening it up front made every
        # textbook_search fail with "Table 'vault' was not found".
        # vault_search opens its own table (v1 or v2, see _vault_search), and
        # vault_related answers from the wiki-link graph alone, so neither is
        # listed here.
        table = None
        if name == "vault_similar":
            table = get_vault_chunks_table() if vault_v2_active() else None
            if table is None:
                if TABLE_NAME not in open_db().table_names():
                    return (
                        "No vault index found. Build one with "
                        "`python server/indexer.py` (v1) or "
                        "`python server/vault_indexer_v2.py` (v2)."
                    )
                table = get_table()

        if name in ("vault_related", "vault_similar"):
            try:
                v_scope, v_smap, v_where = resolve_scope(arguments)
            except ScopeRefused as e:
                return str(e)

        if name == "vault_search":
            return _vault_search(arguments, ctx)

        elif name == "vault_related":
            note = arguments["note"]
            n = min(arguments.get("n_results", 12), 30)
            graph = get_graph()
            adjacency = graph.get("adjacency", {})
            if note not in adjacency:
                return f'No wiki-link graph entry for "{note}".'
            exclude_cowork = bool(arguments.get("exclude_cowork", True))
            ranked = related_notes(adjacency, note, top_k=n * 3)
            meta = graph.get("metadata", {})
            related = []
            for rn, sc in ranked:
                folder = meta.get(rn, {}).get("folder", "")
                file_path = meta.get(rn, {}).get("file", "") or rn
                if exclude_cowork and (_is_cowork_path(file_path) or _is_cowork_path(folder)):
                    continue
                if not S.folder_in_scope(folder, v_scope, v_smap):
                    continue
                related.append({"note": rn, "score": round(sc, 4), "folder": folder})
                if len(related) >= n:
                    break
            return _clean(format_related_markdown(related, note))

        elif name == "vault_similar":
            note_name = arguments["note_name"]
            n = min(arguments.get("n_results", 10), 30)

            # Find the note's first chunk
            note_rows = table.search().where(f"note = '{_escape_sql(note_name)}'").limit(1).to_pandas()
            if note_rows.empty:
                return json.dumps({"error": f"Note '{note_name}' not found in index"})

            query_embedding = note_rows.iloc[0]["vector"]
            cand = max(n * 3, 40)
            sb = table.search(query_embedding).metric("cosine").limit(cand)
            if v_where:
                sb = sb.where(v_where, prefilter=True)
            results_df = sb.to_pandas()
            formatted = format_results(results_df, include_excerpt=True)

            # Semantic candidates (note -> dict), preserving order, source note dropped
            sem, sem_order = {}, []
            for r in formatted:
                note = r["note"]
                if note == note_name or note in sem:
                    continue
                sem[note] = r
                sem_order.append(note)

            # Graph candidates via personalized PageRank over the wiki-link graph
            # (LinearRAG-style) — surfaces multi-hop related notes semantic misses.
            graph = get_graph()
            meta = graph.get("metadata", {})
            adjacency = graph.get("adjacency", {})
            ppr_order = [nn for nn, _ in related_notes(adjacency, note_name, top_k=cand)] \
                if note_name in adjacency else []
            ppr_order = [nn for nn in ppr_order
                         if S.folder_in_scope(_graph_folder(meta, nn), v_scope, v_smap)]
            graph_cosine = _cosine_for_notes(
                table, query_embedding, [nn for nn in ppr_order if nn not in sem]
            )

            # Reciprocal-rank fusion of graph + semantic rankings
            RRF_K = 60
            fused = {}
            for rank, nn in enumerate(sem_order):
                fused[nn] = fused.get(nn, 0.0) + 1.0 / (RRF_K + rank)
            for rank, nn in enumerate(ppr_order):
                fused[nn] = fused.get(nn, 0.0) + 1.0 / (RRF_K + rank)

            deduped = []
            for nn in sorted(fused, key=lambda k: fused[k], reverse=True):
                d = sem.get(nn) or {
                    "note": nn, "file": meta.get(nn, {}).get("file", ""),
                    "folder": meta.get(nn, {}).get("folder", ""), "section": "", "excerpt": "",
                    "similarity": graph_cosine.get(nn, 0.0),
                }
                d["fused"] = fused[nn]
                deduped.append(d)

            # Rerank applies path + recency on the fused score, then trim. The
            # fused score only ranks: it tops out near 0.03, so reporting it as
            # 'similarity' put a meaningless number in front of the model, and
            # broke the same field in the plugin's panel (issue #5).
            deduped = rerank(deduped, rank_key="fused")[:n]

            # Remove internal mtime field from output
            for r in deduped:
                r.pop("mtime", None)
                r.pop("raw_similarity", None)

            return _clean(format_vault_markdown(
                deduped, note_name,
                include_excerpt=True,
                header="vault_similar",
            ))

        elif name == "vault_stats":
            stats = {}
            # A textbook-only install has no notes index; report what exists
            # rather than failing the whole call.
            db0 = open_db()
            if TABLE_NAME in db0.table_names():
                table = get_table()
                all_data = table.to_pandas()
                files = set(all_data["file"].unique())
                folders = all_data["folder"].value_counts().to_dict()
                stats["total_chunks"] = table.count_rows()
                stats["total_files"] = len(files)
                stats["chunks_by_folder"] = dict(sorted(folders.items(), key=lambda x: -x[1]))
            else:
                stats["vault_index"] = f"not built (no '{TABLE_NAME}' table)"

            # Include textbook stats if available (v2 schema: chunks + parents)
            db = open_db()
            if CHUNKS_TABLE in db.table_names():
                ch_table = db.open_table(CHUNKS_TABLE)
                ch_count = ch_table.count_rows()
                ch_books_df = ch_table.search().select(["book", "file"]).limit(10**9).to_pandas()
                ch_books = ch_books_df["book"].value_counts().to_dict()
                stats["textbook_chunks"] = ch_count
                stats["textbook_files"] = len(set(ch_books_df["file"].unique()))
                stats["textbook_books"] = dict(sorted(ch_books.items(), key=lambda x: -x[1]))
            if PARENTS_TABLE in db.table_names():
                p_table = db.open_table(PARENTS_TABLE)
                stats["textbook_parents"] = p_table.count_rows()

            # Vault v2 tables, and which index searches are actually reading.
            if VCHUNKS_TABLE in db.table_names():
                stats["vault_chunks_v2"] = db.open_table(VCHUNKS_TABLE).count_rows()
            if VPARENTS_TABLE in db.table_names():
                stats["vault_parents_v2"] = db.open_table(VPARENTS_TABLE).count_rows()
            stats["vault_v2_active"] = vault_v2_active()

            return _safe_json_dumps(stats, indent=2)

        elif name == "textbook_search":
            return _textbook_search_v2(arguments, ctx)

        elif name == "lecture_search":
            if not lecture_enabled():
                return json.dumps({"error": f"lecture index not configured ({LECTURE_DB_PATH} missing)"})
            return _textbook_search_v2(arguments, ctx, corpus="lecture")

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        tb = traceback.format_exc()
        return json.dumps({"error": str(e), "traceback": tb})


def main():
    """MCP server main loop — reads JSON-RPC from stdin, writes to stdout."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=False)
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "vault-search-mcp",
                        "version": "3.0.0",
                        "author": "P.W. Chen (drpwchen.com)",
                    },
                },
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass  # No response needed

        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            arguments = msg["params"].get("arguments", {})
            result_text = handle_tool_call(tool_name, arguments)
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }
            sys.stdout.write(_safe_json_dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "ping":
            response = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        else:
            # Unknown method — return empty result
            if msg_id is not None:
                response = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
