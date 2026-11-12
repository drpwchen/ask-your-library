"""
Textbook indexer v3 — provenance-carrying chunks into a NEW database.

    python textbook_indexer_v3.py --corpus ~/textbook-md-v2 --db ~/.vault-search/lance_db_v3
    python textbook_indexer_v3.py --corpus ... --db ... --book Braddom_PM-and-R_7e_2021
    python textbook_indexer_v3.py --corpus ... --db ... --audit           # per-chapter coverage
    python textbook_indexer_v3.py --corpus ... --dry-run                  # chunk only, no GPU
    python textbook_indexer_v3.py --corpus ... --holding-report out.md    # suggest sources to retire

What differs from textbook_indexer.py (which keeps serving the v2 index):

* Every child carries its own page range (from its own lines), printed pages
  from the book's ``_manifest.json``, ``page_confidence``, character offsets,
  ``source_hash``, ``chapter_id`` / ``chapter_number``, edition / year / tier /
  status / corpus_type from ``books.json``, and ``section_kind``.
  See ``provenance.py`` and ``chunker_v3.py``.
* Tables are their own retrieval units, never cut at a page break or at the
  child length limit (``table_units.py``).
* References sections and download-site watermarks are filtered here.
* One source per guideline: each ``_Guidelines/<name>`` folder is its own
  ``source``; per folder exactly ONE of ``full_text.md`` or its chapter files is
  indexed, never both.
* Sources marked superseded / derivative / low_trust in ``books.json`` are not
  indexed (``--include-superseded`` keeps them).
* ``--book`` always re-embeds that book (no hash skip); a file that became
  empty has its old rows deleted; ``--audit`` checks coverage chapter by chapter.
* It refuses to write to the live database: ``--db`` is required and may not be
  the configured ``VAULT_SEARCH_DB_PATH`` or the lecture DB.
* A non-default ``--model`` writes its own pair of tables (vector width differs
  per model), and ``indexing_signature`` pins the model digest, the document
  instruction and the normalisation.
* Resumable: the per-table hash cache under ``<db>/_v3_state/`` is saved every
  few files, so an interrupted run continues where it stopped.
* Builds the native full-text index on ``text`` at the end, for hybrid search.
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import provenance as P
import chunker_v3 as C
import books_meta

V3_ALGORITHM_VERSION = "v3.0.0"
CHUNKING_VERSION = "v3.0"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DOC_INSTRUCTION = ""          # documents are embedded without an instruction
NORMALIZATION = "l2"
OLLAMA_NUM_CTX = 4096
BATCH_SIZE = 64
BATCH_TOKEN_BUDGET = 48_000
SAVE_EVERY_FILES = 10
PRUNE_EVERY_FILES = 500
HIDDEN_STATUSES = {"superseded", "derivative", "low_trust"}
SKIP_BASENAMES = {"index.md"}
CHAPTER_COVERAGE_MIN = 0.5    # chapters must hold ≥50% of full_text's bytes to replace it

# Same analyzer as the Phase 1 FTS A/B (eval/build/fts_snapshot.py), so the
# numbers that justified hybrid search carry over.
FTS_PARAMS = dict(use_tantivy=False, base_tokenizer="simple", language="English", stem=True,
                  remove_stop_words=True, lower_case=True, ascii_folding=True, replace=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Table names / signature
# =============================================================================

def table_suffix(model: str) -> str:
    """'' for the default model; a slug otherwise. Different models produce
    vectors of different widths, so they can never share a vector column."""
    if model == DEFAULT_MODEL:
        return ""
    return "__" + re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()


def table_names(model: str) -> tuple[str, str]:
    s = table_suffix(model)
    return f"textbook_chunks{s}", f"textbook_parents{s}"


def signature_parts(model: str, model_digest: str, dim: int, settings: C.Settings) -> list[str]:
    return [V3_ALGORITHM_VERSION, CHUNKING_VERSION, f"model={model}", f"digest={model_digest}",
            f"dim={dim}", f"doc_instruction={DOC_INSTRUCTION!r}", f"norm={NORMALIZATION}",
            *settings.signature_parts()]


def compute_signature(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# =============================================================================
# Safety: never the live DB
# =============================================================================

class UnsafeTarget(SystemExit):
    pass


def _resolved(p) -> Path:
    return Path(p).expanduser().resolve()


def is_v3_db(db) -> bool:
    """A database a v3 tool built: textbook (`_v3_state/`) or vault (`_vault_state/`)."""
    d = Path(db).expanduser()
    return (d / "_v3_state").is_dir() or (d / "_vault_state").is_dir()


def check_target(db: Path, corpus: Path | None, forbidden: list[Path]) -> None:
    """Refuse a --db that is (or contains, or sits inside) a live database or
    the corpus itself. A live database that is ITSELF a v3 database is fine:
    after the switch-over, incremental runs write to it by design."""
    d = _resolved(db)
    for f in forbidden:
        if f is None:
            continue
        f = _resolved(f)
        if d == f and is_v3_db(d):
            continue
        if d == f or f in d.parents or d in f.parents:
            raise UnsafeTarget(f"[refuse] --db {d} overlaps the live database {f}. "
                               f"v3 builds go to a new directory (e.g. ~/.vault-search/lance_db_v3).")
    if corpus is not None:
        c = _resolved(corpus)
        if d == c or c in d.parents:
            raise UnsafeTarget(f"[refuse] --db {d} is inside the corpus {c}.")


def live_paths() -> list[Path]:
    """Databases the running server uses. Read lazily: config needs VAULT_PATH."""
    out = []
    try:
        from config import DB_PATH, LECTURE_DB_PATH, DATA_DIR, TEXTBOOK_DB_PATH
        out += [DB_PATH, TEXTBOOK_DB_PATH, LECTURE_DB_PATH,
                DATA_DIR / "lance_db", DATA_DIR / "lecture_db"]
    except Exception:
        home = Path("~/.vault-search").expanduser()
        out += [home / "lance_db", home / "lecture_db"]
    return out


# =============================================================================
# Corpus walk: units, file choice, exclusions
# =============================================================================

def iter_units(corpus: Path, only: str | None = None):
    """(book, unit_dir) pairs. A unit is a book folder, or for ``_Guidelines``
    each guideline sub-folder. Folders starting with '_' (other than
    _Guidelines) are tooling output, not books."""
    for d in sorted(p for p in corpus.iterdir() if p.is_dir()):
        if d.name == P.GUIDELINE_BOOK:
            for g in sorted(p for p in d.iterdir() if p.is_dir()):
                if only and only not in (d.name, g.name, f"{d.name}/{g.name}"):
                    continue
                yield d.name, g
            continue
        if d.name.startswith((".", "_")):
            continue
        if only and only != d.name:
            continue
        yield d.name, d


def choose_files(unit_dir: Path) -> tuple[list[Path], list[dict]]:
    """Per directory, index full_text.md OR its chapter files, never both.

    Chapters win when they hold at least CHAPTER_COVERAGE_MIN of full_text's
    bytes (a split whose files are mostly empty loses to the whole-book file).
    Returns (files, decisions) — decisions are for the run report.
    """
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in sorted(unit_dir.rglob("*.md")):
        rel_parts = f.relative_to(unit_dir).parts
        if any(p.startswith((".", "_")) for p in rel_parts[:-1]) or "__pycache__" in rel_parts:
            continue
        if f.name.casefold() in SKIP_BASENAMES:
            continue
        by_dir[f.parent].append(f)
    files, decisions = [], []
    for d, fs in sorted(by_dir.items()):
        full = [f for f in fs if f.name.casefold() == "full_text.md"]
        chapters = [f for f in fs if f.name.casefold() != "full_text.md"]
        if full and chapters:
            fsize = full[0].stat().st_size
            csize = sum(f.stat().st_size for f in chapters)
            if fsize and csize / fsize >= CHAPTER_COVERAGE_MIN:
                files += chapters
                decisions.append({"dir": str(d), "chose": "chapters", "ratio": round(csize / fsize, 2)})
            else:
                files += full
                decisions.append({"dir": str(d), "chose": "full_text",
                                  "ratio": round(csize / fsize, 2) if fsize else None})
        else:
            files += fs
    return files, decisions


def find_manifest(f: Path, unit_dir: Path, cache: dict) -> tuple[dict | None, str]:
    """Nearest ``_manifest.json`` from the file's folder up to the unit root.
    Returns (manifest, path of f relative to the manifest's folder)."""
    d = f.parent
    while True:
        if d not in cache:
            cache[d] = P.load_manifest(d)
        if cache[d] is not None:
            return cache[d], str(f.relative_to(d)).replace("\\", "/")
        if d == unit_dir or unit_dir not in d.parents:
            return None, f.name
        d = d.parent


def chapter_title(f: Path, unit_dir: Path, entry: dict) -> str:
    if entry.get("title"):
        return str(entry["title"])
    rel = f.relative_to(unit_dir)
    stem = f.stem
    if stem.casefold() == "full_text" and len(rel.parts) > 1:
        stem = rel.parts[-2]
    elif len(rel.parts) > 1:
        stem = f"{rel.parts[-2]} › {stem}"
    return stem.replace("_", " ").strip()


# =============================================================================
# Embedding
# =============================================================================

class Embedder:
    def __init__(self, model: str, host: str):
        import ollama
        self.model = model
        self.client = ollama.Client(host=host)
        self.failed = 0

    def probe(self) -> tuple[int, str]:
        r = self.client.embed(model=self.model, input=["dimension probe"],
                              options={"num_ctx": OLLAMA_NUM_CTX})
        dim = len(r["embeddings"][0])
        digest = ""
        try:
            for m in self.client.list().get("models", []):
                name = m.get("model") or m.get("name")
                if name == self.model or name == self.model + ":latest":
                    digest = (m.get("digest") or "")[:12]
        except Exception:
            pass
        return dim, digest or "unknown"

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out += self._batch(texts[i:i + BATCH_SIZE])
        return out

    def _batch(self, texts):
        for attempt in range(3):
            try:
                r = self.client.embed(model=self.model, input=texts,
                                      options={"num_ctx": OLLAMA_NUM_CTX}, keep_alive="30m")
                return [self._norm(v) for v in r["embeddings"]]
            except Exception as e:
                if "400" in str(e):
                    break
                time.sleep(3 * (attempt + 1))
        # salvage one by one: a poison input costs itself, not the batch
        res = []
        for t in texts:
            try:
                r = self.client.embed(model=self.model, input=[t], options={"num_ctx": OLLAMA_NUM_CTX})
                res.append(self._norm(r["embeddings"][0]))
            except Exception:
                res.append(None)
                self.failed += 1
        return res

    @staticmethod
    def _norm(v):
        if any(math.isnan(x) for x in v):
            return None
        s = math.sqrt(sum(x * x for x in v))
        return [x / s for x in v] if s else v

    def unload(self):
        try:
            self.client.embed(model=self.model, input=[], keep_alive=0)
        except Exception:
            pass


def truncate_tokens(text: str, max_tokens: int) -> str:
    from textbook_indexer import get_tokenizer
    tk = get_tokenizer()
    ids = tk.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tk.decode(ids[:max_tokens], skip_special_tokens=True)


# =============================================================================
# Schema
# =============================================================================

def schemas(dim: int):
    import pyarrow as pa
    common = [
        ("text", pa.utf8()), ("file", pa.utf8()), ("book", pa.utf8()), ("source", pa.utf8()),
        ("lineage", pa.utf8()), ("chapter", pa.utf8()), ("chapter_id", pa.utf8()),
        ("chapter_number", pa.int32()), ("section_path", pa.utf8()),
        ("section_path_raw", pa.utf8()), ("section_kind", pa.utf8()),
        ("page_start", pa.int32()), ("page_end", pa.int32()),
        ("pdf_page_start", pa.int32()), ("pdf_page_end", pa.int32()),
        ("printed_page_start", pa.utf8()), ("printed_page_end", pa.utf8()),
        ("page_confidence", pa.utf8()), ("char_start", pa.int64()), ("char_end", pa.int64()),
        ("token_count", pa.int32()), ("heading_origin", pa.utf8()),
        ("figure_ids", pa.list_(pa.utf8())), ("has_figure_ref", pa.bool_()),
        ("edition", pa.utf8()), ("year", pa.int32()), ("tier", pa.int32()),
        ("status", pa.utf8()), ("corpus_type", pa.utf8()), ("split_method", pa.utf8()),
        ("source_hash", pa.utf8()), ("chunking_version", pa.utf8()),
        ("embedding_model", pa.utf8()), ("query_template_version", pa.utf8()),
        ("indexing_signature", pa.utf8()), ("mtime", pa.float64()),
    ]
    vec = ("vector", pa.list_(pa.float32(), dim))
    chunks = pa.schema([pa.field(n, t) for n, t in [
        ("id", pa.utf8()), vec, ("parent_id", pa.utf8()), ("chunk_kind", pa.utf8()),
        ("chunk_idx", pa.int32()), ("n_siblings", pa.int32()), ("table_id", pa.utf8()),
        ("table_review_flag", pa.utf8()), ("table_embed_truncated", pa.bool_()),
    ] + common])
    parents = pa.schema([pa.field(n, t) for n, t in [
        ("parent_id", pa.utf8()), vec, ("has_table", pa.bool_()),
    ] + common])
    return chunks, parents


# =============================================================================
# Records
# =============================================================================

def _fig_ids(text: str) -> list[str]:
    from textbook_indexer import extract_figure_ids
    return extract_figure_ids(text)


def base_fields(ctx: C.FileContext, meta: dict, entry: dict, manifest: dict | None,
                sig: str, model: str, qtv: str, lineage: str) -> dict:
    cn = entry.get("chapter_number")
    return {
        "file": ctx.rel_file, "book": ctx.book, "source": ctx.source, "lineage": lineage,
        "chapter": ctx.chapter,
        "chapter_id": f"{ctx.source}#ch{cn}" if isinstance(cn, int) else f"{ctx.source}/{ctx.rel_file.split('/', 1)[-1]}",
        "chapter_number": cn if isinstance(cn, int) else None,
        "edition": str(meta.get("edition") or ""), "year": meta.get("year"),
        "tier": meta.get("tier"), "status": meta.get("status") or "current",
        "corpus_type": meta.get("corpus_type") or "", "split_method": (manifest or {}).get("split_method", ""),
        "source_hash": ctx.source_hash, "chunking_version": CHUNKING_VERSION,
        "embedding_model": model, "query_template_version": qtv,
        "indexing_signature": sig, "mtime": ctx.mtime,
    }


def _common(rec: dict, base: dict) -> dict:
    path = rec["section_path_parts"]
    raw = " > ".join(path)
    return {
        **base,
        "text": rec["text"],
        "section_path": re.sub(r"\s+", " ", raw).strip(), "section_path_raw": raw,
        "section_kind": rec.get("section_kind") or "",
        "page_start": rec["page_start"], "page_end": rec["page_end"],
        "pdf_page_start": rec["pdf_page_start"], "pdf_page_end": rec["pdf_page_end"],
        "printed_page_start": rec["printed_page_start"], "printed_page_end": rec["printed_page_end"],
        "page_confidence": rec["page_confidence"],
        "char_start": rec["char_start"], "char_end": rec["char_end"],
        "token_count": rec["token_count"],
        "figure_ids": _fig_ids(rec["text"]),
    }


def chunk_row(rec: dict, base: dict, vec) -> dict:
    row = _common(rec, base)
    row.update({
        "id": rec["chunk_id"], "vector": vec, "parent_id": rec["parent_id"],
        "chunk_kind": rec["chunk_kind"], "chunk_idx": rec["chunk_idx"],
        "n_siblings": rec["n_siblings"], "table_id": rec["table_id"],
        "table_review_flag": rec["table_review_flag"],
        "table_embed_truncated": bool(rec["table_embed_truncated"]),
        "heading_origin": rec.get("heading_origin", "child"),
    })
    row["has_figure_ref"] = bool(row["figure_ids"])
    return row


def parent_row(rec: dict, base: dict, vec) -> dict:
    row = _common(rec, base)
    row.update({"parent_id": rec["parent_id"], "vector": vec, "has_table": rec["has_table"],
                "heading_origin": rec["heading_origin"]})
    row["has_figure_ref"] = bool(row["figure_ids"])
    return row


# =============================================================================
# State (hash cache = checkpoint)
# =============================================================================

def state_dir(db: Path) -> Path:
    d = Path(db) / "_v3_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def save_json(p: Path, data) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _esc(s: str) -> str:
    return s.replace("'", "''")


# =============================================================================
# Plan: which files, with what context
# =============================================================================

def plan_corpus(corpus: Path, meta_table: dict, *, book: str | None, include_superseded: bool):
    """(work items, excluded units, file-choice decisions)."""
    items, excluded, decisions = [], [], []
    for bk, unit in iter_units(corpus, book):
        key = unit.name if bk == P.GUIDELINE_BOOK else bk
        m = books_meta.lookup(bk, f"{bk}/{unit.name}/x.md" if bk == P.GUIDELINE_BOOK else f"{bk}/x.md",
                              meta_table)
        if not include_superseded and m.get("status") in HIDDEN_STATUSES:
            excluded.append({"source": key, "book": bk, "status": m.get("status"),
                             "superseded_by": m.get("superseded_by", "")})
            continue
        files, dec = choose_files(unit)
        decisions += dec
        cache: dict = {}
        for f in files:
            manifest, mrel = find_manifest(f, unit, cache)
            items.append({"path": f, "book": bk, "source": key, "unit": unit, "meta": m,
                          "manifest": manifest, "manifest_rel": mrel,
                          "rel": str(f.relative_to(corpus)).replace("\\", "/")})
    return items, excluded, decisions


def make_context(it: dict, content: str) -> tuple[C.FileContext, dict]:
    entry = P.file_entry(it["manifest"], it["manifest_rel"])
    ctx = C.FileContext(
        book=it["book"], rel_file=it["rel"], source=it["source"],
        chapter=chapter_title(it["path"], it["unit"], entry),
        manifest=it["manifest"], manifest_rel=it["manifest_rel"],
        file_kind=entry.get("kind") or "",
        chapter_number=entry.get("chapter_number") if isinstance(entry.get("chapter_number"), int) else None,
        source_hash=P.sha256_text(content), mtime=it["path"].stat().st_mtime,
    )
    return ctx, entry


# =============================================================================
# Main run
# =============================================================================

def run(args) -> int:
    corpus = _resolved(args.corpus)
    meta_table = books_meta.load(Path(args.books_meta).expanduser()) if args.books_meta else {}
    settings = C.Settings(keep_references=args.keep_references)
    items, excluded, decisions = plan_corpus(corpus, meta_table, book=args.book,
                                             include_superseded=args.include_superseded)
    print(f"[plan] {len(items)} files, {len(excluded)} sources excluded by books.json, "
          f"{sum(1 for d in decisions if d['chose'] == 'full_text')} folders use full_text "
          f"instead of chapters", flush=True)

    if args.dry_run:
        return dry_run(items, settings, args)

    import lancedb
    db_path = _resolved(args.db)
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))
    chunks_name, parents_name = table_names(args.model)
    st = state_dir(db_path)
    hash_path = st / f"hashes{table_suffix(args.model)}.json"
    meta_path = st / f"index_meta{table_suffix(args.model)}.json"

    from config import OLLAMA_HOST
    emb = Embedder(args.model, OLLAMA_HOST)
    dim, digest = emb.probe()
    sig = compute_signature(signature_parts(args.model, digest, dim, settings))
    qprefix = os.environ.get("VAULT_SEARCH_TEXTBOOK_QUERY_PREFIX",
                             "Instruct: Given a query, retrieve the most relevant reference passages.\nQuery: ")
    from config import template_version
    qtv = template_version(os.environ.get("VAULT_SEARCH_TEXTBOOK_TEMPLATE_VERSION", "qwen3-generic-v1"),
                           qprefix)
    print(f"[model] {args.model} dim={dim} digest={digest} signature={sig} "
          f"tables={chunks_name}/{parents_name}", flush=True)

    if args.rebuild:
        for t in (chunks_name, parents_name):
            if t in db.table_names():
                db.drop_table(t)
        hash_path.unlink(missing_ok=True)
    cs, ps = schemas(dim)
    chunks = db.open_table(chunks_name) if chunks_name in db.table_names() else \
        db.create_table(chunks_name, schema=cs)
    parents = db.open_table(parents_name) if parents_name in db.table_names() else \
        db.create_table(parents_name, schema=ps)

    old = load_json(meta_path)
    if old.get("indexing_signature") and old["indexing_signature"] != sig and not args.rebuild:
        print(f"[WARN] signature mismatch (index={old['indexing_signature']}, now={sig}); "
              f"re-run with --rebuild.", flush=True)
        return 2
    save_json(meta_path, {
        "indexing_signature": sig, "signature_parts": signature_parts(args.model, digest, dim, settings),
        "model": args.model, "model_digest": digest, "dim": dim, "doc_instruction": DOC_INSTRUCTION,
        "normalization": NORMALIZATION, "query_instruction": qprefix, "query_template_version": qtv,
        "tables": [chunks_name, parents_name], "corpus": str(corpus), "updated": now_iso(),
    })

    hashes = load_json(hash_path)
    force_files = bool(args.book)  # I3: --book never skips on hash
    totals = Counter()
    t0 = time.time()

    # Excluded sources: make sure none of their rows linger.
    for ex in excluded:
        col = "source" if ex["book"] == P.GUIDELINE_BOOK else "book"
        for t in (chunks, parents):
            t.delete(f"{col} = '{_esc(ex['source'])}'")

    current = set()
    for n, it in enumerate(items, 1):
        rel = it["rel"]
        current.add(rel)
        content = it["path"].read_text(encoding="utf-8", errors="replace")
        h = P.sha256_text(content)
        if not force_files and hashes.get(rel) == h:
            totals["skipped"] += 1
            continue
        ctx, entry = make_context(it, content)
        prs, kids, stats = C.chunk_file(content, ctx, settings)
        for k in ("watermarks", "reference_parents_dropped", "reference_children_dropped",
                  "tables", "tables_digest_embedded"):
            totals[k] += getattr(stats, k)
        esc = _esc(rel)
        if not kids:
            # I3: a file that became empty (or all references) loses its old rows.
            chunks.delete(f"file = '{esc}'")
            parents.delete(f"file = '{esc}'")
            hashes[rel] = h
            totals["empty"] += 1
            continue
        lineage = P.lineage_key(it["source"], meta_table.get(it["source"]))
        base = base_fields(ctx, it["meta"], entry, it["manifest"], sig, args.model, qtv, lineage)
        pvec = emb.embed([truncate_tokens(p["embed_text"], settings.embed_max) for p in prs])
        cvec = emb.embed([truncate_tokens(c["embed_text"], settings.embed_max) for c in kids])
        if any(v is None for v in pvec + cvec):
            totals["files_failed"] += 1
            print(f"  [fail] {rel}: {sum(v is None for v in pvec + cvec)} embeddings failed; "
                  f"left for the next run", flush=True)
            continue
        chunks.delete(f"file = '{esc}'")
        parents.delete(f"file = '{esc}'")
        parents.add([parent_row(p, base, v) for p, v in zip(prs, pvec)])
        chunks.add([chunk_row(c, base, v) for c, v in zip(kids, cvec)])
        hashes[rel] = h
        totals["files"] += 1
        totals["chunks"] += len(kids)
        totals["parents"] += len(prs)
        if n % SAVE_EVERY_FILES == 0:
            save_json(hash_path, hashes)
            el = time.time() - t0
            print(f"  [{n}/{len(items)}] files={totals['files']} chunks={totals['chunks']} "
                  f"skipped={totals['skipped']} {el / 60:.1f} min", flush=True)
            if _LEASE is not None:
                _LEASE.heartbeat()
        if n % PRUNE_EVERY_FILES == 0:
            _optimize(chunks)
            _optimize(parents)

    save_json(hash_path, hashes)

    # Orphans: files indexed before but no longer selected (deleted, excluded,
    # or replaced by the full_text/chapters choice). Scoped to --book's units.
    gone = _orphans(chunks, current, args.book, corpus)
    if gone:
        q = ", ".join(f"'{_esc(f)}'" for f in gone)
        chunks.delete(f"file IN ({q})")
        parents.delete(f"file IN ({q})")
        for f in gone:
            hashes.pop(f, None)
        save_json(hash_path, hashes)
        print(f"[orphan] removed {len(gone)} files no longer in the corpus selection", flush=True)

    if chunks.count_rows():
        try:
            chunks.create_fts_index("text", **FTS_PARAMS)
            print("[fts] full-text index on text built", flush=True)
        except Exception as e:
            print(f"[fts] could not build the full-text index: {e}", flush=True)
    _optimize(chunks)
    _optimize(parents)
    (db_path / f"READY_V3{table_suffix(args.model)}").write_text(f"{sig} {now_iso()}", encoding="utf-8")
    emb.unload()

    print(f"\n[done] {(time.time() - t0) / 60:.1f} min  {dict(totals)}", flush=True)
    if emb.failed:
        print(f"  ⚠ {emb.failed} embeddings failed — re-run the same command to retry those files.",
              flush=True)
    return 0


def _orphans(chunks, current: set, book: str | None, corpus: Path) -> list[str]:
    try:
        indexed = set(chunks.search().select(["file"]).limit(10 ** 9).to_pandas()["file"].unique())
    except Exception:
        return []
    if book:
        pref = (f"{P.GUIDELINE_BOOK}/{book}/" if (corpus / P.GUIDELINE_BOOK / book).is_dir()
                else f"{book}/")
        indexed = {f for f in indexed if f.startswith(pref)}
    return sorted(indexed - current)


def _optimize(table) -> None:
    from datetime import timedelta
    try:
        table.optimize(cleanup_older_than=timedelta(minutes=5), delete_unverified=True)
    except Exception as e:
        print(f"[optimize] skipped: {e}", flush=True)


# =============================================================================
# Dry run / audit / holding report
# =============================================================================

def dry_run(items, settings, args) -> int:
    tok = C.qwen_tok() if not args.fast_tokens else C.whitespace_tok()
    totals = Counter()
    spans = Counter()
    per_file = []
    for it in items:
        content = it["path"].read_text(encoding="utf-8", errors="replace")
        ctx, _ = make_context(it, content)
        prs, kids, stats = C.chunk_file(content, ctx, settings, tok)
        totals["files"] += 1
        totals["parents"] += len(prs)
        totals["children"] += len(kids)
        totals["tables"] += sum(1 for k in kids if k["chunk_kind"] == "table")
        totals["table_review"] += sum(1 for k in kids if k["table_review_flag"] == "review")
        totals["printed"] += sum(1 for k in kids if k["page_confidence"] == P.PRINTED)
        for k in ("watermarks", "reference_parents_dropped", "reference_children_dropped",
                  "tables_digest_embedded"):
            totals[k] += getattr(stats, k)
        for k in kids:
            w = k["pdf_page_end"] - k["pdf_page_start"] if k["pdf_page_start"] else -1
            spans["no_page" if w < 0 else "0" if w == 0 else "1" if w == 1 else "2-4" if w <= 4 else "5+"] += 1
        per_file.append({"file": it["rel"], "children": len(kids), "parents": len(prs),
                         "refs_dropped": stats.reference_children_dropped + stats.reference_parents_dropped})
    print(json.dumps({"totals": dict(totals), "child_page_span": dict(spans)}, ensure_ascii=False, indent=1))
    if args.report:
        save_json(Path(args.report), {"totals": dict(totals), "child_page_span": dict(spans),
                                      "files": per_file})
    return 0


def audit(args) -> int:
    """Per-chapter coverage: every selected file with indexable text must have
    rows; every indexed file must still be selected."""
    import lancedb
    corpus = _resolved(args.corpus)
    meta_table = books_meta.load(Path(args.books_meta).expanduser()) if args.books_meta else {}
    items, excluded, _ = plan_corpus(corpus, meta_table, book=args.book,
                                     include_superseded=args.include_superseded)
    db = lancedb.connect(str(_resolved(args.db)))
    chunks_name, _ = table_names(args.model)
    if chunks_name not in db.table_names():
        print(f"[audit] no table {chunks_name} in {args.db}")
        return 1
    df = db.open_table(chunks_name).search().select(["file"]).limit(10 ** 9).to_pandas()
    counts = Counter(df["file"])
    missing, ok, empty = [], 0, 0
    for it in items:
        body = P.PAGE_MARKER_RE.sub("", it["path"].read_text(encoding="utf-8", errors="replace"))
        if len(re.sub(r"\s+", "", body)) < 200:
            empty += 1
            continue
        if counts.get(it["rel"], 0) == 0:
            missing.append(it["rel"])
        else:
            ok += 1
    selected = {it["rel"] for it in items}
    stray = sorted(f for f in counts if f not in selected
                   and (not args.book or f.split("/")[0] == args.book or f"/{args.book}/" in f))
    print(f"[audit] chapters with rows: {ok}; missing: {len(missing)}; "
          f"near-empty skipped: {empty}; indexed but not selected: {len(stray)}")
    for f in missing[:200]:
        print(f"  MISSING {f}")
    for f in stray[:200]:
        print(f"  STRAY   {f}")
    if args.report:
        save_json(Path(args.report), {"ok": ok, "missing": missing, "stray": stray, "empty": empty})
    return 1 if (missing or stray) else 0


def holding_report(args) -> int:
    """Suggest sources to move to a holding folder. Never moves anything."""
    corpus = _resolved(args.corpus)
    meta_table = books_meta.load(Path(args.books_meta).expanduser()) if args.books_meta else {}
    units = list(iter_units(corpus))
    keys = [(bk, u.name if bk == P.GUIDELINE_BOOK else bk) for bk, u in units]
    rows = []
    for bk, key in keys:
        m = books_meta.lookup(bk, f"{bk}/{key}/x.md" if bk == P.GUIDELINE_BOOK else f"{bk}/x.md", meta_table)
        if m.get("status") in HIDDEN_STATUSES:
            rows.append({"source": key, "reason": f"books.json status={m['status']}",
                         "replacement": m.get("superseded_by", ""), "basis": "decided"})
    decided = {r["source"] for r in rows}
    # Older editions of a lineage that also has a newer edition (proposal only).
    by_lineage = defaultdict(list)
    for bk, key in keys:
        if bk == P.GUIDELINE_BOOK:
            continue
        by_lineage[P.lineage_key(key, meta_table.get(key))].append(key)
    for lin, ks in sorted(by_lineage.items()):
        if len(ks) < 2:
            continue
        ranked = sorted(ks, key=lambda k: (_edition_num(k), books_meta.infer(k, "")["year"] or 0))
        newest = ranked[-1]
        for k in ranked[:-1]:
            if k in decided:
                continue
            same = _edition_num(k) == _edition_num(newest)
            rows.append({"source": k, "reason": ("same edition as " if same else "older edition of ")
                         + f"lineage {lin}", "replacement": newest,
                         "basis": "duplicate? check which copy is complete" if same else "inferred"})
    # Same author + same edition + same year + near-identical title slug: the
    # same book converted twice under two folder names (e.g. a re-download).
    # Suggest keeping the copy with more chapter text.
    flagged = {r["source"] for r in rows}
    plain = [key for bk, key in keys if bk != P.GUIDELINE_BOOK]
    for i, a in enumerate(plain):
        for b in plain[i + 1:]:
            if a in flagged or b in flagged:
                continue
            if _dup_pair(a, b):
                sa, sb = _chapter_bytes(corpus / a), _chapter_bytes(corpus / b)
                drop, keep = (a, b) if sa < sb else (b, a)
                rows.append({"source": drop, "reason": f"same book as `{keep}` (author, edition, "
                             f"year and title match); {min(sa, sb):,} vs {max(sa, sb):,} chapter bytes",
                             "replacement": keep, "basis": "duplicate? check which copy is complete"})
                flagged.add(drop)
    lines = ["# 建議搬到 holding 的來源（只是建議，沒有搬任何檔案）", "",
             f"語料：`{corpus}`　產生：{now_iso()[:19]}", "",
             "basis：decided＝books.json 已標狀態（v3 索引已排除）；inferred＝依資料夾名推斷的舊版次，"
             "要人確認；duplicate?＝同版次兩份，留章節最完整的一份。", "",
             "| 來源 | 原因 | 取代者 | basis |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['source']}` | {r['reason']} | `{r['replacement']}` | {r['basis']} |")
    out = Path(args.holding_report).expanduser()
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(out.with_suffix(".json"), rows)
    print(f"[holding] {len(rows)} suggestions → {out}")
    return 0


_STOP = {"of", "and", "the", "in", "for", "a", "an", "to"}


def _slug_tokens(key: str) -> set[str]:
    parts = key.split("_")
    title = parts[1] if len(parts) > 1 else key
    return {t for t in re.split(r"[-\s]+", title.casefold()) if t and t not in _STOP}


def _dup_pair(a: str, b: str) -> bool:
    if a.split("_")[0].casefold() != b.split("_")[0].casefold():
        return False
    if _edition_num(a) != _edition_num(b):
        return False
    ya, yb = books_meta.infer(a, "")["year"], books_meta.infer(b, "")["year"]
    if ya != yb:
        return False
    ta, tb = _slug_tokens(a), _slug_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


def _chapter_bytes(d: Path) -> int:
    try:
        return sum(f.stat().st_size for f in d.rglob("*.md") if f.name.casefold() != "full_text.md")
    except Exception:
        return 0


def _edition_num(key: str) -> int:
    m = re.search(r"_(\d{1,2})(?:e|ed|th|nd|rd|st)(?=_|$)", key, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# =============================================================================
# CLI
# =============================================================================

_LEASE = None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Textbook indexer v3 (provenance, table units, new DB)")
    p.add_argument("--corpus", required=True, help="corpus root, e.g. ~/Documents/Projects/textbook-md-v2")
    p.add_argument("--db", help="output LanceDB directory (never the live one)")
    p.add_argument("--book", help="one book folder (or guideline folder); always re-embedded")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--books-meta", default=str(Path("~/.vault-search/books.json").expanduser()),
                   help="books.json for edition/tier/status (excluded statuses are not indexed)")
    p.add_argument("--include-superseded", action="store_true")
    p.add_argument("--keep-references", action="store_true", help="index References sections too")
    p.add_argument("--rebuild", action="store_true", help="drop this model's tables first")
    p.add_argument("--dry-run", action="store_true", help="chunk and report, no embedding, no DB")
    p.add_argument("--fast-tokens", action="store_true", help="dry run with a whitespace tokenizer")
    p.add_argument("--audit", action="store_true", help="per-chapter coverage check against --db")
    p.add_argument("--holding-report", metavar="PATH", help="write suggested holding list and exit")
    p.add_argument("--report", metavar="PATH", help="JSON report for --dry-run / --audit")
    p.add_argument("--no-gpu-lease", action="store_true")
    args = p.parse_args(argv)

    if args.holding_report:
        return holding_report(args)
    if args.dry_run:
        return run(args)
    if not args.db:
        p.error("--db is required (a NEW directory; the live database is refused)")
    check_target(Path(args.db), Path(args.corpus), live_paths())
    if args.audit:
        return audit(args)

    global _LEASE
    if not args.no_gpu_lease:
        from config import (GPU_LEASE_PATH, GPU_LEASE_NAME_TEXTBOOK, GPU_LEASE_ACQUIRE_TIMEOUT,
                            GPU_LEASE_MIN_HOLD, GPU_LEASE_QUEUE_DIR)
        from gpu_lease_client import GpuLease, acquire_or_exit
        _LEASE = GpuLease(GPU_LEASE_NAME_TEXTBOOK + "_v3", script=GPU_LEASE_PATH,
                          queue_dir=GPU_LEASE_QUEUE_DIR, acquire_timeout=GPU_LEASE_ACQUIRE_TIMEOUT,
                          min_hold_s=GPU_LEASE_MIN_HOLD)
        acquire_or_exit(_LEASE)
    try:
        return run(args)
    finally:
        if _LEASE is not None and _LEASE.held:
            ok, out = _LEASE.release()
            print(f"[gpu-lease] release: {'ok' if ok else 'FAILED'} {out.strip()[-120:]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
