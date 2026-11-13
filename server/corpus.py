"""
Corpus folders (format ``t2n-corpus/1``): validate one, then make it searchable.

    python corpus.py check <root> [--json] [--strict]
    python corpus.py add   <root> [--db PATH] [--dry-run] [--books-meta PATH]

The format is specified in ``docs/corpus-format.md``. Producers (textbook-to-note,
lecture-to-notes) write a corpus folder; this module is the one place that turns
it into an index the server reads.

``check`` reads files only. ``add`` runs ``check``, then the v3 indexer
(``textbook_indexer_v3.py --corpus <root> --db <db>``; re-runs re-embed only
changed files), then its per-chapter ``--audit``, and finally prints the setting
the server needs to read that database. The indexer's own live-database guard
(``check_target``) applies unchanged; ``add`` adds one more refusal: a database
already built from a DIFFERENT corpus root is never reused, because the
indexer would delete that corpus's rows as orphans.

Exit codes: 0 ok, 1 check errors (or audit gaps), 2 refused / misconfigured.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import provenance as P  # noqa: E402  (pure: no lancedb / ollama / config)
import lecture_meta  # noqa: E402

FORMAT = "t2n-corpus/1"
MARKER = "corpus.json"
KINDS = ("textbook", "lecture")
SKIP_BASENAMES = {"index.md"}          # same as textbook_indexer_v3.SKIP_BASENAMES
FULL_TEXT = "full_text.md"
HEAD_BYTES = 65536                      # page markers / Source lines sit near the top
DEFAULT_DB_NAME = {"textbook": "textbook_db", "lecture": "lecture_db_v3"}
ENV_FOR_KIND = {"textbook": "VAULT_SEARCH_TEXTBOOK_DB_PATH", "lecture": "VAULT_SEARCH_LECTURE_DB_PATH"}
_H1 = re.compile(r"^#\s+\S")
_H2_OR_DEEPER = re.compile(r"^#{2,6}\s")


# =============================================================================
# check
# =============================================================================

@dataclass
class Report:
    root: str
    kind: str = ""
    kind_source: str = ""          # "corpus.json" | "inferred"
    name: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, code: str, path, msg: str) -> None:
        self.errors.append({"code": code, "path": str(path), "message": msg})

    def warn(self, code: str, path, msg: str) -> None:
        self.warnings.append({"code": code, "path": str(path), "message": msg})

    def as_dict(self) -> dict:
        return {"root": self.root, "format": FORMAT, "kind": self.kind,
                "kind_source": self.kind_source, "name": self.name,
                "ok": not self.errors, "errors": self.errors, "warnings": self.warnings,
                "stats": self.stats}


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p)


def _head(p: Path, n: int = HEAD_BYTES) -> str:
    with open(p, "rb") as fh:
        return fh.read(n).decode("utf-8", errors="replace")


def units(root: Path):
    """(book, unit_dir) pairs exactly as textbook_indexer_v3.iter_units walks them."""
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name == P.GUIDELINE_BOOK:
            for g in sorted(p for p in d.iterdir() if p.is_dir()):
                yield d.name, g
            continue
        if d.name.startswith((".", "_")):
            continue
        yield d.name, d


def unit_files(unit_dir: Path) -> list[Path]:
    """Markdown files the indexer considers in a unit (before the full_text /
    chapters choice): skips files under '.'/'_' sub-folders and index.md."""
    out = []
    for f in sorted(unit_dir.rglob("*.md")):
        parts = f.relative_to(unit_dir).parts
        if any(p.startswith((".", "_")) for p in parts[:-1]) or "__pycache__" in parts:
            continue
        if f.name.casefold() in SKIP_BASENAMES:
            continue
        out.append(f)
    return out


def read_marker(root: Path, rep: Report) -> str:
    """Kind declared by corpus.json, or '' (missing / broken: errors recorded)."""
    m = root / MARKER
    if not m.exists():
        rep.warn("no_marker", MARKER, f"no {MARKER}; kind is inferred. Add "
                 f'{{"format": "{FORMAT}", "kind": "textbook"|"lecture"}} to the corpus root.')
        return ""
    try:
        raw = json.loads(m.read_text(encoding="utf-8-sig"))
    except Exception as e:
        rep.error("marker_json", MARKER, f"{MARKER} is not valid JSON: {e}")
        return ""
    if not isinstance(raw, dict):
        rep.error("marker_json", MARKER, f"{MARKER} must be a JSON object")
        return ""
    fmt = raw.get("format")
    if fmt != FORMAT:
        rep.error("marker_format", MARKER, f'"format" is {fmt!r}; this vault-search reads "{FORMAT}"')
    kind = raw.get("kind")
    if kind not in KINDS:
        rep.error("marker_kind", MARKER, f'"kind" is {kind!r}; expected one of {list(KINDS)}')
        kind = ""
    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        rep.error("marker_name", MARKER, '"name" must be a string when present')
    elif name:
        rep.name = name
    return kind or ""


def infer_kind(root: Path) -> str:
    """lecture when any course-level folder holds a _meta.md, else textbook."""
    for _, u in units(root):
        if (u / lecture_meta.META_FILENAME).is_file():
            return "lecture"
    return "textbook"


def check(root, books_meta: Path | None = None) -> Report:
    root = Path(root).expanduser()
    rep = Report(root=str(root))
    if not root.is_dir():
        rep.error("no_root", root, "corpus root is not a directory")
        return rep
    root = root.resolve()
    rep.root = str(root)
    kind = read_marker(root, rep)
    if kind:
        rep.kind, rep.kind_source = kind, MARKER
    else:
        rep.kind, rep.kind_source = infer_kind(root), "inferred"

    for f in sorted(root.glob("*.md")):
        rep.warn("root_md", f.name, "Markdown directly in the corpus root is not indexed; "
                 "put it in a source / course folder")
    gl = root / P.GUIDELINE_BOOK
    if gl.is_dir():
        for f in sorted(gl.glob("*.md")):
            if f.name.casefold() in SKIP_BASENAMES:
                continue
            rep.warn("guideline_loose_md", _rel(f, root),
                     f"files directly in {P.GUIDELINE_BOOK}/ are not indexed; "
                     f"each guideline needs its own sub-folder")

    if rep.kind == "lecture":
        _check_lecture(root, rep)
    else:
        _check_textbook(root, rep)
    if books_meta is not None and rep.kind == "textbook":
        _check_books_meta(root, Path(books_meta).expanduser(), rep)
    return rep


def _check_textbook(root: Path, rep: Report) -> None:
    n_sources = n_guidelines = n_files = n_manifests = 0
    lecture_like = 0
    # The shared guideline manifest textbook-to-note --resplit writes
    # (entries "<Name>/<file>"); used for guideline folders without their own.
    gl = root / P.GUIDELINE_BOOK
    if (gl / P.MANIFEST_NAME).exists():
        n_manifests += 1
        _check_manifest(gl, root, rep)
    for book, u in units(root):
        rel = _rel(u, root)
        is_gl = book == P.GUIDELINE_BOOK
        n_guidelines += is_gl
        n_sources += not is_gl
        files = unit_files(u)
        if not files:
            rep.warn("empty_source", rel, "no .md files to index")
            continue
        n_files += len(files)
        if (u / lecture_meta.META_FILENAME).is_file():
            lecture_like += 1
        _check_full_text_choice(u, files, root, rep)
        # Manifests: any _manifest.json from a file's folder up to the unit root.
        mdirs = {d for d in [u, *{f.parent for f in files}] if (d / P.MANIFEST_NAME).exists()}
        for d in sorted(mdirs):
            n_manifests += 1
            _check_manifest(d, root, rep)
        if not any(P.PAGE_MARKER_RE.search(_head(f)) for f in files):
            rep.warn("no_page_markers", rel, "no <!-- page N --> markers: hits from this "
                     "source carry no page numbers")
    if n_sources + n_guidelines == 0:
        rep.error("no_sources", ".", "no source folders (textbook kind: <root>/<Source>/*.md "
                  f"or <root>/{P.GUIDELINE_BOOK}/<Name>/*.md)")
    elif n_files == 0:
        rep.error("no_files", ".", "source folders exist but none holds a .md file")
    if lecture_like and rep.kind_source == MARKER:
        rep.warn("kind_mismatch", MARKER, f'kind is "textbook" but {lecture_like} folder(s) '
                 f"hold a {lecture_meta.META_FILENAME} like a lecture course")
    rep.stats = {"sources": n_sources, "guidelines": n_guidelines, "files": n_files,
                 "manifests": n_manifests}


def _check_full_text_choice(u: Path, files: list[Path], root: Path, rep: Report) -> None:
    by_dir: dict = {}
    for f in files:
        by_dir.setdefault(f.parent, []).append(f)
    for d, fs in by_dir.items():
        full = [f for f in fs if f.name.casefold() == FULL_TEXT]
        chapters = [f for f in fs if f.name.casefold() != FULL_TEXT]
        if full and chapters:
            fsize = full[0].stat().st_size
            csize = sum(f.stat().st_size for f in chapters)
            if fsize and csize / fsize < 0.5:
                rep.warn("chapters_ignored", _rel(d, root),
                         f"chapter files hold {csize / fsize:.0%} of {FULL_TEXT}'s bytes (<50%): "
                         f"the indexer uses {FULL_TEXT} and skips the chapters")


def _check_manifest(d: Path, root: Path, rep: Report) -> None:
    mp = d / P.MANIFEST_NAME
    rel = _rel(mp, root)
    try:
        raw = json.loads(mp.read_text(encoding="utf-8-sig"))
    except Exception as e:
        rep.warn("manifest_unreadable", rel, f"not valid JSON ({e}); provenance falls back to "
                 "PDF pages from the page markers")
        return
    if not isinstance(raw, dict) or not isinstance(raw.get("files", []), list):
        rep.warn("manifest_unreadable", rel, "expected an object with a files[] list")
        return
    missing = [e.get("file") for e in raw.get("files") or []
               if isinstance(e, dict) and e.get("file")
               and not (d / str(e["file"]).replace("\\", "/")).exists()]
    if missing:
        rep.warn("manifest_stale", rel, f"{len(missing)} files[] entr{'y' if len(missing) == 1 else 'ies'} "
                 f"name files that do not exist (e.g. {missing[0]}); re-run the split")


def _check_lecture(root: Path, rep: Report) -> None:
    n_courses = n_entries = n_meta = 0
    for book, u in units(root):
        rel = _rel(u, root)
        if book == P.GUIDELINE_BOOK:
            rep.warn("lecture_guidelines", rel, f"{P.GUIDELINE_BOOK}/ in a lecture corpus is "
                     "indexed as a guideline, not a course")
            continue
        n_courses += 1
        if (u / lecture_meta.META_FILENAME).is_file():
            n_meta += 1
        else:
            rep.warn("no_meta", rel, f"course has no {lecture_meta.META_FILENAME}")
        entries = [f for f in unit_files(u) if not lecture_meta.is_meta_file(_rel(f, u))]
        if not entries:
            rep.warn("empty_course", rel, "course has no entry files")
        for f in entries:
            n_entries += 1
            _check_entry(f, root, rep)
    if n_courses == 0:
        rep.error("no_courses", ".", "no course folders (lecture kind: <root>/<course_dir>/*.md)")
    elif n_entries == 0:
        rep.error("no_entries", ".", "course folders exist but none holds an entry file")
    rep.stats = {"courses": n_courses, "entries": n_entries, "meta_files": n_meta}


def _check_entry(f: Path, root: Path, rep: Report) -> None:
    rel = _rel(f, root)
    lines = _head(f).splitlines()
    first = next((ln for ln in lines if ln.strip()), "")
    if not _H1.match(first):
        rep.warn("entry_no_h1", rel, "first line is not an H1 (# Title)")
    head = []
    for ln in lines:
        if _H2_OR_DEEPER.match(ln):
            break
        head.append(ln)
    src = lecture_meta.find_source_line("\n".join(head))
    if not src:
        rep.warn("entry_no_source", rel, "no `Source:` line before the first ## heading: "
                 "date / speaker / course stay empty in lecture_search")
        return
    parsed = lecture_meta.parse_source(src)
    if not parsed["date"]:
        rep.warn("source_no_date", rel, "Source line has no YYYY-MM-DD date")
    if not parsed["course"]:
        rep.warn("source_no_course", rel, "Source line has no 《course title》")


def _check_books_meta(root: Path, path: Path, rep: Report) -> None:
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        rep.error("books_meta_json", path, f"books.json is not valid JSON ({e}); the indexer would stop")
        return
    if not isinstance(raw, dict):
        rep.error("books_meta_json", path, "books.json must be a JSON object")
        return
    keys = {u.name for _, u in units(root)}
    for k, v in raw.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        if k not in keys:
            rep.warn("books_meta_unknown", path, f"key {k!r} matches no source folder in this corpus "
                     "(typo, or an entry for another corpus)")


def print_report(rep: Report, limit: int = 10) -> None:
    src = f"{rep.kind} ({rep.kind_source})" if rep.kind else "?"
    print(f"[check] {rep.root}")
    print(f"  kind: {src}" + (f"   name: {rep.name}" if rep.name else ""))
    if rep.stats:
        print("  " + ", ".join(f"{k}={v}" for k, v in rep.stats.items()))
    for label, items in (("ERROR", rep.errors), ("warn ", rep.warnings)):
        by_code: dict = {}
        for it in items:
            by_code.setdefault(it["code"], []).append(it)
        for code, its in by_code.items():
            for it in its[:limit]:
                print(f"  {label} {code}: {it['path']} — {it['message']}")
            if len(its) > limit:
                print(f"  {label} {code}: … {len(its) - limit} more (--json lists all)")
    print(f"  → {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")


# =============================================================================
# add
# =============================================================================

def load_settings() -> dict:
    """The server's configured paths. Falls back to the documented defaults when
    config cannot load (e.g. VAULT_PATH unset) so --dry-run still works."""
    try:
        import config as cfg
        return {"ok": True, "data_dir": Path(cfg.DATA_DIR),
                "textbook_db": Path(cfg.TEXTBOOK_DB_PATH),
                "textbook_db_set": bool(os.environ.get("VAULT_SEARCH_TEXTBOOK_DB_PATH")),
                "lecture_db": Path(cfg.LECTURE_DB_PATH),
                "lecture_db_set": bool(os.environ.get("VAULT_SEARCH_LECTURE_DB_PATH")),
                "books_meta": Path(cfg.BOOKS_META_PATH),
                "env_file": HERE.parent / ".env", "error": ""}
    except Exception as e:  # RuntimeError for VAULT_PATH, ImportError, ...
        dd = Path(os.environ.get("VAULT_SEARCH_DATA_DIR") or "~/.vault-search").expanduser()
        return {"ok": False, "data_dir": dd, "textbook_db": None, "textbook_db_set": False,
                "lecture_db": None, "lecture_db_set": False, "books_meta": dd / "books.json",
                "env_file": HERE.parent / ".env", "error": str(e).splitlines()[0]}


def default_db(kind: str, s: dict) -> Path:
    """The database the server is already configured to read for this kind when
    that is a v3 database (so a re-run adds to it), else a new one under the
    data directory: textbook → textbook_db, lecture → lecture_db_v3."""
    from textbook_indexer_v3 import is_v3_db
    configured = s.get(f"{kind}_db")
    if s.get(f"{kind}_db_set") and configured is not None and is_v3_db(configured):
        return Path(configured)
    return Path(s["data_dir"]) / DEFAULT_DB_NAME[kind]


def built_from(db: Path) -> str:
    """Corpus root recorded by a previous v3 run into ``db`` ('' if none)."""
    st = Path(db) / "_v3_state"
    if not st.is_dir():
        return ""
    for meta in [st / "index_meta.json", *sorted(st.glob("index_meta__*.json"))]:
        try:
            c = json.loads(meta.read_text(encoding="utf-8")).get("corpus")
        except Exception:
            continue
        if c:
            return str(c)
    return ""


def _same_path(a, b) -> bool:
    return os.path.normcase(str(Path(a).expanduser().resolve())) == \
        os.path.normcase(str(Path(b).expanduser().resolve()))


def indexer_argv(root: Path, db: Path, books_meta: str | None, no_gpu_lease: bool) -> list[str]:
    argv = ["--corpus", str(root), "--db", str(db)]
    if books_meta:
        argv += ["--books-meta", str(books_meta)]
    if no_gpu_lease:
        argv.append("--no-gpu-lease")
    return argv


def _show(argv: list[str]) -> str:
    q = [f'"{a}"' if (" " in a or not a) else a for a in argv]
    return f"python {HERE / 'textbook_indexer_v3.py'} " + " ".join(q)


def env_lines(kind: str, db: Path, books_meta: str | None, s: dict) -> tuple[list[str], bool]:
    """(lines to put in .env, whether every one of them is already in effect)."""
    var = ENV_FOR_KIND[kind]
    lines = [f"{var}={db}"]
    current = s.get(f"{kind}_db") if s.get(f"{kind}_db_set") else None
    in_effect = current is not None and _same_path(current, db)
    if books_meta and not (s.get("books_meta") and _same_path(s["books_meta"], books_meta)):
        lines.append(f"VAULT_SEARCH_BOOKS_META={Path(books_meta).expanduser()}")
        in_effect = False
    return lines, in_effect


def print_env(kind: str, db: Path, books_meta: str | None, s: dict) -> None:
    lines, in_effect = env_lines(kind, db, books_meta, s)
    print("\n[env] the server reads this corpus through:")
    for ln in lines:
        print(f"  {ln}")
    if in_effect:
        print("  Already set. A server running with this setting opens the database on every "
              "search, so the new rows are searchable now — no restart.")
        return
    print(f"  Put the line(s) above in {s['env_file']} (or export them), then restart "
          "api_server.py, and start a new session of any MCP client that runs mcp_server.py.")
    tool = "lecture_search" if kind == "lecture" else "textbook_search"
    print(f"  `{tool}` reads this database after the restart.")


def add(args) -> int:
    root = Path(args.root).expanduser()
    s = load_settings()
    # The server's books.json, so index-time exclusions match what search shows.
    books_meta = args.books_meta or str(s["books_meta"])
    rep = check(root, Path(books_meta).expanduser())
    print_report(rep)
    if rep.errors or (args.strict and rep.warnings):
        print("[add] stopped: fix the errors above first.")
        return 1
    root = Path(rep.root)
    kind = rep.kind
    db = Path(args.db).expanduser() if args.db else default_db(kind, s)
    db = db.resolve()

    import textbook_indexer_v3 as V
    try:
        V.check_target(db, root, V.live_paths())
    except SystemExit as e:
        print(str(e.code if e.code is not None else e))
        configured = s.get(f"{kind}_db") if s.get(f"{kind}_db_set") else None
        if configured is not None and _same_path(configured, db) and not V.is_v3_db(db):
            print(f"  {ENV_FOR_KIND[kind]} already points here, but no v3 index exists there yet. "
                  f"Build first, then set it: remove the line for this run (or pass another --db).")
        return 2
    prev = built_from(db)
    if prev and not _same_path(prev, root):
        print(f"[refuse] {db} was built from another corpus root:\n  {prev}\n"
              f"One database holds one corpus root (a run deletes rows of files outside it). "
              f"Put the new source folder under that root, or pass --db for a new database.")
        return 2

    argv = indexer_argv(root, db, books_meta, args.no_gpu_lease)
    print(f"\n[add] kind={kind}  db={db}" + ("  (new database)" if not V.is_v3_db(db) else
                                             "  (existing v3 database: changed files only)"))
    print(f"  index: {_show(argv)}")
    print(f"  audit: {_show(argv + ['--audit'])}")
    if args.dry_run:
        print("[add] --dry-run: nothing indexed.")
        print_env(kind, db, books_meta, s)
        return 0
    if not s["ok"]:
        print(f"[add] the server configuration did not load: {s['error']}\n"
              f"  Set VAULT_PATH (see env.example) and run again.")
        return 2
    rc = V.main(argv)
    if rc:
        print(f"[add] indexer exited {rc}; nothing more done.")
        return rc
    rc_audit = V.main(argv + ["--audit"])
    print_env(kind, db, books_meta, s)
    if rc_audit:
        print("[add] audit found chapters missing from the index (listed above). Run the same "
              "command again to retry them.")
        return 1
    return 0


# =============================================================================
# CLI
# =============================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=f"Validate a {FORMAT} corpus folder and index it "
                                            "(spec: docs/corpus-format.md)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate the folder layout (reads files only)")
    c.add_argument("root")
    c.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    c.add_argument("--strict", action="store_true", help="warnings also fail")
    c.add_argument("--books-meta", help="also check this books.json against the source folders")
    a = sub.add_parser("add", help="check, index (v3), audit, print the .env line")
    a.add_argument("root")
    a.add_argument("--db", help="target database (default: the configured one if it is v3, "
                                "else ~/.vault-search/textbook_db | lecture_db_v3)")
    a.add_argument("--dry-run", action="store_true", help="check and print the commands; index nothing")
    a.add_argument("--books-meta", help="books.json passed to the indexer (default: the server's)")
    a.add_argument("--strict", action="store_true", help="refuse when check has warnings")
    a.add_argument("--no-gpu-lease", action="store_true", help="passed through to the indexer")
    args = p.parse_args(argv)

    if args.cmd == "check":
        rep = check(args.root, args.books_meta)
        if args.json:
            print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=1))
        else:
            print_report(rep)
        return 1 if rep.errors or (args.strict and rep.warnings) else 0
    return add(args)


if __name__ == "__main__":
    sys.exit(main())
