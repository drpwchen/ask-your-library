"""
vault-search eval — adapters that turn one query into ranked result lists.

Every adapter returns

    {"results": [result, ...], "searched": ["vault", "textbook", ...],
     "latency_ms": {"vault": 812, ...}, "error": None | str}

where each result follows the shape documented in metrics.py.

Systems
-------
mcp       The tool path an agent uses (vault_search / textbook_search /
          lecture_search when the server has it). In-process by default;
          `--base-url` sends the same calls through /api/mcp/call instead,
          which tests the running process exactly as an MCP client sees it.
chat      The retrieval context the plugin chat builds (_search_vault_context),
          parsed back into result lists. In-process only: /api/chat would also
          run the language model, which is the e2e step's job, not retrieval's.
ripgrep   Pooling only — literal term matches over the corpus folders, so the
          judgment pool is not limited to what an embedding model finds.

In-process mode imports the server modules from `server_dir`. Their config is
read from the environment / the `.env` next to that server dir, exactly like
the server itself. Search logs are redirected away from the live logs first
(see `prepare_env`), so an eval run never pollutes the observability data.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- environment ---------------------------------------------------------------

def prepare_env(server_dir: Path, scratch: Path, env_file: Path | None = None) -> None:
    """Point the search logs at `scratch` and make `server_dir` importable.

    Must run before the first server import: config.py resolves paths at import
    time. `env_file` is for a checkout that has no `.env` of its own (e.g. the
    dev repo evaluated against the live configuration).
    """
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["VAULT_SEARCH_LOG_PATH"] = str(scratch / "eval_textbook_log.jsonl")
    os.environ["VAULT_SEARCH_VAULT_LOG_PATH"] = str(scratch / "eval_vault_log.jsonl")
    os.environ.setdefault("VAULT_SEARCH_LECTURE_LOG_PATH", str(scratch / "eval_lecture_log.jsonl"))
    os.environ["VAULT_SEARCH_EVAL"] = "1"
    if env_file is not None:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(env_file, override=False)
        except ImportError:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    sd = str(server_dir.resolve())
    if sd not in sys.path:
        sys.path.insert(0, sd)


# --- markdown parsers (the MCP tools answer in compact Markdown) ---------------

_VAULT_LINE = re.compile(r"^(\d+)\. \[\[(.+?)\]\](?: · (.*))?$")
_TB_LINE = re.compile(r"^(\d+)\. \*\*(.+?)\*\*(?: · (.*))?$")
_SIM = re.compile(r"^sim ([0-9.]+|\?)$")
_PAGES = re.compile(r"^p\.(\d+)(?:-(\d+))?$")
_PDF_PAGES = re.compile(r"^PDF p\.(\d+)(?:-(\d+))?$")
_PRINTED = re.compile(r"^p\.\S+$")


def parse_vault_markdown(md: str, corpus: str = "vault") -> list[dict]:
    out: list[dict] = []
    for line in md.splitlines():
        m = _VAULT_LINE.match(line)
        if m:
            meta = [p for p in (m.group(3) or "").split(" · ") if p]
            sim = None
            if meta and _SIM.match(meta[-1]):
                s = _SIM.match(meta.pop()).group(1)
                sim = float(s) if s != "?" else None
            folder = meta.pop() if meta else ""
            out.append({"corpus": corpus, "rank": int(m.group(1)), "note": m.group(2),
                        "folder": folder, "section": " · ".join(meta), "sim": sim, "text": ""})
            continue
        if not out:
            continue
        if line.startswith(("graph-linked:", "entities:", "relations:")):
            break
        # The excerpt's first line is indented, its continuation lines are not
        # (the note text is printed as-is), so take everything up to the next item.
        if not line.startswith("   tags:"):
            out[-1]["text"] += (line[3:] if line.startswith("   ") else line) + "\n"
    for r in out:
        r["text"] = r["text"].strip()
    return out


def parse_textbook_markdown(md: str, corpus: str = "textbook") -> list[dict]:
    lines = md.splitlines()
    out: list[dict] = []
    i = 0
    while i < len(lines):
        m = _TB_LINE.match(lines[i])
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if m and nxt.startswith("   (") and nxt.rstrip().endswith(")"):
            meta = [p for p in (m.group(3) or "").split(" · ") if p]
            sim, ps, pe = None, 0, 0
            if meta and _SIM.match(meta[-1]):
                s = _SIM.match(meta.pop()).group(1)
                sim = float(s) if s != "?" else None
            printed = ""
            if meta and _PDF_PAGES.match(meta[-1]):
                # v2.11+: "PDF p.N" is the physical page; an optional "p.x"
                # before it is the printed page label.
                pm = _PDF_PAGES.match(meta.pop())
                ps = int(pm.group(1))
                pe = int(pm.group(2) or ps)
                if meta and _PRINTED.match(meta[-1]):
                    printed = meta.pop()[2:]
            elif meta and _PAGES.match(meta[-1]):
                pm = _PAGES.match(meta.pop())
                ps = int(pm.group(1))
                pe = int(pm.group(2) or ps)
            item = {"corpus": corpus, "rank": int(m.group(1)), "book": m.group(2),
                    "file": nxt.strip()[1:-1], "section": " · ".join(meta),
                    "page_start": ps, "page_end": pe, "sim": sim, "text": ""}
            if printed:
                item["printed_pages"] = printed
            out.append(item)
            i += 2
            continue
        if out:
            out[-1]["text"] += lines[i] + "\n"
        i += 1
    for r in out:
        r["text"] = r["text"].strip()
    return out


_CTX_VAULT = re.compile(r"^【(?!📖)(.+?) › (.*?)】\(sim=([0-9.]+)\)$")
_CTX_TB = re.compile(r"^【📖 (.+?) › (.+?) › (.*?)】\(sim=([0-9.]+)\)$")
_CTX_LEC = re.compile(r"^【🎤 (.+?) › (.+?) › (.*?)】\(sim=([0-9.]+)\)$")
_CTX_PAGE = re.compile(r"^\[p\.(\d+)(?:-(\d+))?\]")


def parse_chat_context(ctx: str) -> list[dict]:
    """Parse the context string the chat endpoint feeds the model.

    Header lines are the contract (see api_server._search_vault_context):
      【note › section】(sim=0.71)                vault
      【📖 book › file › section】(sim=0.71)      textbook
      【🎤 course › file › section】(sim=0.71)    lecture
    Textbook bodies may open with "[p.N]" page tags; those give the pages.
    """
    out: list[dict] = []
    ranks = {"vault": 0, "textbook": 0, "lecture": 0}
    cur = None
    for line in ctx.splitlines():
        mt, ml, mv = _CTX_TB.match(line), _CTX_LEC.match(line), _CTX_VAULT.match(line)
        if mt or ml:
            m = mt or ml
            corpus = "textbook" if mt else "lecture"
            ranks[corpus] += 1
            cur = {"corpus": corpus, "rank": ranks[corpus], "book": m.group(1),
                   "file": m.group(2), "section": m.group(3), "sim": float(m.group(4)),
                   "page_start": 0, "page_end": 0, "text": ""}
            out.append(cur)
            continue
        if mv:
            ranks["vault"] += 1
            cur = {"corpus": "vault", "rank": ranks["vault"], "note": mv.group(1),
                   "section": mv.group(2), "folder": "", "sim": float(mv.group(3)), "text": ""}
            out.append(cur)
            continue
        if cur is None or line.strip() == "---":
            continue
        pm = _CTX_PAGE.match(line)
        if pm and cur["corpus"] != "vault":
            p, q = int(pm.group(1)), int(pm.group(2) or pm.group(1))
            cur["page_start"] = p if not cur["page_start"] else min(cur["page_start"], p)
            cur["page_end"] = max(cur["page_end"], q)
        cur["text"] += line + "\n"
    return out


# --- systems -----------------------------------------------------------------

class McpSystem:
    """vault_search + textbook_search (+ lecture_search if the server has it)."""

    name = "mcp"

    def __init__(self, server_dir: Path | None = None, base_url: str | None = None,
                 api_key: str | None = None, n: int = 10):
        self.n = n
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        if self.base_url:
            tools = self._http_get("/api/mcp/tools")["tools"]
        else:
            import mcp_server  # noqa: F401  (imported from server_dir, see prepare_env)
            self._mcp = mcp_server
            tools = mcp_server.TOOLS
        self.tools = {t["name"] for t in tools}

    def _http_get(self, path):
        req = urllib.request.Request(self.base_url + path, headers=self._headers())
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def call(self, name: str, args: dict) -> str:
        if self.base_url:
            body = json.dumps({"name": name, "arguments": args}).encode("utf-8")
            req = urllib.request.Request(self.base_url + "/api/mcp/call", data=body,
                                         headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))["text"]
        return self._mcp.handle_tool_call(name, args)

    def search(self, query: str, corpora: tuple[str, ...] = ("vault", "textbook", "lecture"),
               extra: dict | None = None) -> dict:
        res, searched, lat, errors = [], [], {}, []
        plan = [("vault", "vault_search", {"include_excerpt": True}, parse_vault_markdown),
                ("textbook", "textbook_search", {}, parse_textbook_markdown),
                ("lecture", "lecture_search", {}, lambda md: parse_textbook_markdown(md, "lecture"))]
        for corpus, tool, args, parse in plan:
            if corpus not in corpora or tool not in self.tools:
                continue
            t0 = time.time()
            md = self.call(tool, {"query": query, "n_results": self.n, **args, **(extra or {})})
            lat[corpus] = int((time.time() - t0) * 1000)
            if md.lstrip().startswith('{"error"'):
                errors.append(f"{tool}: {md[:200]}")
                continue
            searched.append(corpus)
            res.extend(parse(md))
        return {"results": res, "searched": searched, "latency_ms": lat,
                "error": "; ".join(errors) or None}


class ChatSystem:
    """The context the plugin chat would have sent to the model for this query."""

    name = "chat"

    def __init__(self, server_dir: Path | None = None, n: int = 10):
        import api_server  # noqa: F401
        self._api = api_server
        self.n = n

    def search(self, query: str, corpora=("vault", "textbook", "lecture"), extra=None) -> dict:
        t0 = time.time()
        ctx = self._api._search_vault_context(query, self.n)
        ms = int((time.time() - t0) * 1000)
        res = parse_chat_context(ctx)
        searched = sorted({"vault", "textbook"} | {r["corpus"] for r in res})
        return {"results": res, "searched": searched, "latency_ms": {"all": ms},
                "error": None, "context_chars": len(ctx)}


class RipgrepSystem:
    """Literal-term matches over corpus folders — a pooling source, not a contender.

    Ranks files by how many distinct query terms they contain; for files with
    `<!-- page N -->` markers it reports the page of the densest match.
    """

    name = "ripgrep"
    _TERM = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}|\d+(?:\.\d+)?\s?(?:mg|mcg|μg|g|kg|ml|mL|%|IU|U|mmHg|bpm)\b")
    _STOP = {"with", "from", "that", "this", "what", "when", "which", "should", "does",
             "have", "into", "than", "then", "there", "their", "about", "after", "before"}

    def __init__(self, roots: dict[str, Path], n: int = 10, rg: str = "rg"):
        import shutil
        self.roots = {c: Path(p) for c, p in roots.items() if p}
        self.n = n
        # Fail loudly: a missing binary used to make every query return zero
        # results with no error, which reads as "ripgrep found nothing".
        found = shutil.which(rg) or (rg if Path(rg).is_file() else None)
        if not found:
            raise FileNotFoundError(f"ripgrep binary not found: {rg!r} (pass --rg PATH)")
        self.rg = found

    def terms(self, query: str, given: list[str] | None = None) -> list[str]:
        if given:
            return given
        seen = []
        for t in self._TERM.findall(query):
            if t.casefold() not in self._STOP and t.casefold() not in [s.casefold() for s in seen]:
                seen.append(t)
        cjk = re.findall(r"[一-鿿]{2,6}", query)
        return (seen + cjk)[:8]

    def search(self, query: str, corpora=("vault", "textbook", "lecture"), extra=None) -> dict:
        terms = self.terms(query, (extra or {}).get("grep_terms"))
        res, searched = [], []
        if not terms:
            return {"results": [], "searched": [], "latency_ms": {}, "error": "no terms"}
        for corpus, root in self.roots.items():
            if corpus not in corpora:
                continue
            searched.append(corpus)
            score: dict[str, set] = {}
            for t in terms:
                try:
                    p = subprocess.run([self.rg, "-l", "-i", "-F", "--glob", "*.md", t, str(root)],
                                       capture_output=True, text=True, encoding="utf-8",
                                       timeout=120)
                except subprocess.TimeoutExpired:
                    continue
                for f in p.stdout.splitlines():
                    score.setdefault(f, set()).add(t)
            ranked = sorted(score, key=lambda f: -len(score[f]))[: self.n]
            for i, f in enumerate(ranked, 1):
                rel = Path(f).resolve().relative_to(root.resolve()).as_posix()
                item = {"corpus": corpus, "rank": i, "sim": len(score[f]) / len(terms), "text": ""}
                if corpus == "vault":
                    item.update(note=Path(f).stem, folder=rel.split("/")[0] if "/" in rel else "")
                else:
                    page = _densest_page(Path(f), list(score[f]))
                    item.update(book=rel.split("/")[0], file=rel, page_start=page, page_end=page)
                res.append(item)
        return {"results": res, "searched": searched, "latency_ms": {}, "error": None}


def _densest_page(path: Path, terms: list[str]) -> int:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if "<!-- page " not in txt:
        return 0
    best, best_n = 0, 0
    for m in re.finditer(r"<!-- page (\d+) -->(.*?)(?=<!-- page \d+ -->|\Z)", txt, re.S):
        body = m.group(2).casefold()
        n = sum(1 for t in terms if t.casefold() in body)
        if n > best_n:
            best, best_n = int(m.group(1)), n
    return best


def make_system(name: str, *, server_dir: Path | None, base_url: str | None,
                api_key: str | None, n: int, roots: dict[str, Path] | None = None,
                rg: str = "rg"):
    if name == "mcp":
        return McpSystem(server_dir, base_url=base_url, api_key=api_key, n=n)
    if name == "chat":
        return ChatSystem(server_dir, n=n)
    if name == "ripgrep":
        return RipgrepSystem(roots or {}, n=n, rg=rg)
    raise ValueError(f"unknown system: {name}")
