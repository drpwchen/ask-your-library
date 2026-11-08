"""Minimal LLM call for the eval: the `claude` CLI in print mode, JSON out.

Fixed model + fixed prompt per step, so two runs of the same step are
comparable. Tools are disabled: the model judges only the text it is given.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")
DEFAULT_MODEL = os.environ.get("VAULT_SEARCH_EVAL_MODEL", "opus")
NO_TOOLS = "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,NotebookEdit,Skill,ToolSearch,mcp__*"


def ask(prompt: str, system: str, model: str = DEFAULT_MODEL, timeout: int = 600) -> str:
    cmd = [CLAUDE_CMD, "-p", "--model", model, "--output-format", "json",
           "--disallowedTools", NO_TOOLS, "--system-prompt", system,
           # A judge only reads a prompt and returns JSON: no user hooks (Stop hooks pushed
           # ntfy per call; session-start hooks read secrets) and no MCP servers.
           "--settings", '{"disableAllHooks": true}', "--strict-mcp-config"]
    p = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=timeout)
    raw = p.stdout.decode("utf-8", errors="replace").strip()
    try:
        out = json.loads(raw).get("result") or ""
    except json.JSONDecodeError:
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", errors="replace")[-500:])
        out = raw
    check_quota(out)
    return out


class QuotaExhausted(RuntimeError):
    """The CLI answered with a usage-limit notice instead of a model reply."""


_QUOTA_RE = re.compile(r"hit your [\w-]+ limit|usage limit reached", re.I)


def check_quota(text: str) -> None:
    # A limit notice comes back as an ordinary result string; treating it as an answer
    # would grade it (and every later question) as a failure.
    if len(text) < 300 and _QUOTA_RE.search(text):
        raise QuotaExhausted(text.strip())


def ask_json(prompt: str, system: str, model: str = DEFAULT_MODEL, retries: int = 1):
    """ask() and parse the first JSON value in the reply."""
    last = ""
    for _ in range(retries + 1):
        last = ask(prompt, system, model)
        m = re.search(r"```(?:json)?\s*(.+?)```", last, re.S)
        body = m.group(1) if m else last
        start = min([i for i in (body.find("{"), body.find("[")) if i >= 0], default=-1)
        if start >= 0:
            try:
                return json.JSONDecoder().raw_decode(body[start:])[0]
            except json.JSONDecodeError:
                pass
        prompt += "\n\nReturn ONLY valid JSON, no prose."
    raise ValueError(f"no JSON in model reply: {last[:300]}")
