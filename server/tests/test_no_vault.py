"""Corpus-only install: VAULT_PATH unset. The config loads, the corpus tools work,
the vault tools and endpoints say no vault is configured, and the vault indexers
stop with the full setup message.

Every check runs in a subprocess: config resolves VAULT_PATH at import time and
conftest.py sets it for this test session. VAULT_PATH="" (not just absent) keeps
a developer's repo .env from filling it back in (python-dotenv never overrides a
variable that is already in the environment)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1]


def run_py(code: str, tmp_path, *args, script=None):
    env = dict(os.environ)
    data = tmp_path / "data"
    env.update({"VAULT_PATH": "", "VAULT_SEARCH_DATA_DIR": str(data),
                "VAULT_SEARCH_DB_PATH": str(data / "lance_db"),
                "VAULT_SEARCH_TEXTBOOK_DB_PATH": str(data / "textbook_db"),
                "VAULT_SEARCH_LECTURE_DB_PATH": str(data / "lecture_db_v3"),
                "VAULT_SEARCH_GPU_LEASE": "", "PYTHONIOENCODING": "utf-8"})
    cmd = [sys.executable, str(SERVER / script), *args] if script else [sys.executable, "-c", code]
    return subprocess.run(cmd, cwd=SERVER, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)


def test_config_loads_without_vault_path(tmp_path):
    r = run_py("import config, json; print(json.dumps([config.VAULT_PATH is None, "
               "config.VAULT_PATH_MISSING]))", tmp_path)
    assert r.returncode == 0, r.stderr
    none, msg = json.loads(r.stdout)
    assert none is True and "setx VAULT_PATH" in msg and "export VAULT_PATH" in msg


@pytest.mark.parametrize("script,args", [("vault_indexer_v2.py", ["--incremental"]),
                                         ("indexer.py", []), ("graph_builder.py", [])])
def test_vault_indexers_stop_with_full_message(tmp_path, script, args):
    r = run_py("", tmp_path, *args, script=script)
    assert r.returncode != 0
    assert "VAULT_PATH is not set" in r.stderr
    assert "(Linux/macOS)" in r.stderr and "(Windows)" in r.stderr and ".env file" in r.stderr


def test_mcp_vault_tools_say_no_vault(tmp_path):
    pytest.importorskip("lancedb")
    code = """
import json, mcp_server as M
from config import NO_VAULT_MESSAGE
out = {n: M.handle_tool_call(n, a) for n, a in [
    ("vault_search", {"query": "x"}), ("vault_similar", {"note_name": "x"}),
    ("vault_related", {"note": "x"})]}
try:
    M._vault_search_raw({"query": "x"})
    raised = False
except M.NoVaultIndex as e:
    raised = str(e) == NO_VAULT_MESSAGE
out["textbook"] = M.handle_tool_call("textbook_search", {"query": "x"})
print(json.dumps({"vault": [v == NO_VAULT_MESSAGE for k, v in out.items() if k != "textbook"],
                  "raised": raised, "textbook": out["textbook"]}))
"""
    r = run_py(code, tmp_path)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["vault"] == [True, True, True] and res["raised"] is True
    # the corpus tool runs (no index here, so it says so) instead of crashing on VAULT_PATH
    assert "VAULT_PATH" not in res["textbook"]


def test_api_vault_endpoints_503(tmp_path):
    pytest.importorskip("fastapi")
    code = """
import json, api_server as A
rs = [A.search(query="x", n_results=5, folder=None), A.related(note="x", n_results=5),
      A.similar(note_name="x", n_results=5)]
print(json.dumps([[r.status_code, json.loads(r.body)["error"][:22]] for r in rs]))
"""
    r = run_py(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [[503, "No vault configured — "]] * 3


def test_corpus_settings_load_without_vault_path(tmp_path):
    r = run_py("import corpus, json; s = corpus.load_settings(); print(json.dumps([s['ok'], s['error']]))",
               tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [True, ""]
