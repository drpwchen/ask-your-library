"""Vault search scopes: fail-closed folder filtering (server/scope.py + mcp_server).

What must hold:
  * a folder not listed in a scope is NOT in it (new folders, typos, vault root);
  * "personal" = everything not listed under another scope;
  * the server default applies when the caller names no scope;
  * an unknown scope or an unreadable map REFUSES — it never widens to "all";
  * the filter really reaches LanceDB (prefilter on `folder`), and graph-linked
    suggestions cannot reintroduce out-of-scope notes.

Run directly (`python server/tests/test_scope.py`) or under pytest.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_tmp = tempfile.mkdtemp(prefix="vsearch_scope_")
os.environ.setdefault("VAULT_PATH", _tmp)
os.environ.setdefault("VAULT_SEARCH_DATA_DIR", _tmp)

import scope as S  # noqa: E402

SMAP = {"medical": ["Clinical", "Lectures", "Inbox"], "work": ["Projects"]}


def test_listed_folders_only():
    assert S.folder_in_scope("Clinical", "medical", SMAP)
    assert not S.folder_in_scope("Journal", "medical", SMAP)       # unlisted
    assert not S.folder_in_scope("", "medical", SMAP)              # vault root
    assert not S.folder_in_scope("clinical", "medical", SMAP)      # exact names only


def test_personal_is_the_complement():
    assert S.folder_in_scope("Journal", "personal", SMAP)
    assert S.folder_in_scope("", "personal", SMAP)
    assert not S.folder_in_scope("Projects", "personal", SMAP)
    assert not S.folder_in_scope("Clinical", "personal", SMAP)


def test_where_clauses():
    assert S.where_clause("medical", SMAP) == "folder IN ('Clinical', 'Lectures', 'Inbox')"
    assert S.where_clause("personal", SMAP) == \
        "folder NOT IN ('Clinical', 'Lectures', 'Inbox', 'Projects')"
    assert S.where_clause("all", SMAP) is None
    # an empty scope matches nothing rather than everything
    w = S.where_clause("empty", {"empty": []})
    assert w and "IN" not in w
    assert S.where_clause("medical", {"medical": ["O'Brien"]}) == "folder IN ('O''Brien')"


def test_resolve_defaults_and_refusals():
    assert S.resolve(None, SMAP, "medical") == "medical"
    assert S.resolve("ALL", SMAP, "medical") == "all"
    for bad in ("medicl", "everything"):
        try:
            S.resolve(bad, SMAP, "medical")
        except S.ScopeError:
            pass
        else:
            raise AssertionError(f"unknown scope {bad!r} must refuse")
    # no map: naming a scope refuses, asking for nothing means "all"
    assert S.resolve(None, {}, "") == "all"
    try:
        S.resolve("medical", {}, "")
    except S.ScopeError:
        pass
    else:
        raise AssertionError("scope requested without a map must refuse")


def test_config_default_and_broken_map():
    d = Path(tempfile.mkdtemp(prefix="vsearch_scope_cfg_"))
    p = d / "scope_map.json"
    p.write_text(json.dumps({"_default": "medical", **SMAP}), encoding="utf-8")
    smap, default = S.load_scope_config(p)
    assert default == "medical" and smap == SMAP
    assert S.load_scope_config(p, "work")[1] == "work"     # env wins
    assert S.load_scope_config(d / "missing.json") == ({}, "all")
    p.write_text("{not json", encoding="utf-8")
    try:
        S.load_scope_config(p)
    except S.ScopeError:
        pass
    else:
        raise AssertionError("an unreadable map must refuse, not disable scoping")


# --- through mcp_server ------------------------------------------------------

try:
    import lancedb  # noqa: F401
    import mcp_server
except ImportError:
    mcp_server = None


if mcp_server is not None:
    import contextlib

    _MAP_PATH = Path(_tmp) / "scope_map_test.json"

    @contextlib.contextmanager
    def _scope_map(obj):
        """Point mcp_server at a test map for one test, then put it back —
        other test modules share this process and must see no map at all."""
        if isinstance(obj, str):
            _MAP_PATH.write_text(obj, encoding="utf-8")
        else:
            _MAP_PATH.write_text(json.dumps(obj), encoding="utf-8")
        saved = (mcp_server.SCOPE_MAP_PATH, mcp_server.DEFAULT_SCOPE_ENV, mcp_server.vault_v2_active)
        mcp_server.SCOPE_MAP_PATH, mcp_server.DEFAULT_SCOPE_ENV = _MAP_PATH, ""
        mcp_server.vault_v2_active = lambda: False
        try:
            yield
        finally:
            mcp_server.SCOPE_MAP_PATH, mcp_server.DEFAULT_SCOPE_ENV, mcp_server.vault_v2_active = saved

    def _rows(*pairs):
        return [{"file": f"{f}/{n}.md", "note": n, "section": "", "folder": f, "tags": "",
                 "similarity": 0.9 - i * 0.05, "mtime": None} for i, (f, n) in enumerate(pairs)]

    MIXED = _rows(("Clinical", "Stroke"), ("Journal", "Diary"), ("", "RootNote"),
                  ("Inbox", "Talk"), ("Projects", "Plan"))

    def _search(args, retrieve=None, graph=None):
        saved = (mcp_server._vault_retrieve_v1, mcp_server.get_graph)
        seen = {}

        def fake(query, folder, fetch_n, include_excerpt, scope_where=None):
            seen["where"] = scope_where
            return [dict(r) for r in (retrieve or MIXED)]

        mcp_server._vault_retrieve_v1 = fake
        mcp_server.get_graph = lambda: graph or {}
        try:
            return mcp_server._vault_search_raw({"query": "q", "n_results": 10, **args}), seen
        finally:
            mcp_server._vault_retrieve_v1, mcp_server.get_graph = saved

    def test_default_scope_applies_without_asking():
        with _scope_map({"_default": "medical", **SMAP}):
            out, seen = _search({})
        assert out["scope"] == "medical"
        assert seen["where"] == "folder IN ('Clinical', 'Lectures', 'Inbox')"
        assert [r["note"] for r in out["results"]] == ["Stroke", "Talk"]

    def test_personal_and_all():
        with _scope_map({"_default": "medical", **SMAP}):
            out, _ = _search({"scope": "personal"})
            assert sorted(r["note"] for r in out["results"]) == ["Diary", "RootNote"]
            out, seen = _search({"scope": "all"})
        assert seen["where"] is None and len(out["results"]) == 5

    def test_unknown_scope_refuses_through_the_tool():
        with _scope_map({"_default": "medical", **SMAP}):
            text = mcp_server.handle_tool_call("vault_search", {"query": "q", "scope": "medicl"})
        assert "unknown scope" in text

    def test_broken_map_refuses_instead_of_searching_everything():
        with _scope_map("{oops"):
            try:
                _search({})
            except mcp_server.ScopeRefused:
                pass
            else:
                raise AssertionError("broken scope map must refuse")

    def test_graph_links_cannot_leak_out_of_scope():
        graph = {"adjacency": {"Stroke": {"Diary": 1, "Rehab": 1}, "Diary": {"Stroke": 1},
                               "Rehab": {"Stroke": 1}},
                 "metadata": {"Diary": {"folder": "Journal"}, "Rehab": {"folder": "Clinical"}}}
        with _scope_map({"_default": "medical", **SMAP}):
            out, _ = _search({}, retrieve=_rows(("Clinical", "Stroke")), graph=graph)
        assert "Diary" not in out["graph_linked"]
        assert "Rehab" in out["graph_linked"]

    def test_prefilter_reaches_lancedb():
        """Real LanceDB table, stubbed embedder: the WHERE must filter rows."""
        import lancedb as ldb
        db = ldb.connect(str(Path(_tmp) / "lance_scope"))
        rows = [{"vector": [1.0, 0.0, 0.0, 0.0], "parent_id": f"p{i}", "file": f"{f}/{n}.md",
                 "note": n, "section_path": "", "folder": f, "tags": "", "mtime": 0.0, "text": n}
                for i, (f, n) in enumerate([("Clinical", "A"), ("Journal", "B"), ("", "C"),
                                            ("Inbox", "D")])]
        tbl = db.create_table("vault_chunks_v2", rows, mode="overwrite")

        class _Emb:
            def embed(self, model, input, options=None):
                return {"embeddings": [[1.0, 0.0, 0.0, 0.0]]}

        saved = (mcp_server.get_vault_chunks_table, mcp_server.get_ollama)
        mcp_server.get_vault_chunks_table = lambda: tbl
        mcp_server.get_ollama = lambda: _Emb()
        try:
            got = mcp_server._vault_retrieve_v2("q", None, 10, False,
                                                S.where_clause("medical", SMAP))
            assert sorted(r["note"] for r in got) == ["A", "D"]
            got = mcp_server._vault_retrieve_v2("q", None, 10, False,
                                                S.where_clause("personal", SMAP))
            assert sorted(r["note"] for r in got) == ["B", "C"]
        finally:
            mcp_server.get_vault_chunks_table, mcp_server.get_ollama = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} scope tests passed")
