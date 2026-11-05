"""
Vault search scopes — which top-level folders a query may see.

A scope map is a JSON file naming the folders that belong to each named scope:

    {"medical": ["Clinical", "Lectures", "Inbox"],
     "work":    ["Projects"]}

Rules
-----
* **Fail-closed.** A folder belongs to a scope only if the map lists it. A new
  top-level folder, a typo, or a note at the vault root therefore stays OUT of
  every named scope until someone adds it — it can never leak into a scoped
  search by default.
* The implicit scope ``personal`` is everything NOT listed under any other
  scope, unless the map defines ``personal`` itself. That is what "unlisted
  folders count as personal" means.
* ``all`` disables the filter.
* The DEFAULT scope is chosen by the server (``VAULT_SEARCH_DEFAULT_SCOPE``),
  never by a prompt. With no map configured the default is ``all`` and nothing
  changes for an install that does not use scopes.

The filter is applied as a LanceDB prefilter on the ``folder`` column, so it
needs no schema change and no reindex.
"""

from __future__ import annotations

import json
from pathlib import Path

ALL = "all"
PERSONAL = "personal"


class ScopeError(ValueError):
    """Unknown scope name or unreadable map — refuse rather than search wide."""


def load_scope_map(path: Path | None) -> dict[str, list[str]]:
    """Read the map. Missing path → {} (scopes disabled). Broken file → ScopeError.

    A map that exists but cannot be parsed must not silently become "no map":
    that would turn every scoped search into an unscoped one.
    """
    if path is None or not Path(path).exists():
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ScopeError(f"scope map {path} is unreadable: {e}") from e
    if not isinstance(raw, dict):
        raise ScopeError(f"scope map {path} must be a JSON object")
    out: dict[str, list[str]] = {}
    for name, folders in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(folders, list) or not all(isinstance(f, str) for f in folders):
            raise ScopeError(f"scope '{name}' must be a list of folder names")
        out[name.strip().lower()] = [f.strip() for f in folders if f.strip()]
    return out


def load_scope_config(path: Path | None, env_default: str = "") -> tuple[dict[str, list[str]], str]:
    """(map, default scope). Default precedence: env value, map "_default", "all"."""
    smap = load_scope_map(path)
    default = (env_default or "").strip().lower()
    if not default and path is not None and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        default = str(raw.get("_default", "") or "").strip().lower()
    return smap, default or ALL


def scope_names(smap: dict[str, list[str]]) -> list[str]:
    names = set(smap) | {ALL}
    if smap:
        names.add(PERSONAL)
    return sorted(names)


def resolve(scope: str | None, smap: dict[str, list[str]], default: str) -> str:
    """Normalise a requested scope; None/"" → the server default."""
    s = (scope or default or ALL).strip().lower()
    if s == ALL:
        return ALL
    if not smap:
        # Scopes are not configured. A caller asking for one by name gets an
        # error, not silently everything.
        raise ScopeError(f"scope '{s}' requested but no scope map is configured")
    if s not in smap and s != PERSONAL:
        raise ScopeError(f"unknown scope '{s}' (known: {', '.join(scope_names(smap))})")
    return s


def _listed_elsewhere(smap: dict[str, list[str]], exclude: str | None = None) -> list[str]:
    seen: list[str] = []
    for name, folders in smap.items():
        if name == exclude:
            continue
        for f in folders:
            if f not in seen:
                seen.append(f)
    return seen


def folder_in_scope(folder: str | None, scope: str, smap: dict[str, list[str]]) -> bool:
    if scope == ALL:
        return True
    folder = folder or ""
    if scope == PERSONAL and PERSONAL not in smap:
        return folder not in _listed_elsewhere(smap)
    return folder in smap.get(scope, [])


def _sql_list(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def where_clause(scope: str, smap: dict[str, list[str]]) -> str | None:
    """SQL prefilter on `folder`, or None for no filter.

    An empty folder list yields a clause that matches nothing — a scope with
    no folders is an empty scope, not an unscoped search.
    """
    if scope == ALL:
        return None
    if scope == PERSONAL and PERSONAL not in smap:
        others = _listed_elsewhere(smap)
        return f"folder NOT IN ({_sql_list(others)})" if others else None
    folders = smap.get(scope, [])
    if not folders:
        return "folder = '\x00no-folder\x00'"
    return f"folder IN ({_sql_list(folders)})"


def combine(*clauses: str | None) -> str | None:
    parts = [f"({c})" for c in clauses if c]
    return " AND ".join(parts) if parts else None
