"""vault_indexer_v2: error-log entries get closed once their file is indexed.

Run directly (`python server/tests/test_vault_error_reconcile.py`) or under pytest.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp(prefix="vsearch_recon_")
os.environ.setdefault("VAULT_PATH", _tmp)
os.environ.setdefault("VAULT_SEARCH_DATA_DIR", _tmp)

try:
    import vault_indexer_v2 as V
except ImportError:  # lancedb / ollama not installed
    V = None


def _log():
    return {
        "a.md:p1:0": {"file": "a.md", "status": "pending"},
        "a.md:p2:1": {"file": "a.md", "status": "pending"},
        "b.md:p3:0": {"file": "b.md", "status": "pending"},
        "c.md:p4:0": {"file": "c.md", "status": "pending"},
        "d.md:p5:0": {"file": "d.md", "status": "fixed"},
    }


if V is not None:
    def test_resolve_file_errors_is_file_level():
        log = _log()
        assert V.resolve_file_errors(log, "a.md") == 2
        assert [e["status"] for e in log.values()] == ["fixed", "fixed", "pending", "pending", "fixed"]
        assert V.resolve_file_errors(log, "a.md") == 0  # idempotent

    def test_reconcile():
        vault = Path(tempfile.mkdtemp(prefix="vsearch_recon_vault_"))
        (vault / "a.md").write_text("indexed and unchanged", encoding="utf-8")
        (vault / "b.md").write_text("changed since it was hashed", encoding="utf-8")
        # c.md was deleted
        cache = {"a.md": V.file_hash("indexed and unchanged"), "b.md": "stale-hash"}
        log = _log()
        res = V.reconcile_error_log(log, cache, {"a.md", "b.md"}, vault)
        assert res == {"fixed": 2, "obsolete": 1, "open": 1}
        assert log["b.md:p3:0"]["status"] == "pending"     # still really needs a run
        assert log["c.md:p4:0"]["status"] == "obsolete"

    def test_hashed_but_not_in_index_stays_open():
        vault = Path(tempfile.mkdtemp(prefix="vsearch_recon_vault_"))
        (vault / "a.md").write_text("x", encoding="utf-8")
        log = {"a.md:p:0": {"file": "a.md", "status": "pending"}}
        res = V.reconcile_error_log(log, {"a.md": V.file_hash("x")}, set(), vault)
        assert res["open"] == 1 and log["a.md:p:0"]["status"] == "pending"


if __name__ == "__main__":
    if V is None:
        print("SKIP: lancedb not installed")
    else:
        test_resolve_file_errors_is_file_level()
        test_reconcile()
        test_hashed_but_not_in_index_stays_open()
        print("OK: 3 tests passed")
