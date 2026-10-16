"""Tests for the end-of-run Ollama unload in vault_indexer_v2.

The indexer embeds with keep_alive="2h" so the model survives idle gaps between
batches. Left alone, that also pins the embedding runner (~4 GB commit) for two
hours AFTER the run, which is what pushed a nightly host to 92% memory commit.
index_vault_v2 therefore sends one keep_alive=0 call when the run ends, on the
success path and the exception path alike, and a failing unload must never turn
a finished index into a failed one.

Plain stdlib — run directly (`python server/tests/test_embedder_unload.py`) or
under pytest. The Ollama client and the indexing body are replaced by fakes.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("VAULT_PATH", tempfile.mkdtemp(prefix="vsearch_test_vault_"))

import pytest  # noqa: E402

import vault_indexer_v2 as v2  # noqa: E402


class FakeClient:
    def __init__(self, fail_unload: bool = False):
        self.calls: list[dict] = []
        self.fail_unload = fail_unload

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_unload and kwargs.get("keep_alive") == 0:
            raise ConnectionError("ollama is gone")
        return {"embeddings": []}

    def unloads(self) -> list[dict]:
        return [c for c in self.calls if c.get("keep_alive") == 0]


def _install(monkeypatch, client, body):
    monkeypatch.setattr(v2.ollama, "Client", lambda host=None: client)
    monkeypatch.setattr(v2, "_index_vault_v2", body)


def test_unload_sent_after_a_normal_run(monkeypatch):
    client = FakeClient()
    seen = {}

    def body(c, incremental, limit):
        seen["args"] = (c, incremental, limit)
        # mid-run embedding keeps the long keep_alive
        c.embed(model=v2.EMBEDDING_MODEL, input=["x"], keep_alive=v2.EMBED_KEEP_ALIVE)

    _install(monkeypatch, client, body)
    v2.index_vault_v2(incremental=True, limit=5)

    assert seen["args"] == (client, True, 5)
    assert client.calls[0]["keep_alive"] == v2.EMBED_KEEP_ALIVE == "2h"
    unloads = client.unloads()
    assert len(unloads) == 1
    assert unloads[0]["model"] == v2.EMBEDDING_MODEL
    assert client.calls[-1] is unloads[0]  # unload is the LAST thing that happens


def test_unload_sent_when_the_run_raises(monkeypatch):
    client = FakeClient()

    def body(c, incremental, limit):
        raise RuntimeError("lancedb write failed")

    _install(monkeypatch, client, body)
    with pytest.raises(RuntimeError, match="lancedb write failed"):
        v2.index_vault_v2()

    assert len(client.unloads()) == 1


def test_failing_unload_does_not_fail_the_run(monkeypatch, capsys):
    client = FakeClient(fail_unload=True)
    _install(monkeypatch, client, lambda c, incremental, limit: None)

    v2.index_vault_v2()  # must not raise

    assert len(client.unloads()) == 1
    assert "could not unload" in capsys.readouterr().out


def test_failing_unload_does_not_mask_the_original_error(monkeypatch):
    client = FakeClient(fail_unload=True)

    def body(c, incremental, limit):
        raise RuntimeError("the real failure")

    _install(monkeypatch, client, body)
    with pytest.raises(RuntimeError, match="the real failure"):
        v2.index_vault_v2()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
