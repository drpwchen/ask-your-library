"""Client side of the optional cross-encoder reranker (see reranker_service.py).

Stdlib only: the api_server process never imports torch. The first call that
wants a rerank starts the service in its own CUDA Python and waits for the
model to load; the service exits by itself when idle or when another job claims
the GPU lease, and the next call starts it again.

Every failure is soft. `score()` raises RerankUnavailable with a short reason
and the caller returns the un-reranked list — a missing reranker must never
fail a search.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    GPU_LEASE_STATE_DIR, RERANK_IDLE_S, RERANK_MODEL, RERANK_PORT, RERANK_PYTHON,
    RERANK_START_TIMEOUT, DATA_DIR,
)

SERVICE = Path(__file__).resolve().parent / "reranker_service.py"
LOG_PATH = DATA_DIR / "reranker_service.log"
_start_lock = threading.Lock()
_last_start_failure = 0.0
START_RETRY_S = 60.0  # after a failed start, don't retry on every query


class RerankUnavailable(RuntimeError):
    pass


def enabled() -> bool:
    return bool(RERANK_PYTHON) and Path(RERANK_PYTHON).exists()


def gpu_claimed() -> bool:
    """Same test the service uses: another job holds, is taking, or awaits the GPU."""
    d = GPU_LEASE_STATE_DIR
    if d is None:
        return False
    try:
        if (d / "PAUSE").exists():
            return True
        q = d / "queue"
        return q.is_dir() and any(q.iterdir())
    except OSError:
        return False


def _url(path: str) -> str:
    return f"http://127.0.0.1:{RERANK_PORT}{path}"


def _healthy(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(_url("/health"), timeout=timeout) as r:
            return bool(json.loads(r.read().decode("utf-8")).get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _spawn() -> subprocess.Popen:
    args = [str(RERANK_PYTHON), str(SERVICE), "--port", str(RERANK_PORT),
            "--model", RERANK_MODEL, "--idle", str(RERANK_IDLE_S)]
    if GPU_LEASE_STATE_DIR is not None:
        args += ["--lease-state", str(GPU_LEASE_STATE_DIR)]
    kw = {}
    if sys.platform == "win32":
        # Detached and windowless: it must outlive neither more nor less than
        # its own idle timer, and never flash a console on the desktop.
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log = open(LOG_PATH, "a", encoding="utf-8")
    try:
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, **kw)
    finally:
        log.close()


def ensure_running() -> str:
    """Make sure the service answers. Returns "warm" or "cold" (just started)."""
    global _last_start_failure
    if _healthy():
        return "warm"
    with _start_lock:
        if _healthy():
            return "warm"
        if time.time() - _last_start_failure < START_RETRY_S:
            raise RerankUnavailable("service failed to start recently")
        proc = _spawn()
        deadline = time.time() + RERANK_START_TIMEOUT
        while time.time() < deadline:
            if _healthy(timeout=2.0):
                return "cold"
            rc = proc.poll()
            if rc is not None:
                # 4 = port taken by a copy another thread/process just started.
                if rc == 4 and _healthy(timeout=5.0):
                    return "warm"
                _last_start_failure = time.time()
                raise RerankUnavailable(f"service exited rc={rc} (see {LOG_PATH.name})")
            time.sleep(0.5)
        _last_start_failure = time.time()
        raise RerankUnavailable(f"service not ready after {RERANK_START_TIMEOUT:.0f}s")


def score(query: str, docs: list[str], max_length: int = 512,
          timeout: float = 60.0) -> tuple[list[float], dict]:
    """Cross-encoder scores for each doc, plus {"start", "ms"} info."""
    if not enabled():
        raise RerankUnavailable("reranker not configured")
    if gpu_claimed():
        raise RerankUnavailable("gpu claimed by another job")
    start = ensure_running()
    body = json.dumps({"query": query, "docs": docs, "max_length": max_length}).encode("utf-8")
    req = urllib.request.Request(_url("/rerank"), data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RerankUnavailable(f"service HTTP {e.code}") from None
    except (OSError, ValueError) as e:
        raise RerankUnavailable(f"{type(e).__name__}: {str(e)[:120]}") from None
    scores = out.get("scores") or []
    if len(scores) != len(docs):
        raise RerankUnavailable(f"service returned {len(scores)} scores for {len(docs)} docs")
    return scores, {"start": start, "ms": out.get("ms")}
