"""
vault-search — cross-encoder reranker service (optional, GPU).
Author: P.W. Chen · https://drpwchen.com · https://github.com/drpwchen

A tiny localhost HTTP server that scores (query, passage) pairs with a
cross-encoder (default BAAI/bge-reranker-v2-m3). It runs in its OWN Python —
one with CUDA torch + transformers — so the api_server stays light; the
api_server starts it on demand (server/reranker.py) and never imports torch.

Lifecycle, by design:
  * Started lazily by the first textbook query that wants a rerank.
  * Exits by itself after RERANK_IDLE_S without a request. Exiting (rather than
    freeing the model) is the only way to give back the CUDA context too.
  * Exits as soon as another job wants the GPU: the machine-wide GPU lease
    presses a PAUSE flag / queues a ticket before it waits for free VRAM, and
    this process watches for both. A nightly pipeline therefore never waits
    on the reranker; queries during that window are simply not reranked.
  * Refuses to run on CPU (measured too slow to be worth it: >10 s per query).

Endpoints (127.0.0.1 only):
  GET  /health   -> {"ok": true, "model": ..., "device": ...}
  POST /rerank   {"query": str, "docs": [str], "max_length": 512}
                 -> {"scores": [float], "ms": float}
                 503 when the GPU has been claimed (the service is exiting).

    <cuda-python> reranker_service.py --port 3791 --idle 600 \
        [--model BAAI/bge-reranker-v2-m3] [--lease-state ~/.claude/state/gpu-lease]
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

WATCH_INTERVAL_S = 2.0
MAX_DOCS = 200


def gpu_claimed(state_dir: Path | None) -> bool:
    """True when another job holds, is taking, or is queued for the GPU lease."""
    if state_dir is None:
        return False
    try:
        if (state_dir / "PAUSE").exists():
            return True
        q = state_dir / "queue"
        return q.is_dir() and any(q.iterdir())
    except OSError:
        return False


class Scorer:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not torch.cuda.is_available():
            raise SystemExit("[reranker] CUDA not available — refusing to run on CPU")
        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(model_name, dtype=torch.float16).cuda().eval())

    def score(self, query: str, docs: list[str], max_length: int = 512, bs: int = 16) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(docs), bs):
                enc = self.tok([[query, d] for d in docs[i:i + bs]], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to("cuda")
                out += self.model(**enc).logits.view(-1).float().tolist()
        return out


class State:
    def __init__(self, scorer, idle_s: float, lease_state: Path | None):
        self.scorer = scorer
        self.idle_s = idle_s
        self.lease_state = lease_state
        self.last_used = time.time()
        self.busy = 0
        self.stopping = False
        self.lock = threading.Lock()  # one GPU batch at a time


def make_handler(st: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": not st.stopping, "model": st.scorer.name, "device": "cuda",
                                 "idle_s": round(time.time() - st.last_used, 1)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/rerank":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                query = str(req["query"])
                docs = [str(d) for d in req["docs"]][:MAX_DOCS]
                max_length = int(req.get("max_length", 512))
            except Exception as e:  # noqa: BLE001
                return self._json(400, {"error": f"bad request: {e}"})
            if st.stopping or gpu_claimed(st.lease_state):
                st.stopping = True
                return self._json(503, {"error": "gpu claimed by another job"})
            with st.lock:
                st.busy += 1
                try:
                    t = time.perf_counter()
                    scores = st.scorer.score(query, docs, max_length) if docs else []
                    ms = (time.perf_counter() - t) * 1000
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                finally:
                    st.busy -= 1
                    st.last_used = time.time()
            self._json(200, {"scores": scores, "ms": round(ms, 1)})

    return Handler


def watcher(st: State, server: HTTPServer):
    while True:
        time.sleep(WATCH_INTERVAL_S)
        reason = None
        if gpu_claimed(st.lease_state):
            reason = "gpu claimed by another job"
        elif time.time() - st.last_used > st.idle_s and not st.busy:
            reason = f"idle > {st.idle_s:.0f}s"
        if reason:
            st.stopping = True
            print(f"[reranker] exiting: {reason}", flush=True)
            # Let an in-flight batch finish; it holds st.lock.
            with st.lock:
                pass
            server.shutdown()
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3791)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--idle", type=float, default=600.0)
    ap.add_argument("--lease-state", default=None)
    a = ap.parse_args()
    lease_state = Path(a.lease_state).expanduser() if a.lease_state else None
    if gpu_claimed(lease_state):
        print("[reranker] gpu claimed by another job — not starting", flush=True)
        sys.exit(3)
    # Bind before loading: a second copy started by a racing query exits here
    # instead of loading a second model onto the card.
    try:
        server = HTTPServer(("127.0.0.1", a.port), None)
    except OSError as e:
        print(f"[reranker] port {a.port} busy ({e}) — another instance is up", flush=True)
        sys.exit(4)
    t = time.time()
    scorer = Scorer(a.model)
    st = State(scorer, a.idle, lease_state)
    server.RequestHandlerClass = make_handler(st)
    print(f"[reranker] {a.model} loaded in {time.time() - t:.1f}s on :{a.port} "
          f"(idle exit {a.idle:.0f}s, pid {os.getpid()})", flush=True)
    threading.Thread(target=watcher, args=(st, server), daemon=True).start()
    server.serve_forever(poll_interval=0.5)
    server.server_close()


if __name__ == "__main__":
    main()
