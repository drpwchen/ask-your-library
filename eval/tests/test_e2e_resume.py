"""e2e.py checkpoint/resume and usage-limit stop. Stdlib-only, no model calls.

Run: python eval/tests/test_e2e_resume.py   (or pytest)
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e2e  # noqa: E402
import llm  # noqa: E402


def _setup(tmp: Path, n: int = 5):
    qs = [{"id": f"Q{i}", "type": "T1", "corpus": "textbook"} for i in range(n)]
    (tmp / "set.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    run = {"label": "x", "answers": {q["id"]: {"results": []} for q in qs}}
    (tmp / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return ["--set", str(tmp / "set.jsonl"), "--run", str(tmp / "run.json"),
            "--out", str(tmp / "out.json"), "--workers", "1", "--model", "m"]


def _fake(calls, quota_on=()):
    def run_one(q, ans, k, model):
        calls.append(q["id"])
        if q["id"] in quota_on:
            raise llm.QuotaExhausted("You've hit your 5-hour limit")
        return {"qid": q["id"], "type": q["type"], "reply": "r",
                "grade": {"answer_correct": True, "claims": []}}
    return run_one


def test_quota_stops_and_rerun_resumes():
    orig = e2e.run_one
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        args = _setup(tmp)
        try:
            calls = []
            e2e.run_one = _fake(calls, quota_on={"Q2"})
            assert e2e.main(args) == 3
            assert not (tmp / "out.json").exists()
            assert calls == ["Q0", "Q1", "Q2"]  # stopped: Q3, Q4 never sent
            ckpt = tmp / "out.json.partial.jsonl"
            assert len(ckpt.read_text(encoding="utf-8").splitlines()) == 3  # header + 2

            calls = []
            e2e.run_one = _fake(calls)
            assert e2e.main(args) == 0
            assert calls == ["Q2", "Q3", "Q4"]  # finished ones not re-run
            out = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
            assert [r["qid"] for r in out["rows"]] == ["Q0", "Q1", "Q2", "Q3", "Q4"]
            assert out["summary"]["all"]["n"] == 5
            assert not ckpt.exists()
        finally:
            e2e.run_one = orig


def test_truncated_line_and_errors_are_retried():
    orig = e2e.run_one
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        args = _setup(tmp, 3)
        try:
            e2e.run_one = _fake([], quota_on={"Q1"})
            e2e.main(args)
            ckpt = tmp / "out.json.partial.jsonl"
            with ckpt.open("a", encoding="utf-8") as fh:
                fh.write('{"qid": "Q2", "gra')  # hard kill mid-write
            calls = []
            e2e.run_one = _fake(calls)
            assert e2e.main(args) == 0
            assert calls == ["Q1", "Q2"]
        finally:
            e2e.run_one = orig


def test_settings_mismatch_refuses():
    orig = e2e.run_one
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        args = _setup(tmp, 3)
        try:
            e2e.run_one = _fake([], quota_on={"Q1"})
            e2e.main(args)
            try:
                e2e.main(args[:-1] + ["other-model"])
                raise AssertionError("should refuse a checkpoint from other settings")
            except SystemExit as e:
                assert "different settings" in str(e)
            calls = []
            e2e.run_one = _fake(calls)
            assert e2e.main(args[:-1] + ["other-model", "--fresh"]) == 0
            assert calls == ["Q0", "Q1", "Q2"]
        finally:
            e2e.run_one = orig


def test_check_quota():
    for t in ("You've hit your 5-hour limit · resets 12am (Asia/Taipei)",
              "You've hit your weekly limit", "Claude usage limit reached"):
        try:
            llm.check_quota(t)
            raise AssertionError(t)
        except llm.QuotaExhausted:
            pass
    llm.check_quota('{"answer_correct": true}')
    llm.check_quota("Patients who hit your target heart rate limit... " * 20)  # long = real text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
