"""Stdlib-only tests for the eval harness: matching, metrics, statistics, parsers.

Run: python eval/tests/test_eval_metrics.py   (or pytest)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import metrics as M  # noqa: E402
from adapters import parse_chat_context, parse_textbook_markdown, parse_vault_markdown  # noqa: E402

Q = {
    "id": "T1-001", "type": "T1", "corpus": "textbook", "cluster": "B/ch12",
    "relevant": [{"corpus": "textbook", "book": "B_2e", "file": "B_2e/ch12.md",
                  "page_start": 341, "page_end": 342, "grade": 2}],
    "contaminants": [{"corpus": "textbook", "book": "B_1e"}],
}


def tb(rank, book, file, ps=0, pe=0):
    return {"corpus": "textbook", "rank": rank, "book": book, "file": file,
            "page_start": ps, "page_end": pe}


def test_doc_and_page_match():
    g = Q["relevant"][0]
    assert M.doc_match(tb(1, "B_2e", "B_2e/ch12.md"), g)
    assert not M.doc_match(tb(1, "B_2e", "B_2e/ch13.md"), g)
    assert M.page_match(tb(1, "B_2e", "B_2e/ch12.md", 343, 344), g)       # within ±2
    assert not M.page_match(tb(1, "B_2e", "B_2e/ch12.md", 345, 350), g)   # outside
    assert not M.page_match(tb(1, "B_2e", "B_2e/ch12.md", 0, 0), g)       # no page = no page hit


def test_score_hit_mrr_contamination():
    res = [tb(1, "B_1e", "B_1e/ch12.md", 300, 301), tb(2, "X", "X/a.md"), tb(3, "B_2e", "B_2e/ch12.md", 341, 341)]
    row = M.score_question(Q, res, jidx={("T1-001", M.result_key(res[1])): {"rel": 0}})
    assert row["hit@3"] == 1 and row["hit@1"] == 0
    assert abs(row["mrr"] - 1 / 3) < 1e-9
    assert row["page_hit@3"] == 1
    assert row["contamination@3"] == 1         # old edition in the top 3
    assert row["unjudged_at_hit"] == 1          # B_1e result was never judged


def test_global_contaminant_and_conflict_judgment():
    q = dict(Q, contaminants=[])
    res = [tb(1, "Cowork_Notes", "Cowork_Notes/a.md")]
    assert M.score_question(q, res, global_contaminants={"cowork_notes"})["contamination@3"] == 1
    v = [{"corpus": "vault", "rank": 1, "note": "n", "folder": "Shared Notes"}]
    qv = {"id": "T5-9", "corpus": "vault", "relevant": [{"corpus": "vault", "note": "x"}]}
    assert M.score_question(qv, v, global_contaminants={"vault:shared notes"})["contamination@3"] == 1
    res = [tb(1, "Y", "Y/a.md")]
    j = {("T1-001", M.result_key(res[0])): {"rel": 1, "conflict": True}}
    assert M.score_question(q, res, jidx=j)["contamination@3"] == 1


def test_abstain_question_not_scored_for_hits():
    q = {"id": "T7-1", "type": "T7", "corpus": "textbook", "expect_abstain": True, "relevant": []}
    row = M.score_question(q, [tb(1, "Z", "Z/a.md")])
    assert row["abstain_question"] and "hit@3" not in row
    agg = M.aggregate([row])
    assert agg["all"]["n"] == 0 and agg["by_type"]["T7"]["n"] == 1


def test_vault_match_is_note_level():
    q = {"id": "T5-1", "corpus": "vault", "relevant": [{"corpus": "vault", "note": "Shoulder Exam"}]}
    res = [{"corpus": "vault", "rank": 1, "note": "shoulder exam"}]
    assert M.score_question(q, res)["hit@1"] == 1


def test_mcnemar_exact():
    r = M.mcnemar_exact([1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0])
    assert r["b10"] == 4 and r["b01"] == 0 and abs(r["p"] - 0.125) < 1e-9
    assert M.mcnemar_exact([1, 0], [1, 0])["p"] == 1.0


def test_cluster_bootstrap_is_wider_than_naive_when_clustered():
    # 10 questions, 2 clusters: all the gain sits in one cluster.
    a = [0.0] * 10
    b = [1.0] * 5 + [0.0] * 5
    clustered = M.cluster_bootstrap_diff(a, b, ["c1"] * 5 + ["c2"] * 5, n_boot=2000)
    naive = M.cluster_bootstrap_diff(a, b, [str(i) for i in range(10)], n_boot=2000)
    assert clustered["diff"] == naive["diff"] == 0.5
    width = lambda r: r["ci95"][1] - r["ci95"][0]  # noqa: E731
    assert width(clustered) > width(naive)


def test_parse_vault_markdown():
    md = ("# vault_search: \"q\" — 2 notes\n\n"
          "1. [[Shoulder exam]] · Evaluation > PE · Medicine · sim 0.75\n"
          "   tags: [a]\n"
          "   first line\nsecond line\n"
          "2. [[Root note]] · sim 0.60\n\n"
          "graph-linked: [[X]]\n")
    r = parse_vault_markdown(md)
    assert [x["note"] for x in r] == ["Shoulder exam", "Root note"]
    assert r[0]["folder"] == "Medicine" and r[0]["section"] == "Evaluation > PE"
    assert r[0]["text"] == "first line\nsecond line" and r[0]["sim"] == 0.75


def test_parse_textbook_markdown():
    md = ("# textbook_search: \"q\" — 2 passages\n\n"
          "1. **BookA_2e** · Ch 3 > Dosing · p.12-13 · sim 0.71\n"
          "   (BookA_2e/ch03.md)\nparent text\n\nmore text\n\n"
          "2. **BookB** · sim 0.50\n"
          "   (BookB/full_text.md)\nbody\n")
    r = parse_textbook_markdown(md)
    assert len(r) == 2
    assert (r[0]["book"], r[0]["file"], r[0]["page_start"], r[0]["page_end"]) == ("BookA_2e", "BookA_2e/ch03.md", 12, 13)
    assert r[0]["section"] == "Ch 3 > Dosing" and "more text" in r[0]["text"]
    assert r[1]["page_start"] == 0 and r[1]["sim"] == 0.5


def test_parse_chat_context():
    ctx = ("以下是 vault 中與問題相關的筆記內容：\n\n"
           "【Note A › Sec】(sim=0.70)\nbody a\n\n---\n\n"
           "【Note B › 】(sim=0.60)\nbody b\n\n"
           "以下是教科書中與問題相關的內容：\n\n"
           "【📖 BookA › BookA/ch1.md › 】(sim=0.81)\n[p.4] x\n\n[p.6] y\n\n---\n\n"
           "【🎤 Course › Course/topic.md › Sec】(sim=0.66)\n[p.3-5] · source: x\nlecture body\n")
    r = parse_chat_context(ctx)
    assert [(x["corpus"], x["rank"]) for x in r] == [("vault", 1), ("vault", 2), ("textbook", 1), ("lecture", 1)]
    assert r[2]["page_start"] == 4 and r[2]["page_end"] == 6
    assert r[1]["note"] == "Note B"
    assert (r[3]["page_start"], r[3]["page_end"]) == (3, 5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"{len(fns)} eval tests passed")
