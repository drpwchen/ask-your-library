# Corpus fixtures

Two tiny corpus folders in the `t2n-corpus/1` format ([`docs/corpus-format.md`](../../docs/corpus-format.md)).
The text is invented filler for testing, not clinical guidance.

```bash
python server/corpus.py check examples/corpus-fixture/textbook
python server/corpus.py check examples/corpus-fixture/lecture
python server/corpus.py add   examples/corpus-fixture/textbook --db /tmp/fixture_textbook_db --dry-run
```

`server/tests/test_corpus.py` checks both, breaks copies of them on purpose, and indexes them with a fake embedder.
