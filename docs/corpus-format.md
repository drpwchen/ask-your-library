# Corpus folder format — `t2n-corpus/1`

A **corpus folder** is how reference material reaches vault-search. A producer
(e.g. [textbook-to-note](https://github.com/drpwchen/textbook-to-note) for textbooks and
guidelines, [lecture-to-notes](https://github.com/drpwchen/lecture-to-notes) for lectures,
or your own script) writes Markdown into a folder laid out as below; vault-search makes it
searchable with one command:

```bash
python server/corpus.py check <root>      # validate (reads files only)
python server/corpus.py add   <root>      # check → index → audit → print the .env line
```

This page is the only specification of the format. It describes what the v3 indexer
(`server/textbook_indexer_v3.py`, `chunker_v3.py`, `provenance.py`, `books_meta.py`,
`lecture_meta.py`) actually reads; nothing here is decorative. Producers do not call the
indexer themselves.

Two kinds exist. One corpus root holds one kind, and one index database holds one corpus
root.

| kind | search tool | default database | setting the server reads |
|---|---|---|---|
| `textbook` (textbooks, handbooks, guidelines) | `textbook_search` | `~/.vault-search/textbook_db` | `VAULT_SEARCH_TEXTBOOK_DB_PATH` |
| `lecture` (talks, courses) | `lecture_search` | `~/.vault-search/lecture_db_v3` | `VAULT_SEARCH_LECTURE_DB_PATH` |

`~/.vault-search` is `VAULT_SEARCH_DATA_DIR`. When the setting for that kind already points
at a v3 database, `add` uses that database instead (so a re-run adds to what the server
reads).

---

## 1. The root marker: `corpus.json`

```json
{"format": "t2n-corpus/1", "kind": "textbook", "name": "My reference shelf"}
```

| key | required | value |
|---|---|---|
| `format` | yes | exactly `"t2n-corpus/1"` |
| `kind` | yes | `"textbook"` or `"lecture"` |
| `name` | no | free text, shown by `check` |

A missing `corpus.json` is a **warning**, not an error (folders made before the marker
existed keep working): `check` infers the kind — `lecture` when any top-level folder holds a
`_meta.md`, else `textbook`. A present but broken marker (invalid JSON, another `format`,
an unknown `kind`) is an **error**.

## 2. What every kind shares

- Each **top-level folder** of the root is one *source* (a book, or a lecture course). Its
  folder name is the source's identity in search results and in `books.json`; keep it
  stable.
- Folders whose name starts with `.` or `_` are tooling output and are **not indexed** —
  except `_Guidelines` (below).
- `.md` files directly in the root are not indexed.
- Inside a source, every `*.md` file is indexed, recursively, except `index.md` and files
  under sub-folders starting with `.` or `_`.
- Per directory, **either** `full_text.md` **or** the other (chapter) files are indexed,
  never both: chapters win when they hold at least 50% of `full_text.md`'s bytes.
- Markdown headings: `#` is the file title; `##` and `###` open the sections that become
  the retrieval units (a hit returns its whole section as context). Tables are kept whole.
- Lines that are only `Source: …`, and HTML comments other than figure `<!-- REF: … -->`
  markers, are treated as bookkeeping and are not part of the indexed text.
- Encoding: UTF-8.

## 3. `kind: "textbook"`

```
<root>/
  corpus.json
  Smith_Shoulder-Rehab_2e_2024/        one book = one folder
    _manifest.json                     optional, written by the converter's --resplit
    ch01_Examination.md
    ch02_Exercise.md
  _Guidelines/                         every guideline in its own sub-folder
    ExampleSociety_LowBackPain_2023/
      full_text.md
```

### Folder names

`<Author>_<Title>_<N>e_<YYYY>` is the convention. From the name alone:

- `_7e_` → edition `7e`
- the last `_YYYY` → year `2024`
- the name without its edition/year tail → *lineage* (`Smith_Shoulder-Rehab`): editions of
  one lineage are recognised as the same work (`--holding-report` uses this to suggest
  retiring older editions).

None of this is required — a folder named `My notes on X` indexes fine, it simply has no
edition or year.

### Guidelines

Each guideline is a sub-folder of `<root>/_Guidelines/`. It is its own source (its
sub-folder name is the key), its `corpus_type` is `guideline`, and search applies a small
guideline prior after reranking. `.md` files directly in `_Guidelines/` are not indexed.

### Page markers

```
<!-- page 12 -->
```

on its own line, where physical PDF page 12 starts. Every chunk records the pages of its own
lines. Without markers a source still indexes, but its hits carry no page numbers (`check`
warns). Text before the first marker of a chapter file gets that file's `pdf_page_start`
from the manifest, when there is one.

### `_manifest.json` (optional)

Written by textbook-to-note `convert.py --resplit` (`T2N_SPLIT_MANIFEST=1`) into the book
folder; the full schema lives in textbook-to-note. The indexer looks for the nearest
manifest from a file's folder up to the source folder and uses these fields:

| field | used for |
|---|---|
| `page_labels` `{"<pdf page>": "<printed label>"}` | printed page numbers (`p.2` next to `PDF p.12`); without it `page_confidence` is `pdf_only` |
| `page_confidence` | `printed` / `pdf_only` |
| `split_method` | recorded per chunk |
| `files[].file` | matches the entry to the `.md` file (relative path; a bare name when unambiguous) |
| `files[].chapter_number` | the real chapter number |
| `files[].title` | the chapter title (else derived from the file name) |
| `files[].pdf_page_start` | page of text before the file's first marker |
| `files[].kind` | `references` files are not indexed; `body` / `front` / `back` are |
| `sections[]` (`title`, `file`, `kind`) | section-level `kind`, e.g. a References section |

A missing or unreadable manifest is not an error: those fields stay empty. `check` warns
when a manifest is unreadable or names files that no longer exist.

Reference lists are kept out of the index: by manifest `kind`, by a References /
Bibliography heading, and — for a chapter file *named* like one (`ch74_REFERENCES.md`) —
from that heading line to the end of the file.

### `books.json` (optional, outside the corpus)

`~/.vault-search/books.json` (`VAULT_SEARCH_BOOKS_META`) adds what a folder name cannot say.
Keys are source folder names (for guidelines, the sub-folder name). All fields optional:

```json
{"Smith_Shoulder-Rehab_2e_2024": {"tier": 1},
 "Smith_Shoulder-Rehab_1e_2019": {"status": "superseded", "superseded_by": "Smith_Shoulder-Rehab_2e_2024"},
 "ExampleSociety_LowBackPain_2023": {"corpus_type": "guideline"}}
```

| field | values |
|---|---|
| `edition`, `year` | override the folder-name inference |
| `tier` | 1–3, a tie-break between equally good hits |
| `corpus_type` | `textbook`, `guideline`, `handbook`, `review`, `exam` |
| `status` | `current`, `superseded`, `derivative`, `low_trust` — the last three are not indexed and are hidden from search |
| `superseded_by` | folder name of the replacement |
| `lineage` | explicit lineage, when the name rule gets it wrong |

See `examples/books.example.json`. `check` (with the server's `books.json`) warns about keys
that match no source folder — usually a typo.

## 4. `kind: "lecture"`

```
<root>/
  corpus.json
  20250301_Example_Shoulder-Ultrasound/      one course (talk, workshop) = one folder
    _meta.md                                 course bookkeeping, never returned by search
    Biceps_tendon_and_subscapularis.md       one entry = one topic of the course
    Supraspinatus_and_bursa.md
```

- The course folder name (`course_dir`) is the course's identity; a date prefix keeps them
  sorted.
- `_meta.md` holds the course bookkeeping (clip list, speakers, where the material lives).
  It is indexed but `lecture_search` never returns it. `check` warns when a course has none.
- Every other `.md` file is an **entry**: the first line is an H1, followed — before the
  first `##` heading — by one `Source:` line:

  ```
  Source: YYYY-MM-DD Speaker《Course title》· course_dir · clips
  ```

  `lecture_search` parses `date`, `speaker` and `course_title` from it. Every part is
  optional and an unreadable part is left empty, never guessed: a date range keeps its
  start date (`2026-04-18/19`), `《…》` delimits the course title, and placeholders such as
  `未載明` / `unknown` count as empty. More metadata lines may follow it (they are indexed
  as text). `check` warns about an entry with no H1, no `Source:` line, no date, or no
  `《title》`.
- Time codes in the text (`[c01 03:45]` = clip 1 at 3:45) are plain text to the indexer;
  they let a reader find the moment in the recording.

### A complete entry

```markdown
# Long head of biceps and subscapularis: scanning sequence

Source: 2025-03-01 Dr. Example《Shoulder Ultrasound Workshop》· 20250301_Example_Shoulder-Ultrasound · c01

## Biceps tendon in the groove

Start with the arm in slight external rotation and the probe transverse over the bicipital
groove [c01 02:10]. The tendon should sit inside the groove; an empty groove means it has moved
out of place or is torn [c01 03:45].

## Subscapularis

Rotate the arm outward to bring the subscapularis tendon into view under the probe [c01 07:30].
Follow the fibres to their insertion on the lesser tuberosity and compare the thickness with the
other side [c01 09:05].
```

A search hit on it reads:

```
1. **20250301_Example_Shoulder-Ultrasound** · Biceps tendon in the groove · sim 0.80
   (20250301_Example_Shoulder-Ultrasound/Biceps_tendon_and_subscapularis.md)
   source: 2025-03-01 · Dr. Example · 《Shoulder Ultrasound Workshop》
```

## 5. `check` and `add`

`check` reports **errors** (exit 1) and **warnings** (exit 0; `--strict` makes them fail;
`--json` for machines).

| errors | |
|---|---|
| `no_root` | the root is not a directory |
| `marker_json`, `marker_format`, `marker_kind`, `marker_name` | broken `corpus.json` |
| `no_sources` / `no_courses` | no source folders |
| `no_files` / `no_entries` | source folders, but no `.md` to index |
| `books_meta_json` | `books.json` is not valid JSON (the indexer would stop) |

Warnings: `no_marker`, `root_md`, `guideline_loose_md`, `empty_source`, `no_page_markers`,
`manifest_unreadable`, `manifest_stale`, `chapters_ignored`, `kind_mismatch`,
`books_meta_unknown` (textbook); `no_meta`, `empty_course`, `entry_no_h1`,
`entry_no_source`, `source_no_date`, `source_no_course`, `lecture_guidelines` (lecture).

`add <root> [--db PATH] [--dry-run] [--books-meta PATH] [--strict] [--no-gpu-lease]`:

1. runs `check`; any error stops it;
2. picks the database (`--db`, else the default above) and refuses — exit 2 — a database
   the server uses for something else (the indexer's own guard), a database inside the
   corpus, or a v3 database **built from another corpus root** (a run removes rows of files
   outside its root, so two roots can never share one database);
3. runs `textbook_indexer_v3.py --corpus <root> --db <db> --books-meta <the server's books.json>`
   — later runs re-embed only files whose content changed and drop rows of removed files;
4. runs the same command with `--audit` (every chapter with text has rows; exit 1 if not);
5. prints the `.env` line, e.g. `VAULT_SEARCH_TEXTBOOK_DB_PATH=/home/you/.vault-search/textbook_db`.
   If the server already reads that database the new rows are live at once; otherwise put
   the line in the repo's `.env` and restart `api_server.py` (and start a new MCP client
   session if you run `mcp_server.py` directly).

`--dry-run` stops after printing the two commands and the `.env` line. Indexing needs Ollama
with `qwen3-embedding:0.6b` (`ollama pull qwen3-embedding:0.6b`); a GPU is optional.

Adding a source later = put its folder under the same root and run `add` again.

Fixtures: `examples/corpus-fixture/textbook` and `examples/corpus-fixture/lecture`.

## Versioning

`t2n-corpus/1` changes only by adding optional things. A change an existing reader would
misread gets a new format string (`t2n-corpus/2`), and `check` names the format it reads.
