# ask-your-library

> [English](./README.md) · **繁體中文**

**問你自己的書庫——教科書、指引、演講筆記和你自己的筆記——得到標明頁碼出處的答案。** 地端優先的搜尋中樞，附 Claude Code 用的 MCP server 與 Obsidian 外掛。

舊名 vault-search。

**相關工具：**[textbook-to-note](https://github.com/drpwchen/textbook-to-note) 與 [lecture-to-notes](https://github.com/drpwchen/lecture-to-notes) 產出這裡要搜尋的語料；[note-supplement](https://github.com/drpwchen/note-supplement) 把新素材補進你用它找到的筆記。

三個工具，把你的 Obsidian vault 變成一個**可語意搜尋、可 AI 對話的知識庫**——而且完全跑在自己的電腦上。沒有雲端索引、搜尋時筆記不離開你的電腦、檢索不需要按次數付 API 費用。

作者是一位醫師，為了從上千則臨床筆記中讀書而打造，但它**與領域無關**：指向任何一個放滿 markdown 筆記的 vault（研究、法律、工程、個人 wiki）都能用。

![搜尋面板截圖](docs/images/search.png)

### 兩種使用方式（共用同一份索引）

這個 repo 提供**兩個前端、共用同一份地端 LanceDB 索引 + 檢索核心**，兩個都用或擇一都行：

- **`vault-search-obsidian`** — Obsidian plugin（側欄的 Search／Related Notes／Chat），連到一支小型地端 API server。→ [快速開始](#vault-search-obsidian--obsidian-plugin)
- **`vault-search-mcp`** — 一支 MCP server，把同一套搜尋以工具（`vault_search`、`vault_similar`、`vault_related`、`vault_stats`、`textbook_search`，以及設定了演講語料時的 `lecture_search`）開給 Claude Code 或任何 MCP client。**不需要 Obsidian**。→ [快速開始](#vault-search-mcp--在-claude-code或任何-mcp-client中使用)

兩者共用同一份 `server/` 核心（indexer · scoring · Personalized PageRank · 知識圖譜），所以你只要索引一次，就能從任何地方查詢。

---

## 三套組

| 工具 | 功能 | 後端 |
|---|---|---|
| 🔍 **Vault Search** | 用自然語言（任何語言）輸入問題，回傳語意上最相關的筆記段落——不是關鍵字比對。 | LanceDB 向量搜尋 + 地端 embedding |
| 🔗 **Related Notes** | 即時側欄：當你閱讀或選取文字時，自動浮現你早已忘記寫過的相關筆記，串起 vault 裡的隱藏連結。 | 同一份索引的「找相似」模式 |
| 💬 **Vault Chat** | 跟你的筆記對話（RAG）。三種模式：**Vault**（只用筆記、100% 地端且免費）、**Hybrid**（筆記 + AI 知識）、**Free**（自由對話 + 文獻搜尋）。 | 地端 LLM 或你的 Claude 訂閱 |

三者都在 Obsidian 側欄。同一份索引還會開出一個 **MCP server**，讓 Claude Code 這類 coding agent 能把「搜尋你的 vault」當成一個工具來呼叫。

### 截圖

| 🔍 Vault Search | 🔗 Related Notes |
|---|---|
| ![Vault Search](docs/images/search.png) | ![Related Notes](docs/images/related.png) |

| 💬 Vault Chat | 👁 Context 預覽 |
|---|---|
| ![Vault Chat](docs/images/chat.png) | ![Context preview](docs/images/chat-context.png) |

---

## 為什麼做這個

Obsidian 內建搜尋是字面比對——它找你打的字。但你幾個月前寫的筆記，常常根本想不起當時用了哪些字。語意搜尋是用**意義**找筆記：搜尋 `脊髓損傷後的鬆弛性膀胱怎麼處理`，就算筆記裡沒出現這幾個字，也能找到你那則標題叫 `Neurogenic Bladder` 的筆記。

設計上的第一優先是**地端、隱私**：

- **embedding 在地端跑**，透過 [Ollama](https://ollama.com)（`bge-m3`，中英雙語、CPU 即可）。你的筆記內容不會被送到任何地方做索引或搜尋。
- **Vault 模式對話完全地端且免費**——地端 LLM 用檢索到的筆記回答，不計任何 token 費用。
- **Hybrid／Free 對話可選擇**呼叫 `claude` CLI，讓你直接用已經在付的 Claude 訂閱，而不必另外申請（並按 token 付費的）API key。

---

## 運作原理

```
                    ┌─────────────────────────────────────────┐
   Obsidian  ◄────► │  Plugin（側欄）：搜尋 · 相關 · 對話       │
                    └───────────────────┬─────────────────────┘
                                        │ HTTP (localhost:3789)
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │  api_server.py  (FastAPI)                 │
                    │   /search  /similar  /chat  /reindex      │
                    └──────┬───────────────────────┬───────────┘
                           │                        │
                      embed query              RAG context
                           ▼                        ▼
        ┌────────────────────────┐     ┌──────────────────────────┐
        │ Ollama  (bge-m3)        │     │ Vault 模式 → 地端 LLM      │
        │ → 向量                  │     │ Hybrid/Free → `claude` CLI │
        └───────────┬────────────┘     └──────────────────────────┘
                    ▼
        ┌────────────────────────┐
        │ LanceDB 向量索引        │  ◄── indexer.py（依標題切塊、
        │ (~/.vault-search)      │       embed、儲存；增量更新）
        └────────────────────────┘

   同一份索引 + 排序也透過 mcp_server.py（MCP）提供給 coding agent。
```

- **`indexer.py`** 走訪你的 vault，把每則筆記依標題切成 chunk，用 `bge-m3` embedding，存進地端 **LanceDB** 表。再次執行是增量的（內容雜湊快取）。
- **`scoring.py`** 對原始向量結果重新排序，可選的**路徑加權**（讓你更信任某些資料夾）、**時近性**（新筆記排前面）、**關係**加權。
- **`ppr.py`** 用 **Personalized PageRank** 在 `[[wiki-link]]` 圖上排序筆記——HippoRAG / LinearRAG 風格的 random walk with restart。它驅動 **Related Notes** 與多跳的**查詢擴展**（以語意搜尋的 top 命中為種子做隨機遊走），找出純向量相似度漏掉的相關筆記。在作者的 vault 上，相較於樸素的 1-hop／共享連結擴展，sparse-bridge recall@10 從 ~60% 提升到 ~84%。
- **`api_server.py`** 服務 plugin：語意搜尋、找相似、單篇重新索引，以及對話（含檢索增強的 context）。
- **`mcp_server.py`** 把同一套搜尋以 MCP 工具開給 Claude Code 等：`vault_search`、`vault_related`、`vault_similar`、`vault_stats`、`textbook_search`（設定了演講語料時再加 `lecture_search`）。
- **選用 add-on：** 第二份長文**參考語料**（`textbook_indexer_v3.py`，parent-child 切塊、混合檢索、可選重排）與 `[[wikilink]]` + 實體的**知識圖譜**（`graph_builder.py`）。

---

## 先備條件

**費用：**搜尋不用花錢。Embedding 與 Vault 模式對話都在本機跑 Ollama；唯一選用的花費是你可能已經在付的 Claude 訂閱（Hybrid／Free 對話用）。

| 需求 | 說明 |
|---|---|
| **Python 3.10+** | 跑 server 用。 |
| 本機執行 **[Ollama](https://ollama.com)** | 提供 embedding（以及選用的地端對話）。`bge-m3` 在 **CPU** 上就跑得動，GPU 只是加速大量索引。 |
| **Obsidian** | 用側欄 plugin。也可不開 plugin，純用 MCP/HTTP API headless 使用。 |
| **Claude CLI**（選用） | 只有 Hybrid／Free 對話需要。Vault 模式對話改用地端模型。 |
| 一台**常開的機器**（選用） | 若想讓手機或其他裝置也能搜尋／對話，就把 server 跑在常開的 PC／NAS 上。單機使用的話，開 Obsidian 時順手啟動即可。 |

先把 embedding 模型拉下來：

```bash
ollama pull bge-m3
# 選用，給零成本的地端 Vault 模式對話：
ollama pull gemma2:9b
```

---

## 安裝

### Step 0 — 共用核心（只做一次，兩個前端都需要）

```bash
git clone https://github.com/drpwchen/ask-your-library.git
cd ask-your-library
python -m venv .venv
source .venv/bin/activate    # Windows Git Bash：source .venv/Scripts/activate · PowerShell：.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp env.example .env          # 然後編輯 .env——取消 VAULT_PATH 的註解，設成你的 vault 路徑

cd server
python indexer.py            # 建索引（之後執行為增量）
```

若還沒拉模型，先 `ollama pull bge-m3`（見[先備條件](#先備條件)）。接著從下面兩個前端擇一或都裝。

> 🧋 省了你時間的話，[請我喝杯飲料](https://drpwchen.com/support/)。

### `vault-search-obsidian` — Obsidian plugin

啟動地端 API server，再安裝 plugin：

```bash
python server/api_server.py                       # 服務於 http://localhost:3789
cp -r plugin "<你的VAULT>/.obsidian/plugins/vault-search-plugin"
```

> **記憶體說明。** `api_server.py` 與 `mcp_server.py` 會在 import 時設 `OPENBLAS_NUM_THREADS=1`。numpy 內建 OpenBLAS，一 import 就依 CPU 核心數各預留一塊暫存區——在 12 執行緒機器上實測 373 MB private bytes，設成 1 之後是 19 MB——而這兩支程式本身不做線性代數（embedding 交給 Ollama，向量運算交給 LanceDB）。要拿回預設值就自己 export `OPENBLAS_NUM_THREADS`。

在 Obsidian：**設定 → 第三方外掛 → 啟用「Vault Semantic Search」**。打開它的設定，確認 **API Server URL** 是 `http://localhost:3789`。側欄會出現三個圖示：🔍 搜尋、💬 對話、🔗 相關筆記。

> plugin 附了一份 `data.json.example`。Obsidian 會在首次執行時寫出真正的 `data.json`；千萬不要把含 API key 的 `data.json` commit 上去。

### `vault-search-mcp` — 在 Claude Code（或任何 MCP client）中使用

不需要 Obsidian、不需要 API server——直接拿 Step 0 建好的索引註冊 MCP server：

```bash
# macOS / Linux
claude mcp add vault-search -- /abs/path/to/ask-your-library/.venv/bin/python /abs/path/to/ask-your-library/server/mcp_server.py
# Windows
claude mcp add vault-search -- C:\path\to\ask-your-library\.venv\Scripts\python.exe C:\path\to\ask-your-library\server\mcp_server.py
```

要填 **Step 0 那個 venv 裡 Python 的絕對路徑**（在啟用的 venv 裡跑 `python -c "import sys; print(sys.executable)"` 就會印出來）。MCP client 是自己啟動 server、不經過你的 shell：那裡的 `python` 可能是另一個沒裝 `lancedb` 的直譯器，server 就會連不上。

之後 Claude Code（或任何 MCP client）就能呼叫 `vault_search`、`vault_related`、`vault_similar`、`vault_stats`、`textbook_search`，以及（若你有索引演講語料）`lecture_search`，全部背後共用同一份地端索引、Personalized PageRank 擴展與重新排序。

**如果你會同時開好幾個 session，改註冊 `mcp_thin.py`。** 每個 `mcp_server.py` session 都會自己開一份索引。在作者機器上實測：剛啟動 105 MB，跑完一次 `vault_search` 是 367 MB，跑完一次 `textbook_search`（會載入 tokenizer 與 parent map）則來到 2.2 GB——同時開幾個 session，一份索引就要吃掉好幾 GB。`mcp_thin.py` 是只用標準函式庫的轉發 client，把工具呼叫轉給正在跑的 `api_server.py`，由它獨自持有索引：

```bash
python server/api_server.py                                  # 必須是啟動狀態
claude mcp add vault-search -- /abs/path/to/ask-your-library/.venv/bin/python /abs/path/to/ask-your-library/server/mcp_thin.py
```

工具、參數、輸出完全相同——API server 轉進去的是 standalone MCP server 用的同一個 `handle_tool_call()`。代價是 API server 必須活著；它若掛掉，client 仍然啟動得起來（會拿上一次快取的工具 schema），工具呼叫則回一段看得懂的錯誤訊息，而不是讓整個 session 死掉。

---

## 設定

一切都由環境變數驅動（完整註解清單見 **`env.example`**）。沒有必填項：`VAULT_PATH` 只有要搜你的筆記時才需要，只用語料可以不設。重點：

| 變數 | 預設 | 用途 |
|---|---|---|
| `VAULT_PATH` | —（不設＝只用語料） | vault 的絕對路徑。 |
| `VAULT_SEARCH_DATA_DIR` | `~/.vault-search` | 索引／快取／log 存放處。 |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama 端點。 |
| `VAULT_SEARCH_EMBEDDING_MODEL` | `bge-m3` | vault embedding 模型。 |
| `VAULT_SEARCH_API_PORT` | `3789` | API server 連接埠。 |
| `CLAUDE_CMD` | `claude` | Hybrid／Free 對話用的 CLI。 |
| `VAULT_SEARCH_CHAT_MODEL` | `gemma2:9b` | Vault 模式對話的地端模型。 |
| `VAULT_SEARCH_PERSONA` / `_LANGUAGE` | 通用 | 自訂助理角色與回答語言。 |
| `VAULT_SEARCH_PATH_WEIGHTS` | `{}` | 在排序中加權／降權資料夾。 |
| `VAULT_SEARCH_EXCLUDE_PATTERNS` | （無） | 預設把衍生筆記（草稿、暫存）排除在搜尋外。 |

---

## 加入一份語料（教科書、指引、演講）

參考資料以**語料資料夾**的形式進入 ask-your-library：就是一般資料夾加 Markdown，排法照 [`docs/corpus-format.md`](docs/corpus-format.md)（格式 `t2n-corpus/1`）。[textbook-to-note](https://github.com/drpwchen/textbook-to-note) 從 PDF 教科書與指引產生，[lecture-to-notes](https://github.com/drpwchen/lecture-to-notes) 從演講錄影產生，你自己的腳本也可以。一行指令就能搜：

```bash
ollama pull qwen3-embedding:0.6b                  # 語料用的 embedding 模型（GPU 非必要）
python server/corpus.py check /path/to/corpus     # 檢查排法（只讀檔）
python server/corpus.py add   /path/to/corpus     # 檢查 → 建索引 → 稽核 → 印出 .env 設定行
```

`add` 從資料夾的 `corpus.json` 讀出種類，建進 `~/.vault-search/textbook_db`（教科書、指引）或 `~/.vault-search/lecture_db_v3`（演講），也可用 `--db PATH` 指定。最後會印出 server 需要的那一行設定，例如

```
VAULT_SEARCH_TEXTBOOK_DB_PATH=/home/you/.vault-search/textbook_db
```

把它加進 `.env`，重啟 `api_server.py`（直接跑 `mcp_server.py` 的話，開一個新的 MCP session），`textbook_search`（或 `lecture_search`）就會從這份索引回答。之後要加書：把書的資料夾放進同一個根目錄，再跑一次 `add`，只有變動過的檔會重新 embed。`--dry-run` 只印出 indexer 指令、不執行。一個資料庫只對應一個語料根目錄：由別的根目錄建出來的資料庫 `add` 會拒絕；server 拿去做別的用途的資料庫，indexer 本身也會拒寫。

- **指引**放在 `<root>/_Guidelines/<名稱>/`，一份一個子資料夾：每份指引是獨立來源，標成 `guideline`，排序時有一點加分。
- **`books.json`**（`VAULT_SEARCH_BOOKS_META`，預設 `~/.vault-search/books.json`；範例見 `examples/books.example.json`）補上資料夾名稱表達不了的資訊——版次、tier、`corpus_type`、`status`（`superseded`／`derivative`／`low_trust` 的來源不建索引、也不出現在搜尋結果）。key 是來源資料夾名稱；`check` 會標出對不到任何資料夾的 key。
- **演講**：每門課一個資料夾、附 `_meta.md`，每個條目以 H1 開頭、接一行 `Source: YYYY-MM-DD 講者《課名》· …`——見[格式第 4 節](docs/corpus-format.md#4-kind-lecture)。

先拿內附的範例試：`python server/corpus.py add examples/corpus-fixture/textbook --db ./fixture_db`。

**只用語料、沒有 Obsidian vault。** 其他都不必設：跳過 `indexer.py`，`VAULT_PATH` 不要設。最小設定就是 Step 0 的 venv、`ollama pull qwen3-embedding:0.6b`、`corpus.py add`，再把它印出的那一行放進 `.env`（只有這一行的新檔即可，或用 export）。照[上面](#vault-search-mcp--在-claude-code或任何-mcp-client中使用)用 venv 的 Python 註冊 MCP server；`textbook_search`／`lecture_search` 可以用，`vault_*` 工具會回「No vault configured — set VAULT_PATH …」。

---

## 選用 add-on

<details>
<summary><b>第二份參考語料（教科書／手冊）</b></summary>

長文參考資料（轉檔的 PDF、手冊、指引）用 parent-child 切塊索引，命中精準又帶上下文。用 [`corpus.py add`](#加入一份語料教科書指引演講) 建；它實際跑的是

```bash
cd server
python textbook_indexer_v3.py --corpus /path/to/corpus --db ~/.vault-search/textbook_db
python textbook_indexer_v3.py --corpus /path/to/corpus --db ~/.vault-search/textbook_db --audit
```

再在 `.env` 加上 `VAULT_SEARCH_TEXTBOOK_DB_PATH=~/.vault-search/textbook_db`。

`textbook_indexer_v3.py` 寫進自己的 DB（server 拿去做別的用途的 DB，例如 vault 的，它會拒寫），每個 chunk 記下語料根目錄、頁碼出處與表格單位，濾掉參考文獻，並建好混合檢索需要的全文索引。重跑只重新 embed 有變動的檔；`--book <資料夾>` 重建單本，`--audit` 逐章檢查有沒有漏。舊的 `textbook_indexer.py`（沒有全文索引、寫進 vault 的 DB）仍可用，但新功能不會再加在那裡。

會新增 `textbook_search` 工具（MCP + HTTP）。輸出會註明 `file` 路徑是相對於哪個語料根目錄；工具說明附上來源衝突時的優先順序：同一本書較新版次優先，不比教科書版次舊的指引優先於教科書，單一研究不推翻指引或教科書共識——要點出分歧並兩邊都引用。重排之後，指引與較新的來源會再加一點分（`VAULT_SEARCH_RERANK_GUIDELINE_PRIOR`、`VAULT_SEARCH_RERANK_YEAR_PRIOR`；設 0 即關閉）。可用 boost 檔讓你信任的來源排前面——見 `examples/source_boost.example.json` 與 `VAULT_SEARCH_SOURCE_BOOST`。

**書目表。** `books.json`（`VAULT_SEARCH_BOOKS_META`，範例見 `examples/books.example.json`）在查詢時併到教科書結果上：標成 `superseded`／`derivative` 的來源預設不出現（傳 `include_superseded` 才看得到）；`corpus_type` 可只搜指引；相似度打平時依 tier、年份排序。不必重建索引。

**混合檢索與重排。** `textbook_search` 預設把向量與全文檢索結果用 RRF 合併（`VAULT_SEARCH_TEXTBOOK_RETRIEVAL=hybrid`；需要 `textbook_indexer_v3.py` 建的全文索引，沒有就退回純向量）。有 CUDA GPU 的話可再加 cross-encoder 重排：把 `VAULT_SEARCH_RERANK_PYTHON` 指向一個裝了 CUDA 版 `torch`＋`transformers` 的 Python，每次搜尋先取 50 筆，用 `BAAI/bge-reranker-v2-m3` 重新打分後留前 n 筆。模型跑在獨立行程（`server/reranker_service.py`）：第一次查詢時啟動（幾秒），閒置 10 分鐘自動結束，別的工作要 GPU lease 時立刻結束——這段期間的搜尋就是不重排。作者自己 435 題的教科書評測上 MRR 約 +0.08，每次查詢約 0.5 秒（RTX 3070 Ti）；`similarity` 仍是 cosine，cross-encoder 分數另放在 `rerank_score`。

**演講語料。** `lecture` 種類的語料（[格式](docs/corpus-format.md#4-kind-lecture)）建進自己的 DB：`corpus.py add` 跑的是 `textbook_indexer_v3.py --corpus /path/to/lecture-corpus --db ~/.vault-search/lecture_db_v3`（`--corpus` 指向「子資料夾就是各門課」的那一層），並提示你把 `VAULT_SEARCH_LECTURE_DB_PATH` 指過去。之後就會多一個 `lecture_search` 工具，與 `textbook_search` 分開排名。每門課資料夾的 `_meta.md` 不會出現在結果裡；每筆結果帶 `date`、`speaker`、`course_title`，從該條目的 `Source:` 行解析（`Source: 2025-02-15 講者《課名》· …`），讀不出來的欄位留空，不猜。
</details>

<details>
<summary><b>Vault 索引 v3（筆記的 parent-child 切塊）</b></summary>

Step 0 的 `indexer.py` 是整篇筆記 embed。parent-child 索引比對小塊、回傳它所在的整段；`v3` profile 另外會正確讀 YAML frontmatter、跳過備份檔與範本資料夾、限制切塊長度，並寫入 `scope` 欄位。需要 `ollama pull qwen3-embedding:0.6b`，建在新的資料庫裡，所以切換前舊索引照常服務：

```bash
cd server
python vault_indexer_v2.py --profile v3 --db ~/.vault-search/lance_db_v3                 # 建立
python vault_indexer_v2.py --profile v3 --db ~/.vault-search/lance_db_v3 --incremental   # 之後：只處理有變動的筆記
```

讓 server 改用它：

1. `.env` 加 `VAULT_SEARCH_DB_PATH=~/.vault-search/lance_db_v3`
2. 建立旗標檔 `~/.vault-search/vault_v2.enabled`（`touch` 即可；刪掉就退回整篇筆記索引）
3. 重啟 `api_server.py`（以及任何 `mcp_server.py` session）。

如果你的教科書資料表原本放在 vault 的資料庫裡（沒設 `VAULT_SEARCH_TEXTBOOK_DB_PATH`），先把這個變數設成舊資料庫——它預設跟著 `VAULT_SEARCH_DB_PATH`。
</details>

<details>
<summary><b>搜尋範圍（讓部分資料夾不進搜尋）</b></summary>

用一份 JSON 對照表（`VAULT_SEARCH_SCOPE_MAP`，範例見 `examples/scope_map.example.json`）列出每個範圍包含哪些頂層資料夾。採 **fail-closed**：沒列在某範圍底下的資料夾就不屬於它，新資料夾不會悄悄混進有範圍限制的搜尋。`personal`＝所有沒被列到的資料夾，`all`＝不過濾；預設範圍寫在 server 端，呼叫端沒指定就用安全的預設值。適用於 `vault_search`、`vault_similar`、`vault_related` 與 chat 的 context。
</details>

<details>
<summary><b>評測工具</b></summary>

`eval/` 量檢索（hit@k、MRR、頁窗命中、舊來源污染率）與逐條 claim 的答案品質，A/B 決策用配對檢定（McNemar、cluster bootstrap）。見 [`eval/README.md`](eval/README.md)。真實題庫請放在 repo 之外。
</details>

<details>
<summary><b>知識圖譜（wikilink + 實體）</b></summary>

```bash
cd server
python graph_builder.py            # 解析 [[wikilink]] 成鄰接圖
python graph_builder.py --ner      # 選用：scispaCy/NER 實體抽取
```

存在時，wiki-link 圖會驅動 `ppr.py` 的 **Personalized PageRank** 檢索（Related Notes + 查詢擴展），搜尋結果也會被連結筆記與抽出的關係加以擴充。NER 步驟需要 `scispacy`（見 `requirements.txt`）。PPR 本身是純 Python，只需要 wiki-link 圖（Phase A），不需要 NER。
</details>

---

## 依你的環境調整（替代方案）

這專案是繞著**一台常開、有 GPU 的 PC + Claude 訂閱**打造的，但這些都不是必要。對照下表挑你的情況：

| 你沒有… | 改這樣做 |
|---|---|
| **GPU** | 不用做什麼——`bge-m3` 在 CPU 上就能 embed。超大 vault 的索引會慢些，但搜尋是即時的。想更輕量可用更小的模型（`nomic-embed-text`），設好 `VAULT_SEARCH_EMBEDDING_MODEL`（與 `_DIM`）再重建索引即可。 |
| **Claude 訂閱** | 對話改用 **Vault 模式**——完全跑在地端 Ollama 模型（`VAULT_SEARCH_CHAT_MODEL`），零成本。把 `CLAUDE_CMD=` 留空即可停用 Claude 後端模式。 |
| **常開的機器** | 開 Obsidian 時再啟動 `api_server.py` 就好（或包成登入啟動項／`systemd --user`／工作排程器）。搜尋只在你使用時需要 server。 |
| **第二台裝置要連** | 把 server 跑在家裡的 PC 或 NAS，把 plugin 的 **API Server URL** 指到那台主機，並設一組 **API key**（server 端 `VAULT_API_KEY` 環境變數 + plugin 設定填同一把 key），確保只有你能查詢。 |
| **完全不想跑 server** | 只用 **MCP server** 搭配終端機的 Claude Code——不用 FastAPI 程序、不用 plugin。 |

想用雲端 embedder 取代 Ollama？embedding 呼叫集中在 `indexer.py` / `api_server.py` 的 `client.embed(...)`；換成任何供應商、保留同一套 LanceDB 流程即可。

---

## 安全須知

- API server 支援 **API key**（`X-API-Key`）。在 server 端設 `VAULT_API_KEY`（環境變數）或 `~/.vault-search/api_key.txt`，並在 plugin 設定填入同一把 key。只要 server 不只在 `localhost` 可達，**務必**設一組。
- `.env`、`plugin/data.json`、整個 `~/.vault-search/` 資料夾都已被 git 忽略。切勿 commit 任何密鑰或索引。
- Hybrid／Free 對話以子程序呼叫 `claude` CLI，並帶**工具拒絕清單**（不能寫檔、不能執行 shell）——它只能讀你傳入的 context，並呼叫唯讀的文獻工具。

---

## 限制

- Embedding 需要本機執行 Ollama，沒有內建雲端 embedder（想換的話，embedding 呼叫都集中在 `indexer.py` / `api_server.py`）。
- Obsidian 外掛是連到本機 API server，搜尋時那個 server 必須在跑。
- 超大 vault 在純 CPU 上建索引很慢；索引建好之後搜尋很快。
- Vault 模式對話的品質取決於你選的本機模型；Hybrid／Free 對話需要 `claude` CLI。
- 它只搜尋你建過索引的內容，不負責取得或轉換教科書與演講（見上面的相關工具）。
- 為了相容，環境變數（`VAULT_SEARCH_*`）、MCP server／tool 名稱與 `~/.vault-search` 仍沿用舊名 vault-search。

---

## 專案結構

```
server/      indexer · scoring · ppr · api_server · mcp_server   （核心三套組）
             mcp_thin.py                                    （省記憶體的 MCP client）
             vault_indexer_v2 · vault_v3                    （筆記的 parent-child 索引、v3 profile）
             corpus.py                                      （檢查／加入語料資料夾）
             textbook_indexer_v3 · chunker_v3 · provenance · table_units   （語料 indexer）
             books_meta · lecture_meta · scope · reranker · reranker_service  （查詢端輔助）
             textbook_indexer.py                            （舊版語料 indexer，留給 v2 DB）
             graph_builder                                  （選用的知識圖譜）
             config.py                                      （所有設定，環境變數驅動）
             tests/                                         （pytest；不需索引、Ollama 或 GPU）
plugin/      main.js · manifest.json · styles.css           （Obsidian plugin）
eval/        檢索／回答評測工具
examples/    books · scope_map · source_boost 範例；corpus-fixture/（t2n-corpus/1 範例語料）
docs/        corpus-format.md（語料資料夾規格）· images/（截圖）
env.example  複製成 .env
```

Obsidian plugin 的版號寫在 `plugin/manifest.json`，只有 `plugin/` 裡真的有改動時才會動——刻意不跟著 repository 的 tag 走。那個數字一升高，Obsidian 就會對使用者顯示「有更新」，所以若讓它跟著只改 server 的版本一起升，等於通知使用者去更新一個對他們毫無改變的東西。

## 作者

由 **陳醫師（Dr. P.W. Chen）** 打造——一個只是想在自己幾千則筆記裡找得到東西的復健科醫師。
🌐 [drpwchen.com](https://drpwchen.com) · 🐙 [github.com/drpwchen](https://github.com/drpwchen)

## 授權

MIT — 見 [LICENSE](./LICENSE)。歡迎 PR 與 issue。

## Support 支持

覺得這個工具有幫助嗎？歡迎[請我喝飲料](https://drpwchen.com/support/) 🧋
If this tool helped you, you can [buy me a drink](https://drpwchen.com/en/support/).
