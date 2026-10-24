# Qgent

把论文从「一份 PDF」变成「一个可 SQL 查询、可语义检索、且每句话都能追到原文章节的数据库」。

上传 PDF，然后用自然语言提问。Agent 自己决定去查实验数据表、还是去检索作者的声明、还是去正文里找公式和实现细节，最后给出带原文出处的回答——**出处不是模型说的，是 Python 回查数据库验证过的**。

```
你 > Transformer (big) 在 newstest2014 EN-DE 上的 BLEU 是多少？

  [tool] execute_sql(SELECT DISTINCT exp_group_id, evidence_path FROM experiments)
  [tool] execute_sql(SELECT model_name, dataset, metric, value, is_ours, ... WHERE exp_group_id = '...')
  [tool] execute_sql(SELECT DISTINCT model_name, dataset, metric FROM experiments WHERE ...)
  [tool] execute_sql(...)

**Transformer (big) 在 newstest2014 EN-DE 上的 BLEU 为 28.4。**

| 模型 | 数据集 | 指标 | BLEU | 是否作者模型 |
|---|---|---|---|---|
| Transformer (big) | newstest2014 EN-DE | BLEU | **28.4** | ✅ 是 |

> [来源: Section: 5.4 Regularization, Block Index: 1]
```

---

## 它解决什么问题

| 痛点 | 现有工具做不到 | Qgent 的做法 |
|---|---|---|
| 最有价值的数值压在表格里，检索粒度太粗 | 全文检索只能匹配字符串，答不了「哪篇参数量最小」 | 把表格抽成 `experiments` 表，跨论文聚合退化成一条 SQL |
| LLM 直接读论文必然幻觉，而学术场景编错一个数字比查不到更糟 | chat-with-PDF 的回答没有结构化溯源 | `evidence_path` 由 Python 按论文结构生成，回答里的引用再被逐条回查 |
| 表格最难解析：跨行跨列、空单元格语义、LaTeX、单位方向符号 | 通用 RAG 对表格基本无能为力 | 每张表一次隔离的 LLM 调用 + 专门的空单元格规则报告 + Python 侧数值归约 |
| 横向对比是纯体力活 | 商业工具闭源、不可本地化 | `--multi` 模式下一次 JOIN 取同一指标，模型自己算差距 |

## 核心设计

**1. 双通道检索，而不是一把向量库**

数值问题走 SQL（精确、可聚合），主观问题走 `claims` 向量检索（按 `claim_type` 分类），细节问题走 `text_chunks` 向量检索（可按 `chunk_types` 收窄）。三种提问三条路径。

**2. 溯源可验证，不靠 prompt 承诺**

- `evidence_path` 形如 `Section: 5.4 Regularization, Block Index: 1`，由 Python 按 `content_tree` 的真实结构生成，**不采信模型回抄**
- `text_chunks.content` 始终是原文，**不是模型的回声**
- 回答里的每条 `Section: ..., Block Index: n` 被回查数据库真实存在的路径集合，对不上就在答案末尾追加「⚠️ 证据校验未通过」

**3. 隔离由代码保证**

`paper_id` 由 SQL guard 按括号深度注入到每条语句的**每个作用域**（子查询和 CTE 各自被过滤在自己那层）。单篇模式连 `papers` 表都不给读，连接本身是只读的，还带 15 秒语句超时。模型即使「忘记」加条件也逃不出去。

**4. 公式块旁路**

公式从不经过模型：`content` 是逐字复制的原始 LaTeX，标签强制为 `formula_latex`。既省调用又零幻觉。

**5. 没有 LangChain，也没有 Agent 框架**

工具调用循环、函数 schema、ReAct 观察处理全是手写的。因为有三件事必须由 Python 保证、不能交给框架抽象掉：SQL 隔离注入、引用真实性校验、工具结果的确定性整形。

---

## 架构

```
┌──────────────────── 写侧（离线批处理）────────────────────┐
│                                                          │
│  PDF ──▶ parse_pdf.py ──▶ parsed/<stem>/content_tree.json │
│         (mineru-kit)              │                       │
│                                   ▼                       │
│                 extract_pipeline.py --scan --write-db     │
│                 1 次 metadata + 每表 1 次 + 每 5 块 1 次    │
│                 公式块 0 次调用                            │
│                                   │                       │
└───────────────────────────────────┼───────────────────────┘
                                    ▼
                      PostgreSQL 16 + pgvector
                   papers / experiments / claims / text_chunks
                                    ▲
┌───────────────────────────────────┼───────────────────────┐
│                    读侧（在线问答） │                       │
│                                   │                       │
│  index.html ──HTTP──▶ main.py (FastAPI) ──▶ chat_agent.py │
│   vanilla JS           /api/chat            ReAct loop    │
│   PDF.js + marked      ChatSession (LRU 8)   ├ execute_sql│
│   + KaTeX                                     ├ search_…  │
│                                               └ search_…  │
└───────────────────────────────────────────────────────────┘
```

两条链路**不共享代码路径**，唯一的耦合是数据库 schema。写侧不知道 Agent 存在，读侧不知道 MinerU 存在。

---

## 快速开始

### 前置条件

- **Python 3.12**（开发环境为 3.12.10）
- **Docker**（跑 PostgreSQL + pgvector）
- 两个模型的 API key：
  - **chat 模型**（默认 `glm-5.2`，走 OpenAI 兼容端点）
  - **embedding 模型**（默认 `qwen3.7-text-embedding-flash`，走 DashScope）

> chat 与 embedding 是**两个不同的服务**，所以要两套 key。默认配置下 chat 走 Compshare，而 Compshare 不提供 embedding 模型。

### 1. 安装依赖

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

版本已按开发环境 pin（Python 3.12.10）。装完确认 `mineru-kit` 可执行文件在不在——`parse_pdf.py` 依赖它：

```bash
ls .venv/Scripts/ | grep mineru-kit
```

> ⚠️ 用**基础安装**的 `mineru`，不要改成 `mineru[all]`。`[all]` 等于 `[full]` = `[torch]` + vllm/lmdeploy，会多拉数 GB 的本地推理栈；本项目在 `standard` tier 下解析走默认的 llama-cpp 后端，用不到 torch。

### 2. 配置 `.env`

```bash
cp .env.example .env
```

然后填写：

```ini
POSTGRES_USER=qgent
POSTGRES_PASSWORD=<自己设一个>
POSTGRES_DB=qgent_db

DATABASE_URL=postgresql://qgent:<同一个密码>@localhost:5432/qgent_db
PGPASSWORD=<同一个密码>

# chat 模型：/v1 前缀是必须的，裸主机名对 /chat/completions 返回 404
LLM_BASE_URL=https://cp.compshare.cn/v1
LLM_API_KEY=...

# embedding 模型（DashScope）
DASHSCOPE_API_KEY=sk-...
```

`.env` 会被所有脚本自动加载（放在仓库根目录，脚本启动时读取）。**已导出的真实环境变量优先级更高**，不会被 `.env` 里的旧值覆盖。

### 3. 起数据库

```bash
docker compose up -d
```

首次启动时 `init.sql` 会自动建表建索引（含 HNSW 向量索引）。数据落在 `./pgdata`。

> `init.sql` 里 `claim_type` 的 CHECK 已经包含 `plan`。只有**在 `plan` 加入之前就已经建好的库**才需要执行一次 `migrate_claims_plan.sql`。`extract_pipeline.py --write-db` 会在花任何 API 调用之前检查这个约束，没放宽就直接拒绝并告诉你该跑哪个脚本。

### 4. 起 Web 界面

```bash
.venv/Scripts/python.exe main.py
```

打开 <http://127.0.0.1:8000>，把 PDF 拖进左侧上传区。解析与抽取在后台跑（侧边栏有进度卡和日志），完成后自动出现在文献库里。

### 5. 或者只用命令行

```bash
# 单篇精读（paper_id 见 Web 界面或 papers 表）
.venv/Scripts/python.exe chat_agent.py --paper-id 4

# 跨库检索与横向对比
.venv/Scripts/python.exe chat_agent.py --multi
```

> 以上是本地开发跑法。要长期运行、放服务器、配反向代理，看 [部署说明.md](部署说明.md)。

---

## 使用

### 问答 Agent（读侧）

```bash
python chat_agent.py --paper-id 4                  # 单篇精读，会话锁定在一篇论文
python chat_agent.py --multi                       # 跨库模式，可横向对比多篇
python chat_agent.py --paper-id 4 --top-k 5 --max-steps 8
python chat_agent.py --paper-id 4 --no-verify-evidence   # 关掉引用回查
python chat_agent.py --paper-id 4 --enable-thinking      # 默认关闭，见下方「已知坑」
```

会话内命令：`/help` `/clear` `/paper` `/exit`。

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--top-k` | 3 | 向量检索返回条数 |
| `--max-steps` | 8（多篇 12） | 每问的工具调用轮上限 |
| `--max-rows` | 100 | SQL 返回行上限 |
| `--history-rounds` | 5 | 保留的历史问答对 |
| `--no-verify-evidence` | 关 | 跳过回答后的引用回查 |

### 写侧流水线

```bash
# 阶段 1：PDF → content_tree.json（已存在的会 SKIP）
python parse_pdf.py --pdf-dir ./pdfs --out-dir ./parsed

# 预览：不发 API、不连库、不写文件，只看会发出去什么
python extract_pipeline.py parsed/1706.03762v7/content_tree.json --dry-run

# 抽取 + 打印结果（不写库）
python extract_pipeline.py parsed/1706.03762v7/content_tree.json --print-rows 0

# 抽取并写库（单事务）
python extract_pipeline.py parsed/1706.03762v7/content_tree.json --write-db

# 批量：扫描 parsed/ 下所有论文，跳过库里已有的
python extract_pipeline.py --scan --write-db

# 断点续跑：按 evidence_path 跳过已入库的表/块/公式，只补缺
python extract_pipeline.py --scan --write-db --resume

# 重新抽取：删掉同标题论文（级联删子行）后重导
python extract_pipeline.py parsed/1706.03762v7/content_tree.json --write-db --replace
```

写库三种模式语义严格互斥：

| 模式 | 行为 | 用在什么时候 |
|---|---|---|
| 默认 | 同标题论文已存在 → 报错中止 | 防止意外重复导入 |
| `--replace` | 删除同标题论文后重导 | 重新抽取 |
| `--resume` | 复用已有 paper 行，按 `evidence_path` 只补缺 | 网络中断后接着跑 |

`--replace` 优先于 `--resume`。

常用调参：`--batch-size`（文本块/批，默认 5）、`--max-chars`（单表字符上限，默认 14000）、`--text-max-chars`（单块上限，默认 6000）、`--llm-sleep`（批间隔，默认 0.5 s）、`--stages paper experiments text`（只跑部分阶段）、`--no-embed`（跳过嵌入）。

---

## 数据模型

四张表（`init.sql`）：

| 表 | 关键字段 |
|---|---|
| `papers` | `title` / `authors` / `abstract` / `domain` / `method_tags` / `content_tree` (JSONB，整篇解析结果) |
| `experiments` | `exp_group_id`(UUID，一张表一个) / `task` / `dataset` / `model_name` / `metric` / `value` / `hyperparams`(JSONB) / **`is_ours`** / `domain_specific` / `evidence_path` |
| `claims` | `claim_type`(contribution / limitation / assumption / position / plan) / `content` / `embedding` VECTOR(1024) / `evidence_path` |
| `text_chunks` | `sec_title` / `content` / `embedding` VECTOR(1024) / `chunk_types` TEXT[] / `evidence_path` |

**`evidence_path` 是整个系统的幂等键与溯源键**：格式固定为 `Section: <章节名>, Block Index: <该章节内第 n 个块>`，`Block Index` 统计该章节内**所有**块（含表和图），所以表阶段与文本阶段的口径一致。resume 去重、引用校验、双写一致性三处都用它。

**`is_ours`** 是实验表里最重要的字段：`true` 是作者自己提出的模型，`false` 是 baseline。问「这个对比公平吗」时模型靠它区分主角和配角。

`chunk_types` 词表在 `chunk_types_vocab.json` 里，**运行时会自增长**：模型遇到词表里没有的内容类型时，符合命名规范（英文小写下划线）的新词会被追加进去，下一批就能用上。当前 12 个标签 + 57 条别名（`code_link → source_code` 这类）。

---

## 项目结构

| 文件 | 行数 | 职责 |
|---|---|---|
| `chat_agent.py` | 2010 | **读侧 Agent**：SQL guard、三个工具、ReAct 循环、引用校验。框架无关 |
| `extract_pipeline.py` | 4856 | **写侧一体化抽取**：metadata + 表 + 文本 + 公式 → DB |
| `main.py` | 582 | FastAPI Web 层：文献库、上传、子进程编排、会话缓存、chat 端点 |
| `index.html` | 2416 | 单页前端：PDF.js 虚拟化阅读 + Markdown/KaTeX 渲染 + 引用徽章 |
| `build_content_tree.py` | 288 | MinerU Markdown → `content_tree.json`（正则逐行状态机） |
| `parse_pdf.py` | 167 | 调用 MinerU（`mineru-kit parse -p all`） |
| `test_pipeline_e2e.py` | 871 | 端到端回归测试（stub 掉 LLM，真数据库，14 个场景） |
| `ab_test_row_coverage.py` | 96 | A/B 实验：验证「行覆盖」提示段是否真的抑制丢行 |
| `init.sql` | 56 | schema + 索引 |
| `migrate_claims_plan.sql` | 29 | 放宽 `claim_type` 以接纳 `plan`（仅旧库需要） |
| `requirements.txt` | 9 个包 | 直接依赖 + 版本 pin（含每条的用途说明） |
| `docker-compose.yml` | 18 | PostgreSQL 16 + pgvector |

`extract_to_db.py` / `save_to_db.py` / `extract_text_chunks.py` / `embed_claims.py` 是**第一代拆分式实现**，已被 `extract_pipeline.py` 取代但保留着。其中 `embed_claims.py` 仍是唯一能独立回填历史 `claims.embedding IS NULL` 行的工具。

---

## 测试

```bash
# 需要 .env 里的 DATABASE_URL 指向一个可用的 PostgreSQL
.venv/Scripts/python.exe test_pipeline_e2e.py
```

这个测试**不需要 LLM key**：它把 `LLMClient` 和 `embed_all` 猴子补丁掉，返回固定形状的假响应和常量向量，所以零 API 调用、零费用，但会真的建一个 `pipeline_check` schema 并跑完整个写库路径。

覆盖 14 个场景：三种写库模式、`--resume` 系列（完整 no-op / 挖洞后精确恢复 / 与 `--replace` 的优先级）、`--scan` 批量、公式旁路不变量、块数守恒、标签归一与词表自增长、LaTeX 破坏 JSON 后的修复、`1.2K steps` 类脏值归约、Win/Lose/Tie 数值化、单表/单批失败隔离、全部失败时 exit 3 且零写入。

必须在**仓库根目录**运行（依赖相对路径的 `init.sql` / `content_tree.json`），并且数据库里要有名为 **`qgent` 的角色**——测试脚本会 `GRANT ALL ON SCHEMA pipeline_check TO qgent`，角色名是硬编码的。如果你的 `POSTGRES_USER` 不是 `qgent`，改一下 `test_pipeline_e2e.py:262` 或建一个同名角色。

> 注意：`chat_agent.py` 目前**没有测试覆盖**。SQL guard 是安全边界，这是当前风险最集中的地方。

---

## 已知坑

**MinerU 必须用 `mineru-kit parse`，不能用 `mineru parse`。**

两个原因：`mineru parse` 的 `-p` 是**页码范围**而不是输入路径，不传 `-p all` 会静默只取前 10 页；而且它只输出 `![Image block]` 占位、**丢掉所有图**，只有 `mineru-kit` 会把图片内联成 base64。`parse_pdf.py` 已经封装好了这一点。

**`enable_thinking` 默认关闭是硬约束，不是偏好。** 混合推理模型在**非流式** tool call 下不开思考——读侧用的是非流式，所以打开就会失败。

**`LLM_BASE_URL` 的 `/v1` 前缀不能省。** 裸主机名对 `/chat/completions` 返回 404。

**相似度地板 `0.35` 是标定值。** 它是在实测分布的空隙里取的（真实命中落在 0.45–0.55，论文没写这个主题的问题最高约 0.33），所以它**绑定当前的嵌入模型**——换 embedding 模型必须重新标定，否则「未找到相关信息」要么失效、要么频繁误报。

**写侧串行，一篇论文十几分钟。** 实测每次 LLM 调用 25–35 秒，单篇（Attention 规模）约 11 分钟，20 篇约 3.8 小时，95% 是串行等待。表与表、批与批之间没有依赖，是最好摘的优化果子。

**并发上传会重复解析。** `parse_pdf.py` 扫描整个 `./pdfs` 目录，而 Web 上传不传具体文件；两个上传同时进行时，同一篇 PDF 会被两个 MinerU 实例各解析一遍，还互相 CPU 争用（实测单篇从 2 分 41 秒涨到 24–30 分钟）。串行上传没这个问题。

---

## 延伸阅读

- [部署说明.md](部署说明.md) —— 长期运行、systemd / nginx 配置、备份与迁移、公网前的安全清单、故障排查
- [项目分析报告.md](项目分析报告.md) —— 完整的架构、Agent 链路、异常处理与成本性能分析，含实测数据、SQL guard 的五步拆解、10 项优化方向与 11 条技术债
