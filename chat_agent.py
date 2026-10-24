#!/usr/bin/env python3
"""Interactive Text-to-SQL + RAG agent for one paper, built on the raw APIs.

The agent answers questions about a single paper by calling three tools: it
queries `experiments` for numbers, semantically searches `claims` for the
authors' own statements, and semantically searches `text_chunks` for the
prose, formulas and code of the body.  There is no LangChain and no agent
framework: the tool-calling loop, the function schemas and the ReAct
observation handling are all implemented here.

Paper isolation is enforced in Python, never by the model.  `--paper-id`
scopes the whole session: every SQL statement gets `paper_id = <ID>` spliced
into its WHERE clause before it reaches PostgreSQL, and both vector searches
carry the filter in their own WHERE.  The model is told not to write the
condition itself, and cannot escape it by forgetting to.

`--multi` is the other mode: the session is scoped to the whole library rather
than to one paper, and `paper_id=None` is what says so.  It changes exactly
three things.  The guard stops splicing a predicate in, because there is no
longer one paper to splice; the vector searches stop filtering and return
`paper_id` and `paper_title` with every hit, because a hit the model cannot
attribute is a hit it cannot cite; and the prompt describes the catalogue
instead of one paper, so the model knows which ids exist before it cites one.
The guard keeps the part that matters either way: a single read-only statement
that must read one of the tables this session is allowed to see.

Citations gain a paper dimension in that mode.  The answer has to name a
`paper_id` alongside the section, and the check below verifies each citation
against that paper's own rows -- a citation naming an id the database does not
have is as fabricated as one naming a section that does not exist.

The SQL guard is deliberately conservative, because the model writes the SQL
and a mistake there is a correctness problem, not just a bad answer:

  * only a single read-only statement is accepted, and it must start with
    SELECT or WITH; comments and string literals are masked before the
    keyword scan so a quoted word cannot trip it;
  * UNION/INTERSECT/EXCEPT and multiple statements are rejected, which keeps
    the statement to one top-level query and one injectable WHERE per scope;
  * `paper_id = <ID>` is injected into every scope that reads one of the three
    tables -- a table inside a subquery or a CTE gets its own predicate, at
    its own paren depth, rather than only the outermost query;
  * a query that reads none of the three tables is rejected, and in single mode
    any reference to `papers` is rejected outright, so the model cannot read
    another paper's rows even by joining to the catalogue;
  * the connection itself is opened read-only with a statement timeout, so
    even a guard bypass cannot write.

Fabricated sourcing is treated as a failure, not a style problem.  The system
prompt forbids answering from the model's own knowledge and forbids inventing
an evidence_path, but a prompt is not an enforcement mechanism: every answer
is also checked in Python.  Each "Section: ..., Block Index: n" it cites is
matched against the evidence paths the paper actually stores, and a citation
that matches no row is shown to the reader as unverified rather than passed
off as a source.  A section-only citation is accepted when the section exists,
by prefix, since a title may contain a comma the citation splits at.

Chat runs against the OpenAI-compatible `/chat/completions` endpoint of the
configured host over plain `requests`, which is what carries the `tools` /
`tool_calls` protocol.  Embeddings still go to DashScope over the dashscope
SDK, matching the rest of the pipeline, because that host does not serve the
embedding model.  `enable_thinking` is off by default: hybrid-reasoning models
do not serve non-streaming tool calls with thinking on.

Exit codes:
    0  the session ended normally (exit, quit, EOF)
    1  fatal error (missing API key, database failure)
    2  missing dependency or bad arguments

Usage:
    python chat_agent.py --paper-id 4
    python chat_agent.py --paper-id 4 --top-k 5 --max-steps 8
    python chat_agent.py --paper-id 4 --enable-thinking
    python chat_agent.py --paper-id 4 --no-verify-evidence
    python chat_agent.py --multi
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple, Sequence
from uuid import UUID

try:
    import dashscope
    import psycopg2
    import psycopg2.extras
    import requests
except ImportError as exc:  # reported with a readable message in main()
    dashscope = None  # type: ignore[assignment]
    psycopg2 = None  # type: ignore[assignment]
    requests = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the .env beside this script.

    Real environment variables win, so an exported DASHSCOPE_API_KEY is never
    overridden by a stale .env entry.
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL = "https://cp.compshare.cn/v1"
DEFAULT_LLM_MODEL = "glm-5.2"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
DEFAULT_DIMENSION = 1024

# No password here: DATABASE_URL (from the environment or .env) is used when
# set, and libpq's PG* variables cover everything else.
DEFAULT_DSN = "postgresql://qgent@localhost:5432/qgent_db"

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_STEPS = 8
# A cross-paper question costs more calls than a single-paper one: the model
# has to find which papers are relevant before it can compare them.
DEFAULT_MULTI_MAX_STEPS = 12
DEFAULT_TOP_K = 3
DEFAULT_SQL_ROWS = 100
DEFAULT_CELL_CHARS = 1_200
DEFAULT_HISTORY_ROUNDS = 5
DEFAULT_EMBED_MAX_CHARS = 8_000
# Titles written into the multi-mode prompt.  Past this the catalogue would
# crowd out the tools' own results, and the model can still query `papers`.
DEFAULT_LIBRARY_PROMPT_LIMIT = 120
# A title has to be this long before finding it verbatim in an answer counts as
# a citation of that paper; short titles occur in prose by accident.
MIN_TITLE_MATCH_CHARS = 12
EMBED_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
STATEMENT_TIMEOUT_MS = 15_000

# Cosine similarity below which a vector hit is treated as unrelated.  The
# agent is told to read the score rather than trust the ranking: three
# neighbours are always returned, even when the paper says nothing about the
# question, and "未找到相关信息" is only reachable if the model can tell.  The
# value was calibrated on the stored papers -- genuine hits land at 0.45-0.55
# and a question the paper does not answer tops out around 0.33, so the floor
# sits in the gap rather than at a round number.
SIMILARITY_FLOOR = 0.35

TARGET_TABLES = ("experiments", "claims", "text_chunks")
# The catalogue is readable only in multi mode: a single-paper session has no
# business walking it, and the guard rejects a query that reads it.
CATALOGUE_TABLE = "papers"
MULTI_TABLES = (CATALOGUE_TABLE, *TARGET_TABLES)

SYSTEM_PROMPT_TEMPLATE = """角色设定

你是一个顶级的 AI 学术评审员（AI Reviewer）和科研数据挖掘专家。你的任务是根据用户的提问，精准调用工具（查数据库或语义检索），对{SCOPE_DESC}进行极其专业、严谨、深度的解答。

核心红线（防幻觉）

1. 绝对禁止凭空猜测论文的数值、图表、结论或观点。所有的事实性回答，必须先调用工具（SQL 或 向量检索）。
   —— 这一条没有例外：即使你认为自己已经知道答案（例如一篇著名论文的标准公式、常见架构的超参数），也必须先检索论文的原文与数据库记录，并以其为准。你记住的版本可能与该论文的写法、符号或数值不同。
2. 如果工具没有查到，你必须如实回答：“根据当前数据库记录，未找到相关信息”，绝对不能瞎编。
3. 如果用户的提问极其模糊（如“这篇论文怎么样？”），请利用你的前置知识，结合当前检索范围内已知的论文标题，反问用户具体的意图（是想看实验结果、局限性、还是网络架构？）。

【反捏造与溯源铁律】（最高优先级，违反即属严重错误）

1. 严禁使用你的前置训练知识来补充论文内容！你对这些论文的所有认知，必须且只能来源于数据库查询结果。你对论文的“印象”“记忆”“常识”一律不是证据，哪怕你非常确信。
2. 禁止伪造 evidence_path：你输出的每一句话，都必须对应工具返回的真实 JSON 数据。你只能原样引用工具返回结果中真实出现过的 content 和 evidence_path；如果你没有查到确切的 content 和 evidence_path，就绝对不能凭空编造一个。编造出来的章节号、Block Index、引文都属于伪造溯源，是本次任务中最严重的错误。
3. 关于 claim_type 的严格限制：如果用户问“局限性（limitation）”，你必须且只能执行 SELECT content, evidence_path FROM claims WHERE claim_type = 'limitation'（或带向量的精确查询）。如果查询结果为空，你必须直接回答：“根据数据库记录，作者并未在论文中明确声明局限性（limitation）。”绝不允许把 position、plan 或你脑子里的常识伪装成 limitation 喂给用户！
   —— 同一个限制适用于其他 claim_type：问贡献就查 'contribution'，问未来计划就查 'plan'，问立场就查 'position'，问假设就查 'assumption'。查到什么就只说什么，没查到就只能说没查到。
4. 如果工具确实返回了间接材料（例如 plan 声明中提到了未来要做的改进），你可以把它作为“相关信息”呈现，但必须明确标注它的真实类型（例如“作者在 plan 中提到……”），绝不能改写成 limitation 或 contribution。

数据库字典 (PostgreSQL)

你掌握下面这些表。表结构及字段含义如下：

{PAPERS_TABLE_DOC}1. experiments 表（用于精确数值对比）：
   · task (TEXT): 任务名称，如 "Machine Translation", "Image Classification"。
   · dataset (TEXT): 数据集名称，如 "newstest2014 EN-DE", "ImageNet"。
   · model_name (TEXT): 参与对比的模型名称，如 "Transformer (big)", "ResNet-50"。
   · metric (TEXT): 评估指标，如 "BLEU", "F1", "FID", "params*10^6" (参数量), "Training Cost (FLOPs)"。
   · value (FLOAT): 指标对应的具体数值。
   · hyperparams (JSONB): 实验配置参数，如 {"batch_size": 4096, "learning_rate": 0.1, "N": 6}。
   · is_ours (BOOLEAN): 极其重要。true 代表这是作者自己提出的模型（主角），false 代表是 baseline 对比模型（配角）。
   · domain_specific (JSONB): 特殊领域的额外配置或条件，如 {"resolution": "224x224"}。
   · evidence_path (JSONB): 数据溯源路径。
2. claims 表（用于查找作者观点）：
   · claim_type (TEXT): 只能是 'contribution'(贡献), 'limitation'(局限), 'assumption'(假设), 'position'(立场), 'plan'(未来计划)。
   · content (TEXT): 声明的具体内容。
   · evidence_path (JSONB): 溯源路径。
3. text_chunks 表（用于查找正文细节、公式、代码）：
   · sec_title (TEXT): 所在的章节名。
   · content (TEXT): 正文原文段落。
   · chunk_types (TEXT[]): 内容标签。如 'implementation_detail', 'source_code', 'formula_latex', 'experiment_setup'。
   · evidence_path (JSONB): 溯源路径。

可用工具 (Tools)

1. execute_sql(sql: str): 接收合法的 PostgreSQL 语句并返回结果。你可以使用 JOIN、GROUP BY 等复杂查询。
2. search_claims(query: str): 接收中文，通过向量语义在 claims 表查找。
3. search_chunks(query: str, chunk_types: list[str] | None = None): 接收中文，通过向量语义在 text_chunks 表查找。chunk_types 可选，用于按内容类型收窄范围（如 ["source_code"]、["formula_latex"]）；不传则全库检索。

通用科研任务拆解与执行协议

本协议规定你思考和调用工具的顺序。每一步都要真正执行，不要在回答里复述这个拆解过程。
唯一可以不调用工具的情形是【第一步】判定为模糊提问、需要向用户反问澄清时；除此之外，任何事实性回答都至少要有一次工具调用作为依据。

【第一步：意图识别与书面化】

1. 接收用户输入后，先在内部完成意图识别，把问题转写成专业的学术书面语再去检索。但绝对不要擅自改写用户提到的专有名词（模型名、数据集名、指标名、章节名等可能是特定代号，必须原样保留）。
2. 判断问题属于哪一类：
   A. 找实验数值 —— 分数、参数量、超参数配置、训练成本、排名、性价比。
   B. 找主观立场 —— 局限性、贡献、假设、作者立场、未来计划。
   C. 找客观细节 —— 公式推导、代码链接、架构描述、实现配方。
   复合问题按【第三步】处理。

【第二步：按意图执行探索协议】

▶ 意图 A（找实验数值 / 超参数）：

1. 宏观扫描（意图 A 的固定起点，必须执行，不得跳过）：先执行 SQL 了解有哪些实验组及其章节归属：
   {INTENT_A_SQL}
   原因：{INTENT_A_REASON}做完这一步再决定往哪查。
2. 排序试探：根据上一步返回的章节名（如 Model Variations、Ablation、Machine Translation 等），对这些实验组“可能包含目标数值”的可能性排序，然后按顺序用 SQL 查询具体的 model_name、metric、value、hyperparams。查询时结合 metric 与 is_ours 过滤；模型名不确定时先用 SELECT DISTINCT model_name 确认库里实际的写法（括号、缩写、[引用编号] 都可能不同）。
3. 降级兜底：如果遍历完所有实验组仍然找不到该模型，或者关键字段（如 hyperparams）全部为空，立刻降级：调用 search_chunks，用英文关键词去正文检索配置（如 "big model hyperparameters"、"training settings"）。如果连正文都没有，如实报告未找到。

▶ 意图 B（找主观立场）：

1. 直接调用 search_claims，传入第一步书面化后的查询词。
2. 如果返回空，或返回结果的 relevant 全为 false，说明数据库里没有提取到这类声明。此时按提问类型使用规定措辞：
   · 问局限性的，回答：“根据数据库记录，作者并未在论文中明确声明局限性（limitation）。”
   · 其他类型的，回答：“数据库中未提取到相关的声明记录。”
   严禁用前置知识瞎编。
3. 如果查到了 'plan' 类型，可以在此基础上多给用户一些后续研究方向的建议，但必须与作者原话明确区分开。

▶ 意图 C（找客观细节）：

1. 首先调用 search_chunks，并尽量用 chunk_types 收窄范围：
   · 找公式、推导：chunk_types = ["formula_latex"]（该论文没有此标签时改用 "formula_context"）
   · 找代码、开源链接：chunk_types = ["source_code"]
   · 找实现配方、实验设置：chunk_types = ["implementation_detail", "experiment_setup"]
   可用的类型以下方【本次会话的固定上下文】中列出的 chunk_types 为准，不要臆造标签。
2. 如果搜到了，提取原话和 evidence_path。
3. 如果带标签搜索没搜到（返回空或全不相关），去掉 chunk_types 限制再全库语义搜索一次。

【第三步：跨表交叉验证（复合问题）】

如果用户问的是复合问题（如“它的分数这么高，有什么局限？”、“这个对比公平吗？”），你必须按顺序调用 execute_sql 和 search_claims（必要时再加 search_chunks），拿到多方结果后再分主次、结构清晰地回答。
· 问“对比是否公平”时：先用 execute_sql 查出双方的 params*10^6、Training Cost 与 hyperparams，自己算出差异（例如几倍差距），由你给出“不公平”或“性价比低”的客观结论，而不是转述作者的措辞。
· 复合问题的回答要分层：先回答客观数据部分，再回答主观判断部分。
{CROSS_PAPER_RULE}
【第四步：回答的严谨性规范】

1. 必须遵循“先结论，后证据”：开头直接给出判断或数值，随后再列证据。
2. 如果实验表里没找到配置、但在正文里找到了，必须明确说明：“主实验表未记录，但从正文（Section X）中提取到配置如下……”，不要把两个来源混为一谈。
3. 多行对比（>=2 行）必须用标准 Markdown 表格输出，以便前端渲染。
4. 引用原话时用引用格式，并在末尾附上 evidence_path，例如：{CITATION_FORMAT}
{CITATION_RULE}
---

【本次会话的固定上下文】

{CONTEXT_BLOCK}

【工具调用补充约定】

{CONVENTIONS_BLOCK}
"""


TOOL_BUDGET_NOTICE = """【系统提示】本次提问的工具调用次数已经用完，接下来不会再执行任何工具调用。

请立即基于上面已经获得的工具结果，直接输出最终回答，不要再描述你打算查询什么。要求：
- 先给结论，再给证据；
- 现有证据不足以回答的部分，如实说明“根据当前数据库记录，未找到相关信息”，绝对不要凭印象补充；
- 如果某类声明在数据库中确实不存在（例如 claims 表中没有任何 limitation 记录），请明确指出这一点。
"""


def log(msg: str) -> None:
    """Progress goes to stderr so stdout carries the answers alone."""
    print(msg, file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def configure_stdio() -> None:
    """Windows consoles default to GBK; paper text and model output are not."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# SQL guard: paper_id injection and read-only enforcement
# --------------------------------------------------------------------------- #

# Every keyword that turns a statement into something other than a read.
# Scanned on the masked text, so a quoted word never matches.
FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "grant",
    "revoke", "truncate", "copy", "vacuum", "analyze", "reindex", "cluster",
    "comment", "refresh", "listen", "notify", "unlisten", "lock", "call",
    "do", "merge", "prepare", "deallocate", "declare", "discard",
    "checkpoint", "begin", "commit", "rollback", "savepoint", "start",
    "set", "reset", "into", "returning", "union", "intersect", "except",
    "tablesample", "lateral",
)
FORBIDDEN_RE = re.compile(
    r"\b(?:" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
)
STATEMENT_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

def _table_ref_re(tables: Sequence[str]) -> re.Pattern[str]:
    """A regex matching a FROM/JOIN reference to one of `tables`.

    The table name is matched as a whole word and the optional alias group
    stops at the words that can follow a table without being one, so "FROM
    claims WHERE ..." does not read WHERE as an alias.
    """
    return re.compile(
        r"(?:\b(?:from|join)\s+|(?<=[A-Za-z0-9_)\"])\s*,\s*)"
        r"(?:only\s+)?(?:[A-Za-z_][A-Za-z0-9_$]*\s*\.\s*)?"
        r"(?P<table>" + "|".join(tables) + r")\b"
        r"(?:\s+(?:as\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_$]*))?",
        re.IGNORECASE,
    )


# Single mode splices a predicate into every scope that reads one of these;
# multi mode only uses its regex to check that something readable was read.
TABLE_REF_RE = _table_ref_re(TARGET_TABLES)
MULTI_TABLE_REF_RE = _table_ref_re(MULTI_TABLES)
CATALOGUE_REF_RE = _table_ref_re((CATALOGUE_TABLE,))

# Words that follow a table name without being its alias.  "FROM claims WHERE
# ..." must not read "WHERE" as the alias of `claims`.
ALIAS_STOPWORDS = {
    "where", "group", "order", "limit", "offset", "having", "window", "fetch",
    "for", "on", "using", "join", "left", "right", "inner", "outer", "full",
    "cross", "natural", "union", "intersect", "except", "as", "select",
    "from", "and", "or", "not", "is", "null", "true", "false", "when", "then",
    "else", "end", "with",
}

WHERE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)
CLAUSE_RE = re.compile(
    r"\b(?:group\s+by|order\s+by|limit|offset|having|window|fetch|"
    r"for\s+(?:update|share|no\s+key\s+update|key\s+share))\b",
    re.IGNORECASE,
)
TRAILING_SEMICOLON_RE = re.compile(r";\s*$")


class SQLGuardError(ValueError):
    """The statement cannot be scoped to the paper, so it is not run."""


def mask_sql(sql: str) -> str:
    """Blank out comments and string literals, keeping every offset intact.

    The result has the same length as the input, so an index found in the mask
    addresses the same character in the original.  Single-quoted literals keep
    their SQL doubling rule (''); a quoted word therefore cannot be mistaken
    for a keyword, and a comment cannot hide one.
    """
    chars = list(sql)
    length = len(sql)
    index = 0

    while index < length:
        char = sql[index]

        if char == "'":
            escaped = (
                index > 0
                and sql[index - 1] in "Ee"
                and (index < 2 or not (sql[index - 2].isalnum() or sql[index - 2] == "_"))
            )
            end = index + 1
            while end < length:
                if escaped and sql[end] == "\\":
                    end += 2
                    continue
                if sql[end] == "'":
                    if end + 1 < length and sql[end + 1] == "'":
                        end += 2
                        continue
                    break
                end += 1
            for position in range(index, min(end + 1, length)):
                chars[position] = " "
            index = end + 1
        elif char == "-" and sql.startswith("--", index):
            end = sql.find("\n", index)
            end = length if end == -1 else end
            for position in range(index, end):
                chars[position] = " "
            index = end
        elif char == "/" and sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            end = length if end == -1 else end + 2
            for position in range(index, min(end, length)):
                chars[position] = " "
            index = end
        else:
            index += 1

    return "".join(chars)


def paren_pairs(masked: str) -> dict[int, int]:
    """Map every '(' to the index of its ')'.  Raises on unbalanced parens."""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, char in enumerate(masked):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise SQLGuardError("unbalanced ')' in the statement")
            pairs[stack.pop()] = index
    if stack:
        raise SQLGuardError("unbalanced '(' in the statement")
    return pairs


def depth_map(masked: str) -> list[int]:
    """Paren depth of every character; a '(' counts as its outer depth."""
    depths = [0] * len(masked)
    depth = 0
    for index, char in enumerate(masked):
        if char == ")":
            depth -= 1
        depths[index] = depth
        if char == "(":
            depth += 1
    return depths


def _scope_end(
    masked: str, depths: list[int], pairs: dict[int, int], position: int, end: int
) -> int:
    """Where the SELECT owning `position` stops.

    At depth 0 that is the end of the statement; inside parentheses it is the
    ')' that closes the innermost enclosing '('.  A '(' before the position
    that is already closed cannot be the enclosing one, so only the last open
    paren before it is considered.
    """
    if depths[position] == 0:
        return end
    opens = [
        open_index
        for open_index in pairs
        if open_index < position and depths[open_index] == depths[position] - 1
    ]
    if not opens:
        return end
    return pairs[max(opens)]


def _first_keyword_at_depth(
    pattern: re.Pattern[str],
    masked: str,
    depths: list[int],
    start: int,
    end: int,
    depth: int,
) -> int | None:
    for match in pattern.finditer(masked, start, end):
        if depths[match.start()] == depth:
            return match.start()
    return None


def guard_sql(sql: str, paper_id: int | None) -> str:
    """Return `sql` scoped to `paper_id`, or raise SQLGuardError.

    One predicate per scope that reads a target table, inserted into that
    scope's own WHERE clause (or added just before its GROUP BY / ORDER BY /
    LIMIT, or at the end of the scope when it has none).  A table reached
    through a subquery or a CTE is filtered inside that subquery, which is the
    only place the predicate would still mean anything.

    `paper_id=None` is multi mode: there is no paper to scope to, so the
    statement is passed through unchanged.  What is left of the guard is the
    part that does not depend on the scope -- one read-only statement, and it
    must read a table this session may see.  `papers` joins that set here and
    only here, because a cross-paper answer needs the catalogue to name the
    papers it compared.
    """
    if not sql or not sql.strip():
        raise SQLGuardError("the SQL statement is empty")

    masked = mask_sql(sql)

    statement_end_match = TRAILING_SEMICOLON_RE.search(masked)
    end = statement_end_match.start() if statement_end_match else len(masked)
    if ";" in masked[:end]:
        raise SQLGuardError(
            "only one statement per call is allowed (a ';' was found before the end)"
        )

    body = masked[:end]
    if not STATEMENT_START_RE.match(body):
        raise SQLGuardError(
            "only a read-only SELECT statement is allowed; it must start with "
            "SELECT or WITH"
        )

    forbidden = FORBIDDEN_RE.search(body)
    if forbidden:
        raise SQLGuardError(
            f"'{forbidden.group(0).upper()}' is not allowed; only a read-only "
            f"SELECT over {' / '.join(MULTI_TABLES)} is accepted"
        )

    if paper_id is None:
        if not MULTI_TABLE_REF_RE.search(body):
            raise SQLGuardError(
                "the query reads none of the tables this session may see; query "
                + ", ".join(MULTI_TABLES)
            )
        return sql

    # `papers` is readable only in multi mode.  Checking for it here rather
    # than relying on the injection loop is what makes that true: the loop
    # only filters the three target tables, so a JOIN to the catalogue would
    # otherwise come back unfiltered -- and `papers` holds the whole
    # content_tree of every paper in the library.
    if CATALOGUE_REF_RE.search(body):
        raise SQLGuardError(
            f"the {CATALOGUE_TABLE} table is not readable in a single-paper "
            f"session; the paper's title is already in your context"
        )

    pairs = paren_pairs(masked[:end])
    depths = depth_map(masked[:end])

    # Keyed by (insert position, paren depth, how to insert); every table read
    # in the same scope lands on the same key, so one clause covers them all.
    insertions: dict[tuple[int, int, str], list[str]] = {}
    for match in TABLE_REF_RE.finditer(masked[:end]):
        alias = match.group("alias")
        if alias and alias.lower() in ALIAS_STOPWORDS:
            alias = None
        reference = alias or match.group("table")
        predicate = f"{reference}.paper_id = {paper_id}"

        position = match.start()
        depth = depths[position]
        scope_end = _scope_end(masked, depths, pairs, position, end)
        # Search from the end of the table name, not from the end of the match:
        # the alias group greedily swallows the next keyword ("FROM claims
        # WHERE ..." reads "WHERE" as an alias), and starting after it would
        # skip the very clause the predicate belongs in.
        search_from = match.end("table")

        where_at = _first_keyword_at_depth(
            WHERE_RE, masked, depths, search_from, scope_end, depth
        )
        if where_at is not None:
            # Just after the scope's own WHERE: extend that clause.
            key = (where_at + len("where"), depth, "extend")
        else:
            clause_at = _first_keyword_at_depth(
                CLAUSE_RE, masked, depths, search_from, scope_end, depth
            )
            if clause_at is not None:
                # Before GROUP BY / ORDER BY / LIMIT: a new WHERE clause.
                key = (clause_at, depth, "prepend")
            else:
                # End of the scope: the end of the statement, or just inside
                # the ')' that closes the enclosing subquery.
                key = (scope_end, depth, "append")
        insertions.setdefault(key, []).append(predicate)

    if not insertions:
        raise SQLGuardError(
            "the query reads none of the tables this session may see; query "
            "experiments, claims or text_chunks"
        )

    guarded = sql
    for (position, _depth, style), predicates in sorted(
        insertions.items(), reverse=True
    ):
        joined = " AND ".join(dict.fromkeys(predicates))
        if style == "extend":
            text = f" ({joined}) AND "
        elif style == "prepend":
            text = f"WHERE ({joined}) "
        else:
            text = f" WHERE ({joined})"
        guarded = guarded[:position] + text + guarded[position:]

    return guarded


# --------------------------------------------------------------------------- #
# Result shaping
# --------------------------------------------------------------------------- #


def json_default(value: Any) -> Any:
    """Serialise the column types psycopg2 hands back but JSON does not know."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    return str(value)


def clamp_text(value: Any, limit: int) -> Any:
    """Keep one long cell from eating the model's context."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=json_default)


# --------------------------------------------------------------------------- #
# Citation verification
# --------------------------------------------------------------------------- #

# A citation as the system prompt asks for it: "Section: 5.4 Regularization"
# or "Section: 3.5 Positional Encoding, Block Index: 4".  The stored paths all
# have the second form, so a cited block index can be checked exactly.
CITATION_RE = re.compile(
    r"Section:\s*(?P<section>[^\]\n。；;，]+?)\s*"
    r"(?:,\s*Block\s*Index:\s*(?P<block>\d+))?"
    # Without this anchor the lazy section group matches a single character,
    # which would silently validate every citation by prefix.
    r"(?=[\]\n。；;，]|$)",
    re.IGNORECASE,
)
STORED_PATH_RE = re.compile(
    r"^Section:\s*(?P<section>.+?),\s*Block\s*Index:\s*(?P<block>\d+)$",
    re.IGNORECASE,
)

# In multi mode a citation names the paper it comes from:
# "[来源: 《Attention Is All You Need》, paper_id: 4, Section: 3.2]".  The whole
# bracket is what carries the paper, so it is parsed as one span and a section
# match is attributed to the span it falls inside.  The section itself is still
# matched on its own, because the model also writes a bare "Section: 5.4" --
# that is a real citation in single mode, and in multi mode it is checked
# against every paper rather than thrown away.
CITATION_SPAN_RE = re.compile(r"\[来源\s*:([^\]]*)\]")
PAPER_ID_RE = re.compile(r"paper[_ ]?id\s*[:=]\s*(?P<paper_id>\d+)", re.IGNORECASE)
CITED_TITLE_RE = re.compile(r"《(?P<title>[^》]+)》")


class Citation(NamedTuple):
    """One "[来源: ...]" bracket: where it is, and which paper it names."""

    start: int
    end: int
    paper_id: int | None
    title: str | None


def citation_spans(answer: str) -> list[Citation]:
    """Every bracketed citation in `answer`, with the paper it points at.

    A bracket without a "paper_id" is still a citation -- in single mode that
    is the normal shape -- so the field is None rather than a reason to skip.
    """
    spans: list[Citation] = []
    for match in CITATION_SPAN_RE.finditer(answer):
        inner = match.group(1)
        id_match = PAPER_ID_RE.search(inner)
        title_match = CITED_TITLE_RE.search(inner)
        try:
            paper_id = int(id_match.group("paper_id")) if id_match else None
        except (TypeError, ValueError):  # pragma: no cover - \d+ cannot fail
            paper_id = None
        spans.append(
            Citation(
                start=match.start(),
                end=match.end(),
                paper_id=paper_id,
                title=title_match.group("title").strip() if title_match else None,
            )
        )
    return spans


def citation_at(spans: Sequence[Citation], position: int) -> Citation | None:
    """The span `position` falls inside, if any."""
    for span in spans:
        if span.start <= position < span.end:
            return span
    return None


def normalise_text(value: str) -> str:
    return " ".join(value.split()).lower()


def stored_section(path: str) -> str:
    """The section part of a stored evidence path, normalised."""
    match = STORED_PATH_RE.match(path.strip())
    return normalise_text(match.group("section")) if match else normalise_text(path)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class Embedder:
    """Query embedder over the dashscope SDK, matching the pipeline's model."""

    def __init__(
        self, model: str, api_key: str, max_chars: int, dimension: int
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_chars = max_chars
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        payload = text[: self.max_chars] if len(text) > self.max_chars else text
        last_error = ""

        for attempt in range(EMBED_RETRIES + 1):
            try:
                response = dashscope.TextEmbedding.call(
                    model=self.model, input=[payload], api_key=self.api_key
                )
            except Exception as exc:  # transport and SDK errors
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                status = getattr(response, "status_code", None)
                if status == 200:
                    output = getattr(response, "output", None) or {}
                    items = output.get("embeddings") if isinstance(output, dict) else None
                    if items:
                        vector = items[0]["embedding"]
                        if len(vector) != self.dimension:
                            raise RuntimeError(
                                f"expected a {self.dimension}-d vector, got {len(vector)}"
                            )
                        return vector
                    last_error = f"no embeddings in the response: {output!r}"
                else:
                    last_error = (
                        f"status_code={status} code={getattr(response, 'code', '')!r} "
                        f"message={getattr(response, 'message', '')!r}"
                    )

            if attempt < EMBED_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise RuntimeError(f"embedding the query failed: {last_error}")


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text form: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


CHUNK_TYPE_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def normalise_chunk_types(raw: Any) -> list[str]:
    """Keep only well-formed labels; a malformed filter is dropped, not fatal.

    The vocabulary is free-form, so the check is on shape rather than on a
    fixed list: lowercase letters, digits and underscores.  An unknown label
    matches nothing and the model can retry without it, which is exactly the
    fallback the search protocol asks for.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    labels: list[str] = []
    for item in raw:
        label = str(item).strip().lower().replace(" ", "_").replace("-", "_")
        if CHUNK_TYPE_LABEL_RE.match(label) and label not in labels:
            labels.append(label)
    return labels


class ToolBox:
    """The three tools, bound to one paper -- or to the whole library.

    `paper_id=None` is the multi-paper mode, and it is the only switch there
    is: the guard stops scoping the SQL, the vector searches stop filtering and
    start reporting which paper each hit came from, and the citation check
    starts asking which paper a citation belongs to.  Everything else -- the
    read-only connection, the row caps, the similarity floor -- is unchanged.
    """

    def __init__(
        self,
        conn: "psycopg2.extensions.connection",
        embedder: Embedder,
        paper_id: int | None,
        top_k: int,
        max_rows: int,
        max_cell_chars: int,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.paper_id = paper_id
        self.multi = paper_id is None
        self.top_k = top_k
        self.max_rows = max_rows
        self.max_cell_chars = max_cell_chars
        # Lazily loaded, read once per session: every evidence path stored for
        # each paper in scope, and the catalogue's id -> title.
        self._evidence: dict[int, tuple[set[str], set[str]]] | None = None
        self._library: dict[int, str] | None = None

    # -- catalogue -------------------------------------------------------- #

    def library(self) -> dict[int, str]:
        """paper_id -> title for every paper in the database.

        One query, cached: the model needs the real titles to cite, and the
        reader needs them to label the picker, so both read from here rather
        than trusting a title the model typed.
        """
        if self._library is None:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, title FROM papers ORDER BY id")
                self._library = {
                    int(paper_id): (title or "(untitled)")
                    for paper_id, title in cur.fetchall()
                }
        return self._library

    # -- SQL ------------------------------------------------------------- #

    def execute_sql(self, sql: str) -> str:
        try:
            guarded = guard_sql(sql, self.paper_id)
        except SQLGuardError as exc:
            return dumps({"error": f"SQL rejected: {exc}"})

        try:
            with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(guarded)
                rows = cur.fetchmany(self.max_rows + 1)
        except psycopg2.Error as exc:
            self.conn.rollback()
            message = str(exc).strip() or exc.__class__.__name__
            return dumps({"error": f"SQL failed: {message}"})

        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        shaped = [
            {key: clamp_text(value, self.max_cell_chars) for key, value in row.items()}
            for row in rows
        ]
        return dumps(
            {
                "sql": guarded,
                "row_count": len(shaped),
                "truncated": truncated,
                "rows": shaped,
            }
        )

    # -- citation verification -------------------------------------------- #

    def evidence_index(self) -> dict[int, tuple[set[str], set[str]]]:
        """paper_id -> (every evidence path, every section) that paper stores.

        Loaded once per session from the three tables, so checking a citation
        costs no query.  This is the only truth about what a paper's rows
        actually cite: a citation outside it was not returned by any tool.  In
        multi mode the index covers the whole library, which is what makes a
        citation naming the wrong paper detectable.
        """
        if self._evidence is None:
            index: dict[int, tuple[set[str], set[str]]] = {}
            with self.conn.cursor() as cur:
                for table in TARGET_TABLES:
                    if self.multi:
                        cur.execute(f"SELECT DISTINCT paper_id, evidence_path FROM {table}")
                    else:
                        cur.execute(
                            f"SELECT DISTINCT paper_id, evidence_path FROM {table} "
                            f"WHERE paper_id = %s",
                            (self.paper_id,),
                        )
                    for paper_id, value in cur.fetchall():
                        if not isinstance(value, str) or not value.strip():
                            continue
                        paths, sections = index.setdefault(int(paper_id), (set(), set()))
                        paths.add(normalise_text(value))
                        sections.add(stored_section(value))
            self._evidence = index
        return self._evidence

    def _citations_to_check(self, answer: str, spans: Sequence[Citation]):
        """Yield (match, span, section, the papers it may legitimately point at).

        A citation that names a paper_id the library does not have is yielded
        with no scope at all: it cannot be checked, and a fabricated paper is
        exactly what the reader needs to be told about.  A citation with no
        paper_id falls back to the session's own paper, or -- in multi mode --
        to every paper, so a section the model forgot to attribute still passes
        when it exists somewhere rather than being reported as invented.
        """
        index = self.evidence_index()
        for match in CITATION_RE.finditer(answer):
            section = normalise_text(match.group("section"))
            if not section:
                continue
            span = citation_at(spans, match.start())
            if span is not None and span.paper_id is not None:
                scopes = [span.paper_id] if span.paper_id in index else []
            elif self.paper_id is not None:
                scopes = [self.paper_id] if self.paper_id in index else []
            else:
                scopes = list(index)
            yield match, span, section, scopes

    @staticmethod
    def _cite_label(match: "re.Match[str]", span: Citation | None) -> str:
        """How a failed citation is shown to the reader.

        In multi mode the section alone does not identify the claim: the same
        "Section: 3.2" exists in every paper, so the paper the citation named
        is part of what has to be reported back.
        """
        text = match.group(0).strip()
        if span is None or span.paper_id is None:
            return text
        paper = f"《{span.title}》" if span.title else ""
        return f"{paper}paper_id={span.paper_id}, {text}".strip()

    def verify_citations(self, answer: str) -> list[str]:
        """Citations in `answer` that match no row stored for their paper.

        A citation carrying a block index must match a stored path exactly.
        A section-only citation is accepted when it names a section the paper
        has; the match is by prefix in both directions, because a section title
        may contain a comma the citation splits at.  Only the explicit
        "Section: ..." form is checked -- a looser reference cannot be told
        apart from ordinary prose, and guessing would raise false alarms.

        In multi mode the paper a citation names is part of the check: a
        paper_id that is not in the database, or a section that paper does not
        have, both fail.  Citing the right section of the wrong paper is a real
        mistake, and one this is the only place to catch.
        """
        index = self.evidence_index()
        if not index:
            return []

        spans = citation_spans(answer)
        bad: list[str] = []
        for match, span, section, scopes in self._citations_to_check(answer, spans):
            block = match.group("block")
            if block is not None:
                cited = f"section: {section}, block index: {int(block)}"
                if not any(cited in index[paper_id][0] for paper_id in scopes):
                    bad.append(self._cite_label(match, span))
                continue

            if any(
                known == section
                or known.startswith(section)
                or section.startswith(known)
                for paper_id in scopes
                for known in index[paper_id][1]
            ):
                continue
            bad.append(self._cite_label(match, span))

        return list(dict.fromkeys(bad))

    def cited_papers(self, answer: str) -> list[dict[str, Any]]:
        """The papers an answer cites, in the order it first mentions them.

        The ids come from the citations themselves, which the prompt requires
        to carry one.  A title matching the catalogue verbatim is accepted too,
        so an answer that names papers without ids still fills the reader's
        picker.  Ids the database does not have are dropped here rather than
        offered as a dead link: verify_citations is what tells the reader about
        them.

        Only meaningful in multi mode -- a single-paper answer cites one paper,
        and the reader is already looking at it.
        """
        if not self.multi:
            return []

        library = self.library()
        positions: dict[int, int] = {}
        for span in citation_spans(answer):
            if span.paper_id is not None and span.paper_id in library:
                positions.setdefault(span.paper_id, span.start)

        for paper_id, title in library.items():
            if paper_id in positions:
                continue
            key = title.strip()
            # A short title ("BERT", "GPT-2") occurs in prose by accident; a
            # long one appearing verbatim is the model naming that paper.
            if len(key) < MIN_TITLE_MATCH_CHARS:
                continue
            at = answer.find(key)
            if at >= 0:
                positions[paper_id] = at

        return [
            {"paper_id": paper_id, "title": library[paper_id]}
            for paper_id, _ in sorted(positions.items(), key=lambda item: item[1])
        ]

    # -- vector search ---------------------------------------------------- #

    def _vector_search(
        self,
        table: str,
        columns: Sequence[str],
        query: str,
        chunk_types: Sequence[str] | None = None,
    ) -> str:
        if not query or not query.strip():
            return dumps({"error": "the query text is empty"})

        try:
            vector = vector_literal(self.embedder.embed(query))
        except Exception as exc:
            return dumps({"error": f"embedding the query failed: {exc}"})

        conditions = ["t.embedding IS NOT NULL"]
        params: list[Any] = [vector]
        source = f"{table} t"
        select = ", ".join(f"t.{column}" for column in columns)

        if self.multi:
            # A hit the model cannot attribute is a hit it cannot cite, so the
            # paper's id and title are joined in rather than left to a second
            # query the model would have to remember to run.
            source = f"{table} t JOIN papers p ON p.id = t.paper_id"
            select = "t.paper_id, p.title AS paper_title, " + select
        else:
            conditions.insert(0, "t.paper_id = %s")
            params.append(self.paper_id)

        labels = normalise_chunk_types(chunk_types)
        # A label filter is intersected with the paper scope, never OR-ed around
        # it, so narrowing the search cannot widen which paper is read.
        if labels:
            conditions.append("t.chunk_types && %s::text[]")
            params.append(labels)
        params.extend([vector, self.top_k])

        sql = f"""
            SELECT {select},
                   1 - (t.embedding <=> %s::vector) AS similarity
            FROM {source}
            WHERE {' AND '.join(conditions)}
            ORDER BY t.embedding <=> %s::vector
            LIMIT %s
        """

        try:
            with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except psycopg2.Error as exc:
            self.conn.rollback()
            return dumps({"error": f"search failed: {str(exc).strip()}"})

        results = []
        for row in rows:
            item = {
                key: clamp_text(value, self.max_cell_chars)
                for key, value in row.items()
            }
            similarity = item.get("similarity")
            if isinstance(similarity, (int, float)):
                item["similarity"] = round(float(similarity), 4)
                item["relevant"] = float(similarity) >= SIMILARITY_FLOOR
            results.append(item)

        payload: dict[str, Any] = {
            "query": query,
            "result_count": len(results),
            "results": results,
        }
        # "the paper" in single mode, "the library" in multi: the same sentence
        # reads wrong if the scope it names is the wrong one.
        scope = "the library" if self.multi else "the paper"
        if labels:
            payload["chunk_types_filter"] = labels
            if not results:
                payload["note"] = (
                    f"no row in {scope} carries any of {labels}; this is a "
                    f"property of the filter, not evidence that {scope} is "
                    f"silent. Search again without chunk_types before concluding "
                    f"that nothing was found."
                )
                return dumps(payload)

        payload["note"] = (
            f"these are the {len(results)} nearest neighbours in {scope}, "
            f"always returned even when {scope} does not cover the topic. "
            f"relevant=false means the similarity is below {SIMILARITY_FLOOR} "
            f"and the row is probably not about the question. If every result "
            f"looks irrelevant, rephrase the search once or twice at most "
            f"(English terms from the paper often retrieve better), and if it "
            f"still returns nothing, say that nothing was found."
        )
        return dumps(payload)

    def search_claims(self, query: str) -> str:
        return self._vector_search(
            "claims", ("claim_type", "content", "evidence_path"), query
        )

    def search_chunks(
        self, query: str, chunk_types: Sequence[str] | None = None
    ) -> str:
        return self._vector_search(
            "text_chunks",
            ("sec_title", "content", "chunk_types", "evidence_path"),
            query,
            chunk_types,
        )

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "execute_sql":
            return self.execute_sql(str(arguments.get("sql", "")))
        if name == "search_claims":
            return self.search_claims(str(arguments.get("query", "")))
        if name == "search_chunks":
            return self.search_chunks(
                str(arguments.get("query", "")),
                arguments.get("chunk_types"),
            )
        return dumps({"error": f"unknown tool: {name}"})


# The SQL tool is the one whose contract differs by mode.  In a single-paper
# session the server splices the paper_id in and `papers` is off limits; in a
# multi-paper session neither is true.  A schema that said the wrong one would
# have the model writing a condition that is already there, or refusing to
# JOIN the catalogue it is supposed to use, so it is keyed by mode.
SQL_TOOL_DESCRIPTION = {
    False: (
        "在 experiments 表上执行一条只读 PostgreSQL SELECT 查询，用于获取精确数值、"
        "排名、参数量与实验配置的对比。支持 JOIN / GROUP BY / ORDER BY 等复杂查询。"
        "paper_id 过滤由服务端自动注入，不要自己写。"
    ),
    True: (
        "执行一条只读 PostgreSQL SELECT 查询，用于跨论文取精确数值、排名、参数量与实验"
        "配置并做对比。支持 JOIN papers / GROUP BY / ORDER BY 等复杂查询。"
        "多篇模式下服务端不会注入 paper_id，需要限定单篇时自己写 WHERE paper_id = <ID>。"
    ),
}
SQL_ARGUMENT_DESCRIPTION = {
    False: (
        "一条以 SELECT 或 WITH 开头的 PostgreSQL 查询，只查询 "
        "experiments / claims / text_chunks 三张表，不含 paper_id 条件。"
    ),
    True: (
        "一条以 SELECT 或 WITH 开头的 PostgreSQL 查询，可查询 "
        "papers / experiments / claims / text_chunks 四张表。要拿论文标题就用 "
        "JOIN papers p ON p.id = t.paper_id；papers 表含 content_tree 等超大字段，"
        "不要 SELECT *。"
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": SQL_TOOL_DESCRIPTION[False],
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": SQL_ARGUMENT_DESCRIPTION[False],
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_claims",
            "description": (
                "用中文语义检索作者本人的声明（贡献 contribution / 局限 limitation / "
                "假设 assumption / 立场 position / 未来计划 plan）。问“局限是什么”“有什么"
                "创新点”“作者接下来想做什么”时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "中文语义检索语句，例如“模型的局限性”或“未来工作计划”。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "用中文语义检索论文正文段落，用于实现细节、数学公式、超参数配置、源码链接等"
                "细节问题。返回段落原文、所在章节与 chunk_types 标签。可选 chunk_types 按"
                "内容类型收窄范围；带类型没搜到时应去掉该参数再检索一次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "中文语义检索语句，例如“位置编码的具体实现”。",
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "可选的内容类型标签，命中任意一个即返回。常用：source_code（代码/"
                            "开源链接）、formula_latex（公式）、formula_context（公式讲解）、"
                            "implementation_detail（实现细节）、experiment_setup（实验设置）。"
                            "只使用【本次会话的固定上下文】里列出的、实际存在的标签。"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def build_tools(multi: bool) -> list[dict[str, Any]]:
    """The three function schemas, with the SQL contract for this mode."""
    if not multi:
        return copy.deepcopy(TOOLS)
    tools = copy.deepcopy(TOOLS)
    function = tools[0]["function"]
    function["description"] = SQL_TOOL_DESCRIPTION[True]
    function["parameters"]["properties"]["sql"]["description"] = SQL_ARGUMENT_DESCRIPTION[True]
    return tools


# --------------------------------------------------------------------------- #
# Chat client
# --------------------------------------------------------------------------- #


class ChatError(RuntimeError):
    pass


class ChatClient:
    """OpenAI-compatible chat client that carries the tool-calling protocol.

    The DashScope compatible-mode endpoint is called over plain `requests`
    rather than through the SDK: `tools` and `tool_calls` are part of the
    request body, and the SDK's chat wrapper does not expose them.
    """

    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.enable_thinking is not None:
            # hybrid-reasoning models serve non-streaming tool calls only with
            # thinking off, which is why the agent defaults it to False.
            payload["enable_thinking"] = self.enable_thinking

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code >= 400:
                    last_error = ChatError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                    if response.status_code not in self.RETRY_STATUS:
                        raise last_error
                else:
                    try:
                        message = response.json()["choices"][0]["message"]
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        last_error = ChatError(f"unreadable response: {exc}")
                    else:
                        if isinstance(message, dict):
                            return message
                        last_error = ChatError("empty completion message")

            if attempt < self.max_retries:
                delay = min(2**attempt, 8)
                warn(f"chat call failed ({last_error}); retrying in {delay}s")
                time.sleep(delay)

        raise ChatError(f"chat call failed after {self.max_retries + 1} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """ReAct loop: the model asks for tools, Python runs them, repeat.

    The loop itself does not know which mode it is in: single or multi lives in
    the ToolBox, and this reads it off `toolbox.multi` for the one thing that
    depends on it -- the function schemas, whose SQL contract differs.  A
    `paper_id=None` toolbox is therefore a complete multi-paper session.
    """

    def __init__(
        self,
        client: ChatClient,
        toolbox: ToolBox,
        system_prompt: str,
        max_steps: int,
        history_rounds: int,
        verify_evidence: bool = True,
    ) -> None:
        self.client = client
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.history_rounds = history_rounds
        self.verify_evidence = verify_evidence
        self.tools = build_tools(toolbox.multi)
        self.history: list[dict[str, Any]] = []
        # The papers the last answer cited, for whoever renders it.  Kept on
        # the agent because it is resolved from the answer the model produced,
        # before the verification notice is appended to it.
        self.last_citations: list[dict[str, Any]] = []

    @property
    def multi(self) -> bool:
        return self.toolbox.multi

    def clear(self) -> None:
        self.history.clear()

    def ask(self, question: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": question})

        answer: str | None = None
        for step in range(self.max_steps):
            message = self.client.complete(messages, tools=self.tools)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                answer = (message.get("content") or "").strip()
                if answer:
                    break
                # An empty answer with no tool call is not recoverable here.
                raise ChatError("the model returned neither content nor a tool call")

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                name, arguments = parse_tool_call(call)
                log(f"  [tool] {name}({preview_arguments(arguments)})")
                result = self.toolbox.dispatch(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or name,
                        "content": result,
                    }
                )
        else:
            # The step budget ran out with the model still asking for tools.
            # Asking again without them is not enough on its own: the model
            # tends to narrate what it would look up next instead of answering,
            # so the request says explicitly that the budget is gone.
            log(f"  [tool] step limit reached ({self.max_steps}); asking for a final answer")
            messages.append({"role": "user", "content": TOOL_BUDGET_NOTICE})
            message = self.client.complete(messages)
            answer = (message.get("content") or "").strip()
            if not answer:
                raise ChatError("the model produced no answer after the step limit")

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        del self.history[: max(0, len(self.history) - 2 * self.history_rounds)]

        # Read off the model's own answer, before the verification notice is
        # appended: the notice quotes the suspect citations back, and a reader
        # that scanned the finished text would offer those as sources.
        self.last_citations = self.toolbox.cited_papers(answer)

        # The prompt forbids inventing evidence, but a prompt is not an
        # enforcement mechanism.  Every "Section: ..., Block Index: n" the
        # answer cites is checked against the paths the papers in scope
        # actually store, and a miss is shown to the reader rather than
        # silently accepted.
        if self.verify_evidence:
            suspect = self.toolbox.verify_citations(answer)
            if suspect:
                log(f"  [evidence] {len(suspect)} citation(s) match no stored row")
                answer += (
                    "\n\n---\n"
                    "> ⚠️ 证据校验未通过：以下引用在数据库记录中不存在，"
                    "对应内容可能是编造的，请勿采信。\n"
                    + "\n".join(f"> - `{item}`" for item in suspect)
                )
        return answer


def parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull (name, arguments) out of one tool_call, tolerating a bad payload."""
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw = function.get("arguments")

    if isinstance(raw, dict):
        return name, raw
    if not raw:
        return name, {}
    try:
        arguments = json.loads(raw)
    except (TypeError, ValueError):
        warn(f"tool call arguments are not JSON: {str(raw)[:120]}")
        return name, {}
    return name, arguments if isinstance(arguments, dict) else {}


def preview_arguments(arguments: dict[str, Any]) -> str:
    text = " ".join(" ".join(str(value).split()) for value in arguments.values())
    return text[:160] + ("..." if len(text) > 160 else "")


# --------------------------------------------------------------------------- #
# Session setup
# --------------------------------------------------------------------------- #


def connect_readonly(dsn: str) -> "psycopg2.extensions.connection":
    """A connection that physically cannot write, whatever SQL reaches it."""
    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    return conn


def paper_context(
    conn: "psycopg2.extensions.connection", paper_id: int
) -> tuple[str, str, str]:
    """Return (title, row counts per table, the chunk_types in use)."""
    with conn.cursor() as cur:
        cur.execute("SELECT title FROM papers WHERE id = %s", (paper_id,))
        row = cur.fetchone()
        if row is None:
            raise LookupError(
                f"papers.id={paper_id} does not exist; import the paper first "
                f"(extract_pipeline.py --write-db) or pass another --paper-id"
            )
        title = row[0] or "(untitled)"

        counts = []
        for table in TARGET_TABLES:
            cur.execute(f"SELECT count(*) FROM {table} WHERE paper_id = %s", (paper_id,))
            counts.append(f"{table}={cur.fetchone()[0]}")

        cur.execute(
            "SELECT DISTINCT unnest(chunk_types) FROM text_chunks WHERE paper_id = %s",
            (paper_id,),
        )
        chunk_types = sorted(value for (value,) in cur.fetchall() if value)

    return title, ", ".join(counts), ", ".join(chunk_types) or "(none)"


def library_context(
    conn: "psycopg2.extensions.connection",
    limit: int = DEFAULT_LIBRARY_PROMPT_LIMIT,
) -> tuple[str, str, str]:
    """Return (the paper catalogue, library-wide counts, the chunk_types in use).

    The multi-mode counterpart of paper_context.  There is no single paper to
    describe, so the model is handed the catalogue instead: it cannot cite a
    paper_id it has never been told exists, and the prompt is the only place
    that can tell it before its first tool call.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, title FROM papers ORDER BY id")
        rows = cur.fetchall()

        counts = []
        for table in TARGET_TABLES:
            cur.execute(f"SELECT count(*) FROM {table}")
            counts.append(f"{table}={cur.fetchone()[0]}")

        cur.execute("SELECT DISTINCT unnest(chunk_types) FROM text_chunks")
        chunk_types = sorted(value for (value,) in cur.fetchall() if value)

    shown = rows[:limit]
    lines = [
        f"  - paper_id={paper_id} 《{title or '(untitled)'}》" for paper_id, title in shown
    ]
    if len(rows) > len(shown):
        lines.append(
            f"  - （另有 {len(rows) - len(shown)} 篇未列出，请用 execute_sql 查询 papers 表）"
        )
    listing = f"当前库中共有 {len(rows)} 篇论文，可用 id 与标题如下：\n" + "\n".join(lines)
    return listing, ", ".join(counts), ", ".join(chunk_types) or "(none)"


def _fill_prompt(values: dict[str, str]) -> str:
    """Substitute one mode's blocks into the shared template."""
    text = SYSTEM_PROMPT_TEMPLATE
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def _context_block(paper_id: int, paper_title: str, counts: str, chunk_types: str) -> str:
    return (
        f"- 当前模式：单篇精读，本次会话绑定且只能看到一篇论文。\n"
        f"- 当前论文 ID (paper_id)：{paper_id}\n"
        f"- 当前论文标题：{paper_title}\n"
        f"- 数据库中本论文的规模：{counts}\n"
        f"- 本论文 text_chunks 中实际出现过的 chunk_types 取值：{chunk_types}"
    )


def _multi_context_block(library: str, counts: str, chunk_types: str) -> str:
    return (
        f"- 当前模式：多篇全局（跨库检索），本次会话没有绑定任何单篇论文，"
        f"所有论文都在你的检索范围内。\n"
        f"- 数据库整体规模：{counts}\n"
        f"- 全库 text_chunks 中实际出现过的 chunk_types 取值：{chunk_types}\n"
        f"- {library}"
    )


def _conventions_block(paper_id: int) -> str:
    return (
        f"- paper_id = {paper_id} 的过滤条件由服务端自动注入，你在 SQL 中绝对不要自己写 "
        f"paper_id 条件，也不要 JOIN 或查询 papers 表。\n"
        f"- execute_sql 只接受单条只读 SELECT 语句，必须以 SELECT 或 WITH 开头；禁止 UNION、"
        f"禁止多语句、禁止任何写操作。查询 experiments / claims / text_chunks 之外的表会被直接拒绝。\n"
        f"- execute_sql 返回 {{\"row_count\": n, \"rows\": [...]}}，rows 的字段名与上面的表结构一致；"
        f"返回行数有上限，被截断时会出现 \"truncated\": true。\n"
        f"- search_claims / search_chunks 返回 {{\"results\": [{{\"content\": ..., "
        f"\"evidence_path\": ..., \"similarity\": 0.xx, \"relevant\": true/false}}, ...]}}。"
        f"这是本论文中距离最近的若干条结果，即使论文根本没写相关内容也会返回。relevant=false "
        f"表示相似度低于 {SIMILARITY_FLOOR:.2f}，很可能与问题无关，不要把它当作证据；如果所有结果"
        f"都无关，可以换一种说法（含英文术语时检索效果通常更好）再检索一次，仍然查不到就如实说明未找到。\n"
        f"- search_chunks 传入 chunk_types 后，结果里的 \"chunk_types_filter\" 会回显实际生效的标签；"
        f"该字段缺失或为空表示没有加过滤。\n"
        f"- 工具返回 {{\"error\": ...}} 表示调用失败（例如 SQL 被拒）。请读懂错误原因后修正参数重试一次，"
        f"不要重复提交同一条语句。"
    )


def _multi_conventions_block() -> str:
    return (
        f"- 你的最终回答必须是一段纯 Markdown 文本：直接写正文，不要把整段回答再包一层 JSON"
        f"（例如 {{\"reply\": ...}}）、不要把正文塞进代码块，也不要输出“以下是回答”之类的包装。"
        f"前端会把你的输出原样按 Markdown 渲染。\n"
        f"- 多篇模式下服务端不会注入任何 paper_id 过滤：要只查某一篇，你自己写 WHERE paper_id = <ID>；"
        f"要一次比较几篇，就用 JOIN papers p ON p.id = t.paper_id 把标题取出来。\n"
        f"- 引用某篇论文时必须写成 [来源: 《标题》, paper_id: <ID>, Section: ...]，paper_id 必须是 "
        f"papers 表里真实存在的值，标题必须是 papers 表里的真实标题，两者都不许编造。\n"
        f"- papers 表里有 content_tree 这类超大字段，绝对不要 SELECT *，只取你真正要用的列"
        f"（id、title、domain）。\n"
        f"- execute_sql 只接受单条只读 SELECT 语句，必须以 SELECT 或 WITH 开头；禁止 UNION、禁止多语句、"
        f"禁止任何写操作。查询 papers / experiments / claims / text_chunks 之外的表会被直接拒绝。\n"
        f"- execute_sql 返回 {{\"row_count\": n, \"rows\": [...]}}，rows 的字段名与上面的表结构一致；"
        f"返回行数有上限，被截断时会出现 \"truncated\": true，此时请用 GROUP BY 或 LIMIT 收窄问题。\n"
        f"- search_claims / search_chunks 返回 {{\"results\": [{{\"paper_id\": ..., \"paper_title\": ..., "
        f"\"content\": ..., \"evidence_path\": ..., \"similarity\": 0.xx, \"relevant\": true/false}}, ...]}}。"
        f"这是全库中距离最近的若干条结果，即使没有任何论文写了相关内容也会返回，且结果可能集中在同一篇论文上。"
        f"relevant=false 表示相似度低于 {SIMILARITY_FLOOR:.2f}，很可能与问题无关，不要把它当作证据。"
        f"回答里用到哪一条，就必须带上它的 paper_id 与 paper_title。\n"
        f"- search_chunks 传入 chunk_types 后，结果里的 \"chunk_types_filter\" 会回显实际生效的标签；"
        f"该字段缺失或为空表示没有加过滤。\n"
        f"- 工具返回 {{\"error\": ...}} 表示调用失败（例如 SQL 被拒）。请读懂错误原因后修正参数重试一次，"
        f"不要重复提交同一条语句。"
    )


CROSS_PAPER_RULE = (
    "· 多篇模式下问“哪篇更好 / 有什么不同”：先用一次 execute_sql 把 papers 与 experiments "
    "JOIN 起来，一次性取出所有候选论文的同一个 metric，再自己比较；不要一篇一篇查完再心算。\n"
    "· 多篇模式下要按论文分层作答：每一条结论都要说清它出自哪篇论文（带 paper_id），"
    "不同论文之间的结论绝不能互相串用。\n"
    "· 如果某篇论文在数据库里根本没有相关记录，就明确说这一篇没有，不要用另一篇的内容顶上。"
)


def build_system_prompt(
    paper_id: int, paper_title: str, counts: str, chunk_types: str
) -> str:
    """The single-paper prompt: one paper, scoped by the server."""
    values = {
        "SCOPE_DESC": "指定的单篇论文",
        "PAPERS_TABLE_DOC": "",
        "INTENT_A_SQL": "SELECT DISTINCT exp_group_id, evidence_path FROM experiments;",
        "INTENT_A_REASON": (
            "这一步告诉你目标数值可能藏在哪些章节（主实验、模型变体、消融），"
            "并暴露库中 model_name 的真实写法，避免你在无关的表格里反复试错。"
        ),
        "CROSS_PAPER_RULE": "",
        "CITATION_FORMAT": "[来源: Section: 5.4 Regularization]。",
        "CITATION_RULE": "",
        "CONTEXT_BLOCK": _context_block(paper_id, paper_title, counts, chunk_types),
        "CONVENTIONS_BLOCK": _conventions_block(paper_id),
    }
    return _fill_prompt(values)


def build_multi_system_prompt(library: str, counts: str, chunk_types: str) -> str:
    """The multi-paper prompt: the whole library, scoped by the model itself."""
    values = {
        "SCOPE_DESC": "库中的多篇论文（跨库检索与横向对比）",
        "PAPERS_TABLE_DOC": (
            "0. papers 表（论文目录，仅多篇模式可用）：\n"
            "   · id (INT): 论文 ID。你回答里的每一条引用都必须带上它。\n"
            "   · title (TEXT): 论文标题。引用时要用这里存的原标题。\n"
            "   · domain (TEXT): 论文领域。\n"
            "   · content_tree (JSONB): 整篇论文的解析结果，极大，绝对不要 SELECT 它。\n"
            "   · 另外三张表都带 paper_id 外键，JOIN papers 才能拿到标题。\n"
        ),
        "INTENT_A_SQL": (
            "SELECT DISTINCT paper_id, exp_group_id, evidence_path FROM experiments;"
        ),
        "INTENT_A_REASON": (
            "这一步告诉你库中有哪些实验组、它们分别属于哪篇论文（paper_id）、"
            "以及目标数值可能藏在哪些章节（主实验、模型变体、消融），"
            "并暴露库中 model_name 的真实写法，避免你在无关的表格里反复试错。"
        ),
        "CROSS_PAPER_RULE": CROSS_PAPER_RULE + "\n",
        "CITATION_FORMAT": (
            "[来源: 《Attention Is All You Need》, paper_id: 4, Section: 3.2]。"
            "多篇模式下引用不写 paper_id 会被判为无法溯源。"
        ),
        "CITATION_RULE": (
            "5. 多篇模式下每一条事实性结论都必须落在具体某篇论文上：同一句话里不要把两篇论文的"
            "数值、结论混在一起说；用表格横向对比时，每行要标出它属于哪篇论文。\n"
        ),
        "CONTEXT_BLOCK": _multi_context_block(library, counts, chunk_types),
        "CONVENTIONS_BLOCK": _multi_conventions_block(),
    }
    return _fill_prompt(values)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paper-id", type=int,
        help="papers.id this session is scoped to; every tool call is filtered by it",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="scope the session to the whole library instead of one paper: no "
             "paper_id filter is injected, `papers` becomes readable, and every "
             "citation has to name the paper it came from",
    )
    parser.add_argument(
        "--dsn", default=os.getenv("DATABASE_URL", DEFAULT_DSN),
        help=f"PostgreSQL connection string (default: {DEFAULT_DSN})",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible chat endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--llm-model", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        help=f"chat model (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=f"DashScope embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--dimension", type=int, default=DEFAULT_DIMENSION,
        help=f"expected vector dimension (default: {DEFAULT_DIMENSION})",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"vector results returned per search (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-rows", type=int, default=DEFAULT_SQL_ROWS,
        help=f"maximum rows execute_sql hands to the model (default: {DEFAULT_SQL_ROWS})",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help=f"tool-calling rounds allowed per question (default: {DEFAULT_MAX_STEPS}, "
             f"or {DEFAULT_MULTI_MAX_STEPS} with --multi, where a comparison has to "
             f"locate its papers before it can compare them)",
    )
    parser.add_argument(
        "--history-rounds", type=int, default=DEFAULT_HISTORY_ROUNDS,
        help=f"previous question/answer pairs kept in context (default: {DEFAULT_HISTORY_ROUNDS})",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max tokens per answer (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"retries per API call (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--no-verify-evidence", dest="verify_evidence", action="store_false",
        help="skip the post-answer check that every cited evidence_path exists "
             "in the paper's stored rows; the check is on by default",
    )
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=False,
        help="let the model think before answering; off by default because "
             "hybrid-reasoning models do not serve non-streaming tool calls "
             "with thinking on",
    )
    return parser


HELP_TEXT = """commands:
  /help    show this help
  /clear   forget the conversation history (the paper stays the same)
  /exit    quit (also: exit, quit, Ctrl-D)
  /paper   show the paper this session is scoped to
"""


def repl(agent: Agent, banner: list[str]) -> int:
    print("\n".join(banner))
    print("输入问题开始提问；/help 查看命令，/exit 退出。\n")

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question in {"/exit", "/quit", "exit", "quit", ":q"}:
            return 0
        if question == "/help":
            print(HELP_TEXT)
            continue
        if question == "/clear":
            agent.clear()
            print("history cleared.\n")
            continue
        if question == "/paper":
            print("\n".join(banner) + "\n")
            continue

        try:
            answer = agent.ask(question)
        except ChatError as exc:
            warn(f"{exc}")
            continue
        except KeyboardInterrupt:
            print()
            warn("interrupted; that question was dropped")
            continue

        print()
        print(answer)
        print()


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    if args.max_steps is None:
        args.max_steps = DEFAULT_MULTI_MAX_STEPS if args.multi else DEFAULT_MAX_STEPS

    if _IMPORT_ERROR is not None:
        print(
            f"error: missing dependency ({_IMPORT_ERROR}); install with: "
            "pip install psycopg2-binary dashscope requests",
            file=sys.stderr,
        )
        return 2

    # Chat and embeddings are served by different services, so they take
    # separate keys: the chat host for the chat model, DashScope for the
    # embedding model, which the chat host does not serve.
    llm_api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    if not llm_api_key:
        print(
            "error: LLM_API_KEY is not set.\n"
            "  Windows PowerShell: $env:LLM_API_KEY = '...'\n"
            "  bash:               export LLM_API_KEY=...",
            file=sys.stderr,
        )
        return 1
    embed_api_key = (
        os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or llm_api_key
    )

    if args.top_k < 1 or args.max_rows < 1 or args.max_steps < 1:
        print("error: --top-k, --max-rows and --max-steps must be positive", file=sys.stderr)
        return 2
    if args.multi == (args.paper_id is not None):
        print(
            "error: pass exactly one of --paper-id <ID> (single-paper session) "
            "or --multi (whole-library session)",
            file=sys.stderr,
        )
        return 2

    try:
        conn = connect_readonly(args.dsn)
    except psycopg2.Error as exc:
        print(f"error: cannot connect to PostgreSQL: {exc}".strip(), file=sys.stderr)
        return 1

    try:
        try:
            if args.multi:
                library, counts, chunk_types = library_context(conn)
                prompt = build_multi_system_prompt(library, counts, chunk_types)
                scope = "多篇全局（跨库）"
            else:
                title, counts, chunk_types = paper_context(conn, args.paper_id)
                prompt = build_system_prompt(args.paper_id, title, counts, chunk_types)
                scope = f"paper_id={args.paper_id}  {title}"
        except LookupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except psycopg2.Error as exc:
            print(f"error: cannot read the session's context: {exc}".strip(), file=sys.stderr)
            return 1

        client = ChatClient(
            base_url=args.base_url,
            api_key=llm_api_key,
            model=args.llm_model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        embedder = Embedder(
            model=args.embedding_model,
            api_key=embed_api_key,
            max_chars=DEFAULT_EMBED_MAX_CHARS,
            dimension=args.dimension,
        )
        toolbox = ToolBox(
            conn=conn,
            embedder=embedder,
            paper_id=args.paper_id,  # None is multi mode
            top_k=args.top_k,
            max_rows=args.max_rows,
            max_cell_chars=DEFAULT_CELL_CHARS,
        )
        agent = Agent(
            client=client,
            toolbox=toolbox,
            system_prompt=prompt,
            max_steps=args.max_steps,
            history_rounds=args.history_rounds,
            verify_evidence=args.verify_evidence,
        )

        isolation = (
            "数据库连接为只读；没有注入 paper_id 过滤，所有查询跨全库，"
            "引用必须自带 paper_id。"
            if args.multi
            else f"数据库连接为只读；所有查询已强制限定在 paper_id={args.paper_id} 之内。"
        )
        banner = [
            "",
            scope,
            f"  {counts}",
            f"  model={args.llm_model}  embedding={args.embedding_model}  top_k={args.top_k}",
            f"  {isolation}",
        ]
        return repl(agent, banner)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
