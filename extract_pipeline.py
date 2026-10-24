#!/usr/bin/env python3
"""One-shot extraction pipeline for a MinerU content tree.

Input:  content_tree.json produced by build_content_tree.py
        {"title": "...", "sections": [{"sec_title": "...", "blocks": [...]}]}

Output: the rows are assembled in memory and printed as a reviewable JSON
        preview; with --out-dir they are also written to JSON files.  With
        --write-db the paper, its experiments, claims and text chunks are
        inserted into PostgreSQL in a single transaction, keyed by the
        paper_id returned by the paper insert.

One traversal, several extractions:

1. paper + experiments -- the table logic of extract_to_db.py, unchanged: one
   isolated LLM call per table (caption plus at most one adjacent paragraph),
   the special-rules report, and the two Python-side blank-cell repairs
   (fill_missing_values, fill_inherited_metrics).

2. text blocks -- every `type == "text"` block outside References-like
   sections is collected, batched 5-10 blocks per call, and sent to the model
   exactly once per batch.  That single call does two jobs per block: it labels
   the block's technical content (chunk_types, from the label vocabulary) and
   decides whether the block states a claim (contribution, limitation,
   assumption, position, plan).  The Preamble (title, authors, affiliations,
   licence) is skipped as front matter unless --include-preamble is given.

The chunk-type vocabulary is not hard-coded in the prompt: it lives in
chunk_types_vocab.json beside this script (--vocab overrides the path) and is
created from the built-in standard labels when the file is absent.  Every batch
prompt carries the vocabulary as it stands, so the model reuses existing labels
instead of coining near-synonyms.  A label the model returns is resolved in
three steps: an exact vocabulary word is used as is; a known synonym, a plural
or the same words in another order is folded onto the vocabulary word it
belongs to; anything else is a genuinely new label and is appended to the file,
so the next run already knows it.  A label that is not English lowercase with
underscores is rejected rather than stored.

3. equations -- every `type == "equation"` block takes a Python-side bypass:
   its `content` is copied verbatim (the raw LaTeX, never re-typed by the
   model), its `sec_title` and `evidence_path` are read from the tree, and it
   is forced to `chunk_types = ["formula_latex"]`.  No model call is made for
   it and no claim is read out of it.  The row then joins the ordinary text
   chunks for embedding and storage, so the two travel the same path from
   there on.  The section rules match the text stage (References-like sections
   and the Preamble are skipped); `--min-chars` deliberately does not apply,
   since a formula is short by nature.

4. assembly -- claims and chunks are collected into two in-memory lists.  A
   block that states a claim lands in both lists; a block without a claim is
   still a text chunk, and an equation is always a chunk.

5. embedding -- the contents of both lists are embedded in one pass with
   qwen3.7-text-embedding-flash (1024-d) and the vectors are bound back onto
   the rows they came from.  Every row keeps its own vector: batching happens
   at the API level, never by concatenating blocks.

6. storage -- with --write-db the paper is inserted first, its `id` is passed
   to every child row, and the whole import commits or rolls back as one.
   Nothing is written without that flag.

With --resume the import becomes incremental.  The rows the database already
holds for this paper are read back first, and every table and text block whose
`evidence_path` is among them is skipped before any API call is made, so a run
that died on a network fault restarts without paying for the work it already
did.  The existing paper row is reused instead of deleted and only the missing
rows are inserted.  --replace wins over --resume: it clears the paper and
extracts everything again.

With --scan the same pipeline runs over a whole directory instead of one file:
every `parsed/<paper>/content_tree.json` under --scan-dir (default: the
parsed/ directory beside this script) is picked up and processed in turn, so a
front-end pipeline can point the extractor at a directory rather than naming
each paper.  Deduplication comes first and costs nothing: the title inside each
content tree is looked up in the `papers` table -- both in its `title` column
and in the `content_tree` stored with it -- and a paper that is already there
is skipped with `SKIP: 论文已存在` before any API call is made.  The remaining
trees then go through the ordinary pipeline one at a time, each with its own
transaction and its own database connection.  A paper that fails does not stop
the batch: the failure is logged and the scan moves on, so one bad tree cannot
take a scheduled import down with it.  The scan exits 0 when every tree was
imported or skipped and 1 when one or more of them failed.  A missing or empty
scan directory is not an error either -- `未找到需要处理的新论文` is printed and
the run exits 0 -- so an empty directory cannot fail the job that called it.
--dry-run keeps its meaning and needs no database: the dedup check is skipped
and every tree found is previewed.

The block text stored for a chunk is always the text from the content tree,
never the model's echo of it: the model is asked to return `content` verbatim
so the answer stays self-describing, but a paraphrase or a truncation must not
reach the database.  Mismatches are counted and reported.

Both the chat calls and the embedding calls go through the official dashscope
SDK, but against two different services.  Chat uses
`dashscope.aigc.chat_completion.Completions`, the SDK's compatible-mode entry
point, with `base_compatible_api_url` pointed at the OpenAI-compatible host
that serves the chat model.  Embeddings go to DashScope, which serves
qwen3.7-text-embedding-flash and the chat host does not.  Every call carries an
explicit request timeout and is retried on transport errors and on the
retryable HTTP statuses.

Every answer is parsed strictly first; only when that fails is json_repair
given the raw text, so a well-formed answer is never reinterpreted.  LaTeX is
the usual reason the strict parser rejects one ("\alpha" is not a legal JSON
escape), and a repaired answer is logged as a warning rather than accepted
silently.  The packed prompts also carry an explicit escape rule, and the
table prompt carries the units, win-rate and cross-domain metric rules that
the columns of real papers need; the Python side backs those up with
`parse_measured_value`, which recovers 1200 from "1.2K steps" and files the
unit away in `domain_specific`, and with the win-rate metric label that makes
RLHF tables comparable across papers.

A call that fails every retry costs its batch and nothing else: the blocks of
that batch are dropped, never stored under a placeholder label and never
embedded.  A block the model answers without usable labels is dropped the same
way.  A table whose call fails is skipped and reported while the rest of the
paper is still extracted.

If *every* call of a stage fails -- all tables, or all text batches -- the run
aborts with exit code 3 rather than preparing an empty import.  An empty
experiments list is otherwise indistinguishable from a paper that genuinely
reports no experiments, and chunks full of placeholder labels are worse than
no chunks at all.  A paper that has no table at all is not that case: its
experiments stage returns 0 rows without a single call and the run continues.

Exit codes:
    0  the run completed (and, with --write-db, committed).  With --scan: every
       tree was imported or skipped, or there was nothing to process
    1  fatal error (database failure, or a stage failed after starting).  With
       --scan: one or more of the trees failed; the others still ran
    2  missing dependency, unreadable input or bad arguments
    3  the model API was unusable: every call of a stage failed

Usage:
    python extract_pipeline.py content_tree.json
    python extract_pipeline.py content_tree.json --stages paper experiments
    python extract_pipeline.py content_tree.json --dry-run
    python extract_pipeline.py content_tree.json --out-dir extracted_pipeline
    python extract_pipeline.py content_tree.json --write-db
    python extract_pipeline.py content_tree.json --write-db --replace
    python extract_pipeline.py content_tree.json --write-db --resume
    python extract_pipeline.py content_tree.json --vocab my_chunk_types.json
    python extract_pipeline.py --scan --write-db
    python extract_pipeline.py --scan --write-db --scan-dir parsed
    python extract_pipeline.py --scan --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

# Imported separately so that a missing optional package cannot mask the
# others: a print-only run needs neither psycopg2 nor numpy, and --no-embed
# skips only the embedding calls.  main() reports whichever ones the chosen
# flags require.
try:
    import dashscope
    from dashscope.aigc.chat_completion import Completions
except ImportError:
    dashscope = None  # type: ignore[assignment]
    Completions = None  # type: ignore[assignment]

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    Json = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConnection
else:
    PgConnection = Any


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
DEFAULT_MODEL = "glm-5.2"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
DEFAULT_DIMENSION = 1024
# Ten minutes per request: a long table or a large text batch can legitimately
# take a while, and it is cheaper to wait than to re-send the whole batch.
DEFAULT_TIMEOUT = 600.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_MAX_CHARS = 14_000
DEFAULT_TEXT_MAX_CHARS = 6_000
DEFAULT_TEXT_BATCH_SIZE = 5
DEFAULT_MIN_TEXT_CHARS = 20
DEFAULT_LLM_SLEEP = 0.5
DEFAULT_EMBED_BATCH_SIZE = 10
EMBED_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
DEFAULT_EMBED_MAX_CHARS = 8_000
DEFAULT_PRINT_ROWS = 5
CONTEXT_MAX_CHARS = 1_200

# No password here: DATABASE_URL (from the environment or .env) is used when
# set, and libpq's PG* variables cover everything else.
DEFAULT_DSN = "postgresql://qgent@localhost:5432/qgent_db"

DOMAINS = ["NLP", "CV", "Multimodal", "RL", "AI4Science", "ML"]

DOMAIN_ALIASES = {
    "natural language processing": "NLP",
    "language modeling": "NLP",
    "computer vision": "CV",
    "image": "CV",
    "multi-modal": "Multimodal",
    "reinforcement learning": "RL",
    "ai for science": "AI4Science",
    "machine learning": "ML",
    "deep learning": "ML",
}

# "plan" is an addition to the claim types already stored by init.sql; the
# `claims.claim_type` CHECK constraint still lists only the original four and
# must be widened before a plan claim can be imported.
CLAIM_TYPES = ("contribution", "limitation", "assumption", "position", "plan")

# The chunk-type vocabulary is not fixed here: it lives in VOCAB_FILE_NAME
# beside this script (--vocab overrides the path), and the model is shown the
# vocabulary that is in the file at the time of the call.  A label outside it
# is accepted only when no vocabulary word covers the content, and it is then
# appended to the file so the next run reuses it instead of coining another
# near-synonym.  These seeds are what the file is created from on first run,
# and what the pipeline falls back to if the file is missing or unreadable.
VOCAB_FILE_NAME = "chunk_types_vocab.json"

# Equation blocks never reach the model: their raw LaTeX is copied out in
# Python and filed under this one fixed label, which is therefore not part of
# the vocabulary the model gets to choose from.
FORMULA_CHUNK_TYPE = "formula_latex"

SEED_CHUNK_TYPES = (
    "implementation_detail",
    "experiment_setup",
    "experiment_result",
    "formula_context",
    "related_work",
    "source_code",
    "evaluation_metric",
    "model_proposal",
)

# Near-misses the vocabulary must absorb instead of storing as new labels.
# The JSON file carries its own copy of this table, which wins where the two
# disagree; these seeds are only the fallback and the source of the file.
SEED_CHUNK_TYPE_ALIASES = {
    # experiment_result
    "experiment_results": "experiment_result",
    "experimental_result": "experiment_result",
    "experimental_results": "experiment_result",
    "results": "experiment_result",
    "result": "experiment_result",
    "result_analysis": "experiment_result",
    "experimental_finding": "experiment_result",
    "experimental_findings": "experiment_result",
    "finding": "experiment_result",
    "findings": "experiment_result",
    # experiment_setup
    "experiments": "experiment_setup",
    "experimental_setup": "experiment_setup",
    "experiment_settings": "experiment_setup",
    "experiment_configuration": "experiment_setup",
    "experimental_settings": "experiment_setup",
    "training_setup": "experiment_setup",
    # implementation_detail
    "implementation_details": "implementation_detail",
    "implementation": "implementation_detail",
    "implementation_notes": "implementation_detail",
    "technical_detail": "implementation_detail",
    "technical_details": "implementation_detail",
    # model_proposal
    "model_proposals": "model_proposal",
    "model_introduction": "model_proposal",
    "model_motivation": "model_proposal",
    "model_description": "model_proposal",
    "model_architecture": "model_proposal",
    "method_proposal": "model_proposal",
    "proposed_model": "model_proposal",
    "proposed_method": "model_proposal",
    "proposed_approach": "model_proposal",
    "architecture_proposal": "model_proposal",
    # formula_context
    "formula": "formula_context",
    "formulas": "formula_context",
    "equation": "formula_context",
    "equations": "formula_context",
    "math": "formula_context",
    "mathematical_formulation": "formula_context",
    # related_work
    "related_works": "related_work",
    "prior_work": "related_work",
    "background": "related_work",
    "literature_review": "related_work",
    "related_literature": "related_work",
    # source_code
    "code": "source_code",
    "source_codes": "source_code",
    "code_snippet": "source_code",
    "pseudocode": "source_code",
    "implementation_code": "source_code",
    "code_link": "source_code",
    "code_links": "source_code",
    "code_repository": "source_code",
    # evaluation_metric
    "metrics": "evaluation_metric",
    "metric": "evaluation_metric",
    "evaluation_metrics": "evaluation_metric",
    "evaluation": "evaluation_metric",
    "evaluation_measure": "evaluation_metric",
    "performance_metric": "evaluation_metric",
    "performance_metrics": "evaluation_metric",
}

CLAIM_TYPE_SYNONYMS = {
    # contribution
    "contributions": "contribution",
    "main_contribution": "contribution",
    "core_contribution": "contribution",
    "key_contribution": "contribution",
    "novelty": "contribution",
    "innovation": "contribution",
    "achievement": "contribution",
    "贡献": "contribution",
    "主要贡献": "contribution",
    # limitation
    "limitations": "limitation",
    "weakness": "limitation",
    "weaknesses": "limitation",
    "drawback": "limitation",
    "shortcoming": "limitation",
    "future_work_as_limitation": "limitation",
    "局限": "limitation",
    "局限性": "limitation",
    "不足": "limitation",
    # assumption
    "assumptions": "assumption",
    "premise": "assumption",
    "hypothesis": "assumption",
    "prerequisite": "assumption",
    "假设": "assumption",
    "前提": "assumption",
    # position
    "positions": "position",
    "stance": "position",
    "argument": "position",
    "opinion": "position",
    "viewpoint": "position",
    "claim_of_position": "position",
    "立场": "position",
    "观点": "position",
    # plan
    "plans": "plan",
    "future_work": "plan",
    "future_works": "plan",
    "future_plan": "plan",
    "future_direction": "plan",
    "next_steps": "plan",
    "roadmap": "plan",
    "计划": "plan",
    "未来工作": "plan",
    "后续工作": "plan",
}

# Last-resort classification when the model names a claim but no usable
# type.  Checked in this order: "future work" is a plan, not a limitation.
CLAIM_TYPE_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "plan",
        re.compile(
            r"\bwe (?:will|plan to|intend to)\b|\bin (?:the )?future work\b|"
            r"\bfuture (?:work|direction|research)\b|\bnext steps?\b|"
            r"计划|未来工作|后续工作",
            re.IGNORECASE,
        ),
    ),
    (
        "limitation",
        re.compile(
            r"\blimitation|\bdrawback|\bweakness|\bdoes not\b|\bcannot\b|"
            r"\bfails? to\b|\bonly (?:considers?|handles?|evaluates?)\b|"
            r"局限|不足|未能",
            re.IGNORECASE,
        ),
    ),
    (
        "assumption",
        re.compile(
            r"\bwe assume\b|\bassum(?:e|es|ption|ing)\b|\bwe take .* as given\b|"
            r"\bpremise\b|\bpresuppos|\bhypothes(?:is|es|ize)\b|假设|前提",
            re.IGNORECASE,
        ),
    ),
    (
        "position",
        re.compile(
            r"\bwe (?:argue|believe|conjecture|contend|maintain)\b|"
            r"\bin our (?:view|opinion)\b|\bwe hold that\b|立场|观点|认为",
            re.IGNORECASE,
        ),
    ),
    (
        "contribution",
        re.compile(
            r"\bwe (?:propose|introduce|present|develop|design|show that)\b|"
            r"\bour (?:main )?contribution|\bthis paper (?:proposes|presents)\b|"
            r"\bnovel\b|本文提出|贡献",
            re.IGNORECASE,
        ),
    ),
)

CLAIM_TYPE_KEY_ALIASES = ("claim_type", "type", "kind", "category", "label", "claim")
CLAIM_CONTENT_KEY_ALIASES = ("content", "claim", "text", "quote", "statement", "summary")
BLOCK_ID_KEY_ALIASES = ("block_index", "id", "index", "idx", "block_id", "block")
CHUNK_TYPE_KEY_ALIASES = ("chunk_types", "chunk_type", "types", "type", "tags", "labels")

REFERENCE_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|acknowledg)", re.IGNORECASE
)
BOILERPLATE_RE = re.compile(
    r"^\s*(?:\*\*)?(acknowledg|author contributions|conflict of interest|"
    r"funding|data availability|code availability)",
    re.IGNORECASE,
)
# MinerU puts the title, authors, affiliations and licence in this section.
PREAMBLE_TITLE = "Preamble"

FENCE_OPEN_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*")
FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")
# A second line of defence in case a data URI ends up inside a text block.
DATA_URI_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")

# Table row self-check.  A model that silently merges same-named rows produces a
# plausible-looking but incomplete result, so the record count is compared with
# the number of rows the table itself prints.  This only ever warns.
HTML_ROW_RE = re.compile(r"<tr\b", re.IGNORECASE)
HTML_ROW_BLOCK_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
MD_SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")
# Tables of pure theory ("Complexity per Layer", "O(n^2)") legitimately yield no
# records, so an empty answer there is not evidence of lost rows.
THEORETICAL_TABLE_RE = re.compile(r"\bcomplexity\b|O\s*\(\s*[nN1][\s\)\^]", re.IGNORECASE)

INHERIT_BLANK_RE = re.compile(
    r"inherit|identical to (?:those of )?(?:the )?base|same as (?:the )?base|"
    r"default (?:value|configuration|setting)|unlisted values?|"
    r"继承|沿用|与\s*base\s*(?:模型)?\s*一致|未列出|默认(?:值|配置|参数)",
    re.IGNORECASE,
)
NOT_MEANINGFUL_BLANK_RE = re.compile(
    r"not\s+(?:reported|tested|applicable|available|measured)|"
    r"no\s+(?:special|explicit)|n/?a\b|"
    r"未测试|未报告|未说明|无特殊|不适用|没有说明",
    re.IGNORECASE,
)

LOG = logging.getLogger("pipeline")


def configure_stdio() -> None:
    """Windows consoles default to GBK; paper text and model output are not."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Content-tree traversal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TableUnit:
    """One table and the minimal context allowed to travel with it."""

    sec_title: str
    block_index: int
    caption: str
    table: str
    preceding: str | None
    following: str | None

    @property
    def evidence_path(self) -> str:
        return f"Section: {self.sec_title}, Block Index: {self.block_index}"


@dataclass(frozen=True)
class TextBlock:
    """One prose block, keeping its position so the evidence path matches."""

    sec_title: str
    block_index: int
    content: str

    @property
    def evidence_path(self) -> str:
        return f"Section: {self.sec_title}, Block Index: {self.block_index}"


@dataclass
class Stats:
    dropped: int = 0
    deduped: int = 0
    failed_tables: int = 0
    empty_tables: int = 0
    filled_rows: int = 0
    inherited_metric_rows: int = 0
    skipped_resume: int = 0
    row_warnings: int = 0


@dataclass
class TextStats:
    blocks: int = 0
    skipped_preamble: int = 0
    skipped_boilerplate: int = 0
    skipped_short: int = 0
    skipped_resume: int = 0
    equations: int = 0
    equations_skipped_resume: int = 0
    batches: int = 0
    failed_batches: int = 0
    blocks_lost_to_failed_batches: int = 0
    unanswered_blocks: int = 0
    unlabeled_blocks: int = 0
    content_mismatches: int = 0
    claims: int = 0
    dropped_claims: int = 0
    duplicate_claims: int = 0
    stray_elements: int = 0


@dataclass
class ResumeState:
    """What the database already holds for the paper being extracted.

    Built from the rows of every paper sharing this title and handed to the
    experiments and text stages, which drop the work items they already cover.
    The two sets are kept apart because they are written by different stages:
    `text_paths` comes from claims and text_chunks, `experiment_paths` from
    experiments.  A block index is unique within a section, so no table and no
    text block can ever share an evidence path.
    """

    paper_ids: list[int]
    text_paths: set[str]
    experiment_paths: set[str]

    @property
    def paper_id(self) -> int | None:
        return self.paper_ids[0] if self.paper_ids else None


def load_content_tree(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError(
            f"{path} is not a content tree: expected a dict with a 'sections' list"
        )
    return data


def block_content(block: dict[str, Any]) -> str:
    """Return a block's content with any inline base64 data URI scrubbed."""
    content = block.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return DATA_URI_RE.sub("[IMAGE DATA REMOVED]", content).strip()


def block_text(block: dict[str, Any]) -> str:
    """Return model-ready text for a prose block, or "" when it must be skipped."""
    if block.get("type") == "image":
        return ""

    content = block_content(block)
    caption = block.get("caption")
    caption = caption.strip() if isinstance(caption, str) else ""

    if block.get("type") == "table":
        return f"Caption: {caption}\n\n{content}" if caption else content
    if caption:
        return f"{content}\n\nCaption: {caption}".strip()
    return content


def neighbour_text(blocks: list[dict[str, Any]], index: int, step: int) -> str | None:
    """Nearest prose block in one direction, stopping at another table."""
    position = index + step
    while 0 <= position < len(blocks):
        block = blocks[position]
        btype = block.get("type")
        if btype == "text":
            text = block_content(block)
            return text[:CONTEXT_MAX_CHARS] if text else None
        if btype == "table":
            return None
        position += step
    return None


def build_table_units(tree: dict[str, Any]) -> list[TableUnit]:
    """Collect every table block; prose blocks are deliberately ignored."""
    units: list[TableUnit] = []
    for section in tree["sections"]:
        title = section.get("sec_title") or ""
        if REFERENCE_SECTION_RE.search(title):
            continue
        blocks = section.get("blocks") or []
        for index, block in enumerate(blocks):
            if block.get("type") != "table":
                continue
            table = block_content(block)
            if not table:
                continue
            caption = block.get("caption")
            units.append(
                TableUnit(
                    sec_title=title,
                    block_index=index,
                    caption=caption.strip() if isinstance(caption, str) else "",
                    table=table,
                    preceding=neighbour_text(blocks, index, -1),
                    following=neighbour_text(blocks, index, +1),
                )
            )
    return units


def build_text_blocks(
    tree: dict[str, Any],
    min_chars: int,
    include_preamble: bool,
    stats: TextStats,
) -> list[TextBlock]:
    """Collect every prose block worth labelling.

    `block_index` counts all blocks in the section, tables and images included,
    so it matches the evidence paths used by the table stage.  Sections whose
    title mentions references, a bibliography or acknowledgements are skipped
    entirely, and so are individual boilerplate blocks (funding, data
    availability, ...), which carry neither technical content nor claims.  The
    Preamble (title, authors, affiliations, licence) is front matter and is
    skipped unless --include-preamble is given; the metadata stage reads it
    separately for the author list.
    """
    blocks_out: list[TextBlock] = []
    for section in tree["sections"]:
        if not isinstance(section, dict):
            continue
        title = section.get("sec_title") or ""
        if REFERENCE_SECTION_RE.search(title):
            continue
        if not include_preamble and title.strip() == PREAMBLE_TITLE:
            stats.skipped_preamble += 1
            continue
        for index, block in enumerate(section.get("blocks") or []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block_content(block)
            if not text:
                continue
            if BOILERPLATE_RE.search(text):
                stats.skipped_boilerplate += 1
                continue
            if len(text) < min_chars:
                stats.skipped_short += 1
                continue
            blocks_out.append(TextBlock(title, index, text))
    return blocks_out


def build_equation_blocks(
    tree: dict[str, Any], include_preamble: bool, stats: TextStats
) -> list[TextBlock]:
    """Collect every equation block, bypassing the model entirely.

    An equation's `content` is already LaTeX, so re-typing it through a chat
    call could only corrupt it: the block is copied out in Python and forced to
    the `formula_latex` label.  The section rules are the text stage's (a
    References-like section and the Preamble are skipped), but `--min-chars` is
    not applied -- a formula is short by nature and is not less worth keeping
    for it.  Returned as TextBlocks, which is all the embedding and insert
    paths need; no claim is ever read out of one.
    """
    blocks_out: list[TextBlock] = []
    for section in tree["sections"]:
        if not isinstance(section, dict):
            continue
        title = section.get("sec_title") or ""
        if REFERENCE_SECTION_RE.search(title):
            continue
        if not include_preamble and title.strip() == PREAMBLE_TITLE:
            continue
        for index, block in enumerate(section.get("blocks") or []):
            if not isinstance(block, dict) or block.get("type") != "equation":
                continue
            content = block_content(block)
            if not content:
                continue
            blocks_out.append(TextBlock(title, index, content))
    stats.equations = len(blocks_out)
    return blocks_out


def equation_chunk(block: TextBlock) -> dict[str, Any]:
    """The insert row for one equation: raw LaTeX under the fixed label."""
    return {
        "sec_title": block.sec_title,
        "content": block.content,
        "evidence_path": block.evidence_path,
        "chunk_types": [FORMULA_CHUNK_TYPE],
    }


def section_text(tree: dict[str, Any], title_re: re.Pattern[str]) -> str:
    parts: list[str] = []
    for section in tree["sections"]:
        title = section.get("sec_title") or ""
        if not title_re.search(title):
            continue
        for block in section.get("blocks") or []:
            if block.get("type") != "text":
                continue
            text = block_text(block)
            if text:
                parts.append(text)
        if parts:
            break
    return "\n\n".join(parts)


def front_matter_text(tree: dict[str, Any]) -> str:
    parts: list[str] = []
    for section in tree["sections"]:
        if section.get("sec_title") == "Preamble":
            for block in section.get("blocks") or []:
                if block.get("type") == "text":
                    text = block_text(block)
                    if text:
                        parts.append(text)
            break
    return "\n".join(parts)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    LOG.warning("Truncating a block from %d to %d characters", len(text), limit)
    return text[:limit] + "\n[TRUNCATED]"


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #


class LLMError(RuntimeError):
    pass


class APIUnavailableError(RuntimeError):
    """The model API is unusable: every call of a stage failed.

    Raised instead of quietly emitting empty or placeholder rows, so a dead
    endpoint or an account problem stops the run rather than filling the
    database with junk.
    """


def _completion_text(response: Any) -> str | None:
    """Pull the assistant text out of a DashScope ChatCompletion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content.strip():
        return content
    return None


def _chat_error(response: Any) -> str:
    """Render the error payload of a failed DashScope chat response."""
    error = getattr(response, "error", None)
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    if error:
        return str(error)[:300]
    return f"status_code={getattr(response, 'status_code', None)}"


class LLMClient:
    """Chat client built on the official DashScope SDK.

    Calls go through `Completions.create`, the SDK's OpenAI-compatible entry
    point, which is the route that serves the configured chat model.  The
    native `Generation.call` route is DashScope-specific and cannot reach a
    third-party host, so it is not used here.

    Retries and the per-request timeout are applied by this class; the SDK's
    own transport is not relied on for either.
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
        enable_thinking: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

        # The SDK resolves the compatible-mode endpoint from this module-level
        # global; there is no per-call override, so --base-url is applied here.
        if dashscope is not None and self.base_url:
            dashscope.base_compatible_api_url = self.base_url

    def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "api_key": self.api_key,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "request_timeout": self.timeout,
        }
        # Hybrid-reasoning models accept this in the request body.  It is
        # only sent when the user explicitly chooses a mode.
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = Completions.create(**kwargs)
            except Exception as exc:  # transport errors raised by the SDK
                last_error = exc
            else:
                status = getattr(response, "status_code", None)
                if status == 200:
                    content = _completion_text(response)
                    if content is not None:
                        return content
                    last_error = LLMError("empty completion content")
                else:
                    error = LLMError(f"HTTP {status}: {_chat_error(response)}")
                    if status not in self.RETRY_STATUS:
                        raise error
                    last_error = error

            if attempt < self.max_retries:
                self._backoff(attempt, last_error)

        raise LLMError(
            f"API call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def _backoff(attempt: int, error: Exception) -> None:
        delay = min(2**attempt, 30)
        LOG.warning("API error (%s); retrying in %ds", error, delay)
        time.sleep(delay)

    def chat_json(
        self, system: str, user: str, expect: str, json_attempts: int = 2
    ) -> Any:
        """Call the model and parse its answer as JSON.

        ``expect`` is "object", "array" or "table".  "table" accepts both the
        full {records, special_rules_report} object and a bare record array.
        A malformed answer is retried once with a corrective suffix rather
        than failing the whole pipeline.
        """
        last_error: Exception | None = None
        for attempt in range(json_attempts):
            prompt = user
            if attempt:
                prompt += (
                    "\n\nIMPORTANT: your previous answer was not valid JSON. "
                    "Return ONLY the requested JSON value, without prose and "
                    "without Markdown code fences."
                )
            raw = self.complete(system, prompt)
            try:
                return parse_json_response(raw, expect)
            except LLMError as exc:
                last_error = exc
                LOG.warning("Invalid JSON from model (attempt %d): %s", attempt + 1, exc)
        raise LLMError(f"Model did not return valid JSON: {last_error}")


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers and a leading BOM."""
    cleaned = text.strip().lstrip(chr(0xFEFF))
    cleaned = FENCE_OPEN_RE.sub("", cleaned, count=1)
    cleaned = FENCE_CLOSE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def extract_json_span(text: str) -> str | None:
    """Return the first balanced JSON object/array, ignoring prose around it."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            expected = "[" if char == "]" else "{"
            if stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


ARRAY_WRAPPER_KEYS = ("blocks", "results", "items", "data", "chunks", "texts")


def repair_json_response(raw: str) -> Any:
    """Last-resort parse of a model answer that json.loads rejected.

    LaTeX is what breaks the strict parser in practice: a model that emits
    "\\alpha" instead of "\\\\alpha" produces the JSON error `Invalid \\escape`,
    and one that emits a raw "\\frac" is read as a form feed.  json_repair
    handles both, plus truncation, trailing commas, unescaped quotes and prose
    around the payload.  It is only ever a fallback: a response the strict
    parser accepted is never re-parsed, so a well-formed answer cannot be
    reinterpreted by the repairer.

    Returns None when no JSON value could be recovered, and for a repair that
    yields a bare string: json_repair turns a refusal ("I cannot help") into
    "", which is not a value any stage can use.
    """
    if json_repair is None:
        return None
    try:
        value = json_repair.loads(raw, skip_json_loads=True)
    except Exception as exc:  # json_repair is best-effort by design
        LOG.warning("json_repair could not recover the answer: %s", exc)
        return None
    if value is None or isinstance(value, str):
        return None
    return value


def decode_json_candidate(value: Any, expect: str) -> Any:
    """Shape one parsed candidate into what the caller asked for.

    Returns ``value`` unchanged when it already fits, unwraps the common
    ``{"blocks": [...]}`` container, and raises LLMError when the shape cannot
    be reconciled -- which is a different failure from invalid JSON and must
    not trigger the repairer.
    """
    if expect == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ARRAY_WRAPPER_KEYS:
                wrapped = value.get(key)
                if isinstance(wrapped, list):
                    return wrapped
            lists = [v for v in value.values() if isinstance(v, list)]
            if len(lists) == 1:
                return lists[0]
            # A single annotated block returned bare.
            if any(key in value for key in CHUNK_TYPE_KEY_ALIASES):
                return [value]
        raise LLMError(f"expected a JSON array, got {type(value).__name__}")

    if expect == "table":
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            # Tolerate a bare record array: the report is simply absent.
            return {"records": value, "special_rules_report": {}}
        raise LLMError(f"expected a JSON object, got {type(value).__name__}")

    if isinstance(value, dict):
        return value
    raise LLMError(f"expected a JSON object, got {type(value).__name__}")


def parse_json_response(raw: str, expect: str) -> Any:
    """Parse a model answer, tolerating fences, prose and wrapper objects.

    A JSON array is also accepted when the model wrapped it in an object
    ({"blocks": [...]}) or when it returned a single element object instead of
    a one-element array.

    When neither the whole answer nor its JSON span parses strictly, the
    answer is handed to json_repair before the attempt is given up.  A repair
    that succeeds is logged as a warning, because it means the model sent
    malformed JSON even if this particular answer could be salvaged.  A
    candidate that parses but has the wrong shape is not repaired: the JSON
    was valid, so the problem is the answer itself.
    """
    cleaned = strip_markdown_fences(raw)
    candidates = [cleaned]
    span = extract_json_span(cleaned)
    if span and span != cleaned:
        candidates.append(span)

    last_error: Exception | None = None
    values: list[Any] = []
    for candidate in candidates:
        try:
            values.append(json.loads(candidate))
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    if not values:
        repaired = repair_json_response(cleaned)
        if repaired is not None:
            LOG.warning(
                "Recovered malformed JSON with json_repair (strict error: %s)",
                last_error,
            )
            values.append(repaired)

    shape_error: Exception | None = None
    for value in values:
        try:
            return decode_json_candidate(value, expect)
        except LLMError as exc:
            shape_error = exc

    if shape_error is not None and values:
        raise shape_error
    raise LLMError(
        f"could not parse JSON ({last_error}); response starts with "
        f"{raw.strip()[:160]!r}"
    )


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

# Added to every system prompt.  A single backslash before a LaTeX command is
# the single most common way a model produces unparseable JSON ("\alpha" is not
# a legal JSON escape), and the repairer is a fallback, not an excuse.
JSON_ESCAPE_RULES = """
【公式输出规范】
- 如果文本包含极复杂的 LaTeX 公式，请务必输出双反斜杠 \\\\（例如 \\\\alpha、\\\\frac{a}{b}）。
- 如果公式过长难以转义，请用纯文本自然语言描述公式含义，严禁输出单反斜杠。
"""

METADATA_SYSTEM = (
    "You are a precise metadata extraction engine for scientific papers. "
    "You output only valid JSON, with no prose and no Markdown fences."
    + JSON_ESCAPE_RULES
)

METADATA_TEMPLATE = """Extract the metadata of the paper below.

Return a single JSON object with exactly this schema:
{
  "title": string,
  "authors": [string],
  "abstract": string,
  "domain": string,
  "method_tags": [string]
}

Rules:
- title: the paper title, cleaned of LaTeX/Markdown noise.
- authors: full person names only, in reading order. Exclude emails, affiliations, footnote markers and contributions. Use [] if no authors are present.
- abstract: the abstract as plain text, verbatim, with line breaks collapsed to spaces. Do not summarize it.
- domain: exactly ONE of ["NLP", "CV", "Multimodal", "RL", "AI4Science", "ML"]. Choose the most fitting. Machine translation, language modeling, parsing and text generation belong to NLP; image/video tasks to CV; combined text+image tasks to Multimodal; environment/reward/agent learning to RL; biology/chemistry/physics applications to AI4Science; general methods with no clear application area to ML.
- method_tags: all that apply, chosen from ["SSL", "Contrastive Learning", "Transformer", "Diffusion", "DPO", "RLHF", "GAN", "GNN", "Few-shot"]. You may add a short extra tag only if the paper clearly names a method not in this list.
- Do not invent information that is not in the source. If a field is unknown, use "" or [].

TITLE:
{TITLE}

FRONT MATTER:
{FRONT_MATTER}

ABSTRACT SECTION:
{ABSTRACT}
"""

TABLE_SYSTEM = (
    "You are a meticulous table extraction engine for empirical results in "
    "scientific papers. You receive exactly one table plus its caption and at "
    "most one adjacent paragraph. You output only valid JSON, with no prose and "
    "no Markdown fences. Never invent numbers."
    + JSON_ESCAPE_RULES
)

TABLE_TEMPLATE = """The paper is titled: {PAPER_TITLE}

You are given exactly ONE table from this paper. Extract its empirical results.

Return a single JSON object with exactly this schema:
{
  "records": [
    {
      "task": string|null,
      "dataset": string|null,
      "model_name": string,
      "metric": string,
      "value": number,
      "is_ours": boolean,
      "hyperparams": object|null,
      "domain_specific": object|null
    }
  ],
  "special_rules_report": {
    "has_blank_cells": boolean,
    "blank_cell_meaning": string|null,
    "blanks_inherit_base": boolean,
    "base_model_name": string|null,
    "columns_traversed": [string],
    "column_blank_rules": [
      {
        "column": string,
        "kind": "evaluation_metric" | "intrinsic_attribute",
        "blank_meaning": "not_tested" | "not_reported" | "inherits_base"
      }
    ],
    "notes": string
  }
}

ISOLATION
- The TABLE is the ONLY source of result values. The caption and adjacent paragraphs are context: use them to interpret headers, task names and blank cells, never as a source of extra numbers.
- If a value is not printed in the table, do not emit it.

COLUMN COVERAGE (most important)
- Walk every data column from left to right. A table often has several metric columns (e.g. PPL, BLEU, params, FLOPs). Emit one record for EVERY (row, metric column) pair that has a concrete numeric value.
- Never stop after the first metric and never omit the rightmost metric column.
- Recombine multi-level headers before naming dataset/metric: "BLEU" over "EN-DE"/"EN-FR" -> metric "BLEU", dataset "EN-DE"/"EN-FR"; "WSJ 23 F1" -> dataset "WSJ 23", metric "F1"; "PPL (dev)" -> metric "PPL", dataset "dev".
- When a parameter-count column sits in the results/metric block (e.g. "params x10^6" next to PPL and BLEU), it is a normal metric column: emit a normal record with metric "params*10^6" and the numeric value. Do NOT create a separate `params` field.
- When a parameter count is only a configuration attribute of the row, keep it in `hyperparams` instead.

ROW COVERAGE (mandatory)
- HARD RULE: 严禁跨行合并或去重. Read and emit EVERY data row of the table, in the order printed. Never collapse two rows into one, and never drop a row because it resembles another one.
- Count the rows before you answer. The number of distinct table rows you report must equal the number of data rows printed in the table; dropping a row is an error, never a simplification.
- The same model name legitimately appears several times, each under a different training setting or on a different subset of the evaluation data. Whenever the training setting, the dataset subset or any other distinguishing cell differs, you MUST emit a separate record for each row.
- Two rows are the same record ONLY when every one of their cells is identical. A different Training cell, a different dataset column or a different value always means two records.
- Worked example: in a table whose Training column offers "WSJ only, discriminative" and "semi-supervised", a parser listed under both settings gets TWO records, one per setting -- even though its name is printed twice. Put that setting in `domain_specific` as {"training_setting": "..."} so the two records stay distinguishable.

CROSS-CELL NUMBERS (mandatory)
- One number is often split across two adjacent cells: the first cell ends with a multiplication sign ("3.3 ·", "2.0 ×", "3.3 *", "3.3 x", "3.3 \\times") and the next cell holds only the exponent, usually as a superscript ("10<sup>18</sup>", "10^18", "10^{18}").
- You MUST recombine such a pair into ONE value before emitting anything: "3.3 ·" + "10<sup>18</sup>" -> 3.3e18.
- Never emit the mantissa and the exponent as two separate records, and never drop the exponent.

CENTRED VALUES SPANNING SEVERAL SUB-COLUMNS (mandatory)
- In the source image a value may be centred across two sub-columns (e.g. "EN-DE" and "EN-FR") instead of being aligned to one of them.
- When such a centred value describes a global property of the whole run -- a training cost, a parameter count, a training time, anything that does not depend on the evaluation set -- you MUST emit one record per sub-column carrying the SAME value: one with dataset "EN-DE", one with dataset "EN-FR".
- This is the ONLY case in which one printed number may appear in several records. It is a duplication across columns, never across rows: copying a value from another row stays forbidden.
- It NEVER applies to a metric measured on a specific test set (BLEU, PPL, F1, accuracy, WER). A single centred BLEU value must not be copied to both datasets: if the table prints no value for a dataset, that dataset gets no record.
- Decide by the metric's nature, not by its position: "Training Cost (FLOPs)" centred over EN-DE/EN-FR -> two records; "BLEU" centred over EN-DE/EN-FR -> no copy.

VALUES
- `value` must be a bare JSON number. Convert scientific notation: 3.3 * 10^18 -> 3.3e18. Remove thousands separators.
- Some HTML tables split a number across adjacent cells (e.g. "3.3 *" and "10^18"); recombine them into one value.
- Skip blank cells; never guess, average, or copy a value from another row. Python may later restore an intrinsic-attribute blank from the base row, but you must not emit it yourself.
- If the table contains only theoretical expressions (O(n^2), formulas, symbols) and no concrete measured number, return "records": [].
- Percentages: keep the numeric part (25.8 for 25.8%) and include "%" in the metric name.

VALUES WITH UNITS AND MARKERS (mandatory)
- `value` is a FLOAT column: it can only hold a bare number. A cell such as "1.2K steps", "8 x A100", "1.5h" or "3.2M params" must still become a number.
- Convert the scale prefix into the number: "1.2K steps" -> 1200, "3.2M params" -> 3200000, "1.5h" -> 1.5.
- Keep the unit and the direction out of `value` and put them in `domain_specific`: {"unit": "steps"}, {"unit": "A100", "count": 8}, {"unit": "hours"}.
- A direction arrow (↑ / ↓) states a direction, not a unit. When the cell is a change or an improvement ("↓ 12%" = a 12% decrease), write it as the decimal 0.12 and record {"direction": "decrease"}. When the arrow only marks a value as better or worse ("88.5 ↑"), keep 88.5 and record {"direction": "increase"} -- do not rescale it.
- If a cell genuinely cannot be reduced to a number (e.g. "Win", "best", "n/a", a formula), DROP that record entirely. Never emit a string in `value` and never invent a number for it.

WIN RATE AND PREFERENCE TABLES (mandatory)
- A table whose header is "Win Rate", "Winrate", "Preference", "Preference Rate" or "% Preferred" reports one metric: name it exactly "Win Rate", whatever wording the header used.
- Convert a verbal cell to the 0-1 scale: "Win" -> 1.0, "Lose" -> 0.0, "Tie" -> 0.5. A numeric cell keeps the number the table printed ("72.3" stays 72.3).
- A tie is 0.5, never 0 and never 1.
- The compared model is part of the record: put the baseline the row was compared against in `domain_specific` as {"baseline": "PPO"} / {"baseline": "DPO"}. It is usually named in the caption or in the column header ("vs. PPO").
- Rows must stay aligned: the model in `model_name` is the one the win rate belongs to, and the baseline is the one it was compared against. Never shift a value onto the neighbouring row while collecting a column, and never swap the two.

DOMAIN-SPECIFIC METRICS (mandatory)
- Domain metrics are extracted as written, never renamed, translated or expanded: RMSD, Docking score, AUC, AUROC, PSNR, SSIM, MAE, MCC, Spearman, Pearson, FID.
- Keep the metric label exactly as the table header prints it, including its unit and any qualifier ("Docking score (kcal/mol)"). Do not normalise it to a "standard" name.
- If a metric is unfamiliar, do not guess what it means and do not substitute a metric you do recognise: emit the header text verbatim as `metric`.
- For biology, medicine, chemistry or physics tasks, add {"domain": "AI4Science"} to `domain_specific` (alongside any unit or baseline already there).

RECORD SHAPE AND HYPERPARAMETERS
- Every record has exactly the fields in the schema above. Never add ad-hoc top-level keys such as `params`, `flops` or `parameters`; a numeric parameter count is an ordinary metric record.
- `hyperparams`: object of the row's explicitly listed configuration settings. Use the column header as a snake_case key (N, d_model, d_ff, h, d_k, d_v, P_drop, epsilon_ls, train_steps). Include ONLY cells that are non-blank for this row. null if none.
- Do not put performance metrics or dataset names in `hyperparams`. A parameter count that is genuinely a configuration attribute of the row belongs here.

NAMES
- `model_name`: the model/configuration exactly as written. For rowspan group labels such as (A), (B), (C), prefix them, e.g. "(A) h=1"; use the base row's label exactly ("base").
- `task`: the task evaluated by this table, inferred from the caption/section (e.g. "Machine Translation", "Constituency Parsing").
- `dataset`: the evaluation dataset as written or implied by the column (e.g. "newstest2014 EN-DE", "WSJ 23", "dev"). null if genuinely absent.
- `metric`: the metric name (e.g. "BLEU", "PPL", "F1", "Training Cost (FLOPs)", "params*10^6").
- `is_ours`: true only for rows belonging to this paper's own model/variants (base/big/(A)... rows of its own architecture); false for cited baselines.

DOMAIN-SPECIFIC
- `domain_specific`: task-specific fields explicitly present, e.g. {"training_setting": "WSJ only, discriminative"} for a Training column. null when none.

BLANK CELLS: TWO KINDS OF COLUMNS
Before reading values, classify EVERY metric column into one of two kinds:
- evaluation_metric: a measured outcome of running the model (BLEU, PPL, F1, accuracy, WER, win-rate). A blank means that variant was not evaluated in that cell: blank_meaning "not_tested" or "not_reported". Such blanks must never be filled.
- intrinsic_attribute: a property determined by the model configuration itself, not by a training/evaluation run (params, FLOPs, training cost, memory footprint, inference latency). A blank here usually means the value is identical to the base model's: blank_meaning "inherits_base" when the caption or adjacent paragraph says so, else "not_reported".
- Emit one `column_blank_rules` entry per metric column, using the column header exactly as printed after joining multi-level headers (e.g. "params*10^6", "Training Cost (FLOPs)", "BLEU").
- A column that is only partly blank is still classified once; the rule describes what its blanks mean.
- If a column has no blanks, you may omit it from `column_blank_rules` or include it with the meaning that would apply; never invent an "inherits_base" rule for an evaluation metric.

SPECIAL RULES REPORT
- `has_blank_cells`: true if any data cell is blank.
- `blank_cell_meaning`: if the caption or adjacent paragraph explicitly explains what a blank means, quote it verbatim; otherwise null.
- `blanks_inherit_base`: true ONLY when the text explicitly says blank/unlisted values equal the base model or default configuration (e.g. "Unlisted values are identical to those of the base model"). false when blanks mean "not reported", "not tested", "N/A", or nothing is said.
- `base_model_name`: exact `model_name` of the base row when blanks_inherit_base is true; otherwise null.
- `columns_traversed`: every data column header, left to right, that you processed.
- `column_blank_rules`: the per-column classification described above. This report decides what Python is allowed to restore later, so be explicit about the kind and the meaning.
- `notes`: one short sentence describing how blanks and multi-level headers were handled.

Return ONLY the JSON object. No prose, no Markdown fences.

SECTION: {SECTION}
CAPTION: {CAPTION}

PRECEDING PARAGRAPH (context only):
{PREV}

FOLLOWING PARAGRAPH (context only):
{NEXT}

TABLE:
{TABLE}
"""

UNIFIED_TEXT_SYSTEM = (
    "You are a precise annotation engine for scientific papers. For every text "
    "block you label its technical content and decide whether it states a "
    "claim. You output only valid JSON, with no prose and no Markdown fences. "
    "Never invent content or claims."
    + JSON_ESCAPE_RULES
)

UNIFIED_TEXT_TEMPLATE = """You are given {COUNT} text blocks from one scientific paper. Handle EACH block on its own: (1) label its technical content, (2) decide whether it states a claim.

Return a JSON array with exactly {COUNT} objects, in the same order as the blocks:
[
  {{
    "block_index": <the number shown in the block header>,
    "content": "<the block's text, verbatim>",
    "chunk_types": ["<label>", ...],
    "claim": null
  }}
]

CHUNK TYPES (job 1)
- VOCABULARY -- the labels already in use. Check this list FIRST and reuse a word from it verbatim whenever one fits: {VOCABULARY}
- MEANINGS: implementation_detail (implementation details / how it works), experiment_setup (experimental settings), experiment_result (results, analysis, discussion of numbers), formula_context (explanation around a formula), related_work (background, prior work), source_code (code or code links), evaluation_metric (discussion of metrics), model_proposal (the proposed model, its motivation and design).
- HARD RULE: 严禁为已有语义创造新词. When a vocabulary word fits, you MUST use it. Never invent a plural, a compound or a near-synonym of a word that is already in the vocabulary: results are only ever experiment_result, never experiment_results / experimental_results / results. A proposed model or its motivation is only ever model_proposal, never model_introduction / model_motivation / proposed_model. A code link is only source_code.
- FREE EXTENSION: only when NO vocabulary word covers the meaning, invent a precise label in English lowercase with underscores (e.g. data_augmentation). A label you invent is added to the vocabulary, so invent one only for a meaning the vocabulary genuinely lacks.
- Give one or more labels per block, ordered most to least important. Never return an empty array.

CLAIMS (job 2)
- claim_type MUST be exactly one of "contribution", "limitation", "assumption", "position", "plan".
  * contribution: what the paper introduces, proposes or achieves.
  * limitation: acknowledged weaknesses, scope restrictions, failure cases.
  * assumption: an explicit premise the method or the evaluation relies on.
  * position: a stance or argument about the field, not a result.
  * plan: planned or future work the authors state they intend to do.
- If the block states no claim, use "claim": null. Most blocks state no claim: prefer precision over recall, and never stretch a plain description into a claim.
- At most ONE claim per block. Keep the single most important one.
- claim.content: the claim as a faithful quote or a concise paraphrase. Keep numbers and qualifiers exactly as stated.
- claim.domain_specific: an object of subfield-specific detail explicitly stated, or null.

OTHER RULES
- Echo `content` exactly as given: do not translate, summarize, shorten or reformat it.
- `block_index` must be the number from the block header, so every block can be matched back.

BLOCKS
{BLOCKS}
"""


def metadata_prompt(title: str, front_matter: str, abstract: str) -> str:
    return (
        METADATA_TEMPLATE.replace("{TITLE}", title or "(unknown)")
        .replace("{FRONT_MATTER}", front_matter or "(none)")
        .replace("{ABSTRACT}", abstract or "(none)")
    )


def table_prompt(unit: TableUnit, paper_title: str, max_chars: int) -> str:
    """Build a prompt containing this table and nothing but its neighbours."""
    return (
        TABLE_TEMPLATE.replace("{PAPER_TITLE}", paper_title or "(unknown)")
        .replace("{SECTION}", unit.sec_title or "(unknown)")
        .replace("{CAPTION}", unit.caption or "(none)")
        .replace("{PREV}", unit.preceding or "(none)")
        .replace("{NEXT}", unit.following or "(none)")
        .replace("{TABLE}", truncate(unit.table, max_chars))
    )


def render_block(block_id: int, block: TextBlock, max_chars: int) -> str:
    """One labelled block inside the unified prompt.

    The header deliberately shows only the batch-local id: the section-scoped
    `block_index` would collide with it, and the model only has to echo a
    single number back.
    """
    return (
        f"[Block {block_id}] Section: {block.sec_title or '(unknown)'}\n"
        f"{truncate(block.content, max_chars)}"
    )


def unified_text_prompt(
    batch: Sequence[TextBlock],
    paper_title: str,
    max_chars: int,
    vocab: ChunkVocabulary,
) -> str:
    """Build one batch prompt carrying the chunk vocabulary as it stands."""
    blocks = "\n\n".join(
        render_block(position, block, max_chars)
        for position, block in enumerate(batch, start=1)
    )
    return (
        UNIFIED_TEXT_TEMPLATE.replace("{COUNT}", str(len(batch)))
        .replace("{VOCABULARY}", vocab.render())
        .replace("{BLOCKS}", blocks)
        + f"\n\nPAPER TITLE: {paper_title or '(unknown)'}\n"
    )


# --------------------------------------------------------------------------- #
# Validation and normalisation
# --------------------------------------------------------------------------- #


def as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value or None
    return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "yes", "y", "1", "ours", "proposed"}:
            return True
        if normalised in {"false", "no", "n", "0", "baseline"}:
            return False
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    return None


def sanitise_json(value: Any) -> Any:
    """Make model output safe for json.dump(allow_nan=False)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): sanitise_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitise_json(item) for item in value]
    return value


SCI_NOTATION_RE = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)\s*(?:[xX*])?\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?$"
)
E_NOTATION_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)$")
SUP_RE = re.compile(r"<sup>\s*([+-]?\d+)\s*</sup>", re.IGNORECASE)


def parse_numeric(value: Any) -> float:
    """Coerce a model-supplied value to float, including scientific notation.

    Accepts numbers already parsed by JSON, "3.3e18", and strings such as
    "3.3 * 10^18", "3.3 x 10^{18}" or "3.3 · 10<sup>18</sup>".
    """
    if value is None or isinstance(value, bool):
        raise ValueError("missing or boolean value")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        text = (
            text.replace("−", "-")
            .replace("–", "-")
            .replace("×", "x")
            .replace("·", "*")
            .replace("⋅", "*")
        )
        text = SUP_RE.sub(r"^\1", text)
        text = text.replace(",", "").replace("%", "").strip()
        # Best-result markers used by many tables, e.g. "28.4*" or "28.4**".
        text = text.strip("*†‡ ")
        text = re.sub(r"\s+", " ", text)
        match = SCI_NOTATION_RE.match(text) or E_NOTATION_RE.match(text)
        if match:
            number = float(match.group(1)) * (10.0 ** int(match.group(2)))
        else:
            number = float(text)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


PARAM_SUFFIX = {"k": 1e-3, "m": 1.0, "b": 1e3, "t": 1e6}


def parse_params(value: Any) -> float | None:
    """Parse a model-size cell into millions of parameters."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "na", "-", "null", "none"}:
        return None
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([kKmMbBtT])?", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = (match.group(2) or "m").lower()
    number *= PARAM_SUFFIX.get(suffix, 1.0)
    return number if math.isfinite(number) else None


def normalise_domain(value: Any) -> str | None:
    text = as_str(value)
    if not text:
        return None
    key = " ".join(text.lower().split())
    for domain in DOMAINS:
        if key == domain.lower():
            return domain
    if key in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[key]
    for domain in DOMAINS:
        if domain.lower() in key:
            return domain
    for alias, domain in DOMAIN_ALIASES.items():
        if alias in key:
            return domain
    return None


def normalise_method_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        tag = as_str(item)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


EXPERIMENT_STANDARD_FIELDS = {
    "exp_group_id",
    "evidence_path",
    "task",
    "dataset",
    "model_name",
    "metric",
    "value",
    "is_ours",
    "hyperparams",
    "domain_specific",
}
# A top-level key (same level as "metric"/"value") that names a parameter
# count.  Only top-level keys are considered: the same key inside
# `hyperparams` is configuration and must stay there.
PARAM_KEY_RE = re.compile(r"^\s*(?:#\s*)?(?:params?|parameters?)\b.*$", re.IGNORECASE)

# A win-rate table names its metric any of these ways; the record is filed
# under the one canonical label so RLHF/DPO results are comparable across
# papers.
WIN_RATE_METRIC_RE = re.compile(
    r"^\s*(?:win\s*[-_ ]?\s*rate|winrate|preference(?:\s*rate)?|"
    r"pct\.?\s*preferred|preferred)\s*(?:\(\s*%?\s*\)|%)?\s*$",
    re.IGNORECASE,
)
WIN_RATE_METRIC_LABEL = "Win Rate"
# Verdicts a win-rate table may print instead of a number.  An exact token only:
# "wins" or "winner" must not be read as "win".
WIN_VERDICT_VALUES = {
    "win": 1.0,
    "winner": 1.0,
    "lose": 0.0,
    "loss": 0.0,
    "loses": 0.0,
    "tie": 0.5,
    "draw": 0.5,
    "tied": 0.5,
}

# A cell the model echoed verbatim instead of reducing to the number the FLOAT
# column needs: "1.2K steps", "8 x A100", "1.5h", "128GB", "12%", "↓ 12%",
# "88.5 ↑".  The number is recovered and the unit/direction are handed back
# separately for `domain_specific`.
MEASURED_VALUE_RE = re.compile(
    r"^(?P<lead>[↑↓+±]{0,2})?\s*"
    r"(?P<number>[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?)\s*"
    r"(?P<scale>[kKmMbBtT](?![A-Za-z]))?\s*"
    r"(?:[x×*]\s*)?"
    r"(?P<unit>[A-Za-z%µ°][A-Za-z0-9%µ°·/^\-]*)?\s*"
    r"(?P<trail>[↑↓]+)?$"
)
MEASURED_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
DIRECTION_MARKERS = {"↑": "increase", "↓": "decrease", "±": "difference", "+": "increase"}


def win_rate_metric(metric: Any) -> bool:
    return bool(WIN_RATE_METRIC_RE.match(as_str(metric) or ""))


def canonical_metric(metric: str) -> str:
    """File every win-rate/preference column under one label."""
    return WIN_RATE_METRIC_LABEL if win_rate_metric(metric) else metric


def parse_measured_value(value: Any, metric: str) -> tuple[float, dict[str, Any]] | None:
    """Recover a number from a cell the model copied out verbatim.

    This is the Python-side half of the units rule: the prompt asks the model
    to reduce "1.2K steps" to 1200 with {"unit": "steps"}, and this rescues the
    row when it answers with the cell instead.  Returns the number and the
    extra `domain_specific` fields it implies, or None when nothing numeric can
    be recovered -- "n/a", "best" and formulas are dropped, never guessed.

    A leading direction marker before a percentage is read as a stated change,
    the same way the prompt describes it: "↓ 12%" is 0.12 with
    {"direction": "decrease"}, not 12.  A trailing marker after a percentage is
    a verdict about a measured value and never rescales it ("88.5% ↑" stays
    88.5).  No other cell is rescaled on the strength of a marker.
    """
    if value is None or isinstance(value, bool):
        return None

    text = as_str(value)
    if not text:
        return None

    verdict = text.lower().strip(" .")
    if win_rate_metric(metric) and verdict in WIN_VERDICT_VALUES:
        # The marker keeps a converted verdict apart from a percentage: a bare
        # 1.0 from "Win" and a bare 1.0 from "1.0%" would otherwise look alike.
        return WIN_VERDICT_VALUES[verdict], {"converted_from": "win_lose_tie"}

    match = MEASURED_VALUE_RE.match(
        text.replace("−", "-").replace("–", "-").replace("×", "x").replace("·", "x")
    )
    if not match:
        return None

    number = float(match.group("number").replace(",", ""))
    scale = match.group("scale")
    if scale:
        number *= MEASURED_SCALE[scale.lower()]

    extra: dict[str, Any] = {}
    unit = (match.group("unit") or "").strip()
    if unit:
        extra["unit"] = unit
    lead = (match.group("lead") or "").strip()
    trail = (match.group("trail") or "").strip()
    marker = lead or trail
    if marker and marker[0] in DIRECTION_MARKERS:
        extra["direction"] = DIRECTION_MARKERS[marker[0]]
    if lead and unit == "%":
        number /= 100.0

    if not math.isfinite(number):
        return None
    return number, extra


def parse_metric_value(value: Any, metric: str) -> tuple[float, dict[str, Any]]:
    """Parse a metric value into (number, extra domain_specific fields).

    Four routes, in order: a plain number or scientific notation; a
    parameter-count cell such as "65M" for a parameter-count metric; a printed
    cell such as "1.2K steps"/"8 x A100"/"1.5h"; and a verdict in a win-rate
    table ("Win"/"Lose"/"Tie").  The second element is what the caller must
    merge into `domain_specific` -- the FLOAT column can hold the number only.

    Raises ValueError when nothing numeric can be recovered; the caller drops
    the record rather than storing a placeholder.
    """
    try:
        return parse_numeric(value), {}
    except (TypeError, ValueError):
        pass

    if PARAM_KEY_RE.match(metric):
        params = parse_params(value)
        if params is not None:
            return params, {}

    measured = parse_measured_value(value, metric)
    if measured is not None:
        return measured

    raise ValueError(f"cannot read a number from {value!r}")


def merge_domain_specific(
    base: dict[str, Any] | None, extra: dict[str, Any]
) -> dict[str, Any] | None:
    """Add Python-recovered unit/direction fields without overriding the model.

    The model's own `domain_specific` wins on conflict: it saw the caption and
    the whole table, and it is the one that is supposed to have put the unit
    there in the first place.
    """
    merged = dict(base or {})
    for key, value in extra.items():
        merged.setdefault(key, value)
    return merged or None


def metric_label_from_key(key: str) -> str:
    """Turn a metric-position key into a metric label: 'params ×10^6' -> 'params*10^6'."""
    text = key.strip()
    text = re.sub(
        r"\b(params?|parameters?)\s*[×x]\s*(10\s*\^?\s*6)",
        r"\1*\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\*\s*", "*", text)
    return re.sub(r"\s+", " ", text).strip()


def build_experiment_shell(
    row: dict[str, Any], unit: TableUnit, group_id: str
) -> dict[str, Any] | None:
    model_name = as_str(row.get("model_name"))
    if not model_name:
        LOG.warning("Ignoring row without model_name (%s): %r", unit.evidence_path, row)
        return None

    is_ours = as_bool(row.get("is_ours"))
    if is_ours is None:
        LOG.warning(
            "Missing/invalid is_ours for %r (%s); defaulting to false",
            model_name,
            unit.evidence_path,
        )
        is_ours = False

    return {
        "exp_group_id": group_id,
        "task": as_str(row.get("task")),
        "dataset": as_str(row.get("dataset")),
        "model_name": model_name,
        "metric": None,
        "value": None,
        "hyperparams": sanitise_json(as_dict(row.get("hyperparams"))),
        "is_ours": is_ours,
        "domain_specific": sanitise_json(as_dict(row.get("domain_specific"))),
        # The block we sent is authoritative, not the model's copy of the path.
        "evidence_path": unit.evidence_path,
    }


def normalise_experiment_row(
    row: Any, unit: TableUnit, group_id: str
) -> dict[str, Any] | None:
    """Build the standard record when the model put a value in metric/value."""
    if not isinstance(row, dict):
        LOG.warning(
            "Ignoring non-object row from %s: %r",
            unit.evidence_path,
            str(row)[:120],
        )
        return None

    shell = build_experiment_shell(row, unit, group_id)
    if shell is None:
        return None

    metric = as_str(row.get("metric"))
    if not metric:
        LOG.warning(
            "Ignoring row without metric (%s, model=%r): %r",
            unit.evidence_path,
            row.get("model_name"),
            str(row)[:160],
        )
        return None

    raw_value = row.get("value")
    try:
        value, recovered = parse_metric_value(raw_value, metric)
    except (TypeError, ValueError):
        LOG.warning(
            "Ignoring non-numeric value %r (%s, model=%r, metric=%r)",
            raw_value,
            unit.evidence_path,
            row.get("model_name"),
            metric,
        )
        return None

    if recovered:
        LOG.info(
            "    recovered %r -> %s from the cell text (%s)",
            raw_value,
            value,
            unit.evidence_path,
        )

    shell["metric"] = canonical_metric(metric)
    shell["value"] = value
    shell["domain_specific"] = merge_domain_specific(shell["domain_specific"], recovered)
    return shell


def embedded_metric_records(
    row: Any, unit: TableUnit, group_id: str
) -> list[dict[str, Any]]:
    """Convert ad-hoc top-level metric keys into standard records.

    If the model answers with {"model_name": ..., "params*10^6": 65} instead
    of {"metric": "params*10^6", "value": 65}, the key is in the metric
    position and becomes a normal record.  Keys inside `hyperparams` are
    deliberately never inspected: a parameter count that is configuration
    must stay in `hyperparams`.
    """
    if not isinstance(row, dict):
        return []

    shell = build_experiment_shell(row, unit, group_id)
    if shell is None:
        return []

    main_metric = as_str(row.get("metric"))
    main_key = " ".join(main_metric.lower().split()) if main_metric else None
    records: list[dict[str, Any]] = []

    for key, raw_value in row.items():
        if key in EXPERIMENT_STANDARD_FIELDS or not PARAM_KEY_RE.match(str(key)):
            continue
        metric = metric_label_from_key(str(key))
        if main_key and " ".join(metric.lower().split()) == main_key:
            continue
        try:
            value, recovered = parse_metric_value(raw_value, metric)
        except (TypeError, ValueError):
            LOG.warning(
                "Ignoring non-numeric top-level key %r=%r (%s)",
                key,
                raw_value,
                unit.evidence_path,
            )
            continue
        record = dict(shell)
        record["metric"] = canonical_metric(metric)
        record["value"] = value
        record["domain_specific"] = merge_domain_specific(
            shell["domain_specific"], recovered
        )
        records.append(record)

    return records


def _blank_kind(value: Any) -> str:
    text = (as_str(value) or "").lower()
    if re.search(r"intrinsic|attribute|architectur|param|flop|footprint|latenc", text):
        return "intrinsic"
    if re.search(r"eval|result|metric|performance|score|accuracy|quality", text):
        return "evaluation"
    return "unknown"


def _blank_meaning(value: Any) -> str:
    text = (as_str(value) or "").lower().replace("_", " ")
    if INHERIT_BLANK_RE.search(text):
        return "inherits_base"
    if NOT_MEANINGFUL_BLANK_RE.search(text):
        return "not_reported"
    if re.search(r"blank|empty|missing|未填|空白", text):
        return "not_reported"
    return "unknown"


def normalise_column_blank_rules(raw: Any) -> list[dict[str, str]]:
    """Normalise the per-column blank classification.

    Accepts the documented array of objects, or a mapping
    {"<column>": "<meaning>"} for models that answer more compactly.
    """
    if isinstance(raw, dict):
        items: list[Any] = []
        for column, value in raw.items():
            if isinstance(value, dict):
                items.append({"column": column, **value})
            else:
                items.append({"column": column, "blank_meaning": value})
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    rules: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        column = as_str(
            item.get("column")
            or item.get("metric")
            or item.get("header")
            or item.get("name")
        )
        if not column:
            continue
        rules.append(
            {
                "column": column,
                "kind": _blank_kind(item.get("kind") or item.get("type")),
                "blank_meaning": _blank_meaning(
                    item.get("blank_meaning") or item.get("meaning")
                ),
            }
        )
    return rules


def normalise_special_rules(raw: Any) -> dict[str, Any]:
    """Validate the model's self-report about blank cells."""
    report = raw if isinstance(raw, dict) else {}

    columns = report.get("columns_traversed")
    if isinstance(columns, list):
        traversed = [text for item in columns if (text := as_str(item))]
    else:
        traversed = []
        text = as_str(columns)
        if text:
            traversed = [part.strip() for part in text.split(",") if part.strip()]

    column_rules = report.get("column_blank_rules")
    if column_rules is None:
        column_rules = report.get("column_blank_semantics")

    return {
        "has_blank_cells": as_bool(report.get("has_blank_cells")),
        "blank_cell_meaning": as_str(report.get("blank_cell_meaning")),
        "blanks_inherit_base": as_bool(report.get("blanks_inherit_base")),
        "base_model_name": as_str(report.get("base_model_name")),
        "columns_traversed": traversed,
        "column_blank_rules": normalise_column_blank_rules(column_rules),
        "notes": as_str(report.get("notes")) or "",
    }


def should_fill(report: dict[str, Any]) -> bool:
    """Decide whether blank cells mean 'inherits the base configuration'.

    An explicit boolean from the model wins.  Only when it is absent do we fall
    back to keyword matching on the quoted caption/notes.  "Not reported" and
    "not tested" never trigger a fill.
    """
    flag = report.get("blanks_inherit_base")
    if flag is True:
        return True
    if flag is False:
        return False

    text = " ".join(
        part
        for part in (report.get("blank_cell_meaning"), report.get("notes"))
        if isinstance(part, str) and part
    )
    if not text:
        return False
    if NOT_MEANINGFUL_BLANK_RE.search(text) and not INHERIT_BLANK_RE.search(text):
        return False
    return bool(INHERIT_BLANK_RE.search(text))


def _norm_model_name(name: Any) -> str:
    text = " ".join(str(name or "").lower().split())
    text = re.sub(r"\[[^\]]*\]", "", text)  # citation markers
    text = text.replace("model", " ")  # "base model" == "base"
    return re.sub(r"[^a-z0-9]+", "", text)


def find_base_record(
    exp_group: list[dict[str, Any]], base_model_name: str
) -> dict[str, Any] | None:
    target = _norm_model_name(base_model_name)
    if not target:
        return None

    exact = [row for row in exp_group if _norm_model_name(row.get("model_name")) == target]
    if exact:
        return exact[0]

    partial = [
        row
        for row in exp_group
        if target in _norm_model_name(row.get("model_name"))
        or _norm_model_name(row.get("model_name")) in target
    ]
    if not partial:
        return None
    # The base row normally carries the most hyperparameters; among ties pick
    # the shortest label (the bare "base" rather than "base (A, h=1)").
    return min(
        partial,
        key=lambda row: (
            -len(row.get("hyperparams") or {}),
            len(str(row.get("model_name") or "")),
        ),
    )


def fill_missing_values(
    exp_group: list[dict[str, Any]], base_model_name: str
) -> dict[str, Any]:
    """Copy the base row's hyperparams into incomplete group members.

    Only called when the special-rules report explicitly states that blank
    cells inherit the base configuration.  Mutates ``exp_group`` in place and
    returns a small audit record.  Metric records (including parameter-count
    metrics) and model names are never copied.
    """
    audit: dict[str, Any] = {
        "triggered": True,
        "base_model_name": base_model_name,
        "base_found": False,
        "records_filled": 0,
        "fields_filled": 0,
        "reason": None,
    }
    if not base_model_name:
        audit["reason"] = "no base_model_name in report"
        return audit

    base = find_base_record(exp_group, base_model_name)
    if base is None:
        audit["reason"] = "base record not found in group"
        LOG.warning(
            "Blank cells inherit the base, but %r is not among the extracted rows",
            base_model_name,
        )
        return audit

    audit["base_found"] = True
    base_hyperparams = dict(base.get("hyperparams") or {})

    for row in exp_group:
        if row is base:
            continue

        changed = 0
        hyperparams = dict(row.get("hyperparams") or {})
        for key, value in base_hyperparams.items():
            if key not in hyperparams or hyperparams[key] in (None, ""):
                hyperparams[key] = value
                changed += 1
        if hyperparams:
            row["hyperparams"] = hyperparams

        if changed:
            audit["records_filled"] += 1
            audit["fields_filled"] += changed

    LOG.info(
        "    fill_missing_values: base=%r, %d rows completed (%d fields)",
        base_model_name,
        audit["records_filled"],
        audit["fields_filled"],
    )
    return audit


def _norm_metric_name(name: Any) -> str:
    """Compare metric labels: 'params ×10^6' == 'params*10^6' == 'Params (x10^6)'."""
    text = "" if name is None else str(name).lower()
    # "params \times 10^6" (MinerU LaTeX) == "params ×10^6" == "params*10^6".
    text = re.sub(r"\\(?:times|cdot|ast)\b", "x", text)
    text = text.replace("×", "x").replace("·", "x").replace("−", "-")
    text = re.sub(
        r"[*x]\s*10\s*(?:\^|\*\*)?\s*\{?\s*(\d+)\s*\}?", r"e\1", text
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def metric_matches_column(metric: Any, column: Any) -> bool:
    left, right = _norm_metric_name(metric), _norm_metric_name(column)
    if not left or not right:
        return False
    if left == right:
        return True
    # "FLOPs" should match "Training Cost (FLOPs)".
    return len(left) >= 3 and len(right) >= 3 and (left in right or right in left)


def inheritable_metric_columns(report: dict[str, Any]) -> list[str]:
    """Columns whose blank cells the report says are inherited from the base row.

    Only intrinsic attributes (params, FLOPs, cost) qualify.  Evaluation
    metrics (BLEU, PPL, F1) are never restored: their blanks mean the variant
    was not evaluated and must stay missing.
    """
    global_inherit = report.get("blanks_inherit_base") is True
    columns: list[str] = []
    for rule in report.get("column_blank_rules") or []:
        if rule.get("kind") != "intrinsic":
            continue
        meaning = rule.get("blank_meaning")
        if meaning == "inherits_base" or (meaning == "unknown" and global_inherit):
            if rule["column"] not in columns:
                columns.append(rule["column"])
    return columns


def _same_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when dataset/task agree, treating an absent value as compatible."""
    def norm(value: Any) -> str:
        return " ".join(str(value).lower().split()) if value else ""

    for field in ("dataset", "task"):
        a, b = norm(left.get(field)), norm(right.get(field))
        if a and b and a != b:
            return False
    return True


def fill_inherited_metrics(
    exp_group: list[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    """Copy an intrinsic-attribute metric (params, FLOPs) from base to variants.

    The model reports which columns are intrinsic attributes whose blanks mean
    "identical to the base model".  For each such column, every variant row
    that has no value gets a record copied from the base row -- with the
    variant's own `model_name` and `hyperparams`.  Evaluation-metric columns
    are excluded before this function is called, so a missing BLEU/PPL is
    never invented.
    """
    audit: dict[str, Any] = {
        "triggered": False,
        "base_model_name": as_str(report.get("base_model_name")),
        "base_found": False,
        "columns": [],
        "metrics": [],
        "records_filled": 0,
        "skipped_non_ours": 0,
        "reason": None,
    }

    columns = inheritable_metric_columns(report)
    if not columns:
        audit["reason"] = "no intrinsic column reported as inheriting the base"
        return audit
    audit["triggered"] = True
    audit["columns"] = columns

    base_model_name = as_str(report.get("base_model_name"))
    if not base_model_name:
        audit["reason"] = "no base_model_name in report"
        return audit

    anchor = find_base_record(exp_group, base_model_name)
    if anchor is None:
        audit["reason"] = "base record not found in group"
        LOG.warning(
            "Blank intrinsic attributes inherit the base, but %r is not among "
            "the extracted rows",
            base_model_name,
        )
        return audit

    base_key = _norm_model_name(anchor.get("model_name"))
    base_rows = [
        row for row in exp_group if _norm_model_name(row.get("model_name")) == base_key
    ]
    templates = [
        row
        for row in base_rows
        if any(metric_matches_column(row.get("metric"), column) for column in columns)
    ]
    if not templates:
        audit["reason"] = "the base row has no record for the inheritable columns"
        return audit
    audit["base_found"] = True

    variants: dict[str, list[dict[str, Any]]] = {}
    for row in exp_group:
        key = _norm_model_name(row.get("model_name"))
        if not key or key == base_key:
            continue
        variants.setdefault(key, []).append(row)

    def already_present(rows: list[dict[str, Any]], template: dict[str, Any]) -> bool:
        target = _norm_metric_name(template.get("metric"))
        return any(
            _norm_metric_name(row.get("metric")) == target
            and _same_context(row, template)
            for row in rows
        )

    additions: list[dict[str, Any]] = []
    for rows in variants.values():
        if not any(row.get("is_ours") for row in rows):
            audit["skipped_non_ours"] += 1
            continue

        variant_name = as_str(rows[0].get("model_name"))
        variant_hp = next(
            (row.get("hyperparams") for row in rows if row.get("hyperparams")), None
        )
        for template in templates:
            if already_present(rows, template):
                continue
            record = dict(template)
            record["model_name"] = variant_name
            record["is_ours"] = True
            if variant_hp:
                record["hyperparams"] = dict(variant_hp)
            additions.append(record)
            rows.append(record)  # keep later templates from duplicating it
            audit["records_filled"] += 1
            metric = as_str(template.get("metric"))
            if metric and metric not in audit["metrics"]:
                audit["metrics"].append(metric)

    if additions:
        exp_group.extend(additions)

    LOG.info(
        "    fill_inherited_metrics: base=%r, %d records restored for %s",
        base_model_name,
        audit["records_filled"],
        ", ".join(audit["metrics"]) or "(none)",
    )
    return audit


def synchronise_row_metadata(exp_group: list[dict[str, Any]]) -> int:
    """Share metadata across records of the same configuration.

    One table row often becomes several records (one per metric column).  The
    model may list `h`, `d_k`, ... on only one of them.  Before filling blanks
    from the base row we propagate values between siblings, but only for keys
    whose explicit values do not conflict -- a generic model name such as "(A)"
    must never merge the h=1 and h=16 configurations.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in exp_group:
        by_model.setdefault(_norm_model_name(row.get("model_name")), []).append(row)

    changed_rows = 0
    for rows in by_model.values():
        if len(rows) < 2:
            continue

        values: dict[str, dict[str, Any]] = {}
        for row in rows:
            for name, value in (row.get("hyperparams") or {}).items():
                if value not in (None, ""):
                    values.setdefault(name, {})[json.dumps(value, sort_keys=True)] = value

        agreed = {
            name: next(iter(options.values()))
            for name, options in values.items()
            if len(options) == 1
        }

        for row in rows:
            changed = False
            hyperparams = dict(row.get("hyperparams") or {})
            for name, value in agreed.items():
                if name not in hyperparams or hyperparams[name] in (None, ""):
                    hyperparams[name] = value
                    changed = True
            if hyperparams:
                row["hyperparams"] = hyperparams
            if changed:
                changed_rows += 1

    return changed_rows


def experiment_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def norm(value: Any) -> Any:
        return " ".join(str(value).lower().split()) if value is not None else ""

    return tuple(
        norm(row.get(field))
        for field in (
            "task",
            "dataset",
            "model_name",
            "metric",
            "value",
            "is_ours",
            json.dumps(row.get("hyperparams"), sort_keys=True, ensure_ascii=False),
            json.dumps(row.get("domain_specific"), sort_keys=True, ensure_ascii=False),
        )
    )


# --------------------------------------------------------------------------- #
# Label and claim normalisation (text stage)
# --------------------------------------------------------------------------- #

WORD_SEP_RE = re.compile(r"[\s_]+")
NON_WORD_RE = re.compile(r"[^\w]+")


def slugify(text: str) -> str:
    """Lowercase a label and join its words with underscores.

    ``\\w`` is Unicode-aware, so a CJK label survives instead of collapsing to
    an empty string the way an ASCII-only filter would.
    """
    cleaned = NON_WORD_RE.sub("_", text.strip().lower())
    return WORD_SEP_RE.sub("_", cleaned).strip("_")


def singularise(tag: str) -> str:
    if tag.endswith("ies") and len(tag) > 4:
        return tag[:-3] + "y"
    if tag.endswith("s") and not tag.endswith("ss") and len(tag) > 3:
        return tag[:-1]
    return tag


# A label the model coins itself must be an English lowercase word joined by
# underscores, so the vocabulary cannot fill up with prose, CJK or code.
NEW_CHUNK_TYPE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
MAX_CHUNK_TYPE_CHARS = 48


def is_usable_chunk_type(tag: str) -> bool:
    return (
        bool(tag)
        and len(tag) <= MAX_CHUNK_TYPE_CHARS
        and NEW_CHUNK_TYPE_RE.match(tag) is not None
    )


class ChunkVocabulary:
    """The chunk_types label library, persisted as JSON.

    The file holds the canonical labels and the synonym table that folds
    near-misses onto them::

        {"chunk_types": ["implementation_detail", ...],
         "aliases": {"code_link": "source_code", ...}}

    A bare JSON array is accepted as well, for a hand-written file.  The
    vocabulary grows as the pipeline runs: a label the model coins that no
    vocabulary word covers is appended to `chunk_types` and saved, so the next
    prompt already offers it and the model stops coining synonyms for it.
    """

    def __init__(
        self,
        path: Path,
        types: Iterable[str] = (),
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.path = path
        self.types: list[str] = list(types)
        self.aliases: dict[str, str] = dict(aliases or {})
        # New labels this run appended to the file, and how many model labels
        # were folded onto an existing word instead of becoming new ones.
        self.added: list[str] = []
        self.mapped: int = 0
        self._dirty = False
        self._reindex()

    @classmethod
    def load(cls, path: Path, create: bool = True) -> ChunkVocabulary:
        """Read the vocabulary file, creating it from the seeds when absent.

        A missing or unreadable file is not fatal: the built-in standard labels
        keep the pipeline running, and a file that cannot be written only costs
        the learning, not the run.
        """
        if not path.is_file():
            vocab = cls(path, SEED_CHUNK_TYPES, SEED_CHUNK_TYPE_ALIASES)
            if create:
                try:
                    vocab.save()
                except OSError as exc:
                    LOG.warning("Cannot create %s: %s", path, exc)
                else:
                    LOG.info(
                        "Created %s with the %d standard label(s)",
                        path,
                        len(vocab.types),
                    )
            return vocab

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning(
                "Cannot read %s (%s); using the built-in standard vocabulary",
                path,
                exc,
            )
            return cls(path, SEED_CHUNK_TYPES, SEED_CHUNK_TYPE_ALIASES)

        if isinstance(data, list):
            raw_types, raw_aliases = data, {}
        elif isinstance(data, dict):
            raw_types = data.get("chunk_types")
            raw_aliases = data.get("aliases")
        else:
            LOG.warning(
                "Ignoring %s: expected a JSON array or object, got %s",
                path,
                type(data).__name__,
            )
            return cls(path, SEED_CHUNK_TYPES, SEED_CHUNK_TYPE_ALIASES)

        types: list[str] = []
        for item in raw_types if isinstance(raw_types, list) else []:
            tag = slugify(str(item))
            if tag and tag not in types:
                types.append(tag)
        if not types:
            LOG.warning(
                "%s lists no usable label; falling back to the standard vocabulary",
                path,
            )
            types = list(SEED_CHUNK_TYPES)

        # The file's own table wins; the seeds fill in what it does not cover.
        aliases = dict(SEED_CHUNK_TYPE_ALIASES)
        if isinstance(raw_aliases, dict):
            for alias, canonical in raw_aliases.items():
                alias_tag = slugify(str(alias))
                canonical_tag = slugify(str(canonical))
                if alias_tag and canonical_tag:
                    aliases[alias_tag] = canonical_tag

        return cls(path, types, aliases)

    def _reindex(self) -> None:
        """Index the labels by their sorted tokens, so word order cannot fork."""
        self._by_tokens: dict[str, str] = {}
        for label in self.types:
            self._by_tokens.setdefault(self._token_key(label), label)

    @staticmethod
    def _token_key(tag: str) -> str:
        return "_".join(sorted(tag.split("_")))

    @property
    def dirty(self) -> bool:
        return self._dirty

    def lookup(self, tag: str) -> str | None:
        """Resolve a slugified label to a vocabulary word, or None if it is new."""
        if not tag:
            return None
        if tag in self.types:
            return tag
        if tag in self.aliases:
            return self.aliases[tag]
        # Generic fallback: "experiment_results" -> "experiment_result".
        singular = singularise(tag)
        if singular != tag:
            if singular in self.types:
                return singular
            if singular in self.aliases:
                return self.aliases[singular]
        # "setup_experiment" is the same label as "experiment_setup".
        return self._by_tokens.get(self._token_key(tag))

    def add(self, tag: str) -> bool:
        """Append a genuinely new label, marking the file for saving."""
        if tag in self.types or not is_usable_chunk_type(tag):
            return False
        self.types.append(tag)
        self._by_tokens.setdefault(self._token_key(tag), tag)
        self.added.append(tag)
        self._dirty = True
        return True

    def save(self) -> None:
        """Write the vocabulary out, replacing the file atomically."""
        payload = {
            "chunk_types": self.types,
            "aliases": dict(sorted(self.aliases.items())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        self._dirty = False

    def render(self) -> str:
        """The vocabulary as one prompt-ready line."""
        return ", ".join(self.types)


def load_chunk_vocabulary(path: Path, create: bool = True) -> ChunkVocabulary:
    vocab = ChunkVocabulary.load(path, create=create)
    LOG.info(
        "Chunk vocabulary: %d label(s) from %s", len(vocab.types), vocab.path
    )
    return vocab


def normalise_chunk_types(raw: Any, vocab: ChunkVocabulary) -> list[str]:
    """Force model output into clean, deduplicated chunk labels.

    Accepts an array, a single string ("a, b"), or an object wrapping either.
    A label the vocabulary knows -- directly, as a synonym, as a plural or as
    the same words in another order -- becomes the vocabulary word.  A label
    the vocabulary does not know is a genuine addition and is appended to it,
    unless it is not English lowercase with underscores, in which case it is
    dropped.  Returns [] when nothing usable is present: an unlabelled block is
    dropped rather than stored under a placeholder label.
    """
    if isinstance(raw, dict):
        for key in CHUNK_TYPE_KEY_ALIASES:
            if key in raw:
                raw = raw[key]
                break
        else:
            lists = [value for value in raw.values() if isinstance(value, list)]
            raw = lists[0] if lists else []
    if isinstance(raw, str):
        raw = [part for part in re.split(r"[,\n;/|]+", raw) if part.strip()]
    if not isinstance(raw, (list, tuple, set)):
        return []

    tags: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = slugify(item)
        canonical = vocab.lookup(tag)
        if canonical is None:
            if not is_usable_chunk_type(tag):
                LOG.warning(
                    "Dropping label %r: a chunk label must be English lowercase "
                    "words joined by underscores",
                    item[:60],
                )
                continue
            canonical = tag
            if vocab.add(tag):
                LOG.info("New chunk type %r added to %s", tag, vocab.path)
        elif canonical != tag:
            vocab.mapped += 1
            LOG.debug("Label %r folded onto the existing %r", tag, canonical)
        if canonical not in tags:
            tags.append(canonical)
    return tags


def normalise_claim_type(value: Any) -> str | None:
    text = as_str(value)
    if not text:
        return None
    tag = slugify(text)
    if not tag:
        return None
    if tag in CLAIM_TYPES:
        return tag
    if tag in CLAIM_TYPE_SYNONYMS:
        return CLAIM_TYPE_SYNONYMS[tag]
    singular = singularise(tag)
    if singular in CLAIM_TYPES:
        return singular
    if singular in CLAIM_TYPE_SYNONYMS:
        return CLAIM_TYPE_SYNONYMS[singular]
    for claim_type in CLAIM_TYPES:
        if claim_type in tag:
            return claim_type
    return None


def infer_claim_type(text: str) -> str | None:
    """Last-resort classification when the model named no usable claim type."""
    for claim_type, pattern in CLAIM_TYPE_HINTS:
        if pattern.search(text):
            return claim_type
    return None


def claim_from_value(
    raw: Any, block: TextBlock, stats: TextStats
) -> list[dict[str, Any]]:
    """Normalise one block's `claim` field into zero or more claim rows.

    Handles the documented object, a list of objects, and the bare string some
    models answer with (a claim type name, or the claim text itself).
    """
    if raw is None or raw is False or raw == "" or raw == [] or raw == {}:
        return []
    if isinstance(raw, list):
        claims: list[dict[str, Any]] = []
        for item in raw:
            claims.extend(claim_from_value(item, block, stats))
        return claims
    if isinstance(raw, str):
        claim_type = normalise_claim_type(raw)
        text = block.content
        if claim_type is None:
            text = raw.strip()
            claim_type = infer_claim_type(text)
        if claim_type is None:
            stats.dropped_claims += 1
            LOG.warning(
                "Dropped string claim with unknown type (%s): %r",
                block.evidence_path,
                raw[:120],
            )
            return []
        return [
            {
                "claim_type": claim_type,
                "content": text,
                "domain_specific": None,
                "evidence_path": block.evidence_path,
            }
        ]

    if not isinstance(raw, dict):
        stats.dropped_claims += 1
        LOG.warning(
            "Dropped claim of type %s (%s)", type(raw).__name__, block.evidence_path
        )
        return []

    # Some models answer {"has_claim": false} or {"claim": null} for no claim.
    has_claim = raw.get("has_claim")
    if has_claim is not None and as_bool(has_claim) is False:
        return []

    content: str | None = None
    for key in CLAIM_CONTENT_KEY_ALIASES:
        value = as_str(raw.get(key))
        if value:
            content = value
            break

    claim_type: str | None = None
    for key in CLAIM_TYPE_KEY_ALIASES:
        if key not in raw:
            continue
        claim_type = normalise_claim_type(raw.get(key))
        if claim_type:
            break
    if claim_type is None and content:
        claim_type = infer_claim_type(content)
        if claim_type:
            LOG.warning(
                "Inferred claim_type=%r from text (%s); please verify",
                claim_type,
                block.evidence_path,
            )
    if claim_type is None:
        stats.dropped_claims += 1
        LOG.warning(
            "Dropped claim without a usable claim_type (%s): %r",
            block.evidence_path,
            str(raw)[:160],
        )
        return []

    if not content:
        # The claim lives in this block; fall back to the block's own text
        # rather than inventing wording.
        content = block.content
        LOG.warning(
            "Claim has no content (%s); using the block text", block.evidence_path
        )

    domain_specific = raw.get("domain_specific")
    if domain_specific is None:
        domain_specific = raw.get("details")

    return [
        {
            "claim_type": claim_type,
            "content": content,
            "domain_specific": sanitise_json(as_dict(domain_specific)),
            "evidence_path": block.evidence_path,
        }
    ]


def claim_key(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("content", "")).lower().split())


# --------------------------------------------------------------------------- #
# Table row self-check
# --------------------------------------------------------------------------- #


def _markdown_data_rows(text: str) -> int | None:
    """Data rows of a markdown pipe table, excluding its header and separators."""
    body = [
        line
        for line in text.splitlines()
        if "|" in line
        and line.strip()
        and not all(
            MD_SEPARATOR_CELL_RE.match(cell.strip())
            for cell in line.strip().strip("|").split("|")
        )
    ]
    if not body:
        return None
    # The first surviving line is the header.
    return max(0, len(body) - 1)


def _html_data_rows(text: str) -> int | None:
    """Data rows of an HTML table, excluding its header row(s).

    A first row that groups columns with colspan means a two-level head (the
    sub-labels sit in the row beneath it); that is how MinerU writes the
    "BLEU" / "EN-DE EN-FR" style headers.
    """
    rows = HTML_ROW_BLOCK_RE.findall(text)
    if not rows:
        return None
    header = 0
    for row in rows[:3]:
        if "<th" in row.lower():
            header += 1
        else:
            break
    if not header:
        header = 2 if "colspan" in rows[0].lower() and len(rows) > 1 else 1
    return max(0, len(rows) - header)


def count_table_data_rows(table: str) -> int | None:
    """Best-effort count of the data rows a table prints; None if unrecognised.

    Heuristic on purpose: the number only feeds a warning, it is never used to
    accept or reject a record.
    """
    if not table.strip():
        return None
    if HTML_ROW_RE.search(table):
        return _html_data_rows(table)
    return _markdown_data_rows(table)


def check_table_rows(unit: TableUnit, record_count: int) -> dict[str, Any]:
    """Compare how many records the model returned with the table's own rows.

    Warns when rows appear to have been lost: either the model returned fewer
    records than there are data rows, or it returned none at all from a table
    that is not pure theory.  Theory tables (Complexity, O(n^2)) are exempt,
    because they carry no measurable value to report.
    """
    data_rows = count_table_data_rows(unit.table)
    theoretical = bool(THEORETICAL_TABLE_RE.search(unit.table))
    check: dict[str, Any] = {
        "table_data_rows": data_rows,
        "model_records": record_count,
        "theoretical": theoretical,
        "lost_rows": None,
        "warned": False,
    }
    if data_rows is None:
        check["reason"] = "row count not recognised"
        return check

    if record_count > 0 and record_count < data_rows:
        check["lost_rows"] = data_rows - record_count
        check["warned"] = True
        LOG.warning(
            "WARNING: 表格行数丢失！原表 %d 行，模型仅返回 %d 行 (%s)",
            data_rows,
            record_count,
            unit.evidence_path,
        )
    elif record_count == 0 and data_rows > 0 and not theoretical:
        check["lost_rows"] = data_rows
        check["warned"] = True
        LOG.warning(
            "WARNING: 表格行数丢失！原表 %d 行，模型返回 0 行 (%s)",
            data_rows,
            unit.evidence_path,
        )
    elif record_count == 0 and theoretical:
        check["reason"] = "theoretical table, no records expected"
    return check


def row_warnings_from_reports(
    reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The row-count checks that fired, for the closing summary."""
    warnings = []
    for report in reports:
        check = report.get("row_check")
        if isinstance(check, dict) and check.get("warned"):
            warnings.append(
                {
                    "evidence_path": report.get("evidence_path"),
                    "table_data_rows": check.get("table_data_rows"),
                    "model_records": check.get("model_records"),
                    "lost_rows": check.get("lost_rows"),
                }
            )
    return warnings


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #


def extract_metadata(client: LLMClient, tree: dict[str, Any]) -> dict[str, Any]:
    tree_title = as_str(tree.get("title")) or ""
    front_matter = truncate(front_matter_text(tree), 4_000)
    abstract = truncate(
        section_text(tree, re.compile(r"\babstract\b", re.IGNORECASE)), 8_000
    )

    result = client.chat_json(
        METADATA_SYSTEM,
        metadata_prompt(tree_title, front_matter, abstract),
        expect="object",
    )

    title = as_str(result.get("title")) or tree_title
    abstract_out = as_str(result.get("abstract")) or abstract
    domain = normalise_domain(result.get("domain"))
    if domain is None:
        LOG.warning(
            "Invalid domain %r; falling back to 'ML' (please verify manually)",
            result.get("domain"),
        )
        domain = "ML"

    authors_raw = result.get("authors")
    authors = []
    if isinstance(authors_raw, list):
        for item in authors_raw:
            name = as_str(item)
            if name and name not in authors:
                authors.append(name)

    if not authors:
        # Deterministic fallback for MinerU front matter such as "**Jane Doe**".
        for name in re.findall(r"\*\*(.+?)\*\*", front_matter):
            name = re.sub(r"[∗*†‡]+", "", name).strip()
            if name and len(name) < 80 and name not in authors:
                authors.append(name)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract_out,
        "domain": domain,
        "method_tags": normalise_method_tags(result.get("method_tags")),
    }


def extract_experiments(
    client: LLMClient,
    tree: dict[str, Any],
    args: argparse.Namespace,
    resume: ResumeState | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one isolated LLM call per table, then apply Python-only repairs.

    A table whose call fails after every retry is skipped and reported; the
    rest of the paper is still extracted.  If *every* table fails the run is
    aborted, because that means the API is unusable -- and an empty result
    would otherwise be indistinguishable from a paper that reports no
    experiments at all.

    A paper with no table at all is not that case: the stage returns no rows
    without making a single call and without aborting, so a purely theoretical
    paper proceeds to its text stage normally.  The abort test is therefore
    `units and every unit failed`, never `no rows came back`.

    With `resume`, tables whose evidence path the database already holds are
    dropped before the first call.  Only the tables left over count towards
    the abort rule above, so a resumed run that has nothing left to do
    finishes cleanly instead of looking like a dead API.
    """
    units = build_table_units(tree)
    paper_title = as_str(tree.get("title")) or ""
    stats = Stats()

    total_units = len(units)
    if resume is not None:
        units = [
            unit for unit in units if unit.evidence_path not in resume.experiment_paths
        ]
        stats.skipped_resume = total_units - len(units)
        LOG.info(
            "Resume: %d of %d table(s) already in the database; %d left to extract",
            stats.skipped_resume,
            total_units,
            len(units),
        )

    if not units:
        LOG.info(
            "Experiments: no table to extract (%d in the tree, %d skipped on "
            "resume); the stage is skipped without a call and with 0 rows",
            total_units,
            stats.skipped_resume,
        )
        return [], []

    LOG.info("Experiment sources: %d table(s); prose is excluded", len(units))

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    last_table_error: Exception | None = None

    for position, unit in enumerate(units, start=1):
        LOG.info(
            "[%d/%d] %s, block %d (%d chars; context: prev=%s next=%s)",
            position,
            len(units),
            unit.sec_title,
            unit.block_index,
            len(unit.table),
            "yes" if unit.preceding else "no",
            "yes" if unit.following else "no",
        )

        report: dict[str, Any] = {
            "evidence_path": unit.evidence_path,
            "sec_title": unit.sec_title,
            "caption": unit.caption or None,
            "table_chars": len(unit.table),
            "group_id": None,
            "record_count": 0,
            "embedded_metric_records": 0,
            "sibling_rows_synchronised": 0,
            "special_rules_report": None,
            "fill": None,
            "fill_metrics": None,
            "row_check": None,
            "error": None,
        }

        try:
            raw = client.chat_json(
                TABLE_SYSTEM,
                table_prompt(unit, paper_title, args.max_chars),
                expect="table",
            )
        except LLMError as exc:
            stats.failed_tables += 1
            last_table_error = exc
            report["error"] = str(exc)
            reports.append(report)
            LOG.error(
                "Table %d/%d failed after every retry, skipping it: %s",
                position,
                len(units),
                exc,
            )
            continue

        group_id = str(uuid.uuid4())
        report["group_id"] = group_id

        raw_records = raw.get("records")
        if not isinstance(raw_records, list):
            LOG.warning(
                "Model returned no 'records' array for %s (keys: %s)",
                unit.evidence_path,
                sorted(raw),
            )
            raw_records = []

        group_rows: list[dict[str, Any]] = []
        embedded_total = 0
        for raw_row in raw_records:
            record = normalise_experiment_row(raw_row, unit, group_id)
            embedded = embedded_metric_records(raw_row, unit, group_id)
            if record is not None:
                group_rows.append(record)
            group_rows.extend(embedded)
            embedded_total += len(embedded)
            if record is None and not embedded:
                stats.dropped += 1

        report["embedded_metric_records"] = embedded_total
        report["sibling_rows_synchronised"] = synchronise_row_metadata(group_rows)

        # Compare what the model returned with the rows the table itself
        # prints.  Measured before the fill steps below, those only ever add
        # rows and would mask a shortfall.
        report["row_check"] = check_table_rows(unit, len(group_rows))
        if report["row_check"]["warned"]:
            stats.row_warnings += 1

        rules = normalise_special_rules(raw.get("special_rules_report"))
        report["special_rules_report"] = rules

        # Rule 4: the report decides whether Python is allowed to fill blanks.
        # Hyperparameters and intrinsic metrics are filled on separate rules:
        # the global flag covers configuration cells, the per-column report
        # covers architecture attributes such as params or FLOPs.
        if should_fill(rules):
            report["fill"] = fill_missing_values(group_rows, rules["base_model_name"])
            stats.filled_rows += report["fill"]["records_filled"]
        else:
            report["fill"] = {
                "triggered": False,
                "reason": "blanks do not mean 'inherit base' (or no blank cells)",
            }
        report["fill_metrics"] = fill_inherited_metrics(group_rows, rules)
        stats.inherited_metric_rows += report["fill_metrics"]["records_filled"]

        kept = 0
        for record in group_rows:
            key = experiment_key(record)
            if not args.keep_duplicates:
                if key in seen:
                    stats.deduped += 1
                    continue
                seen.add(key)
            rows.append(record)
            kept += 1

        report["record_count"] = kept
        reports.append(report)

        if not group_rows:
            stats.empty_tables += 1
            LOG.info("    -> 0 records (theoretical table or no concrete numbers)")
        else:
            LOG.info(
                "    -> %d records (group %s) | blanks=%s inherit=%s base=%r "
                "| intrinsic restored=%d",
                kept,
                group_id,
                rules["has_blank_cells"],
                rules["blanks_inherit_base"],
                rules["base_model_name"],
                report["fill_metrics"]["records_filled"],
            )
            if rules["columns_traversed"]:
                LOG.info(
                    "    columns traversed: %s", ", ".join(rules["columns_traversed"])
                )
            for rule in rules["column_blank_rules"]:
                LOG.info(
                    "    column %r: %s -> %s",
                    rule["column"],
                    rule["kind"],
                    rule["blank_meaning"],
                )

    if units and stats.failed_tables == len(units):
        raise APIUnavailableError(
            f"all {len(units)} table call(s) failed, the model API is "
            f"unusable -- aborting (last error: {last_table_error})"
        )

    LOG.info(
        "Experiments: %d rows from %d tables "
        "(%d empty, %d failed, %d dropped, %d duplicates, "
        "%d rows filled, %d intrinsic metrics restored, %d skipped on resume, "
        "%d row-count warning(s))",
        len(rows),
        len(units),
        stats.empty_tables,
        stats.failed_tables,
        stats.dropped,
        stats.deduped,
        stats.filled_rows,
        stats.inherited_metric_rows,
        stats.skipped_resume,
        stats.row_warnings,
    )
    return rows, reports


def chunk_text_blocks(
    blocks: Sequence[TextBlock], batch_size: int
) -> list[list[TextBlock]]:
    return [
        list(blocks[start : start + batch_size])
        for start in range(0, len(blocks), batch_size)
    ]


def match_element(
    element: dict[str, Any], batch: Sequence[TextBlock], used: set[int]
) -> int | None:
    """Resolve a model element back to a block in the batch.

    The id we printed is tried first, then an exact/near-exact content match,
    which rescues answers that echoed a different numbering.
    """
    raw_id = None
    for key in BLOCK_ID_KEY_ALIASES:
        if key in element:
            raw_id = as_int(element.get(key))
            if raw_id is not None:
                break
    if raw_id is not None and 1 <= raw_id <= len(batch):
        index = raw_id - 1
        if index not in used:
            return index

    content = None
    for key in ("content", "text"):
        content = as_str(element.get(key))
        if content:
            break
    if content:
        needle = " ".join(content.split())[:120]
        for index, block in enumerate(batch):
            if index in used:
                continue
            if " ".join(block.content.split())[:120] == needle:
                return index

    if raw_id is not None and 0 <= raw_id < len(batch) and raw_id not in used:
        # Zero-based numbering.
        return raw_id
    return None


def extract_text_layer(
    client: LLMClient,
    tree: dict[str, Any],
    args: argparse.Namespace,
    resume: ResumeState | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], TextStats]:
    """Label and claim-extract every prose block, then build the insert rows.

    One LLM call per batch of `--batch-size` blocks does both jobs, and the
    answer is fanned out into the two in-memory lists: every block becomes a
    text chunk, and every block carrying a claim becomes a claim row as well.

    Equation blocks are handled before any of that and never reach the model:
    `build_equation_blocks` copies their LaTeX out in Python and they are
    forced to the `formula_latex` label, so they cost no call, carry no claim
    and their content is stored exactly as the tree printed it.  They join the
    text chunks for embedding and storage.

    A batch whose call fails after every retry contributes nothing: its blocks
    are dropped outright rather than stored under a placeholder label, so a
    dead or unreachable API cannot fill the tables with junk.  A block the
    model answered without usable labels is dropped the same way.  If *every*
    batch fails the run is aborted, because that means the API is down rather
    than that a few blocks were awkward.

    With `resume`, a block whose evidence path the database already holds is
    dropped before the first call: both its chunk and its claim are in there,
    so re-sending it would only buy a duplicate.  Equations are filtered the
    same way, so a resumed run does not re-insert them either.  Only the blocks
    left over are batched and count towards the abort rule above.

    The chunk labels come from `--vocab`, which is read once and shown to the
    model with every batch.  Labels are resolved as each batch is answered, so
    a word coined in one batch is part of the vocabulary of the next, and the
    file is saved whenever it has grown.
    """
    stats = TextStats()
    blocks = build_text_blocks(tree, args.min_chars, args.include_preamble, stats)
    equations = build_equation_blocks(tree, args.include_preamble, stats)

    total_blocks = len(blocks)
    total_equations = len(equations)
    if resume is not None:
        blocks = [block for block in blocks if block.evidence_path not in resume.text_paths]
        stats.skipped_resume = total_blocks - len(blocks)
        equations = [
            block for block in equations if block.evidence_path not in resume.text_paths
        ]
        stats.equations_skipped_resume = total_equations - len(equations)
        LOG.info(
            "Resume: %d of %d text block(s) and %d of %d equation(s) already in "
            "the database; %d text block(s) left to label and %d equation(s) "
            "left to copy",
            stats.skipped_resume,
            total_blocks,
            stats.equations_skipped_resume,
            total_equations,
            len(blocks),
            len(equations),
        )

    stats.blocks = len(blocks)
    LOG.info(
        "Text blocks: %d to process (%d preamble, %d boilerplate, "
        "%d too-short and %d already-stored block(s) skipped); %d equation(s) "
        "to copy in Python (%d already stored, no model call)",
        len(blocks),
        stats.skipped_preamble,
        stats.skipped_boilerplate,
        stats.skipped_short,
        stats.skipped_resume,
        len(equations),
        stats.equations_skipped_resume,
    )

    paper_title = as_str(tree.get("title")) or ""
    batches = chunk_text_blocks(blocks, args.batch_size)
    stats.batches = len(batches)

    # Loaded once per run and handed to every batch prompt: a label coined in
    # batch 1 is offered to the model in batch 2, and a label the model reuses
    # is recognised as an existing word rather than added a second time.
    vocab = load_chunk_vocabulary(args.vocab)
    vocab_save_failed = False

    parsed: dict[int, dict[str, Any]] = {}
    labels: dict[int, list[str]] = {}
    failed_blocks: set[int] = set()
    last_batch_error: Exception | None = None
    offset = 0
    for position, batch in enumerate(batches, start=1):
        batch_start, offset = offset, offset + len(batch)
        first, last = batch[0].evidence_path, batch[-1].evidence_path
        LOG.info(
            "[%d/%d] %d block(s): %s ... %s",
            position,
            len(batches),
            len(batch),
            first,
            last,
        )
        try:
            raw_elements = client.chat_json(
                UNIFIED_TEXT_SYSTEM,
                unified_text_prompt(batch, paper_title, args.text_max_chars, vocab),
                expect="array",
            )
        except LLMError as exc:
            stats.failed_batches += 1
            last_batch_error = exc
            failed_blocks.update(range(batch_start, batch_start + len(batch)))
            LOG.error(
                "Batch %d/%d failed after every retry; dropping its %d block(s): %s",
                position,
                len(batches),
                len(batch),
                exc,
            )
            if args.llm_sleep > 0 and position < len(batches):
                time.sleep(args.llm_sleep)
            continue

        used: set[int] = set()
        for element in raw_elements:
            if not isinstance(element, dict):
                stats.stray_elements += 1
                LOG.warning("Ignoring non-object element in batch %d: %r",
                            position, str(element)[:120])
                continue
            index = match_element(element, batch, used)
            if index is None:
                stats.stray_elements += 1
                LOG.warning(
                    "Batch %d: element does not match any block (block_index=%r)",
                    position,
                    element.get("block_index"),
                )
                continue
            used.add(index)
            parsed[batch_start + index] = element
            # Resolved here, not in the fan-out below, so that a label coined
            # in this batch is in the vocabulary of the next batch's prompt.
            labels[batch_start + index] = normalise_chunk_types(
                _first_present(element, CHUNK_TYPE_KEY_ALIASES), vocab
            )

        if len(used) < len(batch):
            LOG.warning(
                "Batch %d: model answered for %d of %d block(s); the unanswered "
                "one(s) will be dropped",
                position,
                len(used),
                len(batch),
            )
        LOG.info("    -> %d element(s) matched", len(used))

        # Persisted after every batch: a run that dies halfway keeps the labels
        # it has already learned instead of coining them again next time.
        if vocab.dirty and not vocab_save_failed:
            try:
                vocab.save()
            except OSError as exc:
                vocab_save_failed = True
                LOG.warning(
                    "Cannot save the chunk vocabulary to %s: %s; new labels "
                    "are used for the rest of this run but not persisted",
                    vocab.path,
                    exc,
                )

        if args.llm_sleep > 0 and position < len(batches):
            time.sleep(args.llm_sleep)

    stats.blocks_lost_to_failed_batches = len(failed_blocks)
    if stats.batches and stats.failed_batches == stats.batches:
        raise APIUnavailableError(
            f"all {stats.batches} text batch(es) failed, the model API is "
            f"unusable -- aborting (last error: {last_batch_error})"
        )

    # --- fan out into the two insert lists ------------------------------- #
    claims_to_insert: list[dict[str, Any]] = []
    text_chunks_to_insert: list[dict[str, Any]] = [
        equation_chunk(block) for block in equations
    ]
    seen_claims: set[str] = set()

    for index, block in enumerate(blocks):
        if index in failed_blocks:
            continue
        element = parsed.get(index)
        if element is None:
            stats.unanswered_blocks += 1
            LOG.warning(
                "No answer for %s; the block is dropped rather than stored "
                "with a placeholder label",
                block.evidence_path,
            )
            continue

        echoed = as_str(element.get("content"))
        if echoed and " ".join(echoed.split()) != " ".join(block.content.split()):
            stats.content_mismatches += 1
            LOG.debug("Model rewrote the text of %s; the original is stored",
                      block.evidence_path)

        chunk_types = labels.get(index) or []
        if chunk_types:
            text_chunks_to_insert.append(
                {
                    "sec_title": block.sec_title,
                    "content": block.content,
                    "evidence_path": block.evidence_path,
                    "chunk_types": chunk_types,
                }
            )
        else:
            stats.unlabeled_blocks += 1
            LOG.warning(
                "No usable chunk_types for %s; the chunk is dropped",
                block.evidence_path,
            )

        for claim in claim_from_value(_first_present(element, ("claim",)), block, stats):
            key = claim_key(claim)
            if key in seen_claims:
                stats.duplicate_claims += 1
                continue
            seen_claims.add(key)
            claims_to_insert.append(claim)

    stats.claims = len(claims_to_insert)
    LOG.info(
        "Text layer: %d chunk(s) and %d claim(s) kept "
        "(%d equation(s) copied in Python; %d block(s) already stored, "
        "%d dropped in %d failed batch(es), "
        "%d unanswered, %d unlabelled, %d duplicate claim(s), %d invalid "
        "claim(s); %d stray element(s), %d rewritten block(s))",
        len(text_chunks_to_insert),
        len(claims_to_insert),
        len(equations),
        stats.skipped_resume,
        stats.blocks_lost_to_failed_batches,
        stats.failed_batches,
        stats.unanswered_blocks,
        stats.unlabeled_blocks,
        stats.duplicate_claims,
        stats.dropped_claims,
        stats.stray_elements,
        stats.content_mismatches,
    )
    if vocab.added:
        LOG.info(
            "Chunk vocabulary: %d label(s), %d new this run: %s",
            len(vocab.types),
            len(vocab.added),
            ", ".join(vocab.added),
        )
    else:
        LOG.info(
            "Chunk vocabulary: %d label(s), none added (%d label(s) folded "
            "onto an existing word)",
            len(vocab.types),
            vocab.mapped,
        )
    return claims_to_insert, text_chunks_to_insert, stats


def _first_present(source: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def _response_error(response: Any) -> str | None:
    status = getattr(response, "status_code", None)
    if status == 200:
        return None
    code = getattr(response, "code", "") or ""
    message = getattr(response, "message", "") or ""
    return f"status_code={status} code={code!r} message={message!r}"


def _extract_embeddings(response: Any, expected: int) -> list[list[float]]:
    output = getattr(response, "output", None) or {}
    items = output.get("embeddings") if isinstance(output, dict) else None
    if not items:
        raise RuntimeError(f"API response has no embeddings: {output!r}")
    ordered = sorted(items, key=lambda item: item.get("text_index", 0))
    if len(ordered) != expected:
        raise RuntimeError(f"expected {expected} embeddings, got {len(ordered)}")
    return [item["embedding"] for item in ordered]


def embed_batch(texts: Sequence[str], model: str, api_key: str) -> list[list[float]] | None:
    """Embed one batch, retrying transient errors.  None if it keeps failing."""
    last_error = ""
    for attempt in range(EMBED_RETRIES + 1):
        try:
            response = dashscope.TextEmbedding.call(
                model=model, input=list(texts), api_key=api_key
            )
        except Exception as exc:  # network errors, SDK errors
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            error = _response_error(response)
            if error is None:
                try:
                    return _extract_embeddings(response, len(texts))
                except Exception as exc:
                    last_error = str(exc)
            else:
                last_error = error

        if attempt < EMBED_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    LOG.warning("embedding batch of %d failed: %s", len(texts), last_error)
    return None


def embed_all(
    texts: Sequence[str], model: str, api_key: str, batch_size: int, max_chars: int
) -> list[list[float] | None]:
    """Embed every text in order; failed entries come back as None.

    Text is truncated to `max_chars` for the API call only -- the untruncated
    text is what gets stored.
    """
    results: list[list[float] | None] = [None] * len(texts)
    payloads = [text[:max_chars] if len(text) > max_chars else text for text in texts]
    truncated = sum(1 for text, payload in zip(texts, payloads) if len(payload) < len(text))
    if truncated:
        LOG.info("Truncated %d text(s) to %d chars for embedding", truncated, max_chars)

    for start in range(0, len(payloads), batch_size):
        window = payloads[start : start + batch_size]
        LOG.info("  embedding %d-%d of %d ...", start + 1, start + len(window), len(payloads))
        vectors = embed_batch(window, model, api_key)
        if vectors is not None:
            for offset, vector in enumerate(vectors):
                results[start + offset] = vector
            continue

        # A whole batch failed: retry each text alone so one bad text cannot
        # sink its neighbours.
        for offset, payload in enumerate(window):
            single = embed_batch([payload], model, api_key)
            if single is not None:
                results[start + offset] = single[0]
    return results


def embed_rows(
    claims: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    model: str,
    api_key: str,
    batch_size: int,
    max_chars: int,
    dimension: int,
) -> tuple[int, int]:
    """Embed claims and chunks in one pass and bind each vector to its row.

    Returns (embedded_rows, missing_rows).  A vector of the wrong dimension is
    treated as missing rather than stored.
    """
    all_rows = list(claims) + list(chunks)
    if not all_rows:
        LOG.info("Nothing to embed")
        return 0, 0

    LOG.info(
        "Embedding %d text(s) (%d claim(s) + %d chunk(s)) with %s ...",
        len(all_rows),
        len(claims),
        len(chunks),
        model,
    )
    vectors = embed_all(
        [row["content"] for row in all_rows], model, api_key, batch_size, max_chars
    )

    embedded = 0
    for row, vector in zip(all_rows, vectors):
        if vector is not None and len(vector) != dimension:
            LOG.warning(
                "Dropping a %d-d vector for %s (expected %d-d)",
                len(vector),
                row["evidence_path"],
                dimension,
            )
            vector = None
        row["embedding"] = vector
        if vector is not None:
            embedded += 1

    missing = len(all_rows) - embedded
    if missing:
        LOG.warning("%d row(s) will be inserted with embedding=NULL", missing)
    return embedded, missing


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

PAPER_INSERT_SQL = """
    INSERT INTO papers (
        title, authors, abstract, domain, method_tags, content_tree, status
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id
"""

EXPERIMENT_INSERT_SQL = """
    INSERT INTO experiments (
        paper_id, exp_group_id, task, dataset, model_name, metric, value,
        hyperparams, is_ours, domain_specific, evidence_path
    ) VALUES %s
"""

CLAIM_INSERT_SQL = """
    INSERT INTO claims (
        paper_id, claim_type, content, embedding, domain_specific, evidence_path
    ) VALUES %s
"""

CHUNK_INSERT_SQL = """
    INSERT INTO text_chunks (
        paper_id, sec_title, content, embedding, evidence_path, chunk_types
    ) VALUES %s
"""

# init.sql constrains claim_type to the original four values; "plan" is an
# addition of this pipeline, so the constraint has to be widened once per
# database.  Kept here so the script can print the exact statement to run.
CLAIM_TYPE_MIGRATION_SQL = """
ALTER TABLE claims DROP CONSTRAINT IF EXISTS claims_claim_type_check;
ALTER TABLE claims ADD CONSTRAINT claims_claim_type_check
    CHECK (claim_type IN ('contribution', 'limitation', 'assumption', 'position', 'plan'));
""".strip()


def as_jsonb(value: Any) -> Any:
    """Wrap a Python value for a JSONB column, preserving None as SQL NULL.

    Strings stay JSON strings: `evidence_path` is a plain string such as
    "Section: 5.4 Regularization, Block Index: 1", so it lands in JSONB as a
    JSON string value rather than SQL text.
    """
    return None if value is None else Json(value)


def to_pgvector(vector: Sequence[float] | None, dimension: int) -> str | None:
    """Convert a Python list to pgvector's text literal: '[0.1,0.2,...]'."""
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dimension:
        raise ValueError(f"expected a {dimension}-d vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector contains NaN or infinity")
    return "[" + ",".join(repr(float(x)) for x in arr) + "]"


def claim_type_check_allows_plan(conn: PgConnection) -> bool | None:
    """True/False when the claims.claim_type CHECK constraint can be inspected.

    None means no such constraint was found, which is not an error: a database
    created from a schema where the CHECK was already widened has none.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE t.relname = 'claims'
              AND n.nspname = current_schema()
              AND c.contype = 'c'
              AND pg_get_constraintdef(c.oid) LIKE '%claim_type%'
            """
        )
        definitions = [row[0] for row in cur.fetchall()]
    if not definitions:
        return None
    return all("'plan'" in definition for definition in definitions)


def preflight_schema(conn: PgConnection) -> None:
    """Fail before any API call when the database cannot take these rows."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name IN ('papers', 'experiments', 'claims', 'text_chunks')
            """
        )
        present = {row[0] for row in cur.fetchall()}

    missing = {"papers", "experiments", "claims", "text_chunks"} - present
    if missing:
        raise RuntimeError(
            f"missing table(s) {sorted(missing)}; create the schema first "
            "with: psql -f init.sql"
        )

    if claim_type_check_allows_plan(conn) is False:
        raise RuntimeError(
            "claims.claim_type still rejects 'plan'.  Widen the constraint "
            "first:\n" + CLAIM_TYPE_MIGRATION_SQL
        )


def load_resume_state(conn: PgConnection, titles: Sequence[str]) -> ResumeState:
    """Read back what the database already holds for a paper, by title.

    `titles` is tried as a whole: the title the model returns and the one in
    the content tree can differ, and either may be the one stored.  Every
    matching paper row is used, so a title that was imported more than once
    does not hide rows behind the first id.

    `evidence_path` is JSONB on claims and experiments but plain TEXT on
    text_chunks; `#>> '{}'` reads the JSON string back as the text it was
    written from.  Returns an empty state when the paper is not there yet,
    which is exactly the behaviour of a first run.
    """
    wanted = [title for title in titles if title]
    if not wanted:
        return ResumeState([], set(), set())

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM papers WHERE title = ANY(%s) ORDER BY id", (wanted,)
        )
        paper_ids = [row[0] for row in cur.fetchall()]
        if not paper_ids:
            LOG.info("Resume: no paper titled %r in the database yet", wanted[0])
            return ResumeState([], set(), set())

        cur.execute(
            "SELECT DISTINCT evidence_path #>> '{}' FROM claims "
            "WHERE paper_id = ANY(%s)",
            (paper_ids,),
        )
        text_paths = {row[0] for row in cur.fetchall() if row[0]}

        cur.execute(
            "SELECT DISTINCT evidence_path FROM text_chunks WHERE paper_id = ANY(%s)",
            (paper_ids,),
        )
        text_paths.update(row[0] for row in cur.fetchall() if row[0])

        cur.execute(
            "SELECT DISTINCT evidence_path #>> '{}' FROM experiments "
            "WHERE paper_id = ANY(%s)",
            (paper_ids,),
        )
        experiment_paths = {row[0] for row in cur.fetchall() if row[0]}

    LOG.info(
        "Resume: paper id(s) %s already hold %d text evidence path(s) and "
        "%d experiment evidence path(s)",
        paper_ids,
        len(text_paths),
        len(experiment_paths),
    )
    return ResumeState(paper_ids, text_paths, experiment_paths)


def store_paper(
    conn: PgConnection,
    paper_info: dict[str, Any],
    tree: dict[str, Any],
    replace: bool,
    status: str,
    reuse_existing: bool = False,
) -> int:
    """Insert the paper and return its id.  Caller owns the transaction.

    With `reuse_existing` (--resume) an existing paper is kept and its id is
    returned: the run is topping it up, so its children must stay and the new
    rows must land under the same paper_id.  Without it an existing paper is
    an error, unless `replace` says to delete it first.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM papers WHERE title = %s ORDER BY id", (paper_info["title"],)
        )
        existing = [row[0] for row in cur.fetchall()]

        if existing and not replace:
            if not reuse_existing:
                raise RuntimeError(
                    f"paper {paper_info['title']!r} already exists (id={existing}); "
                    "re-run with --replace to overwrite it or --resume to top it up"
                )
            LOG.info(
                "Resuming into existing paper id(s) %s; nothing is deleted",
                existing,
            )
            return existing[0]
        if existing:
            # ON DELETE CASCADE removes the old experiments, claims and chunks.
            cur.execute("DELETE FROM papers WHERE title = %s", (paper_info["title"],))
            LOG.info("Replaced existing paper id(s): %s", existing)

        cur.execute(
            PAPER_INSERT_SQL,
            (
                paper_info["title"],
                as_jsonb(paper_info.get("authors")),
                paper_info.get("abstract"),
                paper_info.get("domain"),
                as_jsonb(paper_info.get("method_tags")),
                as_jsonb(tree),
                status,
            ),
        )
        return cur.fetchone()[0]


def store_rows(
    conn: PgConnection,
    paper_id: int,
    experiments: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    dimension: int,
) -> tuple[int, int, int]:
    """Insert every child row for `paper_id`.  Caller owns the transaction."""
    experiment_rows = [
        (
            paper_id,
            row.get("exp_group_id"),
            row.get("task"),
            row.get("dataset"),
            row.get("model_name"),
            row.get("metric"),
            row.get("value"),
            as_jsonb(row.get("hyperparams")),
            row.get("is_ours"),
            as_jsonb(row.get("domain_specific")),
            as_jsonb(row.get("evidence_path")),
        )
        for row in experiments
    ]
    claim_rows = [
        (
            paper_id,
            row["claim_type"],
            row["content"],
            to_pgvector(row.get("embedding"), dimension),
            as_jsonb(row.get("domain_specific")),
            as_jsonb(row.get("evidence_path")),
        )
        for row in claims
    ]
    chunk_rows = [
        (
            paper_id,
            row.get("sec_title"),
            row["content"],
            to_pgvector(row.get("embedding"), dimension),
            row.get("evidence_path"),
            list(row.get("chunk_types") or []),
        )
        for row in chunks
    ]

    with conn.cursor() as cur:
        if experiment_rows:
            execute_values(cur, EXPERIMENT_INSERT_SQL, experiment_rows)
        if claim_rows:
            execute_values(cur, CLAIM_INSERT_SQL, claim_rows)
        if chunk_rows:
            # psycopg2's list adapter renders chunk_types as ARRAY[...] and the
            # column type pins the element type.
            execute_values(cur, CHUNK_INSERT_SQL, chunk_rows)

    return len(experiment_rows), len(claim_rows), len(chunk_rows)


def connect_db(dsn: str, args: argparse.Namespace) -> PgConnection:
    """Open a connection; libpq picks up PG* env vars for anything not passed."""
    if dsn:
        return psycopg2.connect(dsn)

    return psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
    )


def validate_rows(
    paper_info: dict[str, Any] | None,
    tree: dict[str, Any],
    experiments: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    dimension: int,
) -> None:
    """Fail fast on shape problems before the transaction opens.

    Mirrors the checks save_to_db.py runs on its JSON input, extended to the
    text_chunks rows this pipeline builds.  A malformed row must be caught here
    rather than halfway through an insert that would have to be rolled back.
    """
    if not isinstance(paper_info, dict) or not paper_info.get("title"):
        raise ValueError("paper_info must be an object with a non-empty 'title'")
    if not isinstance(tree, dict):
        raise ValueError("the content tree must be a JSON object")

    if tree.get("title") and tree["title"] != paper_info["title"]:
        LOG.warning(
            "content tree title %r does not match paper title %r",
            tree["title"],
            paper_info["title"],
        )

    for index, row in enumerate(experiments):
        if not isinstance(row, dict):
            raise ValueError(f"experiments[{index}] is not an object")
        if row.get("value") is not None and not isinstance(row["value"], (int, float)):
            raise ValueError(
                f"experiments[{index}].value is not numeric: {row['value']!r}"
            )

    for index, row in enumerate(claims):
        if not isinstance(row, dict):
            raise ValueError(f"claims[{index}] is not an object")
        if not row.get("content"):
            raise ValueError(f"claims[{index}].content is empty")
        if row.get("claim_type") not in CLAIM_TYPES:
            raise ValueError(
                f"claims[{index}].claim_type {row.get('claim_type')!r} "
                f"is not one of {sorted(CLAIM_TYPES)}"
            )
        if not row.get("evidence_path"):
            raise ValueError(f"claims[{index}].evidence_path is empty")
        _validate_vector(row.get("embedding"), dimension, f"claims[{index}]")

    for index, row in enumerate(chunks):
        if not isinstance(row, dict):
            raise ValueError(f"text_chunks[{index}] is not an object")
        if not row.get("content"):
            raise ValueError(f"text_chunks[{index}].content is empty")
        if not row.get("evidence_path"):
            raise ValueError(f"text_chunks[{index}].evidence_path is empty")
        chunk_types = row.get("chunk_types")
        if not isinstance(chunk_types, (list, tuple)) or not chunk_types:
            raise ValueError(
                f"text_chunks[{index}].chunk_types must be a non-empty array"
            )
        _validate_vector(row.get("embedding"), dimension, f"text_chunks[{index}]")


def _validate_vector(vector: Any, dimension: int, label: str) -> None:
    if vector is None:
        return
    if not isinstance(vector, (list, tuple)) or len(vector) != dimension:
        got = len(vector) if isinstance(vector, (list, tuple)) else type(vector).__name__
        raise ValueError(f"{label}.embedding is not {dimension}-d: {got}")


def write_to_db(
    conn: PgConnection,
    paper_info: dict[str, Any],
    tree: dict[str, Any],
    experiments: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    replace: bool,
    status: str,
    dimension: int,
    resume: ResumeState | None = None,
) -> tuple[int, int, int, int]:
    """Insert the paper and all of its rows in one transaction.

    Returns (paper_id, n_experiments, n_claims, n_chunks).  Any failure rolls
    the whole import back, so a paper is never left with partial children.

    With `resume` nothing is deleted: an existing paper row is reused and the
    rows handed in are appended to it.  The caller has already dropped the
    work items the database covers, so those rows are the missing ones.
    """
    validate_rows(paper_info, tree, experiments, claims, chunks, dimension)
    try:
        paper_id = store_paper(
            conn, paper_info, tree, replace, status, reuse_existing=resume is not None
        )
        n_experiments, n_claims, n_chunks = store_rows(
            conn, paper_id, experiments, claims, chunks, dimension
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return paper_id, n_experiments, n_claims, n_chunks


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def abbreviate_vectors(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy rows for printing, replacing each vector with a short summary."""
    preview: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        vector = copy.get("embedding")
        if isinstance(vector, (list, tuple)):
            head = ", ".join(f"{value:.4f}" for value in list(vector)[:3])
            copy["embedding"] = f"<{len(vector)}-d vector: {head}, ...>"
        preview.append(copy)
    return preview


def print_prepared(
    paper_info: dict[str, Any] | None,
    experiments: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    limit: int,
    write_db: bool = False,
    row_warnings: Sequence[dict[str, Any]] = (),
) -> None:
    """Print what is ready to insert."""
    print()
    print("=" * 78)
    print(
        "PREPARED FOR INSERT"
        if write_db
        else "PREPARED FOR INSERT -- no database connection was opened"
    )
    print("=" * 78)

    def show(name: str, rows: Sequence[dict[str, Any]]) -> None:
        shown = rows if limit <= 0 else rows[:limit]
        print(f"\n--- {name}: {len(rows)} row(s) ---")
        if not rows:
            print("[]")
            return
        print(json.dumps(abbreviate_vectors(shown), ensure_ascii=False, indent=2))
        if len(shown) < len(rows):
            print(f"... {len(rows) - len(shown)} more row(s) not shown")

    if paper_info is not None:
        print("\n--- papers: 1 row ---")
        print(json.dumps(paper_info, ensure_ascii=False, indent=2))
    show("experiments", experiments)
    show("claims", claims)
    show("text_chunks", chunks)

    print()
    print("=" * 78)
    print(
        f"Totals: 1 paper, {len(experiments)} experiment(s), "
        f"{len(claims)} claim(s), {len(chunks)} text chunk(s)"
    )
    if row_warnings:
        lost = sum(w.get("lost_rows") or 0 for w in row_warnings)
        print()
        print("!" * 78)
        print(
            f"!! 警告：{len(row_warnings)} 张表格疑似丢失行，合计约 {lost} 行未入库"
        )
        print("!! 以下是怀疑模型合并了同名行的表格，请核对后再入库：")
        for warning in row_warnings:
            print(
                f"!!   {warning.get('evidence_path')}: "
                f"原表 {warning.get('table_data_rows')} 行 -> "
                f"模型返回 {warning.get('model_records')} 行 "
                f"(少 {warning.get('lost_rows')} 行)"
            )
        print("!" * 78)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Batch scan mode
# --------------------------------------------------------------------------- #

# What a scan looks for under each subdirectory, and the layout it expects:
# parsed/<paper>/content_tree.json, as build_content_tree.py and parse_pdf.py
# leave it.
SCAN_TREE_NAME = "content_tree.json"
# Printed instead of raising when a scan finds nothing to do.  The front end
# polls stdout, so this is the signal that the directory held no new paper.
NO_NEW_PAPERS_MESSAGE = "未找到需要处理的新论文"


def find_content_trees(root: Path) -> list[Path]:
    """Every content_tree.json one level under `root`, in directory-name order.

    One level only, which is the layout the parsers produce; a nested copy is
    never picked up by accident.  A missing, unreadable or empty directory is
    not an error: it simply yields no tree, and the caller reports that as
    "nothing to do" rather than as a failure.
    """
    if not root.is_dir():
        return []
    try:
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError as exc:
        LOG.error("Cannot list %s: %s", root, exc)
        return []
    return [
        child / SCAN_TREE_NAME
        for child in children
        if (child / SCAN_TREE_NAME).is_file()
    ]


def load_known_titles(conn: PgConnection, titles: Sequence[str]) -> set[str]:
    """The subset of `titles` the papers table already accounts for.

    Both keys are checked, because they can differ: `papers.title` holds what
    the model read out of the paper, and `papers.content_tree ->> 'title'` holds
    the title the content tree itself carried.  A paper imported from this very
    tree is therefore found even when the model named it differently, and a
    title the model normalised on the way in does not make the scan import the
    same paper twice.
    """
    wanted = [title for title in titles if title]
    if not wanted:
        return set()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title FROM papers WHERE title = ANY(%s)
            UNION
            SELECT content_tree ->> 'title' FROM papers
            WHERE content_tree ->> 'title' = ANY(%s)
            """,
            (wanted, wanted),
        )
        return {row[0] for row in cur.fetchall() if row[0]}


def run_scan(args: argparse.Namespace) -> int:
    """Run the pipeline over every content tree under --scan-dir.

    Dedup before extraction: the titles are read out of the trees and checked
    against the `papers` table in one query, so a paper that was already
    imported is skipped without spending a single API call.  Each remaining
    tree then goes through `run_pipeline`, which opens its own connection and
    owns its own transaction, so the trees are isolated from one another: one
    that fails is logged and counted while the rest still run.  Nothing here
    raises for an empty directory -- that is the ordinary "no new papers" case
    and exits 0, which is what a scheduled job needs.
    """
    trees = find_content_trees(args.scan_dir)
    if not trees:
        print(NO_NEW_PAPERS_MESSAGE)
        LOG.info(
            "Scan: no %s under %s; nothing to do", SCAN_TREE_NAME, args.scan_dir
        )
        return 0

    LOG.info("Scan: %d content tree(s) under %s", len(trees), args.scan_dir)

    failures = 0
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in trees:
        try:
            loaded.append((path, load_content_tree(path)))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            LOG.error("Cannot read %s: %s", path, exc)
            failures += 1

    # A dry run makes no API call and writes nothing, so it needs neither the
    # database nor the dedup check: every tree is previewed as it comes.
    known: set[str] = set()
    if not args.dry_run:
        if psycopg2 is None:
            LOG.error(
                "The 'psycopg2' package is required for --scan: "
                "pip install psycopg2-binary"
            )
            return 2
        try:
            conn = connect_db(args.dsn, args)
            try:
                preflight_schema(conn)
                known = load_known_titles(
                    conn, [tree.get("title") for _, tree in loaded]
                )
            finally:
                conn.close()
        except Exception as exc:
            LOG.error("Cannot use the database: %s", exc)
            return 1
        LOG.info(
            "Scan: %d of %d title(s) already in the database",
            len(known),
            len(loaded),
        )

    processed = skipped = 0
    for path, tree in loaded:
        title = as_str(tree.get("title"))
        if title and title in known:
            print(f"SKIP: 论文已存在 ({title})")
            LOG.info("SKIP: %s is already in the database", path)
            skipped += 1
            continue
        if not title:
            LOG.warning(
                "%s carries no title, so it cannot be checked against the "
                "database and is extracted again",
                path,
            )

        LOG.info("=== %s ===", path)
        # --out-dir names one directory, and every paper would write its
        # paper_info.json into it; in a scan each paper gets its own
        # subdirectory, named after the parsed/<paper> folder it came from, so
        # the papers do not overwrite one another.
        paper_args = args
        if args.out_dir is not None:
            paper_args = argparse.Namespace(**vars(args))
            paper_args.out_dir = args.out_dir / path.parent.name

        try:
            code = run_pipeline(tree, paper_args)
        except Exception as exc:  # one bad tree must not end the batch
            LOG.error("Failed on %s: %s", path, exc)
            code = 1
        if code:
            failures += 1
            LOG.error("Failed: %s (exit code %d)", path, code)
        else:
            processed += 1

    LOG.info(
        "Scan done: %d processed, %d skipped as already present, %d failed",
        processed,
        skipped,
        failures,
    )
    if not processed and not failures:
        print(NO_NEW_PAPERS_MESSAGE)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot extraction: paper metadata, experiments, text chunks and "
            "claims.  Rows are assembled in memory and printed; --write-db "
            "additionally inserts them into PostgreSQL in one transaction."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="path to content_tree.json; omit it with --scan, which walks "
             "--scan-dir instead",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="batch mode: process every content_tree.json under --scan-dir, "
             "skipping the papers the database already holds (the paper is "
             "identified by the title in its content tree)",
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(__file__).with_name("parsed"),
        help="directory --scan walks, one level deep, for "
             "<paper>/content_tree.json (default: the parsed/ directory beside "
             "this script)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="optional directory for the prepared JSON files (default: none; "
             "the result is only printed)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("paper", "experiments", "text"),
        default=("paper", "experiments", "text"),
        help="stages to run (default: all three; the text stage includes "
             "claim extraction and embedding)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible endpoint used by the SDK for the chat model "
             f"(default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-key",
        default=None,
        help="chat API key; defaults to $LLM_API_KEY, then $DASHSCOPE_API_KEY, "
             "then $OPENAI_API_KEY",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="disable hybrid-model reasoning; faster and avoids long-table timeouts",
    )
    thinking.add_argument(
        "--thinking",
        dest="enable_thinking",
        action="store_true",
        help="force hybrid-model reasoning on",
    )
    thinking.set_defaults(enable_thinking=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"per-request timeout in seconds, applied to every chat call "
             f"(default: {DEFAULT_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"retries after a failed chat call, so "
             f"{DEFAULT_MAX_RETRIES + 1} attempts in total "
             f"(default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="maximum characters of a table sent to the model",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=DEFAULT_TEXT_MAX_CHARS,
        help=f"maximum characters of one text block in the unified prompt "
             f"(default: {DEFAULT_TEXT_MAX_CHARS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TEXT_BATCH_SIZE,
        help=f"text blocks per unified LLM call, 5-10 recommended "
             f"(default: {DEFAULT_TEXT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_TEXT_CHARS,
        help=f"skip text blocks shorter than this many characters "
             f"(default: {DEFAULT_MIN_TEXT_CHARS})",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(__file__).with_name(VOCAB_FILE_NAME),
        help=f"chunk_types vocabulary file: the labels the model may reuse, "
             f"plus the synonym table that folds near-misses onto them.  It is "
             f"created from the standard labels when absent and grows as new "
             f"labels are coined (default: {VOCAB_FILE_NAME} beside this script)",
    )
    parser.add_argument(
        "--include-preamble",
        action="store_true",
        help="also label the Preamble section (title, authors, affiliations, "
             "licence); skipped by default as front-matter noise",
    )
    parser.add_argument(
        "--llm-sleep",
        type=float,
        default=DEFAULT_LLM_SLEEP,
        help=f"seconds to sleep after each unified batch call "
             f"(default: {DEFAULT_LLM_SLEEP})",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=f"DashScope embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        help=f"expected vector dimension (default: {DEFAULT_DIMENSION})",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help=f"texts per embedding call, 1-{DEFAULT_EMBED_BATCH_SIZE} "
             f"(default: {DEFAULT_EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-max-chars",
        type=int,
        default=DEFAULT_EMBED_MAX_CHARS,
        help=f"truncate text to this many chars before embedding "
             f"(default: {DEFAULT_EMBED_MAX_CHARS})",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="skip the embedding stage (rows are prepared without vectors)",
    )
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="insert the paper, its experiments, claims and text chunks into "
             "PostgreSQL in one transaction (default: print only, no writes)",
    )
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL", DEFAULT_DSN),
        help=f"PostgreSQL connection string; wins over the --host/--port/... "
             f"options (default: $DATABASE_URL or {DEFAULT_DSN})",
    )
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"),
                        help="database host, used only when --dsn is empty")
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")),
                        help="database port, used only when --dsn is empty")
    parser.add_argument("--user", default=os.getenv("PGUSER", "qgent"),
                        help="database user, used only when --dsn is empty")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"),
                        help="database password, used only when --dsn is empty")
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "qgent_db"),
                        help="database name, used only when --dsn is empty")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="with --write-db: delete an existing paper with the same title "
             "(cascades to its experiments, claims and chunks) before importing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="top up an existing paper instead of redoing it: skip every table "
             "and text block the database already holds, keep the paper row and "
             "insert only the missing rows (needs the database; --replace wins)",
    )
    parser.add_argument(
        "--status",
        default="done",
        help="papers.status value for the imported paper (default: done)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="keep exact duplicate experiment rows found in multiple tables",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=DEFAULT_PRINT_ROWS,
        help=f"rows shown per table in the preview, 0 for all "
             f"(default: {DEFAULT_PRINT_ROWS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only report what would be sent to the model; make no API calls",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def dry_run(tree: dict[str, Any], args: argparse.Namespace) -> None:
    front_matter = front_matter_text(tree)
    abstract = section_text(tree, re.compile(r"\babstract\b", re.IGNORECASE))
    tables = build_table_units(tree)
    stats = TextStats()
    blocks = build_text_blocks(tree, args.min_chars, args.include_preamble, stats)
    batches = chunk_text_blocks(blocks, args.batch_size)

    print(f"Title: {tree.get('title')!r}")
    print(f"Front matter: {len(front_matter)} chars")
    print(f"Abstract: {len(abstract)} chars")
    print()
    print(f"Experiment tables: {len(tables)} (one API call each; prose is excluded)")
    for unit in tables:
        preview = " ".join(unit.table.split())[:100]
        print(
            f"  {unit.sec_title!r} block {unit.block_index}: "
            f"{len(unit.table)} chars, caption={'yes' if unit.caption else 'no'}, "
            f"prev={'yes' if unit.preceding else 'no'}, "
            f"next={'yes' if unit.following else 'no'}"
        )
        print(f"      {preview}...")
    print()
    print(
        f"Text blocks: {len(blocks)} in {len(batches)} batch(es) of up to "
        f"{args.batch_size} (one API call each; labels and claims together)"
    )
    print(
        f"  skipped: {stats.skipped_preamble} preamble, "
        f"{stats.skipped_boilerplate} boilerplate, "
        f"{stats.skipped_short} shorter than {args.min_chars} chars"
    )
    for number, batch in enumerate(batches, start=1):
        print(f"  batch {number}: {len(batch)} block(s)")
        for block in batch:
            preview = " ".join(block.content.split())[:80]
            print(f"      {block.evidence_path}: {preview}...")
    equations = build_equation_blocks(tree, args.include_preamble, stats)
    print()
    print(
        f"Equations: {len(equations)} (copied in Python as {FORMULA_CHUNK_TYPE}, "
        f"no API call each)"
    )
    for block in equations:
        preview = " ".join(block.content.split())[:80]
        print(f"      {block.evidence_path}: {preview}...")
    print()
    if args.vocab.is_file():
        vocab = load_chunk_vocabulary(args.vocab, create=False)
        print(f"Chunk vocabulary: {len(vocab.types)} label(s) in {args.vocab}")
        print(f"  {vocab.render()}")
    else:
        print(
            f"Chunk vocabulary: {args.vocab} does not exist yet; the run would "
            f"create it with the {len(SEED_CHUNK_TYPES)} standard label(s)"
        )
    print()
    print("Dry run: no API calls were made and nothing was written.")


def validate_args(args: argparse.Namespace) -> str | None:
    if args.scan and args.input is not None:
        LOG.warning(
            "--scan ignores the positional path %s: every content tree under "
            "%s is processed instead",
            args.input,
            args.scan_dir,
        )
    if not args.scan and args.input is None:
        return "a path to content_tree.json is required unless --scan is given"
    if args.batch_size < 1:
        return "--batch-size must be at least 1"
    if not 5 <= args.batch_size <= 10:
        LOG.warning(
            "--batch-size=%d is outside the recommended 5-10 range",
            args.batch_size,
        )
    if not 1 <= args.embed_batch_size <= DEFAULT_EMBED_BATCH_SIZE:
        return (
            f"--embed-batch-size must be between 1 and {DEFAULT_EMBED_BATCH_SIZE}"
        )
    if args.dimension < 1:
        return "--dimension must be positive"
    if args.embed_max_chars < 1 or args.text_max_chars < 1 or args.max_chars < 1:
        return "--max-chars, --text-max-chars and --embed-max-chars must be positive"
    if args.min_chars < 0:
        return "--min-chars must not be negative"
    if args.write_db and "paper" not in args.stages:
        return "--write-db needs the paper stage: the paper row provides the " \
               "paper_id its children are keyed by"
    if args.resume and args.dry_run:
        return "--resume cannot be combined with --dry-run: a dry run opens " \
               "no database connection, so there is nothing to resume from"
    return None


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    problem = validate_args(args)
    if problem:
        LOG.error("%s", problem)
        return 2

    # Batch mode walks the directory and calls run_pipeline once per tree; the
    # single-file mode below is the same function called once.
    if args.scan:
        return run_scan(args)

    assert args.input is not None  # validate_args guarantees it
    try:
        tree = load_content_tree(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        LOG.error("Cannot read %s: %s", args.input, exc)
        return 2

    return run_pipeline(tree, args)


def run_pipeline(tree: dict[str, Any], args: argparse.Namespace) -> int:
    """Extract one content tree and, with --write-db, import it.

    Owns everything that follows from a loaded tree: the dependency checks, the
    database connection and its transaction, the three stages and the import.
    The connection is opened here and closed on every exit path, so a scan can
    call this once per paper without the papers sharing a transaction -- one
    failure cannot roll back another paper's committed rows.
    """
    if args.dry_run:
        dry_run(tree, args)
        return 0

    # dashscope now backs both the chat calls and the embedding calls, so it
    # is required for every run that reaches the model.
    if dashscope is None or Completions is None:
        LOG.error(
            "The 'dashscope' package is required: pip install dashscope"
        )
        return 2

    # Not fatal: json_repair only backs up the strict parser, and without it a
    # malformed answer is retried and then dropped as before.
    if json_repair is None:
        LOG.warning(
            "The 'json_repair' package is not installed, so a malformed model "
            "answer cannot be salvaged: pip install json_repair"
        )

    # The chat model and the embedding model are served by different hosts
    # (the LLM endpoint and DashScope respectively), so they carry separate
    # keys.  The DashScope key stays the shared fallback so a single-key
    # setup, and the --api-key override, keep working.
    llm_api_key = (
        args.api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not llm_api_key:
        LOG.error(
            "No API key. Set LLM_API_KEY (or pass --api-key).\n"
            "  PowerShell:  $env:LLM_API_KEY=\"...\"\n"
            "  CMD:         set LLM_API_KEY=...\n"
            "  Bash:        export LLM_API_KEY=..."
        )
        return 2
    embed_api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or llm_api_key
    )

    if args.resume and args.replace:
        LOG.warning(
            "--replace wins over --resume: the existing paper is deleted and "
            "everything is extracted again"
        )
    resume_enabled = args.resume and not args.replace

    # Connect and validate the schema before spending anything on API calls.
    # --resume reads the database before the first call, so it needs the
    # connection even when nothing will be written.
    conn: PgConnection | None = None
    if args.write_db or args.resume:
        if psycopg2 is None:
            LOG.error(
                "The 'psycopg2' package is required for --write-db and "
                "--resume: pip install psycopg2-binary"
            )
            return 2
        if args.write_db and np is None:
            LOG.error(
                "The 'numpy' package is required for --write-db: "
                "pip install numpy"
            )
            return 2
        try:
            conn = connect_db(args.dsn, args)
            preflight_schema(conn)
        except Exception as exc:
            LOG.error("Cannot use the database: %s", exc)
            if conn is not None:
                conn.close()
            return 1
        LOG.info("Database ready: %s", args.dsn or f"{args.user}@{args.host}/{args.dbname}")

    client = LLMClient(
        base_url=args.base_url,
        api_key=llm_api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )
    LOG.info("Using model %s at %s", args.model, args.base_url)

    paper_info: dict[str, Any] | None = None
    experiments: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    stats: TextStats | None = None

    stage_list = list(args.stages)

    # The existing rows are keyed by the paper title, which the model only
    # names during the paper stage.  When that stage is not part of this run
    # the content tree's title is the only key available, and is enough: the
    # stored title comes from the same tree.
    resume: ResumeState | None = None
    if resume_enabled and "paper" not in stage_list:
        assert conn is not None
        resume = load_resume_state(conn, [tree.get("title")])

    for stage_number, stage in enumerate(stage_list, start=1):
        LOG.info("=== [%d/%d] %s ===", stage_number, len(stage_list), stage)

        try:
            if stage == "paper":
                paper_info = extract_metadata(client, tree)
                LOG.info(
                    "Paper: title=%r, domain=%s, %d author(s), %d tag(s)",
                    paper_info["title"],
                    paper_info["domain"],
                    len(paper_info["authors"]),
                    len(paper_info["method_tags"]),
                )
                if resume_enabled:
                    assert conn is not None
                    resume = load_resume_state(
                        conn, [paper_info["title"], tree.get("title")]
                    )

            elif stage == "experiments":
                experiments, reports = extract_experiments(client, tree, args, resume)

            elif stage == "text":
                claims, chunks, stats = extract_text_layer(client, tree, args, resume)
                if not args.no_embed:
                    LOG.info("=== embedding ===")
                    embed_rows(
                        claims,
                        chunks,
                        args.embedding_model,
                        embed_api_key,
                        args.embed_batch_size,
                        args.embed_max_chars,
                        args.dimension,
                    )
                else:
                    for row in list(claims) + list(chunks):
                        row["embedding"] = None
                    LOG.info("Embedding skipped (--no-embed)")
        except APIUnavailableError as exc:
            LOG.error(
                "FATAL: the model API is unusable, so the %s stage produced "
                "nothing.  Aborting without preparing or writing any row: %s",
                stage,
                exc,
            )
            if conn is not None:
                conn.close()
            return 3

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if paper_info is not None:
            write_json(args.out_dir / "paper_info.json", paper_info)
        if experiments:
            write_json(args.out_dir / "experiments.json", experiments)
        if reports:
            write_json(args.out_dir / "extraction_reports.json", reports)
        if claims:
            write_json(args.out_dir / "claims.json", claims)
        if chunks:
            write_json(args.out_dir / "text_chunks.json", chunks)
        LOG.info("Wrote the prepared JSON to %s", args.out_dir)

    row_warnings = row_warnings_from_reports(reports)
    print_prepared(
        paper_info,
        experiments,
        claims,
        chunks,
        args.print_rows,
        args.write_db,
        row_warnings,
    )
    if stats is not None:
        dropped = (
            stats.blocks_lost_to_failed_batches
            + stats.unanswered_blocks
            + stats.unlabeled_blocks
        )
        LOG.info(
            "Text stage summary: %d batch(es), %d failed; %d of %d block(s) "
            "dropped, %d chunk(s) and %d claim(s) kept",
            stats.batches,
            stats.failed_batches,
            dropped,
            stats.blocks,
            len(chunks),
            len(claims),
        )
        if dropped:
            LOG.warning(
                "%d block(s) produced no row: %d were in a batch whose call "
                "failed, %d went unanswered by the model, %d came back without "
                "usable labels.  Re-run to retry them.",
                dropped,
                stats.blocks_lost_to_failed_batches,
                stats.unanswered_blocks,
                stats.unlabeled_blocks,
            )

    if not args.write_db:
        if conn is not None:
            conn.close()
        LOG.info(
            "Done. Nothing was written to the database: review the rows above "
            "(or in %s) before importing them.%s",
            args.out_dir if args.out_dir else "the printed preview",
            "  --resume skipped what the database already holds, so these are "
            "the missing rows only." if resume_enabled else "",
        )
        return 0

    assert conn is not None  # guaranteed by the --write-db preflight above
    # The text stage supplies claims and chunks; without it there are none to
    # write, and the paper is imported with experiments only.
    if paper_info is None:
        LOG.error("--write-db needs the paper stage to produce paper_info")
        conn.close()
        return 2
    if args.no_embed:
        LOG.warning(
            "--no-embed: claims and text chunks will be inserted with "
            "embedding=NULL"
        )

    try:
        paper_id, n_experiments, n_claims, n_chunks = write_to_db(
            conn,
            paper_info,
            tree,
            experiments,
            claims,
            chunks,
            args.replace,
            args.status,
            args.dimension,
            resume,
        )
    except Exception as exc:
        LOG.error("Import rolled back, nothing was written: %s", exc)
        return 1
    finally:
        conn.close()

    LOG.info(
        "Committed: paper_id=%d, %d experiment(s), %d claim(s), %d text chunk(s)",
        paper_id,
        n_experiments,
        n_claims,
        n_chunks,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
