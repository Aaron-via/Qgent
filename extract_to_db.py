#!/usr/bin/env python3
"""Agentic table extraction for a MinerU content tree.

Input:  content_tree.json produced by build_content_tree.py
        {"title": "...", "sections": [{"sec_title": "...", "blocks": [...]}]}

Output (JSON files for manual review before DB import):
        paper_info.json          -> one object for the `papers` table
        experiments.json         -> rows for the `experiments` table
        extraction_reports.json  -> per-table audit: blank-cell report, fill actions
        claims.json              -> rows for the `claims` table

Experiment extraction follows three rules:

1. Isolation.  Prose is never used as a source of numbers.  Each API call sees
   exactly one table plus its caption and at most one adjacent paragraph before
   and after it (for header and blank-cell interpretation only).
2. Coverage.  Every metric column is traversed, left to right.  One table row
   with three metric columns yields three records.  Theoretical tables such as
   "O(n^2)" produce no records.
3. Reflection + tool use.  The model returns a `special_rules_report` that
   explains blank cells and classifies every metric column as an
   `evaluation_metric` (BLEU, PPL: blanks mean "not tested") or an
   `intrinsic_attribute` (params, FLOPs: blanks mean "same as the base model").
   Python then applies two separate tools:
     * `fill_missing_values()` copies `hyperparams` from the base row when the
       report says unlisted configuration values are inherited;
     * `fill_inherited_metrics()` recreates a missing params/FLOPs record from
       the base row -- only for columns the report labels intrinsic with an
       `inherits_base` blank meaning.  Missing evaluation metrics are never
       invented.
   A parameter-count column is an ordinary metric (`metric` + `value`), never a
   special field; only a parameter count that is configuration stays inside
   `hyperparams`.

The model is called through an OpenAI-compatible chat endpoint.  The default
is glm-5.2 on Compshare; any compatible endpoint can be selected with
--base-url.
Image blocks are never sent to the model: their content is a base64 data URI.
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
from typing import Any

try:
    import requests
except ImportError:  # handled with a readable message in main()
    requests = None  # type: ignore[assignment]


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
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_MAX_CHARS = 14_000
DEFAULT_MAX_TOKENS = 8_192
CONTEXT_MAX_CHARS = 1_200

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
CLAIM_TYPES = {"contribution", "limitation", "assumption", "position"}

INTRO_RE = re.compile(r"\bintroduction\b", re.IGNORECASE)
CONCLUSION_RE = re.compile(r"\b(conclusions?|discussion|summary)\b", re.IGNORECASE)
LIMITATION_RE = re.compile(r"\blimitations?\b", re.IGNORECASE)
REFERENCE_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|acknowledg)", re.IGNORECASE
)
BOILERPLATE_RE = re.compile(
    r"^\s*(?:\*\*)?(acknowledg|author contributions|conflict of interest|"
    r"funding|data availability|code availability)",
    re.IGNORECASE,
)

FENCE_OPEN_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*")
FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")
# A second line of defence in case a data URI ends up inside a text block.
DATA_URI_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")

# Blank-cell semantics.  These are only a fallback for models that omit the
# boolean; an explicit `blanks_inherit_base` value always wins.
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

LOG = logging.getLogger("extract")


def configure_stdio() -> None:
    """Windows consoles default to GBK; paper text and model output are not."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Content-tree traversal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceBlock:
    sec_title: str
    block_index: int
    block_type: str
    text: str

    @property
    def evidence_path(self) -> str:
        return f"Section: {self.sec_title}, Block Index: {self.block_index}"


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


@dataclass
class Stats:
    dropped: int = 0
    deduped: int = 0
    failed_tables: int = 0
    empty_tables: int = 0
    filled_rows: int = 0
    inherited_metric_rows: int = 0


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


def build_claim_sources(tree: dict[str, Any]) -> list[SourceBlock]:
    """Introduction's last paragraph, plus Conclusion and Limitations sections."""
    sources: list[SourceBlock] = []
    intro_seen = False
    conclusion_seen = False
    limitation_seen = False

    for section in tree["sections"]:
        title = section.get("sec_title") or ""
        if REFERENCE_SECTION_RE.search(title):
            continue

        blocks = section.get("blocks") or []
        eligible: list[SourceBlock] = []
        for block_index, block in enumerate(blocks):
            if block.get("type") != "text":
                continue
            text = block_text(block)
            if len(text) < 40 or BOILERPLATE_RE.search(text):
                continue
            eligible.append(SourceBlock(title, block_index, "text", text))

        if LIMITATION_RE.search(title) and not limitation_seen:
            sources.extend(eligible)
            limitation_seen = True
        elif CONCLUSION_RE.search(title) and not conclusion_seen:
            sources.extend(eligible)
            conclusion_seen = True
        elif INTRO_RE.search(title) and not intro_seen:
            # Only the final paragraph of the introduction.
            if eligible:
                sources.append(eligible[-1])
            intro_seen = True

    return sources


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


class LLMClient:
    """Minimal OpenAI-compatible chat client with retry and JSON repair."""

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

    def complete(self, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        # Hybrid-reasoning Qwen models accept this in the OpenAI-compatible
        # body.  It is only sent when the user explicitly chooses a mode, so
        # other providers are unaffected.
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                if response.status_code >= 400:
                    body = response.text[:500]
                    error = LLMError(f"HTTP {response.status_code}: {body}")
                    if (
                        response.status_code in self.RETRY_STATUS
                        and attempt < self.max_retries
                    ):
                        last_error = error
                        self._backoff(attempt, error)
                        continue
                    raise error
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty completion content")
                return content
            except LLMError:
                raise
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt, exc)

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


def parse_json_response(raw: str, expect: str) -> Any:
    cleaned = strip_markdown_fences(raw)
    candidates = [cleaned]
    span = extract_json_span(cleaned)
    if span and span != cleaned:
        candidates.append(span)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if expect == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                lists = [v for v in value.values() if isinstance(v, list)]
                if len(lists) == 1:
                    return lists[0]
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

    raise LLMError(
        f"could not parse JSON ({last_error}); response starts with "
        f"{raw.strip()[:160]!r}"
    )


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

METADATA_SYSTEM = (
    "You are a precise metadata extraction engine for scientific papers. "
    "You output only valid JSON, with no prose and no Markdown fences."
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

VALUES
- `value` must be a bare JSON number. Convert scientific notation: 3.3 * 10^18 -> 3.3e18. Remove thousands separators.
- Some HTML tables split a number across adjacent cells (e.g. "3.3 *" and "10^18"); recombine them into one value.
- Skip blank cells; never guess, average, or copy a value from another row. Python may later restore an intrinsic-attribute blank from the base row, but you must not emit it yourself.
- If the table contains only theoretical expressions (O(n^2), formulas, symbols) and no concrete measured number, return "records": [].
- Percentages: keep the numeric part (25.8 for 25.8%) and include "%" in the metric name.

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

CLAIM_SYSTEM = (
    "You are a precise claim extraction engine for scientific papers. "
    "You output only valid JSON, with no prose and no Markdown fences. "
    "Do not invent claims."
)

CLAIM_TEMPLATE = """Extract the paper's explicit claims from the SOURCE below.

Return a JSON array. Each element must have exactly this schema:
{
  "claim_type": string,
  "content": string,
  "domain_specific": object|null,
  "evidence_path": string
}

Rules:
- claim_type: MUST be exactly one of "contribution", "limitation", "assumption", "position".
  * contribution: what the paper introduces, proposes or achieves.
  * limitation: acknowledged weaknesses, scope restrictions or failure cases.
  * assumption: an explicit premise the method or evaluation relies on.
  * position: a stance or argument about the field, not a result.
- content: the claim as a faithful quote or a concise paraphrase. Keep numbers and qualifiers exactly as stated.
- domain_specific: optional subfield-specific detail explicitly stated in the source; otherwise null.
- evidence_path: use "Section: <section title>, Block Index: <index>".
- Extract 0-5 claims per block. Prefer precision over recall; return [] if the block contains no claim.

Section: {SECTION}
Block Index: {BLOCK_INDEX}

{CONTENT}
"""


def metadata_prompt(title: str, front_matter: str, abstract: str) -> str:
    return (
        METADATA_TEMPLATE.replace("{TITLE}", title or "(unknown)")
        .replace("{FRONT_MATTER}", front_matter or "(none)")
        .replace("{ABSTRACT}", abstract or "(none)")
    )


def table_prompt(
    unit: TableUnit, paper_title: str, max_chars: int
) -> str:
    """Build a prompt containing this table and nothing but its neighbours."""
    return (
        TABLE_TEMPLATE.replace("{PAPER_TITLE}", paper_title or "(unknown)")
        .replace("{SECTION}", unit.sec_title or "(unknown)")
        .replace("{CAPTION}", unit.caption or "(none)")
        .replace("{PREV}", unit.preceding or "(none)")
        .replace("{NEXT}", unit.following or "(none)")
        .replace("{TABLE}", truncate(unit.table, max_chars))
    )


def claim_prompt(unit: SourceBlock, max_chars: int) -> str:
    return (
        CLAIM_TEMPLATE.replace("{SECTION}", unit.sec_title)
        .replace("{BLOCK_INDEX}", str(unit.block_index))
        .replace("{CONTENT}", truncate(unit.text, max_chars))
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


def parse_metric_value(value: Any, metric: str) -> float:
    """Parse a metric value; parameter-count metrics also accept '65M'/'65K'."""
    try:
        return parse_numeric(value)
    except (TypeError, ValueError):
        if PARAM_KEY_RE.match(metric):
            params = parse_params(value)
            if params is not None:
                return params
        raise


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
        value = parse_metric_value(raw_value, metric)
    except (TypeError, ValueError):
        LOG.warning(
            "Ignoring non-numeric value %r (%s, model=%r, metric=%r)",
            raw_value,
            unit.evidence_path,
            row.get("model_name"),
            metric,
        )
        return None

    shell["metric"] = metric
    shell["value"] = value
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
            value = parse_metric_value(raw_value, metric)
        except (TypeError, ValueError):
            LOG.warning(
                "Ignoring non-numeric top-level key %r=%r (%s)",
                key,
                raw_value,
                unit.evidence_path,
            )
            continue
        record = dict(shell)
        record["metric"] = metric
        record["value"] = value
        records.append(record)

    return records


def normalise_claim_row(
    row: Any, unit: SourceBlock, stats: Stats
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        stats.dropped += 1
        LOG.warning("Dropped non-object claim from %s", unit.evidence_path)
        return None

    claim_type = as_str(row.get("claim_type"))
    claim_type = claim_type.lower() if claim_type else ""
    if claim_type not in CLAIM_TYPES:
        stats.dropped += 1
        LOG.warning(
            "Dropped claim with invalid type %r (%s)",
            row.get("claim_type"),
            unit.evidence_path,
        )
        return None

    content = as_str(row.get("content"))
    if not content:
        stats.dropped += 1
        LOG.warning("Dropped empty claim (%s)", unit.evidence_path)
        return None

    return {
        "claim_type": claim_type,
        "content": content,
        "domain_specific": sanitise_json(as_dict(row.get("domain_specific"))),
        "evidence_path": unit.evidence_path,
    }


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


def claim_key(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("content", "")).lower().split())


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
    client: LLMClient, tree: dict[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one isolated LLM call per table, then apply Python-only repairs."""
    units = build_table_units(tree)
    paper_title = as_str(tree.get("title")) or ""
    LOG.info("Experiment sources: %d table(s); prose is excluded", len(units))

    stats = Stats()
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

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
            report["error"] = str(exc)
            reports.append(report)
            LOG.error("Skipping %s: %s", unit.evidence_path, exc)
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

    LOG.info(
        "Experiments: %d rows from %d tables "
        "(%d empty, %d failed, %d dropped, %d duplicates, "
        "%d rows filled, %d intrinsic metrics restored)",
        len(rows),
        len(units),
        stats.empty_tables,
        stats.failed_tables,
        stats.dropped,
        stats.deduped,
        stats.filled_rows,
        stats.inherited_metric_rows,
    )
    return rows, reports


def extract_claims(
    client: LLMClient, tree: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    units = build_claim_sources(tree)
    LOG.info("Claim sources: %d blocks", len(units))

    stats = Stats()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for position, unit in enumerate(units, start=1):
        LOG.info(
            "[%d/%d] %s, block %d",
            position,
            len(units),
            unit.sec_title,
            unit.block_index,
        )
        try:
            raw_rows = client.chat_json(
                CLAIM_SYSTEM,
                claim_prompt(unit, args.max_chars),
                expect="array",
            )
        except LLMError as exc:
            stats.failed_tables += 1
            LOG.error("Skipping %s: %s", unit.evidence_path, exc)
            continue

        kept = 0
        for raw_row in raw_rows:
            record = normalise_claim_row(raw_row, unit, stats)
            if record is None:
                continue
            key = claim_key(record)
            if key in seen:
                stats.deduped += 1
                continue
            seen.add(key)
            rows.append(record)
            kept += 1
        LOG.info("    -> %d claims kept", kept)

    LOG.info(
        "Claims: %d rows, %d dropped, %d duplicates removed, %d blocks failed",
        len(rows),
        stats.dropped,
        stats.deduped,
        stats.failed_tables,
    )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract paper metadata, experiments and claims from a MinerU content tree."
    )
    parser.add_argument("input", type=Path, help="path to content_tree.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("extracted"),
        help="directory for the output JSON files (default: ./extracted)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("paper", "experiments", "claims"),
        default=("paper", "experiments", "claims"),
        help="stages to run (default: all three)",
    )
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; defaults to $LLM_API_KEY, then $DASHSCOPE_API_KEY, "
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
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="maximum characters of a table sent to the model",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="keep exact duplicate experiment rows found in multiple tables",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only report which tables would be sent; make no API calls",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def dry_run(tree: dict[str, Any]) -> None:
    front_matter = front_matter_text(tree)
    abstract = section_text(tree, re.compile(r"\babstract\b", re.IGNORECASE))
    tables = build_table_units(tree)
    claims = build_claim_sources(tree)

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
    print(f"Claim sources: {len(claims)}")
    for unit in claims:
        preview = " ".join(unit.text.split())[:90]
        print(f"  {unit.sec_title!r} block {unit.block_index}: {preview}...")
    print()
    print("Dry run: no API calls were made.")


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

    try:
        tree = load_content_tree(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        LOG.error("Cannot read %s: %s", args.input, exc)
        return 2

    if args.dry_run:
        dry_run(tree)
        return 0

    if requests is None:
        LOG.error("The 'requests' package is required: pip install requests")
        return 2

    api_key = (
        args.api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        LOG.error(
            "No API key. Set LLM_API_KEY (or pass --api-key).\n"
            "  PowerShell:  $env:LLM_API_KEY=\"...\"\n"
            "  CMD:         set LLM_API_KEY=...\n"
            "  Bash:        export LLM_API_KEY=..."
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )
    LOG.info("Using model %s at %s", args.model, args.base_url)

    stage_list = list(args.stages)
    for stage_number, stage in enumerate(stage_list, start=1):
        LOG.info("=== [%d/%d] %s ===", stage_number, len(stage_list), stage)

        if stage == "paper":
            info = extract_metadata(client, tree)
            path = args.out_dir / "paper_info.json"
            write_json(path, info)
            LOG.info(
                "Wrote %s (title=%r, domain=%s, %d authors, %d tags)",
                path,
                info["title"],
                info["domain"],
                len(info["authors"]),
                len(info["method_tags"]),
            )

        elif stage == "experiments":
            records, reports = extract_experiments(client, tree, args)
            path = args.out_dir / "experiments.json"
            write_json(path, records)
            report_path = args.out_dir / "extraction_reports.json"
            write_json(report_path, reports)
            triggered = sum(
                1
                for report in reports
                if isinstance(report.get("fill"), dict)
                and report["fill"].get("base_found")
            )
            LOG.info(
                "Wrote %s (%d rows) and %s (%d tables, %d fill runs)",
                path,
                len(records),
                report_path,
                len(reports),
                triggered,
            )

        elif stage == "claims":
            records = extract_claims(client, tree, args)
            path = args.out_dir / "claims.json"
            write_json(path, records)
            LOG.info("Wrote %s (%d rows)", path, len(records))

    LOG.info("Done. Review the JSON files in %s before importing them.", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
