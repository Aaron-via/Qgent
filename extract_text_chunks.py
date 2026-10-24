#!/usr/bin/env python3
"""Extract prose blocks from content_tree.json, tag them, embed them, store them.

Pipeline:
    1. Walk every section of content_tree.json and keep the blocks whose
       `type` is "text" (tables, images, equations and footnotes are skipped).
    2. Ask an LLM (Qwen, OpenAI-compatible endpoint) for type labels per block,
       drawn from a standard vocabulary but free to invent precise new labels.
    3. Embed each block's text into a 1024-dimensional vector.
    4. Insert one row per block into `text_chunks`, creating the table and its
       indexes if they do not exist yet.

`text_chunks.chunk_types` is TEXT[]; Python lists are handed to psycopg2's
built-in list adapter and cast explicitly with `%s::text[]`.

The whole run is a single transaction: either every chunk of the paper is
written, or nothing is.  A chunk whose labels or vector could not be produced
is still inserted (with the fallback label / a NULL embedding) and reported at
the end.  Re-running replaces the paper's previous chunks unless --append is
given, so the script is safe to repeat after a partial or failed run.

Exit codes:
    0  the run committed
    1  fatal error (missing API key, DB connection, commit failure)
    2  missing Python dependency or bad input

Usage:
    python extract_text_chunks.py
    python extract_text_chunks.py --paper-id 1 --llm-model qwen-plus
    python extract_text_chunks.py --dry-run
    python extract_text_chunks.py --append --llm-sleep 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
    import psycopg2
    import requests
    import dashscope
except ImportError as exc:  # handled with a readable message in main()
    np = None  # type: ignore[assignment]
    psycopg2 = None  # type: ignore[assignment]
    requests = None  # type: ignore[assignment]
    dashscope = None  # type: ignore[assignment]
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

# No password here: PGPASSWORD (from the environment or .env) is used instead.
DEFAULT_DSN = "postgresql://qgent@localhost:5432/qgent_db"
DEFAULT_BASE_URL = "https://cp.compshare.cn/v1"
DEFAULT_LLM_MODEL = "glm-5.2"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
DEFAULT_DIMENSION = 1024
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 60.0
DEFAULT_LLM_RETRIES = 3
DEFAULT_LLM_SLEEP = 0.5
DEFAULT_EMBED_BATCH_SIZE = 10
EMBED_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
# Characters, not tokens: a safe truncation point for the embedding model's
# input-length limit.  The full text is stored regardless.
DEFAULT_EMBED_MAX_CHARS = 8_000
FALLBACK_TAG = "uncategorized"

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id SERIAL PRIMARY KEY,
    paper_id INT REFERENCES papers(id) ON DELETE CASCADE,
    sec_title TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    evidence_path TEXT,
    chunk_types TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_chunks_vec
    ON text_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_types
    ON text_chunks USING GIN (chunk_types);
"""

TAGGING_SYSTEM = "你是一位论文文本标注专家，只输出 JSON。"

TAGGING_PROMPT = """请阅读以下论文文本片段，为其打上类型标签（返回 JSON 数组，如 ["implementation_detail", "source_code"]）。
> 优先使用标准词库：implementation_detail（实现细节/原理）, experiment_setup（实验设置）, experiment_result（实验结果分析/讨论）, formula_context（公式解释）, related_work（背景）, source_code（代码）, evaluation_metric（指标讨论）, model_proposal（模型提出/动机/介绍）。
> 硬性规则：如果遇到相关内容，必须使用上述标准词，严禁生造复数形式或近义词（例如实验结果只能叫 experiment_result，不得写成 experiment_results；模型提出/动机/介绍只能叫 model_proposal，不得写成 model_introduction、model_motivation 等）。
> 自由扩展规则：如果不符合上述标准，请自行创造精准标签，必须使用英文小写+下划线（如 data_augmentation）。严禁同义词泛滥（例如，代码链接只能叫 source_code）。允许一个文本块有多个标签。

论文文本片段：
{content}
"""

# Reference lists are bibliography entries, not prose worth tagging or
# embedding, so their sections are skipped entirely.
REFERENCE_SECTION_RE = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
FENCE_OPEN_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*")
FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")
NON_TAG_RE = re.compile(r"[^a-z0-9]+")


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def configure_stdio() -> None:
    """Windows consoles default to GBK; paper text and model output are not."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextChunk:
    sec_title: str
    block_index: int
    content: str

    @property
    def evidence_path(self) -> str:
        return f"Section: {self.sec_title}, Block Index: {self.block_index}"


def load_content_tree(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError(
            f"{path} is not a content tree: expected an object with a 'sections' list"
        )
    return data


def extract_text_blocks(tree: dict[str, Any]) -> list[TextChunk]:
    """Collect every `type == "text"` block, keeping its position in the section.

    `block_index` counts all blocks in the section, tables and images included,
    so it matches the evidence paths used by extract_to_db.py.  Sections whose
    title mentions references or a bibliography are skipped.
    """
    chunks: list[TextChunk] = []
    for section in tree["sections"]:
        if not isinstance(section, dict):
            continue
        title = section.get("sec_title") or ""
        if REFERENCE_SECTION_RE.search(title):
            continue
        for index, block in enumerate(section.get("blocks") or []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            content = block.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            chunks.append(TextChunk(title, index, content.strip()))
    return chunks


# --------------------------------------------------------------------------- #
# Label normalisation
# --------------------------------------------------------------------------- #


def normalise_tags(raw: Any) -> list[str]:
    """Force model output into clean, lowercase, underscore-separated labels."""
    if isinstance(raw, dict):
        # Some models wrap the array in an object, e.g. {"tags": [...]}.
        for value in raw.values():
            if isinstance(value, list):
                raw = value
                break
        else:
            raw = [raw]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return [FALLBACK_TAG]

    tags: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = NON_TAG_RE.sub("_", item.strip().lower()).strip("_")
        if tag and tag not in tags:
            tags.append(tag)
    return tags or [FALLBACK_TAG]


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip().lstrip(chr(0xFEFF))
    cleaned = FENCE_OPEN_RE.sub("", cleaned, count=1)
    cleaned = FENCE_CLOSE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def extract_json_span(text: str) -> str | None:
    """Return the first balanced JSON array/object, ignoring prose around it."""
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
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
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                return None
            if stack[-1] != ("[" if char == "]" else "{"):
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def parse_tags(raw: str) -> list[str]:
    """Parse a model answer into labels, tolerating fences and stray prose."""
    for candidate in (strip_markdown_fences(raw), extract_json_span(raw)):
        if not candidate:
            continue
        try:
            return normalise_tags(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    # Not JSON at all: fall back to the comma-separated words it looks like.
    return normalise_tags([part for part in re.split(r"[,\s]+", raw) if part])


# --------------------------------------------------------------------------- #
# LLM tagging
# --------------------------------------------------------------------------- #


class TaggingError(RuntimeError):
    pass


class Tagger:
    """OpenAI-compatible chat client that returns a block's type labels."""

    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        max_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def tag(self, content: str) -> list[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TAGGING_SYSTEM},
                {"role": "user", "content": TAGGING_PROMPT.format(content=content)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
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
                    last_error = TaggingError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                    # 4xx other than rate limiting will not fix itself.
                    if response.status_code not in self.RETRY_STATUS:
                        raise last_error
                else:
                    try:
                        message = response.json()["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        last_error = exc
                    else:
                        if isinstance(message, str) and message.strip():
                            return parse_tags(message)
                        last_error = TaggingError("empty completion content")

            if attempt < self.max_retries:
                delay = min(2**attempt, 8)
                log(f"    tag call failed ({last_error}); retrying in {delay}s")
                time.sleep(delay)

        raise TaggingError(f"tagging failed after {self.max_retries + 1} attempts: {last_error}")


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

    warn(f"embedding batch of {len(texts)} failed: {last_error}")
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
        log(f"truncated {truncated} chunk(s) to {max_chars} chars for embedding")

    for start in range(0, len(payloads), batch_size):
        window = payloads[start : start + batch_size]
        log(f"  embedding {start + 1}-{start + len(window)} of {len(payloads)} ...")
        vectors = embed_batch(window, model, api_key)
        if vectors is not None:
            for offset, vector in enumerate(vectors):
                results[start + offset] = vector
            continue

        # A whole batch failed: retry each text alone so one bad chunk cannot
        # sink its neighbours.
        for offset, payload in enumerate(window):
            single = embed_batch([payload], model, api_key)
            if single is not None:
                results[start + offset] = single[0]
    return results


def to_pgvector(vector: Sequence[float], dimension: int) -> str:
    """Convert a Python list to pgvector's text literal: '[0.1,0.2,...]'."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dimension:
        raise ValueError(f"expected a {dimension}-d vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector contains NaN or infinity")
    return "[" + ",".join(repr(float(x)) for x in arr) + "]"


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #


def ensure_schema(conn: "psycopg2.extensions.connection") -> None:
    """Create the text_chunks table, its indexes and the vector extension."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def paper_exists(conn: "psycopg2.extensions.connection", paper_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM papers WHERE id = %s", (paper_id,))
        return cur.fetchone() is not None


INSERT_SQL = """
    INSERT INTO text_chunks (
        paper_id, sec_title, content, embedding, evidence_path, chunk_types
    ) VALUES (%s, %s, %s, %s::vector, %s, %s::text[])
"""


def store_chunks(
    conn: "psycopg2.extensions.connection",
    paper_id: int,
    rows: Sequence[tuple[TextChunk, list[str], str | None]],
    replace: bool,
    dimension: int,
) -> int:
    """Write every chunk, then commit once.  Rolls back on any error."""
    try:
        with conn.cursor() as cur:
            if replace:
                cur.execute("DELETE FROM text_chunks WHERE paper_id = %s", (paper_id,))
                if cur.rowcount:
                    log(f"removed {cur.rowcount} existing chunk(s) for paper_id={paper_id}")

            params = []
            for chunk, tags, vector in rows:
                literal = None
                if vector is not None:
                    literal = to_pgvector(vector, dimension)
                # psycopg2's list adapter renders the Python list as ARRAY[...],
                # and the `::text[]` cast in INSERT_SQL pins the element type.
                params.append(
                    (paper_id, chunk.sec_title, chunk.content, literal,
                     chunk.evidence_path, list(tags))
                )
            cur.executemany(INSERT_SQL, params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(params)


def count_chunks(conn: "psycopg2.extensions.connection", paper_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM text_chunks WHERE paper_id = %s", (paper_id,))
        return cur.fetchone()[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--content-tree", type=Path, default=Path("content_tree.json"),
        help="content_tree.json path (default: content_tree.json)",
    )
    parser.add_argument(
        "--paper-id", type=int, default=1,
        help="papers.id the chunks belong to (default: 1)",
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
        help=f"model used for tagging (default: {DEFAULT_LLM_MODEL})",
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
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
        help=f"texts per embedding call, 1-{DEFAULT_EMBED_BATCH_SIZE} "
             f"(default: {DEFAULT_EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-max-chars", type=int, default=DEFAULT_EMBED_MAX_CHARS,
        help=f"truncate text to this many chars before embedding "
             f"(default: {DEFAULT_EMBED_MAX_CHARS})",
    )
    parser.add_argument(
        "--llm-sleep", type=float, default=DEFAULT_LLM_SLEEP,
        help=f"seconds to sleep after each tagging call, to avoid rate limits "
             f"(default: {DEFAULT_LLM_SLEEP})",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_LLM_RETRIES,
        help=f"tagging retries per chunk (default: {DEFAULT_LLM_RETRIES})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max tokens for the tagging answer (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="keep existing text_chunks rows for this paper instead of replacing them",
    )
    parser.add_argument(
        "--no-create-schema", action="store_true",
        help="skip the CREATE TABLE / CREATE INDEX statements and assume they exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list the chunks that would be processed; make no API or DB writes",
    )
    return parser


def dry_run(chunks: Sequence[TextChunk]) -> None:
    print(f"Text blocks: {len(chunks)} (blocks of other types are skipped)")
    for chunk in chunks:
        preview = " ".join(chunk.content.split())[:100]
        print(f"  {chunk.evidence_path}: {preview}...")
    print()
    print("Dry run: no API calls and no database writes were made.")


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)

    if _IMPORT_ERROR is not None:
        print(
            f"error: missing dependency ({_IMPORT_ERROR}); install with: "
            "pip install psycopg2-binary dashscope numpy requests",
            file=sys.stderr,
        )
        return 2

    try:
        tree = load_content_tree(args.content_tree)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: cannot read content tree: {exc}", file=sys.stderr)
        return 2

    chunks = extract_text_blocks(tree)
    if not chunks:
        print("error: no text blocks found in the content tree", file=sys.stderr)
        return 2

    if args.dry_run:
        dry_run(chunks)
        return 0

    if not 1 <= args.embed_batch_size <= DEFAULT_EMBED_BATCH_SIZE:
        print(
            f"error: --embed-batch-size must be between 1 and "
            f"{DEFAULT_EMBED_BATCH_SIZE}",
            file=sys.stderr,
        )
        return 2
    if args.embed_max_chars < 1 or args.dimension < 1:
        print("error: --embed-max-chars and --dimension must be positive", file=sys.stderr)
        return 2

    # Tagging runs on the chat host; the embeddings stay on DashScope, which
    # is why the two keys are resolved separately.
    llm_api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    if not llm_api_key:
        print(
            "error: LLM_API_KEY is not set.\n"
            "  Windows PowerShell: $env:LLM_API_KEY = '...'\n"
            "  Windows cmd:        set LLM_API_KEY=...\n"
            "  bash:               export LLM_API_KEY=...",
            file=sys.stderr,
        )
        return 1
    embed_api_key = (
        os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or llm_api_key
    )

    try:
        conn = psycopg2.connect(args.dsn)
    except psycopg2.Error as exc:
        print(f"error: cannot connect to PostgreSQL: {exc}".strip(), file=sys.stderr)
        return 1

    try:
        if not args.no_create_schema:
            ensure_schema(conn)
            log("schema ready: text_chunks (+ vector and GIN indexes)")

        if not paper_exists(conn, args.paper_id):
            print(
                f"error: papers.id={args.paper_id} does not exist; "
                "import the paper first (save_to_db.py) or pass --paper-id",
                file=sys.stderr,
            )
            return 1

        log(
            f"{len(chunks)} text blocks to process, paper_id={args.paper_id}, "
            f"tagging model={args.llm_model}, embedding model={args.embedding_model}"
        )

        tagger = Tagger(
            base_url=args.base_url,
            api_key=llm_api_key,
            model=args.llm_model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        )

        # --- labels ----------------------------------------------------- #
        tags_per_chunk: list[list[str]] = []
        untagged = 0
        for number, chunk in enumerate(chunks, start=1):
            preview = " ".join(chunk.content.split())[:60]
            try:
                tags = tagger.tag(chunk.content)
            except Exception as exc:
                warn(f"tagging failed for {chunk.evidence_path} ({exc}); "
                     f"using [{FALLBACK_TAG}]")
                tags = [FALLBACK_TAG]
                untagged += 1
            tags_per_chunk.append(tags)
            log(f"  [{number}/{len(chunks)}] {chunk.evidence_path} -> {tags}  ({preview}...)")
            if args.llm_sleep > 0 and number < len(chunks):
                time.sleep(args.llm_sleep)

        # --- vectors ---------------------------------------------------- #
        log(f"embedding {len(chunks)} chunks with {args.embedding_model} ...")
        vectors = embed_all(
            [chunk.content for chunk in chunks],
            args.embedding_model,
            embed_api_key,
            args.embed_batch_size,
            args.embed_max_chars,
        )
        missing = sum(1 for vector in vectors if vector is None)

        # --- store ------------------------------------------------------ #
        rows = list(zip(chunks, tags_per_chunk, vectors))
        written = store_chunks(conn, args.paper_id, rows, not args.append, args.dimension)
        total = count_chunks(conn, args.paper_id)

        log(
            f"committed: {written} chunks inserted "
            f"(paper_id={args.paper_id}, text_chunks now holds {total} row(s))"
        )
        if untagged:
            log(f"  {untagged} chunk(s) got the fallback label [{FALLBACK_TAG}]")
        if missing:
            log(f"  {missing} chunk(s) were stored with embedding=NULL "
                f"(re-run the script to retry them)")
        return 0
    except Exception as exc:
        print(f"error: run failed and was rolled back: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
