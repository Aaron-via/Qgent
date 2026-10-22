#!/usr/bin/env python3
"""Embed pending claims with Qwen and write the vectors back to PostgreSQL.

Agent tool contract:
    * Reads credentials from DASHSCOPE_API_KEY (never hard-coded).  A .env file
      beside this script is loaded into the environment at startup; exported
      environment variables take precedence over it.
    * Selects only rows with `embedding IS NULL`, so the tool is idempotent and
      safe to re-run after a partial failure.
    * Calls the DashScope TextEmbedding API in batches of at most 10 texts.
    * Writes 1024-dimensional vectors into `claims.embedding` (vector(1024)).
    * One transaction: every successful update is committed together at the end.
      A failed row is reported, left NULL, and skipped; the rest still commit.

Exit codes:
    0  the run committed (possibly with rows skipped -- they stay NULL)
    1  fatal error (missing API key, DB connection, or commit failure)
    2  missing Python dependency

Usage:
    python embed_claims.py
    python embed_claims.py --batch-size 5 --model qwen3.7-text-embedding
    python embed_claims.py --dsn postgresql://qgent@localhost:5432/qgent_db
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
    import psycopg2
    import dashscope
except ImportError as exc:  # handled with a readable message in main()
    np = None  # type: ignore[assignment]
    psycopg2 = None  # type: ignore[assignment]
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

# Keep Chinese status messages readable when stdout is redirected (agent tools,
# logs) instead of a Windows console, which would otherwise use the ANSI code page.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# No password here: PGPASSWORD (from the environment or .env) is used instead.
DEFAULT_DSN = "postgresql://qgent@localhost:5432/qgent_db"
DEFAULT_MODEL = "qwen3.7-text-embedding"
DEFAULT_DIMENSION = 1024
MAX_BATCH_SIZE = 10
API_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def fetch_pending(conn: "psycopg2.extensions.connection") -> list[tuple[int, str]]:
    """Return (claim_id, content) for every claim that has no embedding yet."""
    with conn.cursor() as cur:
        cur.execute("SELECT claim_id, content FROM claims WHERE embedding IS NULL")
        return cur.fetchall()


def _response_error(response: Any) -> str | None:
    """Return a human-readable error for a failed DashScope response, else None."""
    status = getattr(response, "status_code", None)
    if status == 200:
        return None
    code = getattr(response, "code", "") or ""
    message = getattr(response, "message", "") or ""
    return f"status_code={status} code={code!r} message={message!r}"


def _extract_embeddings(response: Any, expected: int) -> list[list[float]]:
    """Pull embeddings out of a successful response, ordered by text_index."""
    output = getattr(response, "output", None) or {}
    items = output.get("embeddings") if isinstance(output, dict) else None
    if not items:
        raise RuntimeError(f"API response has no embeddings: {output!r}")

    # text_index tells us which input each vector belongs to; never rely on order.
    ordered = sorted(items, key=lambda item: item.get("text_index", 0))
    if len(ordered) != expected:
        raise RuntimeError(f"expected {expected} embeddings, got {len(ordered)}")
    return [item["embedding"] for item in ordered]


def embed_batch(
    texts: Sequence[str], model: str, api_key: str
) -> list[list[float]] | None:
    """Embed one batch, retrying transient errors.  Returns None if it keeps failing."""
    last_error = ""
    for attempt in range(API_RETRIES + 1):
        try:
            response = dashscope.TextEmbedding.call(
                model=model,
                input=list(texts),
                api_key=api_key,
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

        if attempt < API_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    warn(f"batch of {len(texts)} failed after {API_RETRIES + 1} attempts: {last_error}")
    return None


def embed_texts(
    items: Sequence[tuple[int, str]], model: str, api_key: str, batch_size: int
) -> dict[int, list[float]]:
    """Embed all pending claims, skipping any that cannot be embedded.

    If a whole batch fails, each text is retried on its own so one bad row
    (e.g. content rejected by a content filter) cannot sink its nine neighbours.
    """
    results: dict[int, list[float]] = {}

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        texts = [content for _, content in chunk]
        log(f"embedding claims {start + 1}-{start + len(chunk)} of {len(items)} ...")

        vectors = embed_batch(texts, model, api_key)
        if vectors is not None:
            for (claim_id, _), vector in zip(chunk, vectors):
                results[claim_id] = vector
            continue

        # Batch failed: fall back to one call per text.
        for claim_id, content in chunk:
            single = embed_batch([content], model, api_key)
            if single is None:
                warn(f"skipping claim_id={claim_id}: embedding failed after retries")
            else:
                results[claim_id] = single[0]

    return results


def to_pgvector(vector: Sequence[float], dimension: int) -> str:
    """Convert a Python list to pgvector's text literal: '[0.1,0.2,...]'."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dimension:
        raise ValueError(f"expected a {dimension}-d vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector contains NaN or infinity")
    return "[" + ",".join(repr(float(x)) for x in arr) + "]"


def store_embeddings(
    conn: "psycopg2.extensions.connection",
    results: dict[int, list[float]],
    dimension: int,
) -> int:
    """UPDATE all vectors, then commit once.  Rolls back on any error."""
    written = 0
    try:
        with conn.cursor() as cur:
            for claim_id, vector in results.items():
                try:
                    literal = to_pgvector(vector, dimension)
                except ValueError as exc:
                    warn(f"skipping claim_id={claim_id}: {exc}")
                    continue
                cur.execute(
                    "UPDATE claims SET embedding = %s::vector WHERE claim_id = %s",
                    (literal, claim_id),
                )
                written += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dsn", default=os.getenv("DATABASE_URL", DEFAULT_DSN),
        help=f"PostgreSQL connection string (default: {DEFAULT_DSN})",
    )
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        help=f"DashScope embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=MAX_BATCH_SIZE,
        help=f"texts per API call, 1-{MAX_BATCH_SIZE} (default: {MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dimension", type=int, default=DEFAULT_DIMENSION,
        help=f"expected vector dimension (default: {DEFAULT_DIMENSION})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if _IMPORT_ERROR is not None:
        print(f"error: missing dependency ({_IMPORT_ERROR}); "
              "install with: pip install psycopg2-binary dashscope numpy", file=sys.stderr)
        return 2

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print(
            "error: DASHSCOPE_API_KEY is not set.\n"
            "  Windows PowerShell: $env:DASHSCOPE_API_KEY = 'sk-...'\n"
            "  Windows cmd:        set DASHSCOPE_API_KEY=sk-...\n"
            "  bash:               export DASHSCOPE_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 1

    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        print(f"error: --batch-size must be between 1 and {MAX_BATCH_SIZE} "
              f"({args.model} accepts at most {MAX_BATCH_SIZE} texts per call)",
              file=sys.stderr)
        return 1

    try:
        conn = psycopg2.connect(args.dsn)
    except psycopg2.Error as exc:
        print(f"error: cannot connect to PostgreSQL: {exc}".strip(), file=sys.stderr)
        return 1

    try:
        pending = fetch_pending(conn)
        if not pending:
            log("所有 claims 均已向量化，无需处理")
            return 0

        # Blank content cannot be embedded; leave those rows NULL and out of the run.
        items: list[tuple[int, str]] = []
        for claim_id, content in pending:
            if content and content.strip():
                items.append((claim_id, content))
            else:
                warn(f"skipping claim_id={claim_id}: empty content")

        log(f"pending claims: {len(pending)}, to embed: {len(items)}, "
            f"model: {args.model}, batch size: {args.batch_size}")

        results = embed_texts(items, args.model, api_key, args.batch_size)
        if not results:
            warn("no embeddings were produced; nothing committed")
            return 1

        written = store_embeddings(conn, results, args.dimension)
        skipped = len(items) - len(results)
        log(f"committed: {written} claims updated, {skipped} skipped")
        if skipped:
            log("skipped claims keep embedding=NULL and will be retried on the next run")
        return 0
    except Exception as exc:
        print(f"error: run failed and was rolled back: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
