#!/usr/bin/env python3
"""Load extraction JSON files into the qgent PostgreSQL schema.

Inputs (produced by extract_to_db.py / build_content_tree.py):
    extracted/paper_info.json   -> one row in `papers`
    content_tree.json           -> `papers.content_tree` (JSONB)
    extracted/experiments.json  -> rows in `experiments`
    extracted/claims.json       -> rows in `claims`

The whole import runs in a single transaction: either the paper and all of its
experiments/claims are written, or nothing is.  A failure rolls everything back.

Re-running against the same paper title is refused unless --replace is given,
in which case the existing paper row (and its children, via ON DELETE CASCADE)
is deleted before the new import.

Connection settings come from the usual libpq environment variables
(PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE) and default to the values in
docker-compose.yml.

Usage:
    python save_to_db.py
    python save_to_db.py --replace
    python save_to_db.py --host 127.0.0.1 --dbname qgent_db --user qgent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:  # handled with a readable message in main()
    psycopg2 = None  # type: ignore[assignment]
    Json = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]

def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the .env beside this script.

    Real environment variables win, so an exported PGPASSWORD is never
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

CLAIM_TYPES = {"contribution", "limitation", "assumption", "position"}

EXP_INSERT_SQL = """
    INSERT INTO experiments (
        paper_id, exp_group_id, task, dataset, model_name, metric, value,
        hyperparams, is_ours, domain_specific, evidence_path
    ) VALUES %s
"""

CLAIM_INSERT_SQL = """
    INSERT INTO claims (
        paper_id, claim_type, content, embedding,
        domain_specific, evidence_path
    ) VALUES %s
"""


def load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file, failing with a clear message if it is missing."""
    if not path.is_file():
        raise SystemExit(f"error: input file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def as_jsonb(value: Any) -> Any:
    """Wrap a Python value for a JSONB column, preserving None as SQL NULL.

    Strings stay JSON strings: `evidence_path` is a plain string such as
    "Section: 5.4 Regularization, Block Index: 1", so it lands in JSONB as a
    JSON string value rather than SQL text.
    """
    return None if value is None else Json(value)


def validate(paper: Any, tree: Any, experiments: Any, claims: Any) -> None:
    """Fail fast on shape problems before touching the database."""
    if not isinstance(paper, dict) or not paper.get("title"):
        raise SystemExit("error: paper_info.json must be an object with a non-empty 'title'")
    if not isinstance(tree, dict):
        raise SystemExit("error: content_tree.json must be a JSON object")
    if not isinstance(experiments, list):
        raise SystemExit("error: experiments.json must be a JSON array")
    if not isinstance(claims, list):
        raise SystemExit("error: claims.json must be a JSON array")

    if tree.get("title") and tree["title"] != paper["title"]:
        print(
            f"warning: content_tree title {tree['title']!r} does not match "
            f"paper title {paper['title']!r}",
            file=sys.stderr,
        )

    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise SystemExit(f"error: experiments[{i}] is not an object")
        if exp.get("value") is not None and not isinstance(exp["value"], (int, float)):
            raise SystemExit(f"error: experiments[{i}].value is not numeric: {exp['value']!r}")

    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise SystemExit(f"error: claims[{i}] is not an object")
        if not claim.get("content"):
            raise SystemExit(f"error: claims[{i}].content is empty")
        if claim.get("claim_type") not in CLAIM_TYPES:
            raise SystemExit(
                f"error: claims[{i}].claim_type {claim.get('claim_type')!r} "
                f"is not one of {sorted(CLAIM_TYPES)}"
            )


def connect(args: argparse.Namespace) -> "psycopg2.extensions.connection":
    """Open a connection; libpq picks up PG* env vars for anything not passed."""
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)

    return psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
    )


def import_all(
    conn: "psycopg2.extensions.connection",
    paper: dict,
    tree: dict,
    experiments: list[dict],
    claims: list[dict],
    replace: bool,
    status: str,
) -> tuple[int, int, int]:
    """Insert one paper with its experiments and claims in a single transaction.

    Returns (paper_id, n_experiments, n_claims).  Raises on any error after
    rolling the transaction back.
    """
    paper_id = None
    n_experiments = 0
    n_claims = 0

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM papers WHERE title = %s ORDER BY id", (paper["title"],))
            existing = [row[0] for row in cur.fetchall()]

            if existing and not replace:
                raise RuntimeError(
                    f"paper {paper['title']!r} already exists (id={existing}); "
                    "re-run with --replace to overwrite it"
                )
            if existing:
                # ON DELETE CASCADE removes the old experiments and claims.
                cur.execute("DELETE FROM papers WHERE title = %s", (paper["title"],))
                print(f"replaced existing paper id(s): {existing}")

            cur.execute(
                """
                INSERT INTO papers (
                    title, authors, abstract, domain, method_tags, content_tree, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    paper["title"],
                    as_jsonb(paper.get("authors")),
                    paper.get("abstract"),
                    paper.get("domain"),
                    as_jsonb(paper.get("method_tags")),
                    as_jsonb(tree),
                    status,
                ),
            )
            paper_id = cur.fetchone()[0]

            if experiments:
                rows = [
                    (
                        paper_id,
                        exp.get("exp_group_id"),
                        exp.get("task"),
                        exp.get("dataset"),
                        exp.get("model_name"),
                        exp.get("metric"),
                        exp.get("value"),
                        as_jsonb(exp.get("hyperparams")),
                        exp.get("is_ours"),
                        as_jsonb(exp.get("domain_specific")),
                        as_jsonb(exp.get("evidence_path")),
                    )
                    for exp in experiments
                ]
                execute_values(cur, EXP_INSERT_SQL, rows)
                n_experiments = len(rows)

            if claims:
                rows = [
                    (
                        paper_id,
                        claim["claim_type"],
                        claim["content"],
                        None,  # embedding is filled later by the embedding job
                        as_jsonb(claim.get("domain_specific")),
                        as_jsonb(claim.get("evidence_path")),
                    )
                    for claim in claims
                ]
                execute_values(cur, CLAIM_INSERT_SQL, rows)
                n_claims = len(rows)

        conn.commit()
        return paper_id, n_experiments, n_claims

    except Exception:
        conn.rollback()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paper-info", type=Path, default=Path("extracted/paper_info.json"),
        help="paper_info.json path (default: extracted/paper_info.json)",
    )
    parser.add_argument(
        "--content-tree", type=Path, default=Path("content_tree.json"),
        help="content_tree.json path (default: content_tree.json)",
    )
    parser.add_argument(
        "--experiments", type=Path, default=Path("extracted/experiments.json"),
        help="experiments.json path (default: extracted/experiments.json)",
    )
    parser.add_argument(
        "--claims", type=Path, default=Path("extracted/claims.json"),
        help="claims.json path (default: extracted/claims.json)",
    )
    parser.add_argument(
        "--status", default="done",
        help="papers.status value for the imported paper (default: done)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="delete an existing paper with the same title (cascades) before importing",
    )
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--user", default=os.getenv("PGUSER", "qgent"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "qgent_db"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if psycopg2 is None:
        print("error: psycopg2 is not installed (pip install psycopg2-binary)", file=sys.stderr)
        return 2

    paper = load_json(args.paper_info)
    tree = load_json(args.content_tree)
    experiments = load_json(args.experiments)
    claims = load_json(args.claims)
    validate(paper, tree, experiments, claims)

    try:
        conn = connect(args)
    except psycopg2.Error as exc:
        print(f"error: cannot connect to PostgreSQL: {exc}".strip(), file=sys.stderr)
        return 1

    try:
        paper_id, n_experiments, n_claims = import_all(
            conn, paper, tree, experiments, claims, args.replace, args.status
        )
    except Exception as exc:
        print(f"error: import rolled back: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(
        f"import committed: paper_id={paper_id}, "
        f"experiments={n_experiments}, claims={n_claims}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
