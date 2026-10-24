#!/usr/bin/env python3
"""Regression test for the extract_pipeline.py end-to-end pipeline.

Run this after changing any extraction, labelling, embedding or database
logic.  It stubs the LLM and the embedding model, so it makes no API calls and
costs nothing, and it works inside a throwaway PostgreSQL schema that it drops
on the way out.  A failure means the pipeline is broken, not that the database
needs cleaning up.

    python test_pipeline_e2e.py

What it covers:

* all three stages (paper, experiments, text) run through main() to completion
* --write-db, --write-db --replace and the read-only default each exit 0
* the paper row is inserted once and every child row carries its paper_id
* a block that states a claim is written to both claims and text_chunks
* every equation block is copied in Python under the fixed "formula_latex"
  label, with its LaTeX stored verbatim, without any model call and without
  producing a claim
* chunk labels are normalised to the standard vocabulary
  ("model_proposals" -> "model_proposal")
* a synonym is folded onto the existing word ("code_link" -> "source_code")
  instead of being stored as a label of its own
* a genuinely new label is stored and appended to the vocabulary file, and a
  second run reuses it instead of adding it again
* the shipped chunk_types_vocab.json is never modified: the runs work on a
  throwaway copy
* every claim and chunk row lands with its own 1024-d vector
* a failed batch costs only its own blocks; a failed table costs only its table
* no row is ever stored with an empty or placeholder chunk_types
* when every call of a stage fails the run aborts with exit code 3 and writes
  nothing, rather than importing an empty or junk-filled paper
* a model answer whose LaTeX broke the strict JSON parser ("\alpha" is an
  invalid escape) is salvaged by json_repair: the table and every text batch
  still come back whole instead of being dropped
* a cell the model echoed verbatim ("1.2K steps") is reduced in Python to
  1200 with the unit filed in domain_specific, instead of being dropped
* a win-rate table is filed under the single "Win Rate" metric with
  Win/Lose/Tie converted to 1.0/0.0/0.5
* a paper whose content tree holds no table at all skips the experiments stage
  with 0 records and no call -- it is not mistaken for a dead API -- and its
  text chunks and equations are still extracted
* --resume re-asks nothing and inserts nothing when the paper is complete,
  restores exactly the rows that were removed when it is not, and yields to
  --replace
* --scan walks a directory: a paper whose title the database already holds is
  skipped with "SKIP: 论文已存在" without a model call, an unseen paper is
  imported, and a missing or empty scan directory prints
  "未找到需要处理的新论文" and exits 0
* the real schema is left untouched: everything happens in a scratch schema

Exit codes:
    0  the pipeline behaved as expected
    1  a check failed (the details are printed)

Requires DATABASE_URL (or the usual PG* variables) to point at a database
where the caller may create and drop a schema.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_pipeline as p  # noqa: E402

DSN = os.environ["DATABASE_URL"]
SCHEMA = "pipeline_check"


def make_client(
    *,
    fail_batches=(),
    fail_all_text=False,
    drop_labels_for=(),
    fail_tables=(),
    fail_all_tables=False,
    labels=None,
    underreport_tables=False,
    latex_json=False,
    verbatim_values=False,
    win_rate_rows=False,
):
    """Stub LLMClient answering the three prompt shapes, with injected faults.

    fail_batches       1-based text batch numbers whose call raises LLMError
    fail_all_text      every text batch raises LLMError
    drop_labels_for    batch-local block ids answered without any chunk_types
    fail_tables        1-based table call numbers whose call raises LLMError
    fail_all_tables    every table call raises LLMError
    labels             batch-local block id -> the chunk_types to answer with,
                       replacing the default ["model_proposals"]
    underreport_tables answer each table with only half its data rows, the
                       silent row loss the row-count warning must catch
    latex_json         answer as a raw string carrying a single-backslash LaTeX
                       escape ("\alpha"), which json.loads rejects, and run it
                       through the pipeline's own parser: the answer must be
                       recovered by json_repair rather than losing the batch
    verbatim_values    answer each table with the cell text ("1.2K steps")
                       instead of a number, so the Python-side unit recovery is
                       what produces the stored value
    win_rate_rows      answer each table as an RLHF win-rate table ("Win",
                       "Lose", "Tie") so the metric label and the 0/0.5/1
                       conversion are exercised end to end

    Each table answer carries a distinct model_name and value per row, so
    neither the cross-table duplicate filter nor the within-table one can mask
    how many rows actually came back.
    """
    labels = labels or {}
    verdicts = ("Win", "Lose", "Tie")

    class _Client:
        def __init__(self, **kwargs):
            self.batch = 0
            self.table = 0

        @staticmethod
        def _titled(user):
            """The title the caller put in the prompt, so each input file gets
            its own paper row instead of colliding on a hard-coded name."""
            tail = user.split("TITLE:", 1)[-1]
            return tail.split("FRONT MATTER:", 1)[0].strip() or "E2E Attention"

        def _answer(self, value, expect, needle=None):
            """Return `value`, or a LaTeX-mangled raw answer run through the parser.

            `needle` is the JSON-encoded substring to leave a single backslash
            in front of: legal LaTeX, illegal JSON, and the exact shape
            parse_json_response has to repair.  Only the needle is touched, so
            the paper title the metadata stage returns stays the one the rest
            of the test keys its paper rows on.
            """
            if not latex_json or needle is None:
                return value
            text = json.dumps(value)
            assert needle in text, needle
            text = text.replace(needle, needle[:-1] + chr(92) + 'alpha"')
            return p.parse_json_response(text, expect)

        def chat_json(self, system, user, expect):
            if expect == "object":
                return self._answer(
                    {
                        "title": self._titled(user),
                        "authors": ["A V"],
                        "abstract": "x",
                        "domain": "NLP",
                        "method_tags": ["Transformer"],
                    },
                    expect,
                )
            if expect == "table":
                self.table += 1
                if fail_all_tables or self.table in fail_tables:
                    raise p.LLMError(
                        f"simulated API failure (table {self.table})"
                    )
                rows = table_rows_in_prompt(user)
                if underreport_tables:
                    rows //= 2
                return self._answer(
                    {
                        "records": [
                            {
                                "task": "MT",
                                "dataset": "EN-DE",
                                "model_name": f"base-{self.table}-{i}",
                                "metric": (
                                    "Win Rate"
                                    if win_rate_rows
                                    else "Training steps" if verbatim_values
                                    else "BLEU"
                                ),
                                "value": (
                                    verdicts[i % len(verdicts)]
                                    if win_rate_rows
                                    else "1.2K steps" if verbatim_values
                                    else 10.0 + i
                                ),
                                "is_ours": True,
                                "hyperparams": {"N": 6},
                                "domain_specific": None,
                            }
                            for i in range(rows)
                        ],
                        "special_rules_report": {},
                    },
                    expect,
                    needle='"MT"',
                )

            self.batch += 1
            if fail_all_text or self.batch in fail_batches:
                raise p.LLMError(f"simulated API failure (batch {self.batch})")
            return self._answer(
                [
                    {
                        "block_index": int(i),
                        "content": "echo",
                        "chunk_types": (
                            []
                            if int(i) in drop_labels_for
                            else labels.get(int(i), ["model_proposals"])
                        ),
                        "claim": (
                            {"claim_type": "contributions", "content": f"claim {i}"}
                            if int(i) % 3 == 0
                            else None
                        ),
                    }
                    for i in re.findall(r"\[Block (\d+)\]", user)
                ],
                expect,
                needle='"echo"',
            )

    return _Client


def table_rows_in_prompt(user):
    """How many data rows the table inside a rendered prompt prints."""
    table = user.split("TABLE:", 1)[-1]
    return p.count_table_data_rows(table) or 1


def run_main(argv):
    """Call main() with its console output suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return p.main(argv)


def run_main_capturing(argv):
    """Call main() and return (exit code, what it printed on stdout).

    Logging goes to stderr and is not captured here; the closing summary is
    printed on stdout, which is the part a user reads.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = p.main(argv)
    return code, buffer.getvalue()


def main() -> int:
    admin = psycopg2.connect(DSN)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    cur.execute(f"CREATE SCHEMA {SCHEMA}")
    cur.execute(f"GRANT ALL ON SCHEMA {SCHEMA} TO qgent")
    cur.execute(f"SET search_path TO {SCHEMA}, public")
    cur.execute(open("init.sql", encoding="utf-8").read())
    cur.execute(open("migrate_claims_plan.sql", encoding="utf-8").read())
    os.environ["PGOPTIONS"] = f"-csearch_path={SCHEMA},public"

    p.LLMClient = make_client()
    p.embed_all = lambda t, m, k, b, c: [[0.01] * p.DEFAULT_DIMENSION for _ in t]

    # The runs below grow the chunk vocabulary, so they work on a throwaway
    # copy: the shipped file must come out of this test untouched.
    shipped_vocab = Path(__file__).with_name(p.VOCAB_FILE_NAME)
    shipped_bytes = shipped_vocab.read_bytes()
    vocab_path = Path(tempfile.mkdtemp(prefix="qgent_vocab_")) / p.VOCAB_FILE_NAME
    shutil.copyfile(shipped_vocab, vocab_path)
    vocab_argv = ["--vocab", str(vocab_path)]

    write_argv = [
        "content_tree.json", "--write-db", "--no-thinking",
        "--llm-sleep", "0", "--print-rows", "1",
    ] + vocab_argv
    calls = {}
    calls["write"] = run_main(write_argv)
    calls["replace"] = run_main(write_argv + ["--replace"])
    calls["readonly"] = run_main(
        ["content_tree.json", "--no-thinking", "--llm-sleep", "0",
         "--print-rows", "0", "--no-embed"] + vocab_argv
    )
    # Nothing coined a label above, so those runs must have left the copy of
    # the vocabulary exactly as it shipped.
    vocab_untouched_by_plain_runs = (
        json.loads(vocab_path.read_text(encoding="utf-8"))["chunk_types"]
        == json.loads(shipped_bytes)["chunk_types"]
    )

    # Blocks per batch and batch count, so the expected drop counts are exact.
    # Equations are counted apart from the prose: they are inserted by the
    # Python bypass and never take part in a batch, so a failed or unlabelled
    # batch can never cost one.
    batch_size = p.DEFAULT_TEXT_BATCH_SIZE
    tree = p.load_content_tree(Path("content_tree.json"))
    text_blocks = p.build_text_blocks(tree, p.DEFAULT_MIN_TEXT_CHARS, False, p.TextStats())
    equations = p.build_equation_blocks(tree, False, p.TextStats())
    total_blocks = len(text_blocks) + len(equations)
    batch_count = -(-len(text_blocks) // batch_size)  # ceil

    check = psycopg2.connect(DSN)
    check.autocommit = True
    checker = check.cursor()
    checker.execute(f"SET search_path TO {SCHEMA}, public")
    results = {}
    for table in ("papers", "experiments", "claims", "text_chunks"):
        checker.execute(f"SELECT count(*) FROM {table}")
        results[table] = checker.fetchone()[0]
    checker.execute("SELECT count(*), count(embedding) FROM claims")
    results["claims_embedded"] = checker.fetchone()
    checker.execute("SELECT count(*), count(embedding) FROM text_chunks")
    results["chunks_embedded"] = checker.fetchone()
    checker.execute(
        "SELECT count(*) FROM text_chunks "
        "WHERE chunk_types = ARRAY['model_proposal']::text[]"
    )
    results["normalised_types"] = checker.fetchone()[0]
    # Every equation is present under the fixed label, with the LaTeX the tree
    # printed -- not a model echo -- and with no claim of its own.
    checker.execute(
        "SELECT count(*) FROM text_chunks "
        "WHERE chunk_types = ARRAY['formula_latex']::text[]"
    )
    results["formula_chunks"] = checker.fetchone()[0]
    stored_latex = {}
    checker.execute(
        "SELECT evidence_path, content FROM text_chunks "
        "WHERE chunk_types = ARRAY['formula_latex']::text[]"
    )
    stored_latex = {row[0]: row[1] for row in checker.fetchall()}
    results["formula_content_verbatim"] = stored_latex == {
        block.evidence_path: block.content for block in equations
    }
    results["formula_preamble_excluded"] = all(
        block.sec_title.strip() != p.PREAMBLE_TITLE for block in equations
    )
    checker.execute(
        "SELECT count(*) FROM claims WHERE evidence_path #>> '{}' = ANY(%s)",
        ([block.evidence_path for block in equations],),
    )
    results["claims_from_equations"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM claims c JOIN text_chunks t "
        "ON t.evidence_path = c.evidence_path #>> '{}'"
    )
    results["dual_written"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM experiments e JOIN papers p ON p.id = e.paper_id"
    )
    results["experiments_linked"] = checker.fetchone()[0]
    checker.execute("SELECT id FROM papers")
    results["paper_id"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM claims WHERE paper_id = %s", (results["paper_id"],)
    )
    results["claims_for_paper"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM text_chunks WHERE paper_id = %s", (results["paper_id"],)
    )
    results["chunks_for_paper"] = checker.fetchone()[0]
    results["total_blocks"] = len(text_blocks) + len(equations)
    results["block_count_agrees"] = results["text_chunks"] == results["total_blocks"]
    checker.execute("SELECT evidence_path FROM claims LIMIT 1")
    results["claim_evidence"] = checker.fetchone()[0]
    checker.execute("SELECT DISTINCT sec_title FROM text_chunks ORDER BY 1 LIMIT 3")
    results["some_sections"] = [row[0] for row in checker.fetchall()]

    # --- failure handling ------------------------------------------------ #
    # The stub answers every table with one record per data row, so a healthy
    # run stores exactly as many experiments as the tables print rows.
    units = p.build_table_units(tree)
    table_rows = [p.count_table_data_rows(unit.table) or 0 for unit in units]
    expected_experiments = sum(table_rows)
    results["expected_experiments"] = expected_experiments

    # Every table call fails: abort with exit code 3 and write nothing, so an
    # empty experiments list can never be mistaken for "the paper has none".
    p.LLMClient = make_client(fail_all_tables=True)
    results["exit_all_tables_failed"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT count(*) FROM experiments")
    results["experiments_after_total_table_failure"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_total_table_failure"] = checker.fetchone()[0]

    # One table fails: the rest of the paper is still extracted.
    p.LLMClient = make_client(fail_tables={1})
    results["exit_one_table_failed"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT count(*) FROM experiments")
    results["experiments_after_one_failed_table"] = checker.fetchone()[0]

    # The model silently answers each table with half its rows: the run must
    # still complete, but the row-count warning must fire and say so.
    p.LLMClient = make_client(underreport_tables=True)
    code, output = run_main_capturing(write_argv + ["--replace"])
    results["exit_underreported"] = code
    # one "原表 X 行 -> 模型返回 Y 行" line per warned table in the summary
    results["underreport_warned_tables"] = output.count("原表")
    results["underreport_highlighted"] = "疑似丢失行" in output
    checker.execute("SELECT count(*) FROM experiments")
    results["experiments_after_underreport"] = checker.fetchone()[0]

    # ...and the per-table check itself reports both counts (its LOG.warning
    # goes to stderr, so it is exercised directly rather than via stdout).
    unit_check = p.check_table_rows(units[0], 1)
    results["unit_check_warned"] = unit_check["warned"]
    results["unit_check_rows"] = unit_check["table_data_rows"]
    results["unit_check_lost"] = unit_check["lost_rows"]
    # a theoretical table answered with nothing stays silent
    theoretical = [
        u for u in units if p.THEORETICAL_TABLE_RE.search(u.table)
    ]
    results["theoretical_exempt"] = (
        not p.check_table_rows(theoretical[0], 0)["warned"]
        if theoretical
        else None
    )

    # --- malformed JSON and non-standard cells ---------------------------- #
    # LaTeX in a JSON string breaks the strict parser ("\alpha" is an invalid
    # escape).  json_repair must salvage the answer instead of the batch or
    # table being dropped.  Both the table stage and the text stage are
    # exercised, because the repair happens inside parse_json_response, which
    # both go through.
    p.LLMClient = make_client(latex_json=True)
    calls["latex_json"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT count(*) FROM experiments")
    results["experiments_after_latex_json"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_latex_json"] = checker.fetchone()[0]

    # A cell the model echoes verbatim ("1.2K steps") instead of reducing it to
    # the FLOAT the schema wants: Python recovers 1200 and files the unit away
    # in domain_specific.  The record must survive rather than being dropped.
    p.LLMClient = make_client(verbatim_values=True)
    calls["verbatim_values"] = run_main(write_argv + ["--replace"])
    checker.execute(
        "SELECT count(*) FROM experiments WHERE value = 1200 "
        "AND domain_specific ->> 'unit' = 'steps'"
    )
    results["unit_recovered_rows"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM experiments WHERE value IN (1200, 1200.0)"
    )
    results["unit_recovered_from_verbatim"] = checker.fetchone()[0]

    # An RLHF win-rate table answered with Win/Lose/Tie: the metric is renamed
    # to the one canonical label and the verdicts become 1.0 / 0.0 / 0.5.
    p.LLMClient = make_client(win_rate_rows=True)
    calls["win_rate"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT DISTINCT metric FROM experiments")
    results["win_rate_metrics"] = sorted(row[0] for row in checker.fetchall())
    checker.execute("SELECT count(*) FROM experiments WHERE value = 1.0")
    results["win_rate_wins"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM experiments WHERE value = 0.0")
    results["win_rate_losses"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM experiments WHERE value = 0.5")
    results["win_rate_ties"] = checker.fetchone()[0]

    # --- a paper with no table at all ------------------------------------- #
    # The stage must return 0 records without a single call and without
    # tripping the "every table failed" abort: a purely theoretical paper has
    # no experiments, it is not a dead API.
    tableless = {
        "title": "A Purely Theoretical Paper",
        "sections": [
            {
                "sec_title": "1 Introduction",
                "blocks": [
                    {
                        "type": "text",
                        "content": (
                            "We propose a purely theoretical framework with no "
                            "empirical evaluation whatsoever, only proofs."
                        ),
                    }
                ],
            },
            {
                "sec_title": "2 Method",
                "blocks": [
                    {"type": "equation", "content": r"E = mc^2"},
                    {
                        "type": "text",
                        "content": (
                            "The derivation follows from the definition above "
                            "and holds for every finite configuration."
                        ),
                    },
                ],
            },
        ],
    }
    tableless_path = Path(tempfile.mkdtemp(prefix="qgent_notable_")) / "no_tables.json"
    tableless_path.write_text(json.dumps(tableless), encoding="utf-8")
    p.LLMClient = make_client()
    calls["no_tables"] = run_main(
        [str(tableless_path), "--write-db", "--no-thinking", "--llm-sleep", "0",
         "--print-rows", "0"] + vocab_argv
    )
    checker.execute(
        "SELECT count(*) FROM experiments e JOIN papers p ON p.id = e.paper_id "
        "WHERE p.title = %s",
        (tableless["title"],),
    )
    results["experiments_for_tableless_paper"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM text_chunks t JOIN papers p ON p.id = t.paper_id "
        "WHERE p.title = %s",
        (tableless["title"],),
    )
    results["chunks_for_tableless_paper"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM text_chunks t JOIN papers p ON p.id = t.paper_id "
        "WHERE p.title = %s AND t.chunk_types = ARRAY['formula_latex']::text[]",
        (tableless["title"],),
    )
    results["formulas_for_tableless_paper"] = checker.fetchone()[0]
    # Drop the paper again: the resume checks below count paper rows, and this
    # one is a visitor rather than part of the fixture.
    checker.execute("DELETE FROM papers WHERE title = %s", (tableless["title"],))

    # Every text batch fails: abort with exit code 3 and write nothing at all.
    p.LLMClient = make_client(fail_all_text=True)
    results["exit_all_batches_failed"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_total_failure"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM claims")
    results["claims_after_total_failure"] = checker.fetchone()[0]

    # One batch fails: only its blocks are lost, and the run still completes.
    p.LLMClient = make_client(fail_batches={1})
    results["exit_one_batch_failed"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_one_failed_batch"] = checker.fetchone()[0]

    # The first block of every batch comes back without labels: each one is
    # dropped, and the run still completes.
    p.LLMClient = make_client(drop_labels_for={1})
    results["exit_unlabelled"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_unlabelled"] = checker.fetchone()[0]

    # --- the chunk vocabulary --------------------------------------------- #
    # One block per batch answers with a synonym plus a label the vocabulary
    # does not have: the synonym must land on the existing word, and the new
    # label must be stored as is and remembered in the vocabulary file.
    p.LLMClient = make_client(labels={1: ["code_link", "data_augmentation"]})
    calls["exit_vocab"] = run_main(write_argv + ["--replace"])
    checker.execute(
        "SELECT count(*) FROM text_chunks WHERE chunk_types @> ARRAY['source_code']"
    )
    results["synonym_folded"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM text_chunks WHERE 'code_link' = ANY(chunk_types)"
    )
    results["synonym_stored_raw"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM text_chunks "
        "WHERE chunk_types @> ARRAY['data_augmentation']"
    )
    results["new_label_chunks"] = checker.fetchone()[0]
    vocab_after = json.loads(vocab_path.read_text(encoding="utf-8"))
    results["vocab_grew"] = vocab_after["chunk_types"] == (
        json.loads(shipped_bytes)["chunk_types"] + ["data_augmentation"]
    )

    # The label is in the vocabulary now, so a second run reuses it instead of
    # appending it a second time.
    calls["exit_vocab_again"] = run_main(write_argv + ["--replace"])
    results["vocab_stable"] = (
        json.loads(vocab_path.read_text(encoding="utf-8")) == vocab_after
    )
    checker.execute(
        "SELECT count(*) FROM text_chunks "
        "WHERE chunk_types @> ARRAY['data_augmentation']"
    )
    results["new_label_chunks_again"] = checker.fetchone()[0]

    # No run may leave an unlabelled or placeholder-labelled chunk behind.
    checker.execute(
        "SELECT count(*) FROM text_chunks "
        "WHERE chunk_types IS NULL OR chunk_types = '{}'::text[] "
        "OR 'uncategorized' = ANY(chunk_types)"
    )
    results["placeholder_chunks"] = checker.fetchone()[0]

    # --- resume ----------------------------------------------------------- #
    # A clean, complete import is the baseline every resume check compares to.
    p.LLMClient = make_client()
    calls["resume_base"] = run_main(write_argv + ["--replace"])
    checker.execute("SELECT id FROM papers ORDER BY id")
    paper_ids_before = [row[0] for row in checker.fetchall()]
    checker.execute("SELECT count(*) FROM experiments")
    experiments_before = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM claims")
    claims_before = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM text_chunks")
    chunks_before = checker.fetchone()[0]

    # Nothing is missing, so a resumed run re-asks nothing and adds nothing --
    # and above all it does not touch the paper row it resumes into.
    calls["resume_noop"] = run_main(write_argv + ["--resume"])
    checker.execute("SELECT id FROM papers ORDER BY id")
    results["paper_ids_after_resume_noop"] = [row[0] for row in checker.fetchall()]
    for table, key in (
        ("experiments", "experiments_after_resume_noop"),
        ("claims", "claims_after_resume_noop"),
        ("text_chunks", "chunks_after_resume_noop"),
    ):
        checker.execute(f"SELECT count(*) FROM {table}")
        results[key] = checker.fetchone()[0]

    # Punch a hole in the paper -- one table's records, one chunk that has no
    # claim of its own, and one equation -- then resume.  Exactly the missing
    # rows come back, under the same paper, and nothing already stored is
    # duplicated.  The equation matters on its own: it is inserted by the
    # Python bypass, so resume has to skip and restore it without any call.
    tree = p.load_content_tree(Path("content_tree.json"))
    first_table = p.build_table_units(tree)[0].evidence_path
    checker.execute(
        "DELETE FROM experiments WHERE evidence_path #>> '{}' = %s", (first_table,)
    )
    checker.execute(
        "SELECT t.evidence_path FROM text_chunks t WHERE NOT EXISTS ("
        "  SELECT 1 FROM claims c WHERE c.evidence_path #>> '{}' = t.evidence_path"
        ") AND NOT ('formula_latex' = ANY(t.chunk_types)) "
        "ORDER BY t.chunk_id LIMIT 1"
    )
    missing_chunk = checker.fetchone()[0]
    equation_path = equations[0].evidence_path
    checker.execute(
        "DELETE FROM text_chunks WHERE evidence_path = ANY(%s)",
        ([missing_chunk, equation_path],),
    )

    calls["resume_partial"] = run_main(write_argv + ["--resume"])
    checker.execute("SELECT count(*) FROM papers")
    results["papers_after_resume_partial"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM experiments")
    results["experiments_after_resume_partial"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM claims")
    results["claims_after_resume_partial"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_resume_partial"] = checker.fetchone()[0]
    checker.execute(
        "SELECT content FROM text_chunks WHERE evidence_path = %s", (equation_path,)
    )
    row = checker.fetchone()
    results["equation_restored_on_resume"] = (
        row is not None and row[0] == equations[0].content
    )

    # --replace beats --resume: the paper is deleted and extracted again, so
    # the counts hold while the paper row itself is a new one.
    calls["resume_and_replace"] = run_main(write_argv + ["--resume", "--replace"])
    checker.execute("SELECT id FROM papers ORDER BY id")
    results["paper_ids_after_resume_replace"] = [row[0] for row in checker.fetchall()]
    checker.execute("SELECT count(*) FROM experiments")
    results["experiments_after_resume_replace"] = checker.fetchone()[0]
    checker.execute("SELECT count(*) FROM text_chunks")
    results["chunks_after_resume_replace"] = checker.fetchone()[0]

    # --- --scan: batch mode over a directory ------------------------------- #
    # Two trees: the fixture's own, whose title the scratch schema already
    # holds, and a paper the scan has never seen.  The first must be skipped
    # with "SKIP: 论文已存在" before any model call -- if the dedup failed it
    # would instead be extracted again and die on the existing paper row, so a
    # zero exit code is the evidence that nothing was re-sent.  The second must
    # go through the ordinary pipeline and land in the database.
    scan_dir = Path(tempfile.mkdtemp(prefix="qgent_scan_"))
    (scan_dir / "1706.03762v7").mkdir()
    shutil.copyfile("content_tree.json", scan_dir / "1706.03762v7" / "content_tree.json")
    scan_paper = {
        "title": "A Paper Only The Scan Has Seen",
        "sections": [
            {
                "sec_title": "1 Introduction",
                "blocks": [
                    {
                        "type": "text",
                        "content": (
                            "This paper is only ever reached through the scan "
                            "mode, which must import it like any other tree."
                        ),
                    }
                ],
            }
        ],
    }
    (scan_dir / "new_paper").mkdir()
    (scan_dir / "new_paper" / "content_tree.json").write_text(
        json.dumps(scan_paper), encoding="utf-8"
    )

    p.LLMClient = make_client()
    scan_code, scan_output = run_main_capturing(
        ["--scan", "--scan-dir", str(scan_dir), "--write-db", "--no-thinking",
         "--llm-sleep", "0", "--print-rows", "0"] + vocab_argv
    )
    calls["scan"] = scan_code
    results["scan_skipped_message"] = "SKIP: 论文已存在" in scan_output
    checker.execute(
        "SELECT count(*) FROM papers WHERE title = %s", (scan_paper["title"],)
    )
    results["scan_imported"] = checker.fetchone()[0]
    checker.execute(
        "SELECT count(*) FROM text_chunks t JOIN papers p ON p.id = t.paper_id "
        "WHERE p.title = %s",
        (scan_paper["title"],),
    )
    results["scan_chunks"] = checker.fetchone()[0]
    # A directory with nothing in it, and a directory that is not there at all:
    # both are "no new papers", never a crash.
    empty_code, empty_output = run_main_capturing(
        ["--scan", "--scan-dir", str(tempfile.mkdtemp(prefix="qgent_scan_empty_")),
         "--write-db"]
    )
    missing_code, missing_output = run_main_capturing(
        ["--scan", "--scan-dir", str(scan_dir / "not_a_directory"), "--write-db"]
    )
    results["scan_empty_exit"] = empty_code
    results["scan_empty_message"] = p.NO_NEW_PAPERS_MESSAGE in empty_output
    results["scan_missing_exit"] = missing_code
    results["scan_missing_message"] = p.NO_NEW_PAPERS_MESSAGE in missing_output
    # Drop the paper again: the checks below count paper rows, and this one is
    # a visitor rather than part of the fixture.
    checker.execute("DELETE FROM papers WHERE title = %s", (scan_paper["title"],))

    checker.execute("SET search_path TO public")
    checker.execute("SELECT count(*) FROM papers")
    results["public_papers"] = checker.fetchone()[0]
    # Every run above worked on a copy: the shipped vocabulary is untouched.
    results["shipped_vocab_untouched"] = shipped_vocab.read_bytes() == shipped_bytes
    # Release the read locks before dropping the schema.
    check.close()

    print("exit codes:", calls, flush=True)
    for key, value in results.items():
        print(f"{key}: {value}", flush=True)
    ok = (
        calls == {
            "write": 0, "replace": 0, "readonly": 0, "resume_base": 0,
            "resume_noop": 0, "resume_partial": 0, "resume_and_replace": 0,
            "exit_vocab": 0, "exit_vocab_again": 0,
            # the malformed-JSON, non-standard-cell and no-table runs
            "latex_json": 0, "verbatim_values": 0, "win_rate": 0, "no_tables": 0,
            "scan": 0,
        }
        and results["papers"] == 1
        and results["claims"] > 0
        and results["text_chunks"] > 0
        and results["claims_embedded"][1] == results["claims"]
        and results["chunks_embedded"][1] == results["text_chunks"]
        # every equation is stored verbatim under formula_latex, with no claim
        and results["formula_chunks"] == len(equations)
        and results["formula_content_verbatim"]
        and results["formula_preamble_excluded"]
        and results["claims_from_equations"] == 0
        and results["block_count_agrees"]
        and results["normalised_types"] + results["formula_chunks"]
        == results["text_chunks"]
        and results["dual_written"] == results["claims"]
        and results["claims_for_paper"] == results["claims"]
        and results["chunks_for_paper"] == results["text_chunks"]
        # a healthy run stores one record per data row of every table
        and results["experiments"] == expected_experiments
        # a dead API aborts instead of importing an empty paper
        and results["exit_all_batches_failed"] == 3
        and results["chunks_after_total_failure"] == total_blocks
        and results["claims_after_total_failure"] == results["claims"]
        # ...and the same guard covers the table stage
        and results["exit_all_tables_failed"] == 3
        and results["experiments_after_total_table_failure"] == expected_experiments
        and results["chunks_after_total_table_failure"] == total_blocks
        # a single failed table costs only that table
        and results["exit_one_table_failed"] == 0
        and results["experiments_after_one_failed_table"]
        == expected_experiments - table_rows[0]
        # silently dropping half the rows is warned about, never accepted quietly
        and results["exit_underreported"] == 0
        and results["underreport_warned_tables"] == len(units)
        and results["underreport_highlighted"]
        and results["experiments_after_underreport"]
        == sum(rows // 2 for rows in table_rows)
        # the per-table check names the table, the row count and the shortfall
        and results["unit_check_warned"]
        and results["unit_check_lost"] == table_rows[0] - 1
        # a table that is pure theory is exempt from the empty-answer warning
        and results["theoretical_exempt"] in (True, None)
        # a single failed batch costs exactly its own blocks
        and results["exit_one_batch_failed"] == 0
        and results["chunks_after_one_failed_batch"] == total_blocks - batch_size
        # an unlabelled block is dropped, not stored under a placeholder
        and results["exit_unlabelled"] == 0
        and results["chunks_after_unlabelled"] == total_blocks - batch_count
        and results["placeholder_chunks"] == 0
        # a synonym is folded onto the existing word, never stored as a label
        and results["synonym_folded"] == batch_count
        and results["synonym_stored_raw"] == 0
        # a genuinely new label is stored and appended to the vocabulary file
        and results["new_label_chunks"] == batch_count
        and results["vocab_grew"]
        # ...and the next run reuses it instead of adding it a second time
        and results["vocab_stable"]
        and results["new_label_chunks_again"] == batch_count
        # no run may rewrite the vocabulary the pipeline ships with
        and vocab_untouched_by_plain_runs
        and results["shipped_vocab_untouched"]
        # --resume re-asks nothing and duplicates nothing when all is stored
        and results["paper_ids_after_resume_noop"] == paper_ids_before
        and results["experiments_after_resume_noop"] == experiments_before
        and results["claims_after_resume_noop"] == claims_before
        and results["chunks_after_resume_noop"] == chunks_before
        # --resume restores exactly the rows that were removed, and no more
        and results["papers_after_resume_partial"] == len(paper_ids_before)
        and results["experiments_after_resume_partial"] == experiments_before
        and results["claims_after_resume_partial"] == claims_before
        and results["chunks_after_resume_partial"] == chunks_before
        # ...including an equation, restored by the Python bypass on resume
        and results["equation_restored_on_resume"]
        # malformed LaTeX JSON is repaired, not thrown away: every table and
        # every batch still comes back whole
        and results["experiments_after_latex_json"] == expected_experiments
        and results["chunks_after_latex_json"] == total_blocks
        # a cell the model echoed verbatim is reduced in Python, with its unit
        # filed away instead of the record being dropped
        and results["unit_recovered_rows"] == expected_experiments
        and results["unit_recovered_from_verbatim"] == expected_experiments
        # a win-rate table is filed under one metric with Win/Lose/Tie -> 1/0/0.5
        and results["win_rate_metrics"] == ["Win Rate"]
        and results["win_rate_wins"] + results["win_rate_losses"] \
            + results["win_rate_ties"] == expected_experiments
        and results["win_rate_wins"] > 0
        and results["win_rate_losses"] > 0
        and results["win_rate_ties"] > 0
        # a paper with no table at all skips the experiments stage cleanly and
        # still gets its text chunks and its equation
        and results["experiments_for_tableless_paper"] == 0
        and results["chunks_for_tableless_paper"] > 0
        and results["formulas_for_tableless_paper"] == 1
        # --replace wins over --resume: same rows, a brand new paper row
        and len(results["paper_ids_after_resume_replace"]) == 1
        and results["paper_ids_after_resume_replace"][0] > paper_ids_before[0]
        and results["experiments_after_resume_replace"] == experiments_before
        and results["chunks_after_resume_replace"] == chunks_before
        # --scan walks a directory: a paper the database already holds is
        # skipped with "SKIP: 论文已存在" and costs no model call, a paper it does
        # not hold is imported, and a missing or empty directory prints
        # "未找到需要处理的新论文" and exits 0 instead of failing the job
        and results["scan_skipped_message"]
        and results["scan_imported"] == 1
        and results["scan_chunks"] > 0
        and results["scan_empty_exit"] == 0
        and results["scan_empty_message"]
        and results["scan_missing_exit"] == 0
        and results["scan_missing_message"]
    )
    cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    print("RESULT:", "ALL GOOD" if ok else "PROBLEM", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
