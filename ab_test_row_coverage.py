"""A/B trial: does the ROW COVERAGE rule suppress the row-dropping behaviour?

Runs the real Table 4 extraction a few times with the shipped prompt and a few
times with the same prompt minus its ROW COVERAGE section, and compares how
many records come back.  Costs a handful of API calls; nothing is written to
the database.

    python ab_test_row_coverage.py [trials]
"""

from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

import extract_pipeline as p

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")


def section_boundaries(template: str) -> tuple[int, int]:
    start = template.index("ROW COVERAGE")
    end = template.index("CROSS-CELL NUMBERS")
    return start, end


def main() -> int:
    new_template = p.TABLE_TEMPLATE
    start, end = section_boundaries(new_template)
    old_template = new_template[:start] + new_template[end:]

    print(f"ROW COVERAGE section: {end - start} chars")
    print(f"new prompt has it: {'ROW COVERAGE' in new_template}")
    print(f"old prompt has it: {'ROW COVERAGE' in old_template}")
    print(f"old prompt is otherwise identical: "
          f"{old_template.replace(chr(10), '') == (new_template[:start] + new_template[end:]).replace(chr(10), '')}")

    tree = p.load_content_tree(Path("content_tree.json"))
    tree = {
        "title": tree.get("title"),
        "sections": [
            s for s in tree["sections"] if "Constituency" in (s.get("sec_title") or "")
        ],
    }
    unit = p.build_table_units(tree)[0]
    table_rows = p.count_table_data_rows(unit.table)
    print(f"\nTable 4 data rows as printed: {table_rows}")

    client = p.LLMClient(
        base_url=os.getenv("LLM_BASE_URL", p.DEFAULT_BASE_URL),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        timeout=p.DEFAULT_TIMEOUT,
        max_retries=p.DEFAULT_MAX_RETRIES,
        max_tokens=p.DEFAULT_MAX_TOKENS,
    )
    args = Namespace(max_chars=p.DEFAULT_MAX_CHARS, keep_duplicates=False)

    def trial(template: str) -> tuple[int, int]:
        p.TABLE_TEMPLATE = template
        rows, reports = p.extract_experiments(client, tree, args)
        names = [r["model_name"] for r in rows]
        repeated = len(names) - len(set(names))
        check = reports[0]["row_check"]
        assert check["warned"] == (len(rows) < table_rows), check
        return len(rows), repeated

    try:
        print(f"\n=== A. 有 ROW COVERAGE ({TRIALS} 次) ===")
        with_rule = [trial(new_template) for _ in range(TRIALS)]
        for n, (count, repeated) in enumerate(with_rule, 1):
            print(f"  第{n}次: {count} records, 重复模型名 {repeated} 个")

        print(f"\n=== B. 删掉 ROW COVERAGE ({TRIALS} 次) ===")
        without_rule = [trial(old_template) for _ in range(TRIALS)]
        for n, (count, repeated) in enumerate(without_rule, 1):
            print(f"  第{n}次: {count} records, 重复模型名 {repeated} 个")
    finally:
        p.TABLE_TEMPLATE = new_template

    a = [c for c, _ in with_rule]
    b = [c for c, _ in without_rule]
    print(f"\n原表行数: {table_rows}")
    print(f"A 有规则: {a}  全对: {all(x == table_rows for x in a)}")
    print(f"B 无规则: {b}  全对: {all(x == table_rows for x in b)}")
    print(f"B 触发告警次数: {sum(1 for x in b if x < table_rows)}/{len(b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
