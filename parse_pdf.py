#!/usr/bin/env python3
"""Pipeline stage 1: PDF -> content_tree.json.

For every PDF in ./pdfs, MinerU converts the document to Markdown, then
build_content_tree turns that Markdown into the structured JSON tree:

    ./pdfs/<name>.pdf  ->  ./parsed/<name>/content_tree.json

MinerU is driven through the `mineru-kit` CLI from ./.venv.  It runs
standalone — no doclib/parse-server daemon is involved — and its markdown
output inlines figure images as base64 data URIs, which is what
build_content_tree expects for image blocks.

Two MinerU CLI details this script works around:

  * `-p` is a page range, not the input path, and `mineru parse` silently
    defaults to the first 10 pages.  We always pass "-p all".
  * Only the `mineru-kit` route inlines images; `mineru parse` renders them
    as `![Image block]` placeholders, which would drop every figure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import build_content_tree

ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = ROOT / "pdfs"
DEFAULT_PARSED_DIR = ROOT / "parsed"

OUTPUT_NAME = "content_tree.json"
DEFAULT_TIER = "standard"
# MinerU's own default is the first 10 pages; papers are longer than that.
PAGES = "all"


def find_mineru_kit() -> str:
    found = shutil.which("mineru-kit")
    if found:
        return found
    for candidate in (
        ROOT / ".venv" / "Scripts" / "mineru-kit.exe",
        ROOT / ".venv" / "bin" / "mineru-kit",
    ):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("mineru-kit CLI not found (looked on PATH and in ./.venv)")


def detail(proc: subprocess.CompletedProcess[str]) -> str:
    """Last non-empty line of stderr, else stdout — MinerU's actual error."""
    for stream in (proc.stderr, proc.stdout):
        lines = [line.strip() for line in (stream or "").splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"exit code {proc.returncode}"


def to_markdown(
    kit: str, pdf: Path, md_path: Path, tier: str, timeout: int
) -> None:
    cmd = [
        kit,
        "parse",
        str(pdf),
        "-p",
        PAGES,
        "-o",
        str(md_path),
        "-f",
        "markdown",
        "--tier",
        tier,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"mineru-kit parse 超时（>{timeout}s）") from None
    if proc.returncode != 0:
        raise RuntimeError(f"mineru-kit parse 失败: {detail(proc)}")
    if not md_path.is_file():
        raise RuntimeError("mineru-kit parse 未生成 markdown 文件")


def process(pdf: Path, kit: str, parsed_dir: Path, tier: str, timeout: int) -> str:
    """Convert one PDF. Returns a one-line status; raises on failure."""
    out_dir = parsed_dir / pdf.stem
    dest = out_dir / OUTPUT_NAME
    if dest.is_file():
        return "SKIP: 已存在"

    # MinerU's scratch markdown lives only for the duration of the conversion.
    with tempfile.TemporaryDirectory(prefix="qgent_mineru_") as tmp:
        md_path = Path(tmp) / f"{pdf.stem}.md"
        to_markdown(kit, pdf, md_path, tier, timeout)
        tree = build_content_tree.build(md_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(tree, fh, ensure_ascii=False, indent=2)

    blocks = sum(len(section["blocks"]) for section in tree["sections"])
    return f"OK: {len(tree['sections'])} sections, {blocks} blocks"


def main(argv: list[str] | None = None) -> int:
    # The console codepage on Windows may not encode every character we print.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="input PDF directory"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_PARSED_DIR, help="output directory"
    )
    parser.add_argument(
        "--tier",
        default=DEFAULT_TIER,
        choices=("flash", "basic", "standard", "advanced"),
        help="MinerU parse tier (flash is text-only: no tables or images)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="seconds to let MinerU finish one document",
    )
    args = parser.parse_args(argv)

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"没有找到 PDF: {args.pdf_dir}")
        return 0

    kit = find_mineru_kit()
    print(f"mineru-kit: {kit}\ntier      : {args.tier}\npdfs      : {len(pdfs)} 个\n")

    failures = 0
    for pdf in pdfs:
        print(pdf.name)
        try:
            print(f"  {process(pdf, kit, args.out_dir, args.tier, args.timeout)}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {pdf.name} — {exc}", flush=True)

    print(f"\n完成: {len(pdfs) - failures} 成功, {failures} 失败")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
