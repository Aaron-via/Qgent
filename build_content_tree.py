#!/usr/bin/env python3
"""Parse a MinerU-produced Markdown file into a structured content tree (JSON).

The Markdown is scanned line by line; no model inference and no third-party
Markdown parser are involved. Recognised blocks:

    text      paragraph (consecutive non-blank lines merged)
    table     pipe table rows, or a raw HTML <table>...</table>
    image     Markdown image with an inline data: URI (base64 body kept as-is)
    equation  $$ ... $$ display math
    code      fenced code block
    footnote  MinerU <small>...</small> page footnote (tags stripped)

Table captions ("Table 1: ...") normally precede the table in MinerU output,
image captions ("Figure 1: ...") normally follow the image.  Both are attached
to the corresponding block and removed from the text stream.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
DISPLAY_MATH_RE = re.compile(r"^[ \t]*\$\$")
HTML_TABLE_RE = re.compile(r"^[ \t]*<table\b", re.IGNORECASE)
FOOTNOTE_RE = re.compile(r"^[ \t]*<small\b", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
# Base64 payloads never contain whitespace or ")", so [^)\s]+ is safe on
# multi-hundred-KB lines and avoids a backtracking scan over the whole string.
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*(data:image/[^)\s]+)\s*\)")
CAPTION_RE = re.compile(
    r"^(?:Table|Figure|Fig\.?|Algorithm|Listing|图|表)\s*\d+(?:[.\-]\d+)*\s*[:：.]",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")


def strip_tags(text: str) -> str:
    return html.unescape(TAG_RE.sub("", text)).strip()


def is_blank(line: str) -> bool:
    return not line.strip()


class ContentTreeBuilder:
    def __init__(self) -> None:
        self.title = ""
        self.sections: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._paragraph: list[str] = []

    # -- container helpers --------------------------------------------------
    def _section(self) -> dict[str, Any]:
        if self._current is None:
            # Content appearing before the first "##" heading (copyright line,
            # authors, affiliation footnotes) gets its own leading section.
            self._current = {"sec_title": "Preamble", "blocks": []}
            self.sections.append(self._current)
        return self._current

    def _start_section(self, title: str) -> None:
        self._current = {"sec_title": title, "blocks": []}
        self.sections.append(self._current)

    def _add_block(self, block: dict[str, Any]) -> None:
        self._section()["blocks"].append(block)

    def _last_block(self) -> dict[str, Any] | None:
        blocks = self._section()["blocks"]
        return blocks[-1] if blocks else None

    def _flush_paragraph(self) -> None:
        if self._paragraph:
            content = "\n".join(self._paragraph).strip()
            self._paragraph.clear()
            if content:
                self._add_block({"type": "text", "content": content})

    # -- caption helpers ----------------------------------------------------
    def _take_leading_caption(self) -> str | None:
        """Pop a caption paragraph that sits immediately before a table."""
        block = self._last_block()
        if (
            block is not None
            and block["type"] == "text"
            and CAPTION_RE.match(block["content"].strip())
        ):
            self._section()["blocks"].pop()
            return block["content"].strip()
        return None

    # -- main entry ---------------------------------------------------------
    def parse(self, lines: list[str]) -> dict[str, Any]:
        i, n = 0, len(lines)
        while i < n:
            raw = lines[i].rstrip("\r\n")
            stripped = raw.strip()

            if is_blank(raw):
                self._flush_paragraph()
                i += 1
                continue

            # Fenced code ---------------------------------------------------
            fence = FENCE_RE.match(raw)
            if fence:
                self._flush_paragraph()
                marker = fence.group(1)
                body = [raw]
                i += 1
                while i < n:
                    body.append(lines[i].rstrip("\r\n"))
                    if lines[i].strip().startswith(marker):
                        i += 1
                        break
                    i += 1
                self._add_block({"type": "code", "content": "\n".join(body)})
                continue

            # Heading -------------------------------------------------------
            heading = HEADING_RE.match(raw)
            if heading:
                self._flush_paragraph()
                level, title = len(heading.group(1)), heading.group(2).strip()
                if level == 1 and not self.title:
                    self.title = title
                else:
                    self._start_section(title)
                i += 1
                continue

            # Display math --------------------------------------------------
            if DISPLAY_MATH_RE.match(raw):
                self._flush_paragraph()
                inner = stripped[2:]
                if inner.endswith("$$") and len(inner) >= 2:
                    equation = inner[:-2].strip()
                    i += 1
                else:
                    body = [inner] if inner else []
                    i += 1
                    while i < n:
                        line = lines[i].rstrip("\r\n")
                        i += 1
                        if line.strip().endswith("$$"):
                            tail = line.strip()[:-2].strip()
                            if tail:
                                body.append(tail)
                            break
                        body.append(line)
                    equation = "\n".join(body).strip()
                self._add_block({"type": "equation", "content": equation})
                continue

            # Image(s) ------------------------------------------------------
            images = list(IMAGE_RE.finditer(raw))
            if images:
                self._flush_paragraph()
                for match in images:
                    self._add_block(
                        {"type": "image", "content": match.group(1), "caption": None}
                    )
                # A figure caption follows the image, possibly after blank lines.
                j = i + 1
                while j < n and is_blank(lines[j]):
                    j += 1
                if j < n:
                    candidate = lines[j].strip()
                    if CAPTION_RE.match(candidate):
                        last = self._last_block()
                        if last is not None and last["type"] == "image":
                            last["caption"] = candidate
                        i = j + 1
                        continue
                i += 1
                continue

            # HTML table ----------------------------------------------------
            if HTML_TABLE_RE.match(raw):
                self._flush_paragraph()
                caption = self._take_leading_caption()
                body = [raw]
                i += 1
                while i < n and "</table>" not in body[-1].lower():
                    body.append(lines[i].rstrip("\r\n"))
                    i += 1
                self._add_block(
                    {"type": "table", "content": "\n".join(body), "caption": caption}
                )
                continue

            # Markdown pipe table -------------------------------------------
            if TABLE_ROW_RE.match(raw):
                self._flush_paragraph()
                rows: list[str] = []
                while i < n and TABLE_ROW_RE.match(lines[i].rstrip("\r\n")):
                    rows.append(lines[i].rstrip("\r\n").strip())
                    i += 1
                caption = self._take_leading_caption()
                self._add_block(
                    {"type": "table", "content": "\n".join(rows), "caption": caption}
                )
                continue

            # Footnote ------------------------------------------------------
            if FOOTNOTE_RE.match(raw):
                self._flush_paragraph()
                body: list[str] = [raw]
                while "</small>" not in body[-1].lower() and i + 1 < n:
                    i += 1
                    body.append(lines[i].rstrip("\r\n"))
                self._add_block({"type": "footnote", "content": strip_tags("\n".join(body))})
                i += 1
                continue

            # Ordinary text -------------------------------------------------
            self._paragraph.append(stripped)
            i += 1

        self._flush_paragraph()
        return self._tree()

    def _tree(self) -> dict[str, Any]:
        return {"title": self.title, "sections": self.sections}


def build(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8-sig") as fh:
        lines = fh.readlines()
    return ContentTreeBuilder().parse(lines)


def summarize(tree: dict[str, Any], output_path: Path) -> str:
    counts: dict[str, int] = {}
    for section in tree["sections"]:
        for block in section["blocks"]:
            counts[block["type"]] = counts.get(block["type"], 0) + 1
    size = output_path.stat().st_size if output_path.exists() else 0
    detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return (
        f"title    : {tree['title']!r}\n"
        f"sections : {len(tree['sections'])}\n"
        f"blocks   : {sum(counts.values())} ({detail})\n"
        f"output   : {output_path} ({size / 1024 / 1024:.2f} MB)"
    )


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=root / "output_test" / "1706.03762v7.md",
        help="MinerU Markdown file to parse",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "content_tree.json",
        help="destination JSON file",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    tree = build(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(tree, fh, ensure_ascii=False, indent=2)

    print(summarize(tree, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
