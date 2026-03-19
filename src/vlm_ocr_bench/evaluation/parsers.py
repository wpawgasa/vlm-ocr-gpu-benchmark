"""Markdown structure extraction — split output into typed blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from vlm_ocr_bench.evaluation.metrics._utils import _normalize


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FORMULA = "formula"
    CODE = "code"
    LIST = "list"


@dataclass
class Block:
    """A typed block extracted from Markdown output."""

    type: BlockType
    content: str
    level: int = 0  # heading level (1-6), 0 for non-headings


def parse_markdown(text: str) -> list[Block]:
    """Split Markdown text into typed blocks.

    Identifies headings, tables, LaTeX formulas ($$...$$), code blocks,
    lists, and paragraphs.
    """
    text = _normalize(text)
    blocks: list[Block] = []
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Code block (``` ... ```)
        if stripped.startswith("```"):
            content_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                content_lines.append(lines[i])
                i += 1
            if i < len(lines):
                content_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.CODE, content="\n".join(content_lines)))
            continue

        # Display formula ($$...$$) — multi-line
        if stripped.startswith("$$"):
            content_lines = [line]
            if not stripped.endswith("$$") or stripped == "$$":
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    content_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    content_lines.append(lines[i])
                    i += 1
            else:
                i += 1
            formula_text = "\n".join(content_lines)
            # Strip $$ delimiters
            formula_text = re.sub(r"^\$\$\s*", "", formula_text)
            formula_text = re.sub(r"\s*\$\$$", "", formula_text)
            blocks.append(Block(type=BlockType.FORMULA, content=formula_text.strip()))
            continue

        # Heading (# ... ######)
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append(
                Block(
                    type=BlockType.HEADING,
                    content=heading_match.group(2).strip(),
                    level=level,
                )
            )
            i += 1
            continue

        # Table (lines starting with |)
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.TABLE, content="\n".join(table_lines)))
            continue

        # List (lines starting with - , * , or 1. )
        if re.match(r"^(\s*[-*+]|\s*\d+\.)\s", stripped):
            list_lines = []
            while i < len(lines):
                ls = lines[i].strip()
                if not ls:
                    break
                if re.match(r"^(\s*[-*+]|\s*\d+\.)\s", ls) or ls.startswith("  "):
                    list_lines.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append(Block(type=BlockType.LIST, content="\n".join(list_lines)))
            continue

        # Paragraph (default — collect consecutive non-empty, non-special lines)
        para_lines = []
        while i < len(lines):
            ls = lines[i].strip()
            if not ls:
                break
            if (
                ls.startswith("#")
                or ls.startswith("|")
                or ls.startswith("```")
                or ls.startswith("$$")
                or re.match(r"^(\s*[-*+]|\s*\d+\.)\s", ls)
            ):
                break
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            blocks.append(Block(type=BlockType.PARAGRAPH, content="\n".join(para_lines)))

    return blocks


def extract_tables(text: str) -> list[str]:
    """Extract all Markdown tables from text."""
    blocks = parse_markdown(text)
    return [b.content for b in blocks if b.type == BlockType.TABLE]


def extract_formulas(text: str) -> list[str]:
    """Extract all display-level LaTeX formulas from text."""
    blocks = parse_markdown(text)
    return [b.content for b in blocks if b.type == BlockType.FORMULA]
