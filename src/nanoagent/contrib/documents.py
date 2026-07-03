# -*- coding: utf-8 -*-
"""Document reading utilities — PDF, Word, Excel, PowerPoint, and more.

Converts office documents and other file formats to Markdown text suitable
for LLM consumption. Powered by Microsoft's ``markitdown`` library.

Supported formats (via markitdown):
    - PDF (.pdf)
    - Word (.docx)
    - Excel (.xlsx)
    - PowerPoint (.pptx)
    - Images (.jpg, .png, .gif, .bmp, .webp) — via OCR/vision description
    - HTML (.html, .htm)
    - CSV (.csv)
    - JSON (.json)
    - XML (.xml)
    - ZIP archives (.zip)
    - Plain text (.txt, .md, .py, .ts, etc.)

Usage::

    from nanoagent.contrib.documents import read_document

    text = read_document("report.pdf")
    text = read_document("data.xlsx")

As a nanoagent tool::

    from nanoagent.contrib.documents import document_reader_tool

    agent = Agent(..., tools=[document_reader_tool()])

With truncation::

    from nanoagent.contrib.documents import read_document
    from nanoagent.contrib.truncation import truncate_head

    full = read_document("large_report.pdf")
    safe = truncate_head(full, max_lines=2000).content
"""

from __future__ import annotations

import os
import pathlib
from typing import Any


def _get_markitdown():
    """Lazy-import markitdown with a helpful error if not installed."""
    try:
        from markitdown import MarkItDown
        return MarkItDown
    except ImportError:
        raise ImportError(
            "Document reading requires the 'markitdown' package. "
            "Install with: pip install markitdown\n"
            "Or for all extras: pip install markitdown[all]"
        )


# ── File type detection ──────────────────────────────────────────────────


_KNOWN_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".html", ".htm",
    ".csv", ".json", ".xml",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".zip",
    ".txt", ".md", ".py", ".ts", ".js", ".tsx", ".jsx",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".rst", ".tex",
}


def is_supported(path: str | pathlib.Path) -> bool:
    """Check if a file's extension is supported by the document reader."""
    ext = os.path.splitext(str(path))[1].lower()
    return ext in _KNOWN_EXTENSIONS or ext == ""


# ── Main API ─────────────────────────────────────────────────────────────


def read_document(path: str | pathlib.Path) -> str:
    """Convert a document file to Markdown text.

    Uses markitdown under the hood. The output is Markdown-formatted
    text suitable for LLM consumption — headings, lists, tables, and
    links are preserved.

    Args:
        path: Path to the document file (any supported format).

    Returns:
        Markdown text content of the document.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ImportError: If markitdown is not installed.
        ValueError: If the file format is not supported.
    """
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext and ext not in _KNOWN_EXTENSIONS:
        # Not in known list — try anyway, markitdown may handle it
        pass

    MarkItDown = _get_markitdown()
    md = MarkItDown()
    result = md.convert(str(path))
    return result.text_content


def read_document_with_metadata(path: str | pathlib.Path) -> dict[str, Any]:
    """Convert a document and return both text and metadata.

    Returns:
        Dict with keys: ``text`` (str), ``path`` (str), ``extension`` (str),
        ``size_bytes`` (int).
    """
    path = pathlib.Path(path)
    text = read_document(path)
    return {
        "text": text,
        "path": str(path),
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


# ── Tool factory ─────────────────────────────────────────────────────────


def document_reader_tool(
    *,
    tool_name: str = "read_document",
    max_size_mb: int = 50,
) -> Any:
    """Create a nanoagent Tool for reading documents.

    The tool accepts a file path and returns the document content as
    Markdown text. Large files (> max_size_mb) are rejected.

    Args:
        tool_name: Name for the tool (default: ``"read_document"``).
        max_size_mb: Maximum file size in MB before rejection.

    Returns:
        A nanoagent ``Tool`` instance.
    """
    from nanoagent.tools import Tool

    max_bytes = max_size_mb * 1024 * 1024

    async def reader(raw_args: dict[str, Any]) -> str:
        file_path = raw_args.get("path", raw_args.get("file_path", ""))
        if not file_path:
            return "Error: no path provided."

        path = pathlib.Path(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"

        if path.stat().st_size > max_bytes:
            from nanoagent.contrib.truncation import _format_bytes
            return (
                f"Error: file too large ({_format_bytes(path.stat().st_size)}). "
                f"Maximum size is {max_size_mb}MB."
            )

        try:
            ext = path.suffix.lower()
            text = read_document(path)

            # Add a small header so the LLM knows what it's looking at
            header = f"# Document: {path.name}\n"
            header += f"Type: {ext or 'unknown'}\n"
            header += f"Size: {path.stat().st_size:,} bytes\n\n"

            # Truncate if very large
            from nanoagent.contrib.truncation import truncate_head

            result = truncate_head(text, max_lines=3000, max_bytes=80 * 1024)
            return header + result.content

        except ImportError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading document: {e}"

    return Tool(
        name=tool_name,
        description=(
            "Read a document file and return its content as Markdown text. "
            "Supports: PDF, DOCX, XLSX (Excel), PPTX (PowerPoint), "
            "images (with OCR/description), HTML, CSV, JSON, XML, ZIP, "
            "and plain text files. "
            "Use this tool to inspect the contents of any document "
            "the user references."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the document file to read (absolute or relative).",
                }
            },
            "required": ["path"],
        },
        fn=reader,
    )