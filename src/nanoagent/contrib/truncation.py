# -*- coding: utf-8 -*-
"""Output truncation utilities for tools.

Large tool outputs can overwhelm the LLM's context window and increase
costs. These utilities help you truncate output to safe limits while
optionally saving the full output to a file.

Inspired by pi's ``truncateHead`` / ``truncateTail`` utilities.

Usage::

    from nanoagent.contrib.truncation import truncate_head, DEFAULT_LIMITS

    output = run_some_command()
    result = truncate_head(output, max_lines=2000, max_bytes=50000)

    if result.truncated:
        result.save_to_tempfile("/tmp/full_output.txt")

    return result.content  # truncated text with notice appended
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass


@dataclass
class TruncationLimits:
    """Recommended truncation limits.

    Defaults align with typical models' context windows (~10K tokens for
    a single tool output, leaving room for system prompt, conversation
    history, and other tool results).
    """

    max_lines: int = 2000
    max_bytes: int = 50 * 1024  # 50KB

    @classmethod
    def generous(cls) -> TruncationLimits:
        """1000 lines / 100KB — for tools where detail matters."""
        return cls(max_lines=1000, max_bytes=100 * 1024)

    @classmethod
    def tight(cls) -> TruncationLimits:
        """200 lines / 10KB — for tools where only key output matters."""
        return cls(max_lines=200, max_bytes=10 * 1024)


DEFAULT_LIMITS = TruncationLimits()
"""Default limits: 2000 lines / 50KB."""


@dataclass
class TruncationResult:
    """Result of applying truncation to a string."""

    content: str
    """The (possibly truncated) output text."""

    truncated: bool
    """Whether the output was truncated."""

    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    omitted_lines: int = 0
    omitted_bytes: int = 0

    def save_to_tempfile(self, full_text: str, prefix: str = "nanoagent-") -> str:
        """Save the full untruncated text to a temp file.

        Returns the path to the temp file.
        """
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(full_text)
        return path

    @classmethod
    def not_truncated(cls, text: str) -> TruncationResult:
        """Create a result for text that fits within limits."""
        lines = text.split("\n")
        nbytes = len(text.encode("utf-8"))
        return cls(
            content=text,
            truncated=False,
            total_lines=len(lines),
            total_bytes=nbytes,
            output_lines=len(lines),
            output_bytes=nbytes,
        )


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def truncate_head(
    text: str,
    max_lines: int | None = None,
    max_bytes: int | None = None,
    limits: TruncationLimits | None = None,
) -> TruncationResult:
    """Keep the first N lines/bytes. Best for search results, file reads.

    Truncates at whichever limit is hit first.

    Args:
        text: The full output text.
        max_lines: Override line limit.
        max_bytes: Override byte limit.
        limits: Use a predefined TruncationLimits (overridden by explicit args).
    """
    lim = limits or DEFAULT_LIMITS
    ml = max_lines if max_lines is not None else lim.max_lines
    mb = max_bytes if max_bytes is not None else lim.max_bytes

    all_lines = text.split("\n")
    total_lines = len(all_lines)
    total_bytes = len(text.encode("utf-8"))

    if total_lines <= ml and total_bytes <= mb:
        return TruncationResult.not_truncated(text)

    # Truncate by lines first
    truncated_lines = all_lines[:ml]
    joined = "\n".join(truncated_lines)

    # Then check bytes
    encoded = joined.encode("utf-8")
    if len(encoded) > mb:
        # Truncate bytes (preserving UTF-8)
        encoded = encoded[:mb]
        # Find last complete UTF-8 character
        while encoded and (encoded[-1] & 0xC0) == 0x80:
            encoded = encoded[:-1]
        joined = encoded.decode("utf-8", errors="replace")
        output_lines = joined.count("\n") + 1
    else:
        output_lines = min(total_lines, ml)

    output_bytes = len(joined.encode("utf-8"))
    omitted_lines = total_lines - output_lines
    omitted_bytes = total_bytes - output_bytes

    notice = (
        f"\n\n[Output truncated: {output_lines} of {total_lines} lines "
        f"({_format_bytes(output_bytes)} of {_format_bytes(total_bytes)}). "
        f"{omitted_lines} lines ({_format_bytes(omitted_bytes)}) omitted.]"
    )

    return TruncationResult(
        content=joined + notice,
        truncated=True,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=output_bytes,
        omitted_lines=omitted_lines,
        omitted_bytes=omitted_bytes,
    )


def truncate_tail(
    text: str,
    max_lines: int | None = None,
    max_bytes: int | None = None,
    limits: TruncationLimits | None = None,
) -> TruncationResult:
    """Keep the last N lines/bytes. Best for logs, command output.

    Truncates at whichever limit is hit first.
    """
    lim = limits or DEFAULT_LIMITS
    ml = max_lines if max_lines is not None else lim.max_lines
    mb = max_bytes if max_bytes is not None else lim.max_bytes

    all_lines = text.split("\n")
    total_lines = len(all_lines)
    total_bytes = len(text.encode("utf-8"))

    if total_lines <= ml and total_bytes <= mb:
        return TruncationResult.not_truncated(text)

    # Truncate by lines (keep last N)
    truncated_lines = all_lines[-ml:] if total_lines > ml else all_lines
    joined = "\n".join(truncated_lines)

    # Then check bytes
    encoded = joined.encode("utf-8")
    if len(encoded) > mb:
        encoded = encoded[-mb:]
        # Find first complete UTF-8 after potential cut
        while encoded and (encoded[0] & 0xC0) == 0x80:
            encoded = encoded[1:]
        joined = encoded.decode("utf-8", errors="replace")
        output_lines = joined.count("\n") + 1
    else:
        output_lines = min(total_lines, ml)

    output_bytes = len(joined.encode("utf-8"))
    omitted_lines = total_lines - output_lines
    omitted_bytes = total_bytes - output_bytes

    notice = (
        f"[Output truncated: last {output_lines} of {total_lines} lines "
        f"({_format_bytes(output_bytes)} of {_format_bytes(total_bytes)}). "
        f"{omitted_lines} lines ({_format_bytes(omitted_bytes)}) omitted.]\n\n"
    )

    return TruncationResult(
        content=notice + joined,
        truncated=True,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=output_bytes,
        omitted_lines=omitted_lines,
        omitted_bytes=omitted_bytes,
    )