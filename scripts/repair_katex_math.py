#!/usr/bin/env python3
"""Repair a narrow class of copied Markdown/LaTeX corruption.

The script intentionally does not reflow prose or try to validate the
mathematics.  It repairs only patterns that are unambiguous in these notes:

* standalone ``[`` / ``]`` display delimiters -> ``$$``;
* setext-style runs of ``=`` inside such a display -> one mathematical ``=``;
* Markdown heading markers accidentally copied into a display -> removed;
* blank paragraph lines inside a display -> removed;
* simple math expressions written as ``(x_t)`` in prose -> ``$x_t$``.

Use ``--check`` for a dry run and ``--write`` to update the file.
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path


REPEATED_EQUALS = re.compile(r"^\s*={3,}\s*$")
MATH_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")
INLINE_PARENS = re.compile(r"\(([^()\n]{1,120})\)")


def is_inline_math(body: str) -> bool:
    """Return true only for conservative, equation-like parentheticals."""

    body = body.strip()
    if not body:
        return False

    # Single identifiers and numeric literals are common copied formulae:
    # (t), (R), (I), (N), and (1.19).
    if re.fullmatch(r"[A-Za-z](?:[A-Za-z0-9]*)?", body):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", body):
        return True

    # Operators, LaTeX commands, subscripts, and superscripts make the math
    # intent explicit.  This deliberately leaves prose such as (LP, greedy)
    # unchanged.
    return bool(re.search(r"\\|[_^=<>/]", body))


def repair_display_delimiters(text: str) -> str:
    """Repair standalone display delimiters without touching fenced code."""

    output: list[str] = []
    in_fence = False
    in_display = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            output.append(line)
            continue

        if in_fence:
            output.append(line)
            continue

        if stripped == "$$":
            in_display = not in_display
            output.append(line)
            continue

        if stripped == "[" and not in_display:
            output.append("$$\n")
            in_display = True
            continue

        if stripped == "]" and in_display:
            output.append("$$\n")
            in_display = False
            continue

        if in_display and not stripped:
            continue

        if in_display and REPEATED_EQUALS.fullmatch(line):
            output.append("=\n")
            continue

        if in_display:
            heading = MATH_HEADING.fullmatch(line.rstrip("\n"))
            if heading:
                output.append(heading.group(1) + ("\n" if line.endswith("\n") else ""))
                continue

        output.append(line)

    if in_display:
        raise ValueError("unterminated standalone '[' display block")

    return "".join(output)


def repair_inline_parentheses(text: str) -> str:
    """Convert conservative math parentheticals outside code/math blocks."""

    output: list[str] = []
    in_fence = False
    in_display = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            output.append(line)
            continue

        if in_fence:
            output.append(line)
            continue

        delimiter_count = line.count("$$")
        if delimiter_count % 2:
            in_display = not in_display
            output.append(line)
            continue
        if in_display or "$$" in line:
            output.append(line)
            continue

        def replace(match: re.Match[str]) -> str:
            body = match.group(1)
            return f"${body.strip()}$" if is_inline_math(body) else match.group(0)

        output.append(INLINE_PARENS.sub(replace, line))

    return "".join(output)


def repair(text: str) -> str:
    return repair_inline_parentheses(repair_display_delimiters(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="show the diff without writing")
    mode.add_argument("--write", action="store_true", help="write the repaired file")
    args = parser.parse_args()

    original = args.path.read_text(encoding="utf-8")
    repaired = repair(original)

    if repaired == original:
        print(f"No changes needed: {args.path}")
        return 0

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        repaired.splitlines(keepends=True),
        fromfile=str(args.path),
        tofile=str(args.path),
    )
    print("".join(diff), end="")

    if args.write:
        args.path.write_text(repaired, encoding="utf-8")
        print(f"Repaired: {args.path}")
    elif not args.check:
        print("Dry run only. Pass --write to update the file.")
    else:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
