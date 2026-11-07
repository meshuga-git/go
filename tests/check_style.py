"""Report style and compatibility findings for the package.

Requires Python 3.10 or newer, with requirements-dev.txt installed.

Runs four checks and prints a summary of each:

* ruff: PEP 8 with 79-character code lines and 72-character docstring
  and comment lines, Google-style docstrings, likely bugs;
* pydoclint: docstrings that disagree with the signatures;
* vermin: the minimum Python version each module needs, which must not
  exceed the 3.10 ceiling (annotations are evaluated, so ``X | Y``
  counts as 3.10);
* clang-format 19.1.7: C++ formatting against ``.clang-format``.

The checks only report until the plan makes them binding (task 0.6);
``--strict`` turns any finding into exit code 1.

Usage:
    python tests/check_style.py [--strict]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CEILING = (3, 10)
# vermin and pydoclint have no ``python -m`` entry point; call their
# console-script functions through the current interpreter instead.
VERMIN = "import sys, vermin; sys.argv[0] = 'vermin'; sys.exit(vermin.main())"
PYDOCLINT = "import sys; from pydoclint.main import main; sys.exit(main())"


def tracked(*suffixes: str) -> list[str]:
    """List tracked files with the given suffixes.

    Args:
        *suffixes: File suffixes, such as ``.py``.

    Returns:
        Repository-relative paths.
    """
    done = subprocess.run(["git", "ls-files"], cwd=REPO, check=True,
                          capture_output=True, text=True)
    return [line for line in done.stdout.splitlines()
            if line.endswith(suffixes)]


def run(command: list[str]) -> tuple[int, str]:
    """Run a check from the repository root.

    Args:
        command: Command line.

    Returns:
        Exit code and combined output.
    """
    try:
        done = subprocess.run(command, cwd=REPO, capture_output=True,
                              text=True, errors="replace")
    except OSError as exc:
        return 127, f"cannot run {command[0]}: {exc}"
    return done.returncode, done.stdout + done.stderr


def ruff(python_files: list[str]) -> int:
    """Report ruff findings by rule.

    Args:
        python_files: Files to check.

    Returns:
        Number of findings.
    """
    _, out = run([sys.executable, "-m", "ruff", "check",
                  "--output-format=concise", "--no-cache", *python_files])
    codes: dict[str, int] = {}
    for line in out.splitlines():
        match = re.search(r":\d+:\d+: ([A-Z]+\d+)", line)
        if match:
            codes[match.group(1)] = codes.get(match.group(1), 0) + 1
    total = sum(codes.values())
    summary = ", ".join(f"{code} {n}" for code, n in sorted(codes.items()))
    print(f"ruff: {total} finding(s){': ' + summary if summary else ''}")
    return total


def pydoclint(python_files: list[str]) -> int:
    """Report pydoclint findings by violation code.

    Args:
        python_files: Files to check.

    Returns:
        Number of findings.
    """
    _, out = run([sys.executable, "-c", PYDOCLINT, "--quiet",
                  *python_files])
    codes: dict[str, int] = {}
    for line in out.splitlines():
        match = re.match(r"\s+\d+: (DOC\d+)", line)
        if match:
            codes[match.group(1)] = codes.get(match.group(1), 0) + 1
    total = sum(codes.values())
    summary = ", ".join(f"{code} {n}" for code, n in sorted(codes.items()))
    print(f"pydoclint: {total} finding(s){': ' + summary if summary else ''}")
    return total


def vermin(python_files: list[str]) -> int:
    """Report each module's minimum Python version.

    Args:
        python_files: Files to check.

    Returns:
        Number of modules that need a version above the ceiling.
    """
    over = 0
    print("vermin: minimum Python version per module "
          f"(ceiling {CEILING[0]}.{CEILING[1]})")
    for name in python_files:
        _, out = run([sys.executable, "-c", VERMIN, "--no-tips",
                      "--eval-annotations", "--no-parse-comments", name])
        match = re.search(r"Minimum required versions: (\S+)", out)
        version = match.group(1) if match else "unknown"
        parts = tuple(int(p) for p in re.findall(r"\d+", version)[:2])
        flag = ""
        if len(parts) == 2 and parts > CEILING:
            flag = "  <- above the ceiling"
            over += 1
        print(f"  {name:<34} {version}{flag}")
    return over


def clang_format(cpp_files: list[str]) -> int:
    """Report C++ files that differ from their formatted form.

    Args:
        cpp_files: Files to check.

    Returns:
        Number of formatting findings.
    """
    _, out = run(["clang-format", "--dry-run", "--style=file", *cpp_files])
    counts: dict[str, int] = {}
    for line in out.splitlines():
        match = re.match(r"(.+?):\d+:\d+: warning:", line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    total = sum(counts.values())
    detail = ", ".join(f"{Path(f).name} {n}"
                       for f, n in sorted(counts.items()))
    print(f"clang-format: {total} finding(s){': ' + detail if detail else ''}")
    return total


def main() -> int:
    """Run every check and report.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="exit with 1 when any check reports findings")
    args = parser.parse_args()
    python_files = tracked(".py")
    findings = (ruff(python_files) + pydoclint(python_files)
                + vermin(python_files)
                + clang_format(tracked(".cpp", ".hpp")))
    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
