"""Shared fixtures for the regression tests.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.

The fixtures reuse the golden harness: they build the 22-sample
workflow with the toolchain of this machine, or the one named by the
``TAVNIT_TOOLCHAIN`` environment variable.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("golden",
                                               TESTS / "golden.py")
golden = importlib.util.module_from_spec(_spec)
sys.modules["golden"] = golden
_spec.loader.exec_module(golden)

DEFAULT_TOOLCHAINS = {"linux": "linux-gcc", "darwin": "macos-clang",
                      "win32": "windows-mingw"}


def toolchain() -> str:
    """Return the golden-harness toolchain to build with.

    Returns:
        A key of ``golden.TOOLCHAINS``.
    """
    return os.environ.get("TAVNIT_TOOLCHAIN",
                          DEFAULT_TOOLCHAINS.get(sys.platform, "linux-gcc"))


def run_script(cwd: Path, script: str, *args: str,
               stdin: str = "") -> subprocess.CompletedProcess:
    """Run one pipeline script as a user would.

    Args:
        cwd: Folder to run in.
        script: Script file name, relative to ``cwd``.
        *args: Command-line arguments.
        stdin: Answers to the prompts.

    Returns:
        The finished process, with text output.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1",
               CXX=golden.TOOLCHAINS[toolchain()][0])
    return subprocess.run([sys.executable, script, *args], cwd=cwd,
                          input=stdin, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env,
                          timeout=1800)


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    """Build Part 1 and the variance stage on the 22-sample set.

    Each test gets a fresh build in its own folder: provenance records
    name absolute paths, so a copied folder would no longer link its
    outputs to their parents.

    Args:
        tmp_path: Pytest's per-test temporary folder.

    Returns:
        The finished scenario folder.
    """
    base = golden.build_scenarios()["pipeline22"]
    scenario = golden.Scenario(
        "workflow22",
        golden.part1("2", golden.CANNOT, "None", "None")
        + golden.part2()[:2],
        cohorts=base.cohorts,
    )
    outcome = golden.Runner(toolchain(), tmp_path, measure=False).run(
        scenario)
    assert all(code == 0 for code in outcome["stages"].values()), outcome
    return tmp_path / scenario.name
