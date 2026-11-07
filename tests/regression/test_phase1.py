"""Regression tests for the Phase 1 fixes, one or more per plan task.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.
"""

from __future__ import annotations

import re
from pathlib import Path

from conftest import golden, run_script

REPO = golden.REPO


def test_1_1_quitting_at_the_first_prompt_keeps_every_file(
        workflow: Path) -> None:
    """Task 1.1: no output is deleted before the user has answered."""
    before = golden.snapshot(workflow)
    done = run_script(workflow, "clustering.py", stdin="")
    assert done.returncode != 0
    assert golden.snapshot(workflow) == before


def _stage_names(pattern: str, text: str) -> set[str]:
    """Return the quoted names inside the first match of ``pattern``."""
    match = re.search(pattern, text, re.S)
    assert match is not None, pattern
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def test_1_6_variance_engine_knows_every_stage_name() -> None:
    """Task 1.6: the C++ engine accepts the stages Python writes.

    A stage the engine does not know makes it ignore that manifest when
    it invalidates dependents, as it did with Tjala's "signatures".
    """
    python = (REPO / "compute_variances.py").read_text(encoding="utf-8")
    cpp = (REPO / "compute_variances.cpp").read_text(encoding="utf-8")
    assert (_stage_names(r"_STAGE_NAMES = \{(.*?)\}", python)
            == _stage_names(r"std::set<std::string> STAGES\{(.*?)\};", cpp))
