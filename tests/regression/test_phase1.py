"""Regression tests for the Phase 1 fixes, one or more per plan task.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.
"""

from __future__ import annotations

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
