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


def _load(name: str):
    """Import one of the package's scripts as a module."""
    import importlib.util
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(name, REPO / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def test_1_3_a_million_active_terms_pass_the_pheromone_check() -> None:
    """Task 1.3: the sum check holds for 10^6 terms on Python 3.10."""
    tjala = _load("tjala")
    count = 1_000_000
    pheromones = dict.fromkeys(range(count), 1.0 / count)
    graph = {"pheromones": pheromones, "active_term_ids": set(pheromones)}
    best = tjala.Antecedent(term_ids=[0, 1, 2], quality=0.75)
    tjala.update_pheromones(graph, best)
    assert abs(sum(pheromones.values()) - 1.0) < 1e-9


def test_1_4_cleanup_removes_a_symlinked_output(
        workflow: Path, tmp_path: Path) -> None:
    """Task 1.4: cleanup removes a link to an output, not the target."""
    import shutil

    import pytest

    page = workflow / "dendrogram.html"
    target = tmp_path / "kept.html"
    shutil.copyfile(page, target)
    content = target.read_bytes()
    page.unlink()
    try:
        page.symlink_to(target)
    except OSError:
        pytest.skip("this system does not allow symbolic links")
    answers = f"2\n{golden.CANNOT}\nNone\nNone\n"
    done = run_script(workflow, "clustering.py", stdin=answers)
    assert done.returncode == 0, done.stderr
    assert not page.is_symlink() and not page.exists()
    assert target.read_bytes() == content
