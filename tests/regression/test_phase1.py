"""Regression tests for the Phase 1 fixes, one or more per plan task.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from conftest import build_workflow, golden, run_script

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


QUICK_TJALA = "3\n2\n10\n10\n10\n3\n12\ncluster_1.txt\n0.05\n"


def test_1_2_identical_reexport_keeps_downstream_outputs(
        workflow: Path) -> None:
    """Task 1.2: an identical cluster re-export deletes nothing."""
    results = workflow / "clustering_results"
    for script, stdin in (("samples_by_cluster.py", "1\n1\nN\nExit\n"),
                          ("tjala.py", QUICK_TJALA),
                          ("samples_by_signature.py", "1\nExit\n")):
        done = run_script(results, script, stdin=stdin)
        assert done.returncode == 0, (script, done.stderr[-2000:])
    before = golden.snapshot(workflow)
    assert "clustering_results/tjala_output.txt" in before
    assert "clustering_results/signature_1.txt" in before
    done = run_script(results, "samples_by_cluster.py",
                      stdin="1\n1\nN\nExit\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert golden.snapshot(workflow) == before


def test_1_5_unchanged_files_are_hashed_once(
        tmp_path: Path, monkeypatch) -> None:
    """Task 1.5: the hash memo reads an unchanged file only once."""
    import hashlib
    import os

    shared = _load("compute_variances")
    real = hashlib.sha256
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(shared.hashlib, "sha256", counting)
    path = tmp_path / "data.txt"
    path.write_text("one\n", encoding="utf-8")
    first = shared.sha256_file(path)
    assert shared.sha256_file(path) == first
    assert len(calls) == 1
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("two\n", encoding="utf-8")
    os.replace(replacement, path)
    assert shared.sha256_file(path) != first
    assert len(calls) == 2


def test_1_11_json_identifiers_stay_text(tmp_path: Path) -> None:
    """Task 1.11: JSON input keeps identifiers such as "001" as text."""
    clustering = _load("clustering")
    source = tmp_path / "matrix.json"
    source.write_text(
        '[{"gene": "001", "S1": 5.25}, {"gene": "010", "S1": 0.1}]',
        encoding="utf-8")
    table = clustering._read_json(str(source))
    assert table["gene"].tolist() == ["001", "010"]
    assert table["S1"].tolist() == [5.25, 0.1]


def _assert_duplicate_rejected(clustering, source: Path) -> None:
    """Check that loading ``source`` names the file and the header."""
    import pytest

    reader = clustering.FILE_READERS[source.suffix]
    with pytest.raises(RuntimeError) as error:
        clustering._load_dataframe(str(source), reader)
    assert "duplicate sample header 'S1'" in str(error.value)
    assert source.name in str(error.value)


def test_1_12_duplicate_csv_and_json_headers_are_rejected(
        tmp_path: Path) -> None:
    """Task 1.12: CSV and JSON input refuse a repeated sample name."""
    clustering = _load("clustering")
    csv_source = tmp_path / "matrix.csv"
    csv_source.write_text("gene,S1,S1\nG1,1,2\n", encoding="utf-8")
    _assert_duplicate_rejected(clustering, csv_source)
    json_source = tmp_path / "matrix.json"
    json_source.write_text('[{"gene": "G1", "S1": 1, "S1": 2}]',
                           encoding="utf-8")
    _assert_duplicate_rejected(clustering, json_source)


def test_1_12_duplicate_excel_header_is_rejected(tmp_path: Path) -> None:
    """Task 1.12: Excel input refuses a repeated sample name."""
    import pytest

    openpyxl = pytest.importorskip("openpyxl")
    clustering = _load("clustering")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["gene", "S1", "S1"])
    sheet.append(["G1", 1, 2])
    source = tmp_path / "matrix.xlsx"
    workbook.save(source)
    _assert_duplicate_rejected(clustering, source)


def test_1_9_par_y_genes_stay_distinct(tmp_path: Path) -> None:
    """Task 1.9: stripping versions keeps _PAR_Y IDs apart."""
    clustering = _load("clustering")
    source = tmp_path / "matrix.csv"
    source.write_text(
        "gene,S1\nENSG00000182378.14,5\nENSG00000182378.14_PAR_Y,0\n"
        "ENSG00000000003.15,7\n", encoding="utf-8")
    table = clustering._load_dataframe(str(source),
                                       clustering.FILE_READERS[".csv"])
    clustering._normalize_expression_columns(table, str(source))
    assert table["gene"].tolist() == [
        "ENSG00000182378", "ENSG00000182378_PAR_Y", "ENSG00000000003"]


def _tjala_answers(cannot_file: str) -> str:
    """Return quick Tjala answers that use ``cannot_file``."""
    return f"3\n2\n10\n10\n10\n3\n12\n{cannot_file}\n0.05\n"


def test_1_10_cannot_cluster_list_checks(workflow: Path) -> None:
    """Task 1.10: lenient IDs, a tree check and a missing-manifest note.

    017 and +17 name node 17; a list whose manifest names another tree
    is refused; a list without a manifest is used with a warning that
    names each cluster and its size.
    """
    results = workflow / "clustering_results"
    (results / "typed.txt").write_text("+1\n", encoding="utf-8")
    done = run_script(results, "tjala.py", stdin=_tjala_answers("typed.txt"))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "typed.txt has no provenance record" in done.stderr
    assert "cluster(s) 1 (12 samples)" in done.stderr
    (results / "padded.txt").write_text("01\n", encoding="utf-8")
    done = run_script(results, "tjala.py",
                      stdin=_tjala_answers("padded.txt"))
    assert done.returncode == 0, done.stderr[-2000:]

    done = run_script(results, "samples_by_cluster.py",
                      stdin="1\n1\nN\nExit\n")
    assert done.returncode == 0, done.stderr[-2000:]
    other = build_workflow(workflow.parent / "other", metric="1")
    other_results = other / "clustering_results"
    for name in ("cluster_1.txt", "cluster_1.txt.provenance"):
        shutil.copyfile(results / name, other_results / name)
    done = run_script(other_results, "tjala.py",
                      stdin=_tjala_answers("cluster_1.txt"))
    assert done.returncode != 0
    assert "exported from a different tree" in done.stderr
