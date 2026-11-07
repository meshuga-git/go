"""Unit tests for every shared format: each writer against its reader.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from unit_support import make_tree

import pipeline_formats as formats
import pipeline_io

# Identifiers may hold commas, quotes and any non-space character.
IDENTIFIERS = ["ENSG01", 'gene,"7"', "\u00e9t\u00e9", "x;y"]


def _write(path: Path, text: str) -> Path:
    """Write ``text`` as UTF-8 with LF line ends and return ``path``."""
    path.write_bytes(text.encode("utf-8"))
    return path


def test_identifier_lists_round_trip(tmp_path: Path) -> None:
    """render_identifiers writes what read_identifiers reads."""
    path = _write(tmp_path / "genes.txt",
                  pipeline_io.render_identifiers(IDENTIFIERS))
    assert pipeline_io.read_identifiers(path, label="gene") == IDENTIFIERS
    _write(path, pipeline_io.render_identifiers(["A", "A"]))
    with pytest.raises(ValueError, match="duplicate"):
        pipeline_io.read_identifiers(path, label="gene")


def test_per_sample_csv_round_trips_every_value(tmp_path: Path) -> None:
    """Values survive the CSV exactly, quoted names included."""
    values = [0.1, -2.5e-310, 1e300, 3.0, 123456789.123456789]
    path = _write(tmp_path / "s.csv",
                  formats.render_sample_expression(IDENTIFIERS + ["z"],
                                                   values))
    series = formats.read_sample_expression(path, "s")
    assert list(series.index) == IDENTIFIERS + ["z"]
    assert series.to_list() == values
    with pytest.raises(ValueError):
        formats.render_sample_expression(["a", "b"], [1.0])


def _small_tree(folder: Path) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    """Write a three-sample topology and return its lineages.

    The root 0 splits into 1 and 2, then 1 splits into 3 and 4.
    """
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "tree_splits.txt",
           "root 0\nsplit 1 0 1 2\nsplit 2 1 3 4\n")
    samples = ["a", "b", "c"]
    return samples, {"a": (0, 2), "b": (0, 1, 3), "c": (0, 1, 4)}


def test_lineages_round_trip(tmp_path: Path) -> None:
    """render_sample_lineages writes what read_sample_lineages reads."""
    samples, lineages = _small_tree(tmp_path)
    path = _write(tmp_path / "sample_lineages.txt",
                  formats.render_sample_lineages(samples, lineages))
    assert formats.read_sample_lineages(path, samples) == lineages
    assert formats.read_sample_lineages(path, samples,
                                        history=lineages) == lineages
    topology = formats.read_tree_topology(tmp_path / "tree_splits.txt", 3)
    assert topology.leaves == (2, 3, 4)
    for broken in ({"a": (2, 0)}, {"a": (0, 0, 2)}, {"a": (0, 3)}):
        _write(path, formats.render_sample_lineages(
            samples, dict(lineages, **broken)))
        with pytest.raises(ValueError):
            formats.read_sample_lineages(path, samples)


def test_matrix_round_trips_the_int32_range(tmp_path: Path) -> None:
    """The matrix keeps int32 extremes and refuses anything wider."""
    info = np.iinfo(np.int32)
    values = np.array([[info.min, -1, 0], [1, 7, info.max]], dtype=np.int64)
    path = tmp_path / "matrix_values.txt"
    formats.write_matrix_values(path, values)
    assert path.read_text(encoding="utf-8") == (
        f"{info.min} -1 0\n1 7 {info.max}\n")
    assert np.array_equal(formats.read_matrix_values(path, 2, 3), values)
    assert [index for index, _ in formats.iter_matrix_rows(path, 3)] == [
        0, 1]
    with pytest.raises(ValueError):
        formats.write_matrix_values(path, values + info.max)
    with pytest.raises(ValueError):
        formats.write_matrix_values(path, values.astype(np.float64))
    _write(path, f"0 {info.max + 1} 0\n")
    with pytest.raises(ValueError):
        formats.read_matrix_values(path, 1, 3)


def test_cluster_lists_round_trip(tmp_path: Path) -> None:
    """Exported lists read back; hand-written spellings are accepted."""
    path = _write(tmp_path / "cluster_1.txt",
                  formats.render_cluster_list([3, 17, 4]))
    assert path.read_text(encoding="utf-8") == "3, 17, 4\n"
    assert formats.read_cluster_list(path, max_node_id=20) == (3, 17, 4)
    _write(path, "017; +4\n3")
    assert formats.read_cluster_list(path, max_node_id=20) == (17, 4, 3)
    _write(path, "")
    assert formats.read_cluster_list(path, max_node_id=20) == ()
    for bad in ("21", "3, 3", "-1", "x"):
        _write(path, bad)
        with pytest.raises(ValueError):
            formats.read_cluster_list(path, max_node_id=20)


def _report() -> formats.SignatureReport:
    """Return a report with every part the grammar has."""
    terms = (formats.SignatureTerm("G:1,x", ">", 5),
             formats.SignatureTerm("G2", "<", -7))
    scaling = tuple(
        formats.ScaledTerm(
            scale, term.gene, term.operator,
            term.threshold / scale if term.operator == ">"
            else float(term.threshold * scale), 12.5)
        for term in terms for scale in formats.SCALE_FACTORS)
    return formats.SignatureReport(
        (formats.Signature(1, "max_iterations", terms,
                           (formats.ClusterCoverage("3", 100.0),
                            formats.ClusterCoverage("0", 0.0)),
                           (formats.ClusterCoverage("7", 45.5),), scaling),
         formats.Signature(2, "convergence",
                           (formats.SignatureTerm("G3", "<", 0),), (), ())),
        formats.RunSummary("no_terms_removed", 3, 40, 1234, 5, 17),
    )


@pytest.mark.parametrize("report", [
    _report(),
    formats.SignatureReport((), formats.RunSummary(
        "no_active_terms_remaining", 0, 0, 0, 22, 0)),
])
def test_report_round_trip(report: formats.SignatureReport,
                           tmp_path: Path) -> None:
    """The report parser reads back exactly what the writer wrote."""
    text = formats.render_signature_report(report)
    path = _write(tmp_path / "tjala_output.txt", text)
    parsed = formats.read_signature_report(path)
    assert parsed == report
    assert formats.render_signature_report(parsed) == text


def test_report_writer_refuses_what_its_reader_rejects() -> None:
    """The writer checks its own text against the grammar."""
    report = _report()
    bad = report.signatures[1]
    for signature in (
            formats.Signature(2, "converged", bad.terms, (), ()),
            formats.Signature(2, "convergence", bad.terms,
                              (formats.ClusterCoverage("1", 150.0),), ()),
            formats.Signature(3, "convergence", bad.terms, (), ()),
    ):
        with pytest.raises(ValueError):
            formats.render_signature_report(formats.SignatureReport(
                (report.signatures[0], signature), report.summary))


@pytest.mark.parametrize("samples", [2, 3, 64])
def test_history_round_trips_through_its_reader(
        samples: int, tmp_path: Path) -> None:
    """render_tree_states writes a history read_tree_states accepts."""
    results = make_tree(tmp_path, samples, seed=samples)
    paths, states = formats.read_snapshots(results)
    assert paths == formats.read_sample_paths(results / "filenames.txt")
    path = _write(tmp_path / "total_data.csv",
                  formats.render_tree_states(paths, states))
    data = formats.read_tree_states(path)
    order = {path: row for row, path in enumerate(paths)}
    rows = [order[raw] for raw in data["0"]]
    assert np.array_equal(data.iloc[:, 1:].to_numpy(), states[rows])
    topology = formats.read_tree_topology(results / "tree_splits.txt",
                                          samples)
    formats.validate_tree_states(data, topology)


def test_topology_and_snapshot_checks(tmp_path: Path) -> None:
    """Splits follow the immutable ID rules; snapshots follow splits."""
    results = make_tree(tmp_path, 5, seed=1)
    splits = (results / "tree_splits.txt").read_text(encoding="utf-8")
    formats.read_tree_topology(results / "tree_splits.txt", 5)
    for old, new in (("split 1 0 1 2", "split 1 0 2 1"),
                     ("split 1 0 1 2", "split 1 9 1 2"),
                     ("root 0", "root 1")):
        _write(tmp_path / "bad.txt", splits.replace(old, new))
        with pytest.raises(ValueError):
            formats.read_tree_topology(tmp_path / "bad.txt", 5)
    # Split 1 divides the root: sending every sample to child 1
    # leaves child 2 empty.
    _write(results / "iter1.txt", " ".join(["1"] * 5) + "\n")
    with pytest.raises(ValueError, match="empty"):
        formats.read_snapshots(results)


def test_term_variances_reader(tmp_path: Path) -> None:
    """Records are checked; numbers convert as Tjala reads them."""
    path = _write(tmp_path / "term_variances.txt",
                  "G1 5 1 0.25\nG1 5 2 125e-3\nG2 -3 0 0\n")
    table = formats.read_term_variances(path, ["G1", "G2"])
    assert list(table.columns) == ["gene", "value", "operator", "variance"]
    assert table["value"].to_list() == [5, 5, -3]
    assert table["operator"].to_list() == [1, 2, 0]
    assert table["variance"].to_list() == [0.25, 0.125, 0.0]
    for bad in ("G1 5 1 0.25\nG1 5 1 0.5\n", "G9 5 1 0.25\n",
                "G1 5 3 0.25\n", "G1 5 1 -1\n", "G1 5.0 1 0.25\n",
                "G1 5 1\n"):
        _write(path, bad)
        with pytest.raises(ValueError):
            formats.read_term_variances(path, ["G1", "G2"])


def test_sample_paths_name_unique_samples(tmp_path: Path) -> None:
    """A sample path's last component is a unique sample ID."""
    path = _write(tmp_path / "filenames.txt",
                  "data/a/s1.csv\nC:\\data\\s2.csv\n")
    assert formats.read_sample_paths(path) == ["data/a/s1.csv",
                                               "C:\\data\\s2.csv"]
    _write(path, "a/s1.csv\nb/s1.csv\n")
    with pytest.raises(ValueError, match="duplicate"):
        formats.read_sample_paths(path)
