"""Readers and writers of the pipeline's data formats.

Requires Python 3.10 or newer, with NumPy and pandas.

Each format has one reader and one writer here, with the content
checks its producer guarantees: the per-sample CSVs, the tree topology,
the snapshots (iterK.txt), the compressed history (total_data.csv),
the sample lineages, the expression matrix, Tjala's signature report
and the cannot-cluster list. The Python writer of the history produces
the bytes reconstruct_hierarchy writes today.
"""

from __future__ import annotations

import csv
import io
import ntpath
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_io import (
    SAMPLE_PATHS_FILE,
    TREE_TOPOLOGY_FILE,
    iter_utf8_lines,
    parse_canonical_int,
    parse_decimal_float,
    read_csv_rows,
    read_utf8_lines,
    snapshot_file,
    validate_identifier,
)

INT32_INFO = np.iinfo(np.int32)

#: Header row of every per-sample CSV.
SAMPLE_CSV_HEADER = ("gene", "expression")


SampleLineages = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class TreeSplit:
    """One immutable binary-tree split.

    Attributes:
        iteration: One-based global split number.
        parent: Retired immutable parent ID.
        retained_child: Fresh retained-child ID.
        extracted_child: Fresh extracted-child ID.
    """

    iteration: int
    parent: int
    retained_child: int
    extracted_child: int


@dataclass(frozen=True)
class TreeTopology:
    """A validated immutable topology shared by pipeline consumers.

    Attributes:
        root: Permanent root ID, always zero.
        splits: Splits in global iteration order.
        children: Parent IDs mapped to retained/extracted children.
        parent_by_node: Child IDs mapped to their immediate parent.
        creation_split: Split creating each node; root maps to zero.
        leaves: Terminal immutable IDs in numeric order.
        node_count: Total immutable node count.
    """

    root: int
    splits: tuple[TreeSplit, ...]
    children: dict[int, tuple[int, int]]
    parent_by_node: dict[int, int]
    creation_split: dict[int, int]
    leaves: tuple[int, ...]
    node_count: int


def read_tree_topology(path: Path, sample_count: int) -> TreeTopology:
    """Parse canonical topology records and validate every split.

    Args:
        path: Authoritative ``tree_splits.txt`` file.
        sample_count: Number of samples and terminal leaves.

    Returns:
        Complete topology with one root and two fresh children per
        split.

    Raises:
        ValueError: If syntax, allocation, or active-node invariants
            fail.
        OSError: If reading fails.
    """
    node_count = tree_node_count(sample_count)
    lines = read_utf8_lines(path)
    if len(lines) != sample_count or lines[0].split() != ["root", "0"]:
        raise ValueError(f"{path}: expected root 0 and N-1 split records")
    active = {0}
    children: dict[int, tuple[int, int]] = {}
    parents: dict[int, int] = {}
    creation = {0: 0}
    splits: list[TreeSplit] = []
    for iteration, line in enumerate(lines[1:], 1):
        fields = line.split()
        if len(fields) != 5 or fields[0] != "split":
            raise ValueError(f"{path}:{iteration + 1}: invalid split record")
        number, parent, left, right = [
            parse_canonical_int(token, source=f"{path}:{iteration + 1}")
            for token in fields[1:]
        ]
        if number != iteration or (left, right) != (
            2 * iteration - 1, 2 * iteration
        ) or parent not in active:
            raise ValueError(
                f"{path}:{iteration + 1}: invalid immutable split")
        active.remove(parent)
        active.update((left, right))
        children[parent] = (left, right)
        parents[left] = parents[right] = parent
        creation[left] = creation[right] = iteration
        splits.append(TreeSplit(iteration, parent, left, right))
    return TreeTopology(
        0, tuple(splits), children, parents, creation,
        tuple(sorted(active)), node_count
    )


def read_tree_states(path: Path) -> pd.DataFrame:
    """Decode strict, compressed immutable hierarchy history.

    Args:
        path: Headerless ``total_data.csv`` file.

    Returns:
        Paths followed by N integer state columns, with string labels.

    Raises:
        ValueError: If dimensions, paths, blanks, or node fields are
            invalid.
        OSError: If reading fails.
    """
    rows = read_csv_rows(path)
    count = len(rows)
    node_count = tree_node_count(count)
    states = np.empty((count, count), dtype=np.int64)
    paths: list[str] = []
    ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if len(row) != count + 1 or not row[0]:
            raise ValueError((
                f'{path}:{row_index + 1}: expected path and N '
                f'states'
            ))
        sid = sample_id(row[0])
        validate_identifier(sid, source=f"{path}:{row_index + 1}")
        if sid in ids:
            raise ValueError((
                f'{path}:{row_index + 1}: duplicate sample ID '
                f'{sid!r}'
            ))
        ids.add(sid)
        paths.append(row[0])
        for column, token in enumerate(row[1:]):
            if token == "" and row_index:
                value = int(states[row_index - 1, column])
            else:
                value = parse_canonical_int(
                    token, source=f"{path}:{row_index + 1}:{column + 2}"
                )
            if not 0 <= value < node_count:
                raise ValueError(
                    f"{path}:{row_index + 1}:{column + 2}: bad node ID")
            states[row_index, column] = value
    if not np.all(states[:, 0] == 0):
        raise ValueError(f"{path}: state 0 must contain root 0")
    if set(np.unique(states)) != set(range(node_count)):
        raise ValueError(f"{path}: history omits immutable nodes")
    if len(set(states[:, -1])) != count:
        raise ValueError(f"{path}: terminal sample leaves must be distinct")
    data = pd.DataFrame(states, columns=[str(i) for i in range(1, count + 1)])
    data.insert(0, "0", paths)
    return data


def validate_tree_states(data: pd.DataFrame, topology: TreeTopology) -> None:
    """Cross-check every sample transition against authoritative.

    Cross-check every sample transition against the authoritative tree.

    Args:
        data: Decoded path-plus-state table in sample order.
        topology: Tree topology for the same samples.

    Raises:
        ValueError: If dimensions or any split transition disagree.
    """
    states = data.iloc[:, 1:].to_numpy(dtype=np.int64)
    count = len(data)
    if states.shape != (count, count) or topology.node_count != 2 * count - 1:
        raise ValueError("History and topology dimensions disagree")
    if not np.all(states[:, 0] == 0):
        raise ValueError("History state 0 must contain root 0")
    for split in topology.splits:
        previous = states[:, split.iteration - 1]
        current = states[:, split.iteration]
        mask = previous == split.parent
        if not mask.any() or np.any(current[~mask] != previous[~mask]):
            raise ValueError((
                f'History changes wrong samples at split '
                f'{split.iteration}'
            ))
        if set(current[mask]) != {split.retained_child, split.extracted_child}:
            raise ValueError((
                f'History has wrong children at split '
                f'{split.iteration}'
            ))
    if set(states[:, -1]) != set(topology.leaves):
        raise ValueError("History terminal nodes disagree with topology")


def sample_id(raw_path: str) -> str:
    """Derive a sample identifier from a manifest path.

    Manifest paths may use Windows separators even when the script runs
    on POSIX, so ``ntpath.basename`` is used unconditionally.

    Args:
        raw_path: Path string stored in the manifest.

    Returns:
        The final path component, interpreted with Windows path
        semantics.
    """
    return ntpath.basename(raw_path)


def tree_node_count(sample_count: int) -> int:
    """Return the number of nodes in a full binary tree with N leaves.

    Args:
        sample_count: Number of samples/leaves in the clustering tree.

    Returns:
        ``2 * sample_count - 1`` for a non-empty tree.

    Raises:
        ValueError: If ``sample_count`` is not positive or the largest
            node ID cannot be represented by the signed 32-bit
            downstream contract.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    node_count = 2 * sample_count - 1
    max_node_id = node_count - 1
    if max_node_id > int(INT32_INFO.max):
        raise ValueError(
            (
                f'A fully immutable tree for {sample_count} '
                f'samples requires node ID {max_node_id}, '
                f'exceeding the signed 32-bit hierarchy contract'
            )
        )
    return node_count


def load_tree_states(
    tree_states_path: Path,
) -> tuple[SampleLineages, list[str]]:
    """Load and validate sample memberships in the immutable tree.

    ``reconstruct_hierarchy.cpp`` writes one path and N compressed state
    columns for N samples. Only empty fields repeat the preceding value.
    Validate every transition against the accompanying tree topology.

    Args:
        tree_states_path: Headerless ``total_data.csv`` history path.

    Returns:
        Sample-ID-to-lineage mapping and paths in history row order.

    Raises:
        ValueError: If history, identities, or topology are
            inconsistent.
        OSError: If either input cannot be read.
    """
    data = read_tree_states(tree_states_path)
    topology = read_tree_topology(
        tree_states_path.with_name(TREE_TOPOLOGY_FILE), len(data)
    )
    validate_tree_states(data, topology)
    hierarchy: SampleLineages = {}
    raw_paths: list[str] = []
    for row in data.itertuples(index=False, name=None):
        raw = str(row[0])
        hierarchy[sample_id(raw)] = (
            tuple(dict.fromkeys(int(x) for x in row[1:]))
        )
        raw_paths.append(raw)
    return hierarchy, raw_paths


def read_sample_expression(resolved: Path, sid: str) -> pd.Series:
    """Read one per-sample CSV using the strict pipeline dialect.

    Args:
        resolved: Existing CSV path.
        sid: Sample ID assigned at ingestion and validated globally.

    Returns:
        Binary64 values indexed by literal, unique gene identifiers.

    Raises:
        ValueError: If headers, fields, names, or expressions are
            invalid.
        OSError: If the file cannot be read.
    """
    rows = read_csv_rows(resolved)
    if rows[0] != list(SAMPLE_CSV_HEADER) or len(rows) < 2:
        raise ValueError(f"{resolved}: expected gene,expression and data rows")
    genes: list[str] = []
    values: list[float] = []
    seen: set[str] = set()
    for number, row in enumerate(rows[1:], 2):
        if len(row) != 2:
            raise ValueError(
                f"{resolved}:{number}: expected exactly two fields")
        gene, token = row
        validate_identifier(gene, source=f"{resolved}:{number}:1")
        if gene in seen:
            raise ValueError(f"{resolved}:{number}: duplicate gene {gene!r}")
        seen.add(gene)
        genes.append(gene)
        values.append(parse_decimal_float(token,
            source=f"{resolved}:{number}:2"))
    return pd.Series(values, index=genes, name=sid, dtype=np.float64)


def render_sample_expression(
    genes: Iterable[str], values: Iterable[float]
) -> str:
    """Return the text of one per-sample CSV.

    The file holds the ``gene,expression`` header, then one row per
    gene in the order given, its value written with 17 significant
    digits so that it reads back exactly. Fields are quoted only when
    they need it.

    Args:
        genes: Gene identifiers, validated at ingestion.
        values: The sample's value for each gene.

    Returns:
        The CSV text, LF-terminated.

    Raises:
        ValueError: If ``genes`` and ``values`` differ in length.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SAMPLE_CSV_HEADER)
    for gene, value in zip(genes, values, strict=True):
        writer.writerow([gene, format(value, ".17g")])
    return buffer.getvalue()


# A snapshot: one line of canonical node IDs, single spaces between.
_SNAPSHOT_RE = re.compile(r"(?:0|[1-9][0-9]*)(?: (?:0|[1-9][0-9]*))*\Z")


def read_snapshots(directory: Path) -> tuple[list[str], np.ndarray]:
    """Read every snapshot of a clustering run and check it.

    Snapshot k is ``iter<k>.txt``: one line giving each sample's node
    after split k. Each must agree with the topology: only the members
    of the split's parent move, and they move to its two children.

    Args:
        directory: The ``clustering_results`` folder.

    Returns:
        The sample paths, and an N-by-N array of node IDs whose column
        k holds the state after split k (column 0 is the root).

    Raises:
        ValueError: If a snapshot is malformed or disagrees with the
            topology.
        OSError: If a file cannot be read.
    """
    paths = read_utf8_lines(directory / SAMPLE_PATHS_FILE)
    count = len(paths)
    topology = read_tree_topology(directory / TREE_TOPOLOGY_FILE, count)
    states = np.zeros((count, count), dtype=np.int32)
    for split in topology.splits:
        name = snapshot_file(split.iteration)
        text = (directory / name).read_text(encoding="utf-8")
        line = text[:-1] if text.endswith("\n") else text
        if "\n" in line or not _SNAPSHOT_RE.match(line):
            raise ValueError(f"{name}: not one line of canonical IDs")
        current = np.array(line.split(" "), dtype=np.int64)
        if current.shape != (count,):
            raise ValueError(
                f"{name}: {current.size} node IDs for {count} samples")
        previous = states[:, split.iteration - 1]
        moved = previous == split.parent
        children = (split.retained_child, split.extracted_child)
        stray = np.flatnonzero(~moved & (current != previous))
        if stray.size:
            k = int(stray[0])
            raise ValueError(
                f"{name}: sample {k + 1} moved from node {previous[k]} to "
                f"{current[k]}; only members of split parent "
                f"{split.parent} may move")
        lost = np.flatnonzero(moved & ~np.isin(current, children))
        if lost.size:
            k = int(lost[0])
            raise ValueError(
                f"{name}: sample {k + 1} left split parent {split.parent} "
                f"for node {current[k]}, not for child {children[0]} or "
                f"{children[1]}")
        for child in children:
            if not np.any(current[moved] == child):
                raise ValueError(
                    f"{name}: split {split.iteration} leaves child {child} "
                    "empty")
        states[:, split.iteration] = current
    return paths, states


def _csv_field(field: str) -> str:
    """Quote one CSV field the way reconstruct_hierarchy does."""
    if not any(char in field for char in ',"\n\r'):
        return field
    return '"' + field.replace('"', '""') + '"'


def render_tree_states(paths: list[str], states: np.ndarray) -> str:
    """Return the compressed history that reconstruct_hierarchy writes.

    Rows are sorted by their node IDs. A row leaves empty the fields it
    shares with the row above: adjacent samples agree on a prefix of
    splits and, once separated, sit in disjoint subtrees.

    Args:
        paths: Sample paths, in snapshot order.
        states: N-by-N node IDs from ``read_snapshots``.

    Returns:
        The text of ``total_data.csv``.
    """
    count = len(paths)
    order = np.lexsort(states.T[::-1])
    ordered = states[order]
    differs = ordered[1:] != ordered[:-1]
    first_diff = np.where(differs.any(axis=1), differs.argmax(axis=1),
                          count)
    lines = [_csv_field(paths[order[0]]) + ","
             + ",".join(map(str, ordered[0].tolist()))]
    for row in range(1, count):
        k = int(first_diff[row - 1])
        lines.append(_csv_field(paths[order[row]]) + "," * (k + 1)
                     + ",".join(map(str, ordered[row, k:].tolist())))
    return "\n".join(lines) + "\n"


def read_sample_lineages(
    path: Path, samples: list[str]
) -> dict[str, tuple[int, ...]]:
    """Read ``sample_lineages.txt`` with the rules its producer keeps.

    The variance preparation writes one row per sample: the sample ID,
    the number of nodes on its root-to-leaf path, and those node IDs in
    canonical decimal, root 0 first and strictly increasing. Together
    the rows cover every node of the tree and end at distinct leaves.

    Args:
        path: ``sample_lineages.txt`` path.
        samples: Unique sample IDs from ``matrix_samples.txt``.

    Returns:
        Each sample's lineage, in ``samples`` order.

    Raises:
        ValueError: If a row breaks a rule above.
        OSError: If the file cannot be read.
    """
    sample_count = len(samples)
    node_count = tree_node_count(sample_count)
    known = set(samples)
    if len(known) != sample_count:
        raise ValueError(f"{path.name}: matrix samples are not unique")
    lineages: dict[str, tuple[int, ...]] = {}
    seen_nodes: set[int] = set()
    leaves: set[int] = set()
    rows = read_utf8_lines(path)
    if len(rows) != sample_count:
        raise ValueError(
            f"{path.name} has {len(rows)} row(s) for {sample_count} "
            "sample(s)"
        )
    for number, line in enumerate(rows, start=1):
        source = f"{path.name}:{number}"
        tokens = line.split()
        if len(tokens) < 3:
            raise ValueError(f"{source}: expected sample, count and nodes")
        sample = tokens[0]
        if sample not in known:
            raise ValueError(f"{source}: unknown sample {sample!r}")
        if sample in lineages:
            raise ValueError(f"{source}: duplicate sample {sample!r}")
        count = parse_canonical_int(tokens[1], source=f"{source}:2")
        nodes = tuple(
            parse_canonical_int(token, source=f"{source}:{column}")
            for column, token in enumerate(tokens[2:], start=3)
        )
        if not 0 < count <= sample_count or count != len(nodes):
            raise ValueError(f"{source}: count {count} does not match")
        if nodes[0] != 0:
            raise ValueError(f"{source}: lineage must begin at root 0")
        if any(left >= right
               for left, right in zip(nodes, nodes[1:], strict=False)):
            raise ValueError(f"{source}: node IDs must strictly increase")
        if nodes[-1] >= node_count:
            raise ValueError(
                f"{source}: node {nodes[-1]} is outside 0..{node_count - 1}")
        lineages[sample] = nodes
        seen_nodes.update(nodes)
        leaves.add(nodes[-1])
    if seen_nodes != set(range(node_count)):
        raise ValueError(f"{path.name}: lineages do not cover the tree")
    if len(leaves) != sample_count:
        raise ValueError(f"{path.name}: lineages must end at distinct leaves")
    return {sample: lineages[sample] for sample in samples}


def render_sample_lineages(
    samples: Sequence[str], lineages: dict[str, Sequence[int]]
) -> str:
    """Return the text of ``sample_lineages.txt``.

    Args:
        samples: Sample IDs, in ``matrix_samples.txt`` order.
        lineages: Each sample's root-to-leaf node IDs, root 0 first.

    Returns:
        One row per sample: its ID, the number of nodes on its lineage
        and the node IDs, separated by single spaces.
    """
    return "".join(
        f"{sample} {len(lineages[sample])} "
        + " ".join(str(node) for node in lineages[sample]) + "\n"
        for sample in samples
    )


def iter_matrix_rows(
    path: Path, sample_count: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream ``matrix_values.txt`` one gene row at a time.

    The variance preparation writes one row per gene with one value per
    sample: canonical decimal integers within the int32 range.

    Args:
        path: ``matrix_values.txt`` path.
        sample_count: Number of values each row must hold.

    Yields:
        The zero-based row index and its values as int64.

    Raises:
        ValueError: If a row has the wrong length or a value breaks the
            grammar or the int32 range.
        OSError: If the file cannot be read.
    """
    for number, line in enumerate(iter_utf8_lines(path), start=1):
        tokens = line.split()
        if len(tokens) != sample_count:
            raise ValueError(
                f"{path.name}:{number}: {len(tokens)} value(s) for "
                f"{sample_count} sample(s)")
        row = np.empty(sample_count, dtype=np.int64)
        for column, token in enumerate(tokens):
            value = parse_canonical_int(
                token, source=f"{path.name}:{number}:{column + 1}")
            if not INT32_INFO.min <= value <= INT32_INFO.max:
                raise ValueError(
                    f"{path.name}:{number}:{column + 1}: outside int32")
            row[column] = value
        yield number - 1, row


def read_matrix_values(
    path: Path, gene_count: int, sample_count: int
) -> np.ndarray:
    """Read the whole expression matrix into one array.

    Args:
        path: ``matrix_values.txt`` path.
        gene_count: Number of rows, one per gene.
        sample_count: Number of values per row.

    Returns:
        A ``gene_count`` by ``sample_count`` int64 array.

    Raises:
        ValueError: If the shape or any value is wrong.
        OSError: If the file cannot be read.
    """
    values = np.empty((gene_count, sample_count), dtype=np.int64)
    rows = 0
    for index, row in iter_matrix_rows(path, sample_count):
        if index >= gene_count:
            raise ValueError(f"{path.name}: more than {gene_count} rows")
        values[index] = row
        rows = index + 1
    if rows != gene_count:
        raise ValueError(f"{path.name}: {rows} row(s) for {gene_count} genes")
    return values



def write_matrix_values(path: Path, values: np.ndarray) -> None:
    """Write ``matrix_values.txt``.

    Args:
        path: Destination file.
        values: Genes-by-samples integers within the int32 range.

    Raises:
        ValueError: If ``values`` is not a 2-D integer array within
            int32.
        OSError: If the file cannot be written.
    """
    array = np.asarray(values)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("the matrix must be a 2-D integer array")
    if array.size and (array.min() < INT32_INFO.min
                       or array.max() > INT32_INFO.max):
        raise ValueError("matrix values must lie within the int32 range")
    pd.DataFrame(array).to_csv(path, header=False, index=False, sep=" ",
                               lineterminator="\n")

# ======================================================================
# Tjala's signature report (tjala_output.txt)
# ======================================================================

#: Line that opens and closes a signature's scaling section.
REPORT_RULE = "----------"

#: What the report says in place of signatures when Tjala finds none.
NO_SIGNATURES_LINE = "No signatures were generated."

#: Factors the scaling section applies to each term's threshold.
SCALE_FACTORS = tuple(range(10, 101, 10))

#: Why the search for one signature stopped, as the report says it.
SIGNATURE_STOP_LABELS = {
    "convergence": "convergence",
    "max_iterations": "max iterations",
}

#: Why the whole run stopped, as the report says it.
RUN_STOP_LABELS = {
    "uncovered_threshold_reached": "uncovered-sample target reached",
    "no_active_terms_remaining": "no active terms remain",
    "insufficient_antecedent_coverage": (
        "No further valid signatures met the constraints"
    ),
    "no_terms_removed": "used-term removal made no progress",
}

_OPERATORS = ("<", ">")
_SIGNATURE_STOP_LINES = {
    f"({label})": code for code, label in SIGNATURE_STOP_LABELS.items()
}
_RUN_STOP_LINES = {
    f"Termination: {label} ({code})": code
    for code, label in RUN_STOP_LABELS.items()
}
# A share in percent as the report writes it: one decimal, 0.0 to 100.0.
_PERCENT = r"(100\.0|[1-9]?[0-9]\.[0-9])"
_COVERAGE_RE = re.compile(r"(0|[1-9][0-9]*): " + _PERCENT + r"% covered\Z")
_SCALED_RE = re.compile(
    r"Scale = ([^,]+), ([^ ]+) ([<>]) ([^ ]+): " + _PERCENT
    + r"% covered in cannot clusters\Z"
)
_SUMMARY_FIELDS = (
    "Signatures generated",
    "Epochs completed",
    "Inner-loop iterations",
    "Generated antecedents",
    "Final uncovered samples",
    "Remaining active terms",
)


@dataclass(frozen=True)
class SignatureTerm:
    """One condition of a signature: a gene against a threshold.

    Attributes:
        gene: Gene identifier from ``matrix_genes.txt``.
        operator: ``<`` or ``>``.
        threshold: An observed expression value, within int32.
    """

    gene: str
    operator: str
    threshold: int


@dataclass(frozen=True)
class ClusterCoverage:
    """The share of one cluster's samples that a signature covers.

    Attributes:
        cluster_id: Canonical decimal node ID.
        percent: The covered share in percent, which the report rounds
            to one decimal.
    """

    cluster_id: str
    percent: float


@dataclass(frozen=True)
class ScaledTerm:
    """One line of a signature's scaling section.

    Attributes:
        scale: Factor that divides the threshold of a ``>`` term and
            multiplies that of a ``<`` term.
        gene: The term's gene.
        operator: The term's operator.
        threshold: The scaled threshold.
        percent: Share of the cannot-cluster samples that the scaled
            term covers, in percent, rounded to one decimal.
    """

    scale: int
    gene: str
    operator: str
    threshold: float
    percent: float


@dataclass(frozen=True)
class Signature:
    """One signature of the report.

    Attributes:
        number: One-based position in the report.
        stop_reason: Why its search stopped, a key of
            ``SIGNATURE_STOP_LABELS``.
        terms: The antecedent's conditions, all of which must hold.
        can_clusters: Covered clusters that are not cannot clusters,
            in report order.
        cannot_clusters: Covered cannot clusters, in report order.
        scaling: The scaling section: for each term, one line per
            factor in ``SCALE_FACTORS``. Empty when the run has no
            cannot-cluster samples.
    """

    number: int
    stop_reason: str
    terms: tuple[SignatureTerm, ...]
    can_clusters: tuple[ClusterCoverage, ...]
    cannot_clusters: tuple[ClusterCoverage, ...]
    scaling: tuple[ScaledTerm, ...] = ()

    @property
    def cluster_ids(self) -> tuple[str, ...]:
        """Return the covered cluster IDs, can clusters first."""
        return tuple(
            coverage.cluster_id
            for coverage in self.can_clusters + self.cannot_clusters
        )


@dataclass(frozen=True)
class RunSummary:
    """The report's closing summary of the run.

    Attributes:
        stop_reason: Why the run stopped, a key of ``RUN_STOP_LABELS``.
        epochs: Epochs completed.
        iterations: Inner-loop iterations over all epochs.
        antecedents: Antecedents generated over all epochs.
        uncovered: Samples that no signature covers.
        active_terms: Terms still active at the end.
    """

    stop_reason: str
    epochs: int
    iterations: int
    antecedents: int
    uncovered: int
    active_terms: int


@dataclass(frozen=True)
class SignatureReport:
    """The content of ``tjala_output.txt``.

    Attributes:
        signatures: The signatures, numbered from one.
        summary: The run summary.
    """

    signatures: tuple[Signature, ...]
    summary: RunSummary


def render_antecedent(terms: Sequence[SignatureTerm]) -> str:
    """Return a signature's ``IF ... THEN:`` line.

    Args:
        terms: The antecedent's conditions, in order.

    Returns:
        The line; ``<empty>`` stands for no condition.
    """
    if not terms:
        return "IF <empty> THEN:"
    conditions = " AND ".join(
        f"{term.gene} {term.operator} {term.threshold}" for term in terms
    )
    return f"IF {conditions} THEN:"


def _stop_label(labels: dict[str, str], code: str) -> str:
    """Return the report's wording of a stop reason.

    Args:
        labels: ``SIGNATURE_STOP_LABELS`` or ``RUN_STOP_LABELS``.
        code: The stop reason.

    Returns:
        The wording.

    Raises:
        ValueError: If ``code`` is not a known stop reason.
    """
    try:
        return labels[code]
    except KeyError:
        raise ValueError(f"unknown stop reason {code!r}") from None


def _render_coverage(coverage: Sequence[ClusterCoverage]) -> list[str]:
    """Return the lines of one cluster section, ``None`` if empty.

    Args:
        coverage: The section's clusters, in report order.

    Returns:
        One line per cluster, or the single line ``None``.
    """
    if not coverage:
        return ["None"]
    return [f"{item.cluster_id}: {item.percent:.1f}% covered"
            for item in coverage]


def _render_signature(signature: Signature) -> str:
    """Return the text of one signature, without a final newline.

    Args:
        signature: The signature.

    Returns:
        Its lines: header, stop reason, antecedent, the two cluster
        sections and, if any, the scaling section between rules.

    Raises:
        ValueError: If the stop reason is unknown.
    """
    lines = [
        f"Signature #{signature.number}:",
        f"({_stop_label(SIGNATURE_STOP_LABELS, signature.stop_reason)})",
        render_antecedent(signature.terms),
        "Can clusters:",
        *_render_coverage(signature.can_clusters),
        "Cannot clusters:",
        *_render_coverage(signature.cannot_clusters),
    ]
    if signature.scaling:
        lines.append(REPORT_RULE)
        lines.extend(
            f"Scale = {line.scale}, {line.gene} {line.operator} "
            f"{float(line.threshold)!r}: {line.percent:.1f}% covered in "
            "cannot clusters"
            for line in signature.scaling
        )
        lines.append(REPORT_RULE)
    return "\n".join(lines)


def render_signature_report(report: SignatureReport) -> str:
    """Return the text of ``tjala_output.txt``.

    Signatures are separated by blank lines, and a blank line precedes
    the run summary. The text is parsed back before it is returned, so
    the writer never produces a report that its reader rejects.

    Args:
        report: The report's content.

    Returns:
        The report text, LF-terminated.

    Raises:
        ValueError: If the content breaks the report grammar.
    """
    sections = [_render_signature(signature)
                for signature in report.signatures]
    if not sections:
        sections.append(NO_SIGNATURES_LINE)
    summary = report.summary
    label = _stop_label(RUN_STOP_LABELS, summary.stop_reason)
    counts = (len(report.signatures), summary.epochs, summary.iterations,
              summary.antecedents, summary.uncovered, summary.active_terms)
    sections.append("\n".join([
        "Run summary",
        f"Termination: {label} ({summary.stop_reason})",
        *(f"{field}: {count}"
          for field, count in zip(_SUMMARY_FIELDS, counts, strict=True)),
    ]))
    text = "\n\n".join(sections) + "\n"
    parse_signature_report(text.split("\n")[:-1], source="new report")
    return text


class _ReportLines:
    """A cursor over the lines of a report, for located errors."""

    def __init__(self, lines: Sequence[str], source: str) -> None:
        """Start before the first line.

        Args:
            lines: The report's lines, without terminators.
            source: The report's name, for diagnostics.
        """
        self._lines = lines
        self._source = source
        self.number = 0

    def where(self) -> str:
        """Return the location of the line taken last."""
        return f"{self._source}:{self.number}"

    def located(self, message: str) -> str:
        """Return a message about the line taken last.

        Args:
            message: What is wrong with it.

        Returns:
            The message, prefixed with the line's location.
        """
        return f"{self.where()}: {message}"

    def peek(self) -> str | None:
        """Return the next line without taking it; None at the end."""
        if self.number < len(self._lines):
            return self._lines[self.number]
        return None

    def take(self, expected: str) -> str:
        """Take the next line.

        Args:
            expected: What should come next, for diagnostics.

        Returns:
            The line.

        Raises:
            ValueError: If the report has ended.
        """
        if self.number >= len(self._lines):
            raise ValueError(f"{self._source}: ends before {expected}")
        self.number += 1
        return self._lines[self.number - 1]

    def expect(self, text: str) -> None:
        """Take the next line, which must be ``text``.

        Args:
            text: The required line.

        Raises:
            ValueError: If the line differs or the report has ended.
        """
        line = self.take(repr(text))
        if line != text:
            raise ValueError(
                self.located(f"expected {text!r}, found {line!r}"))


def _parse_terms(lines: _ReportLines) -> tuple[SignatureTerm, ...]:
    """Take and parse a signature's ``IF ... THEN:`` line.

    Args:
        lines: The report, positioned before the line.

    Returns:
        The conditions, in order.

    Raises:
        ValueError: If a condition is malformed or repeats a gene.
    """
    line = lines.take("the antecedent")
    if not (line.startswith("IF ") and line.endswith(" THEN:")):
        raise ValueError(
            lines.located(f"expected 'IF ... THEN:', found {line!r}"))
    body = line[len("IF "):-len(" THEN:")]
    if body == "<empty>":
        return ()
    terms: list[SignatureTerm] = []
    for condition in body.split(" AND "):
        fields = condition.split(" ")
        if len(fields) != 3 or fields[1] not in _OPERATORS:
            raise ValueError(
                lines.located(f"malformed condition {condition!r}"))
        gene, operator, token = fields
        validate_identifier(gene, source=lines.where())
        threshold = parse_canonical_int(token, source=lines.where())
        if not INT32_INFO.min <= threshold <= INT32_INFO.max:
            raise ValueError(
                lines.located(f"threshold {threshold} is outside int32"))
        if any(term.gene == gene for term in terms):
            raise ValueError(
                lines.located(f"more than one condition on {gene!r}"))
        terms.append(SignatureTerm(gene, operator, threshold))
    return tuple(terms)


def _parse_coverage(lines: _ReportLines) -> tuple[ClusterCoverage, ...]:
    """Take the lines of one cluster section.

    Args:
        lines: The report, positioned after the section's heading.

    Returns:
        The section's clusters; empty for ``None``.

    Raises:
        ValueError: If the section is neither ``None`` nor at least
            one coverage line.
    """
    if lines.peek() == "None":
        lines.take("None")
        return ()
    coverage: list[ClusterCoverage] = []
    while (line := lines.peek()) is not None:
        match = _COVERAGE_RE.fullmatch(line)
        if match is None:
            break
        lines.take("a coverage line")
        coverage.append(ClusterCoverage(match[1], float(match[2])))
    if not coverage:
        line = lines.take("a cluster section")
        raise ValueError(lines.located(
            f"expected a coverage line or None, found {line!r}"))
    return tuple(coverage)


def _parse_scaling(
    lines: _ReportLines, terms: tuple[SignatureTerm, ...]
) -> tuple[ScaledTerm, ...]:
    """Take a signature's scaling section, if it has one.

    Args:
        lines: The report, positioned after the cluster sections.
        terms: The signature's conditions, which the section follows.

    Returns:
        The section's lines; empty without a section.

    Raises:
        ValueError: If a line is malformed or out of order.
    """
    if lines.peek() != REPORT_RULE:
        return ()
    lines.take(REPORT_RULE)
    if not terms:
        raise ValueError(
            lines.located("a scaling section needs conditions"))
    scaled: list[ScaledTerm] = []
    for term in terms:
        for scale in SCALE_FACTORS:
            line = lines.take("a scaling line")
            match = _SCALED_RE.fullmatch(line)
            if match is None or match.group(1, 2, 3) != (
                str(scale), term.gene, term.operator
            ):
                raise ValueError(lines.located(
                    f"expected scale {scale} of {term.gene} "
                    f"{term.operator}, found {line!r}"))
            threshold = parse_decimal_float(match[4], source=lines.where())
            if repr(threshold) != match[4]:
                raise ValueError(
                    lines.located(f"noncanonical threshold {match[4]!r}"))
            scaled.append(ScaledTerm(scale, term.gene, term.operator,
                                     threshold, float(match[5])))
    lines.expect(REPORT_RULE)
    return tuple(scaled)


def _parse_signature(lines: _ReportLines, number: int) -> Signature:
    """Take one signature.

    Args:
        lines: The report, positioned before the signature's header.
        number: The number the header must carry.

    Returns:
        The signature.

    Raises:
        ValueError: If a part is missing, malformed or out of order,
            or a cluster appears twice.
    """
    lines.expect(f"Signature #{number}:")
    line = lines.take("the stop reason")
    stop_reason = _SIGNATURE_STOP_LINES.get(line)
    if stop_reason is None:
        raise ValueError(lines.located(f"unknown stop reason {line!r}"))
    terms = _parse_terms(lines)
    lines.expect("Can clusters:")
    can_clusters = _parse_coverage(lines)
    lines.expect("Cannot clusters:")
    cannot_clusters = _parse_coverage(lines)
    clusters = [coverage.cluster_id
                for coverage in can_clusters + cannot_clusters]
    if len(set(clusters)) != len(clusters):
        raise ValueError(
            lines.located(f"signature #{number} lists a cluster twice"))
    scaling = _parse_scaling(lines, terms)
    return Signature(number, stop_reason, terms, can_clusters,
                     cannot_clusters, scaling)


def _parse_summary(lines: _ReportLines, signature_count: int) -> RunSummary:
    """Take the run summary.

    Args:
        lines: The report, positioned before ``Run summary``.
        signature_count: Number of signatures the report holds.

    Returns:
        The summary.

    Raises:
        ValueError: If a line is malformed or the signature count
            disagrees.
    """
    lines.expect("Run summary")
    line = lines.take("the termination line")
    stop_reason = _RUN_STOP_LINES.get(line)
    if stop_reason is None:
        raise ValueError(lines.located(f"unknown termination {line!r}"))
    counts: list[int] = []
    for field in _SUMMARY_FIELDS:
        line = lines.take(repr(field))
        prefix = f"{field}: "
        if not line.startswith(prefix):
            raise ValueError(
                lines.located(f"expected {prefix!r}, found {line!r}"))
        count = parse_canonical_int(line[len(prefix):], source=lines.where())
        if count < 0:
            raise ValueError(lines.located(f"negative count {count}"))
        if not counts and count != signature_count:
            raise ValueError(lines.located(
                f"{count} signature(s) announced, {signature_count} given"))
        counts.append(count)
    return RunSummary(stop_reason, *counts[1:])


def parse_signature_report(
    lines: Sequence[str], *, source: str
) -> SignatureReport:
    """Parse the lines of ``tjala_output.txt``.

    The grammar is the one ``render_signature_report`` writes: the
    signatures, numbered from one, or ``NO_SIGNATURES_LINE``, then the
    run summary, each followed by one blank line except the last.

    Args:
        lines: The report's lines, without terminators.
        source: The report's name, for diagnostics.

    Returns:
        The report's content.

    Raises:
        ValueError: If the lines break the grammar.
    """
    report = _ReportLines(lines, source)
    signatures: list[Signature] = []
    if report.peek() == NO_SIGNATURES_LINE:
        report.take(NO_SIGNATURES_LINE)
        report.expect("")
    else:
        while True:
            signatures.append(_parse_signature(report, len(signatures) + 1))
            report.expect("")
            if report.peek() == "Run summary":
                break
    summary = _parse_summary(report, len(signatures))
    if report.peek() is not None:
        line = report.take("the end")
        raise ValueError(report.located(
            f"unexpected line after the summary: {line!r}"))
    return SignatureReport(tuple(signatures), summary)


def read_signature_report(path: Path) -> SignatureReport:
    """Read and check ``tjala_output.txt``.

    Args:
        path: The report.

    Returns:
        The report's content.

    Raises:
        ValueError: If the file breaks the transport rules or the
            report grammar.
        OSError: If the file cannot be read.
    """
    return parse_signature_report(
        read_utf8_lines(path, allow_blank=True), source=path.name
    )


# ======================================================================
# Cannot-cluster list (cluster_<n>.txt)
# ======================================================================

# People write these lists by hand, so a plus sign and leading zeros
# are accepted: 017 and +17 both name node 17.
_LISTED_NODE_RE = re.compile(r"\+?[0-9]+\Z", re.ASCII)


def render_cluster_list(node_ids: Iterable[int]) -> str:
    """Return a cluster list as samples_by_cluster exports it.

    Args:
        node_ids: Node IDs, in the order to list them.

    Returns:
        The IDs separated by a comma and a space, on one line.
    """
    return ", ".join(map(str, node_ids)) + "\n"


def read_cluster_list(path: Path, *, max_node_id: int) -> tuple[int, ...]:
    """Read a cannot-cluster list, exported or written by hand.

    Commas, semicolons and whitespace separate the entries; each names
    one node of the current tree, at most once. An empty file is an
    empty list.

    Args:
        path: The list.
        max_node_id: Largest node ID of the current tree.

    Returns:
        The node IDs, in file order.

    Raises:
        ValueError: If an entry is not a node ID of the tree or repeats.
        OSError: If the file cannot be read.
    """
    source = f"cannot-clusters file {str(path)!r}"
    raw = "\n".join(read_utf8_lines(path, allow_blank=True, allow_empty=True))
    node_ids: list[int] = []
    for token in raw.replace(",", " ").replace(";", " ").split():
        if _LISTED_NODE_RE.fullmatch(token) is None:
            raise ValueError(f"{source}: invalid node ID {token!r}")
        node_id = int(token.lstrip("+"), 10)
        if not 0 <= node_id <= max_node_id:
            raise ValueError(
                f"{source}: node {node_id} is outside 0..{max_node_id}")
        node_ids.append(node_id)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(
            "Cannot-clusters file contains duplicate immutable node IDs.")
    return tuple(node_ids)
