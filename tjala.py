"""Tjala hierarchical signature-extraction algorithm and CLI.

Requires Python 3.10 or later, NumPy, and pandas. SciPy is optional.
Inputs and outputs are located beside this script. The gene-by-sample
integer matrix follows the orders in matrix_genes.txt and
matrix_samples.txt.
Inputs must have valid producer provenance; the signature report keeps
its numeric cluster-line grammar. Search and numerical objectives are
unchanged by the I/O and documentation conventions.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

from compute_variances import (
    BINARY_VECTORS_FILE,
    MATRIX_GENES_FILE,
    MATRIX_SAMPLES_FILE,
    MATRIX_VALUES_FILE,
    SAMPLE_LINEAGES_FILE,
    TERM_VARIANCES_FILE,
    TJALA_OUTPUT_FILE,
    invalidate_dependents,
    parse_canonical_int,
    parse_decimal_float,
    prompt_input,
    provenance_path,
    publication_unchanged,
    read_identifiers,
    read_utf8_lines,
    validate_variance_output,
    verify_artifacts,
    write_provenance,
)

try:
    from scipy import sparse as scipy_sparse
except (
    ImportError
):  # Optional acceleration; dense NumPy fallback is supported.
    scipy_sparse = None


# ======================================================================
# Block 0: Runtime Configuration
# ======================================================================
@dataclass(frozen=True)
class ConsoleOptions:
    """Controls optional console and inspection behavior.

    Attributes:
        inspect_binary_vectors: Whether to offer interactive
            binary-vector inspection after the membership matrix is
            built.
        show_setup_details: Whether to print term and graph setup
            details.
        show_iteration_progress: Whether to print one progress line per
            ACO iteration.
    """

    inspect_binary_vectors: bool = False
    show_setup_details: bool = False
    show_iteration_progress: bool = True


DEFAULT_CONSOLE_OPTIONS = ConsoleOptions()

# ======================================================================
# Block 1: Typed State Schemas
# ======================================================================
class RunParameters(TypedDict):
    """Validated user-supplied algorithm parameters.

    Attributes:
        max_terms: Maximum number of distinct-gene terms per
            antecedent.
        min_sample_coverage: Minimum samples covered by a feasible
            antecedent.
        ant_count: Number of candidate-construction ants per iteration.
        max_bins: Maximum representative threshold bins per gene.
        max_iterations: Maximum inner-loop iterations in one signature
            epoch.
        convergence_window: Unchanged-best valid iterations needed
            for convergence.
        max_uncovered_samples: Allowed uncovered-sample count
            at normal completion.
        cannot_clusters_path: Optional user-supplied cannot-cluster list
            path.
        max_cannot_cluster_share: Maximum forbidden share of covered
            samples, or None.
    """

    max_terms: int
    min_sample_coverage: int
    ant_count: int
    max_bins: int
    max_iterations: int
    convergence_window: int
    max_uncovered_samples: int
    cannot_clusters_path: str | None
    max_cannot_cluster_share: float | None


class TermRecord(TypedDict):
    """One generated expression-threshold term.

    Attributes:
        id: Stable zero-based term identifier.
        gene: Literal normalized gene identifier.
        gene_index: Zero-based expression-matrix row index.
        operator: Strict comparison operator, either < or >.
        value: Observed integer expression threshold.
        variance: Unweighted hierarchy variance imported for this term.
    """

    id: int
    gene: str
    gene_index: int
    operator: str
    value: int
    variance: float


class StaticAlgorithmData(TypedDict):
    """Validated source data that remains stable during one run.

    Attributes:
        genes: Unique gene identifiers in expression-row order.
        samples: Unique sample IDs in expression-column order.
        expression_values: Genes-by-samples integer expression matrix.
        variances: Validated per-term gene, value, operator, and
            variance table.
        hierarchies: Sample IDs mapped to immutable-node membership
            sets.
        cannot_clusters: Configured forbidden immutable node IDs, or
            None.
        sample_count: Number of samples and terminal hierarchy leaves.
        node_count: Complete immutable node count, equal to 2N - 1.
    """

    genes: list[str]
    samples: list[str]
    expression_values: np.ndarray
    variances: pd.DataFrame
    hierarchies: dict[str, set[str]]
    cannot_clusters: set[str] | None
    sample_count: int
    node_count: int


class DerivedAlgorithmData(TypedDict, total=False):
    """Run-scoped state populated during preprocessing.

    Attributes:
        binary_matrix: Samples-by-nodes dense or CSR membership matrix.
        cluster_ids: Canonical node-ID strings in membership-column
            order.
        terms: Generated term records in stable term-ID order.
        terms_by_gene: Gene names mapped to their generated term IDs.
        term_count: Number of represented terms in this state.
        coverage_words: Packed uint64 sample coverage, indexed by term
            ID.
        term_coverage_counts: Covered-sample counts indexed by term ID.
        term_variances: Cached unweighted binary-membership variances
            per
            term.
        cluster_sample_counts: Sample counts in membership-column order.
        global_msd: Unweighted variance of the complete membership
            dataset.
        cannot_sample_mask: Boolean membership in the union of cannot
            clusters.
        cannot_sample_words: Packed sample membership in the
            cannot-cluster union.
    """

    binary_matrix: Any
    cluster_ids: list[str]
    terms: list[TermRecord]
    terms_by_gene: dict[str, list[int]]
    term_count: int
    coverage_words: np.ndarray
    term_coverage_counts: np.ndarray
    term_variances: np.ndarray
    cluster_sample_counts: np.ndarray
    global_msd: float
    cannot_sample_mask: np.ndarray
    cannot_sample_words: np.ndarray


class AlgorithmData(StaticAlgorithmData, DerivedAlgorithmData):
    """Source data with optional derived preprocessing fields."""


class PreparedAlgorithmData(StaticAlgorithmData):
    """Algorithm data with completed preprocessing state.

    Attributes:
        binary_matrix: Samples-by-nodes dense or CSR membership matrix.
        cluster_ids: Canonical node-ID strings in membership-column
            order.
        terms: Generated term records in stable term-ID order.
        terms_by_gene: Gene names mapped to their generated term IDs.
        term_count: Number of represented terms in this state.
        coverage_words: Packed uint64 sample coverage, indexed by term
            ID.
        term_coverage_counts: Covered-sample counts indexed by term ID.
        term_variances: Cached unweighted binary-membership variances
            per
            term.
        cluster_sample_counts: Sample counts in membership-column order.
        global_msd: Unweighted variance of the complete membership
            dataset.
        cannot_sample_mask: Boolean membership in the union of cannot
            clusters.
        cannot_sample_words: Packed sample membership in the
            cannot-cluster union.
    """

    binary_matrix: Any
    cluster_ids: list[str]
    terms: list[TermRecord]
    terms_by_gene: dict[str, list[int]]
    term_count: int
    coverage_words: np.ndarray
    term_coverage_counts: np.ndarray
    term_variances: np.ndarray
    cluster_sample_counts: np.ndarray
    global_msd: float
    cannot_sample_mask: np.ndarray
    cannot_sample_words: np.ndarray


class GraphStructure(TypedDict):
    """Stable structural portion of an antecedent construction graph.

    Attributes:
        term_by_id: Stable term IDs mapped to complete term records.
        parts: Gene partitions containing their represented term IDs.
        genes_in_graph: Gene names in partition-index order.
        part_index_by_gene: Gene names mapped to their partition
            indices.
        term_to_part: Term IDs mapped to their gene-partition indices.
        active_term_ids: Term IDs still eligible for construction.
        active_term_ids_array: Active term IDs in deterministic numeric
            order.
        active_term_parts_array: Partition indices aligned with active
            term IDs.
        start_node: Label of the implicit antecedent-construction start
            node.
        stop_node: Label of the implicit antecedent-construction stop
            node.
        term_count: Number of represented terms in this state.
        part_count: Number of represented gene partitions.
    """

    term_by_id: dict[int, TermRecord]
    parts: list[list[int]]
    genes_in_graph: list[str]
    part_index_by_gene: dict[str, int]
    term_to_part: dict[int, int]
    active_term_ids: set[int]
    active_term_ids_array: np.ndarray
    active_term_parts_array: np.ndarray
    start_node: str
    stop_node: str
    term_count: int
    part_count: int


class GraphWeights(TypedDict, total=False):
    """Active-set-dependent heuristic and pheromone state.

    Attributes:
        heuristics: Active term IDs mapped to their heuristic weights.
        pheromones: Active term IDs mapped to their pheromone weights.
    """

    heuristics: dict[int, float]
    pheromones: dict[int, float]


class GraphState(GraphStructure, GraphWeights):
    """Explicit schema for the antecedent construction graph."""


# ======================================================================
# Block 2: Prompt the User to Input the Required Parameters
# ======================================================================
def prompt_int(
    prompt: str, min_val: int = 1, default: int | None = None
) -> int:
    """Prompt for and validates an integer.

    Args:
        prompt: Text displayed to the user.
        min_val: Minimum accepted value. Defaults to 1.
        default: Value returned when the user presses Enter without
            input.

    Returns:
        The validated integer.
    """
    while True:
        prompt_text = (
            f"{prompt} (default: {default}): "
            if default is not None
            else f"{prompt}: "
        )
        val = prompt_input(prompt_text).strip()
        if not val and default is not None:
            return default
        try:
            val = int(val)
            if val < min_val:
                print(f"Invalid input. Value must be at least {min_val}.",
                    file=sys.stderr)
                continue
            return val
        except ValueError:
            print((
                'Invalid input. Please enter a whole number.'
            ), file=sys.stderr)


def prompt_float(
    prompt: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
    default: float | None = None,
) -> float:
    """Prompt for and validates a floating-point value.

    Args:
        prompt: Text displayed to the user.
        min_val: Minimum accepted value. Defaults to 0.0.
        max_val: Maximum accepted value. Defaults to 1.0.
        default: Value returned when the user presses Enter without
            input.

    Returns:
        The validated floating-point value.
    """
    while True:
        prompt_text = (
            f"{prompt} (default: {default}): "
            if default is not None
            else f"{prompt}: "
        )
        val = prompt_input(prompt_text).strip()
        if not val and default is not None:
            return default
        try:
            val = float(val)
            if not (min_val <= val <= max_val):
                print(
                    f"Invalid input. Value must be between {min_val} "
                    f"and {max_val}.", file=sys.stderr)
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number.", file=sys.stderr)


def prompt_parameters() -> RunParameters:
    """Collect and validates all interactive run parameters.

    Returns:
        A complete set of validated algorithm parameters.
    """
    params: dict[str, Any] = {}
    print((
        'Enter run parameters (press Enter to accept each '
        'default):'
    ), file=sys.stderr)
    params["max_terms"] = prompt_int(
        "Maximum number of terms per signature",
        min_val=1,
        default=5,
    )
    params["min_sample_coverage"] = prompt_int(
        "Minimum number of samples that must be covered by a signature",
        min_val=1,
        default=10,
    )
    params["ant_count"] = prompt_int(
        "Ant colony size (number of ants)",
        min_val=1,
        default=50,
    )
    params["max_bins"] = prompt_int(
        "Number of terms (discretization bins) per gene",
        min_val=1,
        default=100,
    )

    while True:
        max_iter = prompt_int(
            "Maximum number of iterations per epoch",
            min_val=1,
            default=100,
        )
        conv_iter = prompt_int(
            "Number of iterations yielding the same best signature for "
            "identifying signature convergence",
            min_val=1,
            default=10,
        )
        if conv_iter > max_iter:
            print(
                "Invalid input: convergence iterations cannot exceed the "
                "maximum allowed iterations.", file=sys.stderr)
        else:
            params["max_iterations"] = max_iter
            params["convergence_window"] = conv_iter
            break

    params["max_uncovered_samples"] = prompt_int(
        "Maximum number of uncovered samples allowed before termination",
        min_val=0,
        default=10,
    )

    while True:
        val = prompt_input(
            "Path to a file listing cannot-cluster IDs (Enter to skip): "
        ).strip()
        if not val:
            params["cannot_clusters_path"] = None
            params["max_cannot_cluster_share"] = None
            break
        elif os.path.isfile(val):
            params["cannot_clusters_path"] = val
            params["max_cannot_cluster_share"] = prompt_float(
                "Share of covered samples that may belong to cannot "
                "clusters [0–1]",
                default=0.05,
            )
            break
        else:
            print(
                f"File not found at '{val}'. Please check the path "
                "and try again.", file=sys.stderr)

    cannot_limit = params.get("max_cannot_cluster_share")
    cannot_text = (
        "disabled"
        if cannot_limit is None
        else f"{cannot_limit:.1%} maximum covered-sample share"
    )
    print("\nConfiguration", file=sys.stderr)
    print(
        (
            f"  terms/signature <= "
            f"{params['max_terms']} | minimum "
            f"coverage = {params['min_sample_coverage']} | "
            f"ants = {params['ant_count']} | bins = "
            f"{params['max_bins']}"
        ), file=sys.stderr)
    print(
        (
            f"  iterations/epoch <= "
            f"{params['max_iterations']} | convergence window "
            f"= {params['convergence_window']} | "
            f"uncovered target <= "
            f"{params['max_uncovered_samples']}"
        ), file=sys.stderr)
    print(f"  Cannot-cluster constraint: {cannot_text}\n", file=sys.stderr)
    return cast(RunParameters, params)


# ======================================================================
# Block 3: Load, Normalize, and Validate Data
# ======================================================================
def _tree_node_count(sample_count: int) -> int:
    """Return the node count for a full immutable binary hierarchy.

    A full binary hierarchy with ``N`` leaves contains exactly ``2N -
    1`` immutable nodes. Node 0 is the root, and split ``k`` creates
    sibling nodes ``2k - 1`` and ``2k``. The upstream native variance
    reader stores node IDs as signed 32-bit integers, so the same
    representability contract is enforced here.

    Args:
        sample_count: Number of samples/leaves in the hierarchy.

    Returns:
        Total immutable tree-node count.

    Raises:
        ValueError: If the sample count is empty or the largest node ID
            would exceed the signed 32-bit hierarchy-ID contract.
    """
    if sample_count < 1:
        raise ValueError((
            'At least one sample is required for an immutable '
            'tree.'
        ))

    max_node_id = 2 * sample_count - 2
    if max_node_id > int(np.iinfo(np.int32).max):
        raise ValueError(
            "Dataset is too large for the immutable hierarchy-ID contract: "
            f"largest node ID would be {max_node_id}, exceeding signed int32."
        )

    return 2 * sample_count - 1


def _parse_tree_node_id(
    token: str, *, source: str, max_node_id: int
) -> str:
    """Validate canonical decimal spelling of an node ID.

    Validate the canonical decimal spelling of an immutable node ID.

    Args:
        token: Literal machine-field text without quote removal.
        source: File and field context.
        max_node_id: Maximum node ID allowed by this dataset.

    Returns:
        Canonical node string used by Tjala's report model.

    Raises:
        ValueError: If syntax, sign, or dataset range is invalid.
    """
    node_id = parse_canonical_int(token, source=source)
    if not 0 <= node_id <= max_node_id:
        raise ValueError((
            f'{source}: node {node_id} is outside '
            f'0..{max_node_id}'
        ))
    return str(node_id)


def load_algorithm_data(
    base_path: str,
    params: RunParameters,
    cannot_clusters_path: str | None = None,
) -> AlgorithmData:
    """Load and validates inputs for fully tree-node model.

    Load and validates inputs for the fully immutable tree-node model.

    ``sample_lineages.txt`` must describe a complete full binary
    hierarchy
    with ``N`` samples and therefore exactly ``2N - 1`` immutable node
    IDs. Every sample must include explicit root node ``0``.
    Cannot-cluster IDs are parsed under the same immutable ID contract
    and may therefore range through ``2N - 2`` rather than stopping at
    ``N - 1``.

    Args:
        base_path: Directory containing the required source files.
        params: Validated user-defined parameters.
        cannot_clusters_path: Optional path to the cannot-clusters
            definition file.

    Returns:
        Typed algorithm data containing validated immutable hierarchy
        data.

    Raises:
        FileNotFoundError: If a required source file is missing.
        ValueError: If data values, dimensions, identifiers, hierarchy
            memberships, or parameters are invalid or inconsistent.
    """
    file_names = {
        "genes": MATRIX_GENES_FILE,
        "columns": MATRIX_SAMPLES_FILE,
        "values": MATRIX_VALUES_FILE,
        "variances": TERM_VARIANCES_FILE,
        "hierarchy": SAMPLE_LINEAGES_FILE,
    }

    file_paths = {
        name: os.path.join(base_path, fname)
        for name, fname in file_names.items()
    }

    for name, path in file_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing required file: {file_names[name]}"
            )

    if (Path(base_path) / ".compute_variances_prepare_in_progress").exists():
        raise ValueError("Input preparation is incomplete; rerun prepare")
    verify_artifacts(Path(path) for path in file_paths.values())
    genes = read_identifiers(Path(file_paths["genes"]), label="gene")
    samples = read_identifiers(Path(file_paths["columns"]), label="sample")

    if any(not gene for gene in genes):
        raise ValueError("Empty gene name detected.")

    if any(not sample for sample in samples):
        raise ValueError("Empty sample ID detected.")

    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate gene names detected.")

    if len(set(samples)) != len(samples):
        raise ValueError("Duplicate sample IDs detected.")

    gene_count = len(genes)
    sample_count = len(samples)
    node_count = _tree_node_count(sample_count)
    max_tree_node_id = node_count - 1

    expression_rows: list[list[int]] = []
    for number, line in (
        enumerate(read_utf8_lines(Path(file_paths["values"])), 1)
    ):
        tokens = line.split()
        if len(tokens) != sample_count:
            raise ValueError(f"matrix_values.txt:{number}: wrong sample count")
        expression_rows.append([
            parse_canonical_int(token,
                source=f"matrix_values.txt:{number}:{column}")
            for column, token in enumerate(tokens, 1)
        ])
    expression_values_df = pd.DataFrame(expression_rows)

    expression_values_df = expression_values_df.apply(
        pd.to_numeric, errors="coerce"
    )

    if expression_values_df.isna().any().any():
        invalid_positions = np.argwhere(expression_values_df.isna().to_numpy())
        first_row, first_column = invalid_positions[0]

        raise ValueError(
            "Non-numeric expression value detected in matrix_values.txt "
            f"at row {first_row + 1}, column {first_column + 1}."
        )

    values_arr_float = expression_values_df.to_numpy(dtype=np.float64)

    if not np.isfinite(values_arr_float).all():
        raise ValueError(
            "Non-finite expression values detected in matrix_values.txt."
        )

    if not np.equal(values_arr_float, np.floor(values_arr_float)).all():
        raise ValueError("Expression values must be integers.")

    int32_info = np.iinfo(np.int32)
    values_min = float(values_arr_float.min())
    values_max = float(values_arr_float.max())
    expression_dtype = (
        np.int32
        if values_min >= int32_info.min and values_max <= int32_info.max
        else np.int64
    )
    expression_values_df = expression_values_df.astype(expression_dtype)

    validate_variance_output(Path(file_paths["variances"]), Path(base_path))
    variance_rows = [
        line.split() for line in read_utf8_lines(Path(file_paths["variances"]))
    ]
    variances = pd.DataFrame(variance_rows)

    if expression_values_df.shape[0] != gene_count:
        raise ValueError(
            f"Mismatch: {gene_count} genes vs "
            f"{expression_values_df.shape[0]} value rows."
        )

    if expression_values_df.shape[1] != sample_count:
        raise ValueError(
            f"Mismatch: {sample_count} samples vs "
            f"{expression_values_df.shape[1]} value columns."
        )

    expression_values_df.index = genes
    expression_values_df.columns = samples

    sample_set = set(samples)
    hierarchies: dict[str, set[str]] = {}
    observed_node_ids: set[str] = set()

    hierarchy_file = read_utf8_lines(Path(file_paths["hierarchy"]))
    if True:
        for row_number, line in enumerate(hierarchy_file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                raise ValueError(
                    "Empty hierarchy row detected at physical line "
                    f"{row_number}."
                )

            row = stripped_line.split()
            if len(row) < 2:
                raise ValueError(
                    f"Hierarchy row {row_number} must contain a sample ID "
                    "and cluster count."
                )

            sample = row[0]
            if sample not in sample_set:
                raise ValueError(
                    f"Hierarchy row {row_number} names unknown sample "
                    f"{sample!r}."
                )
            if sample in hierarchies:
                raise ValueError(
                    f"Hierarchy row {row_number} duplicates sample {sample!r}."
                )

            try:
                declared_count = parse_canonical_int(
                    row[1], source=f"sample_lineages.txt:{row_number}:2"
                )
            except ValueError as exc:
                raise ValueError(
                    f"Hierarchy row {row_number} has a non-integer "
                    f"cluster count: {row[1]!r}."
                ) from exc

            if declared_count < 1:
                raise ValueError(
                    f"Hierarchy row {row_number} must contain at least root "
                    "node 0."
                )
            if declared_count > sample_count:
                raise ValueError(
                    f"Hierarchy row {row_number} declares {declared_count} "
                    f"nodes, exceeding the maximum root-to-leaf path length "
                    f"of {sample_count} for {sample_count} samples."
                )

            raw_cluster_ids = row[2:]
            if declared_count != len(raw_cluster_ids):
                raise ValueError(
                    f"Hierarchy row {row_number} declares {declared_count} "
                    f"cluster(s) but contains {len(raw_cluster_ids)}."
                )

            normalized_cluster_ids = [
                _parse_tree_node_id(
                    cluster_id,
                    source=f"hierarchy row {row_number}",
                    max_node_id=max_tree_node_id,
                )
                for cluster_id in raw_cluster_ids
            ]

            if len(normalized_cluster_ids) != len(set(normalized_cluster_ids)):
                raise ValueError(
                    f"Hierarchy row {row_number} contains duplicate "
                    "immutable node IDs."
                )

            cluster_set = set(normalized_cluster_ids)
            if "0" not in cluster_set:
                raise ValueError(
                    f"Hierarchy row {row_number} for sample {sample!r} does "
                    "not contain immutable root node 0."
                )

            hierarchies[sample] = cluster_set
            observed_node_ids.update(cluster_set)

    missing_samples = (
        [sample for sample in samples if sample not in hierarchies]
    )
    if missing_samples:
        raise ValueError(
            "Hierarchy is missing row(s) for sample(s): "
            f"{missing_samples}."
        )

    expected_node_ids = {str(node_id) for node_id in range(node_count)}
    missing_node_ids = expected_node_ids - observed_node_ids
    if missing_node_ids:
        missing_numeric = sorted(int(node_id) for node_id in missing_node_ids)
        preview = missing_numeric[:20]
        suffix = "..." if len(missing_numeric) > len(preview) else ""
        raise ValueError(
            "sample_lineages.txt does not represent a complete "
            "fully immutable "
            f"tree: expected every node ID 0..{max_tree_node_id}, but missing "
            f"{preview}{suffix}."
        )

    if variances.shape[1] < 4:
        raise ValueError(
            "term_variances.txt must have ≥4 columns "
            "(gene, value, operator, variance)."
        )

    variances = variances.iloc[:, :4].copy()
    variances.columns = ["gene", "value", "operator", "variance"]
    variances["gene"] = variances["gene"].astype(str)
    variances["value"] = pd.to_numeric(variances["value"], errors="coerce")
    variances["variance"] = pd.to_numeric(
        variances["variance"], errors="coerce"
    )
    variances["operator"] = pd.to_numeric(
        variances["operator"], errors="coerce"
    )

    if variances["value"].isna().any():
        raise ValueError(
            "Non-numeric transcription-level values found in "
            "term_variances.txt."
        )

    if variances["variance"].isna().any():
        raise ValueError(
            "Non-numeric values of variances found in term_variances.txt."
        )

    if variances["operator"].isna().any():
        raise ValueError(
            "Non-numeric operator codes found in term_variances.txt"
        )

    numeric_variance_data = variances[
        ["value", "operator", "variance"]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_variance_data).all():
        raise ValueError(
            "Non-finite value, operator, or variance detected "
            "in term_variances.txt."
        )

    operator_values = variances["operator"].to_numpy(dtype=np.float64)

    if not np.equal(operator_values, np.floor(operator_values)).all():
        invalid_operator_values = sorted(
            variances.loc[
                variances["operator"] != np.floor(variances["operator"]),
                "operator",
            ].unique()
        )

        raise ValueError(
            "Operator codes must be integers. "
            f"Invalid values: {invalid_operator_values}."
        )

    variances["operator"] = variances["operator"].astype(np.int64)

    allowed_operator_codes = {0, 1, 2}
    invalid_operator_codes = (
        set(variances["operator"].unique()) - allowed_operator_codes
    )

    if invalid_operator_codes:
        raise ValueError(
            "Unsupported operator codes in term_variances.txt: "
            f"{sorted(invalid_operator_codes)}. "
            f"Expected only {sorted(allowed_operator_codes)}."
        )

    # Code 0 denotes equality and is intentionally not used to generate
    # terms.
    variances["operator"] = variances["operator"].map(
        {
            0: "==",
            1: "<",
            2: ">",
        }
    )

    variances = variances[variances["operator"].isin({"<", ">"})].copy()

    cannot_clusters: set[str] | None = None

    if cannot_clusters_path:
        if not os.path.isfile(cannot_clusters_path):
            raise FileNotFoundError(
                f"Cannot-clusters file not found: {cannot_clusters_path}"
            )

        constraint = Path(cannot_clusters_path)
        raw = "\n".join(read_utf8_lines(
            constraint, allow_blank=True, allow_empty=True
        ))
        if provenance_path(constraint).is_file():
            verify_artifacts([constraint])

        if raw:
            tokens = raw.replace(",", " ").replace(";", " ").split()
            normalized_tokens = [
                _parse_tree_node_id(
                    token,
                    source=f"cannot-clusters file {cannot_clusters_path!r}",
                    max_node_id=max_tree_node_id,
                )
                for token in tokens
            ]
            if len(normalized_tokens) != len(set(normalized_tokens)):
                raise ValueError(
                    (
                        'Cannot-clusters file contains duplicate '
                        'immutable node IDs.'
                    )
                )
            cannot_clusters = set(normalized_tokens)

    if cannot_clusters:
        invalid = cannot_clusters - observed_node_ids
        if invalid:
            raise ValueError(
                "Invalid cannot-cluster IDs not found in immutable hierarchy: "
                f"{sorted(invalid, key=int)}"
            )

    if params["max_terms"] > gene_count:
        raise ValueError(
            f"Antecedent length ({params['max_terms']}) "
            f"exceeds number of genes ({gene_count})."
        )

    if params["min_sample_coverage"] > sample_count:
        raise ValueError(
            f"Minimum sample coverage ({params['min_sample_coverage']}) "
            f"exceeds number of samples ({sample_count})."
        )

    if params["max_uncovered_samples"] > sample_count:
        raise ValueError(
            "Max uncovered samples threshold exceeds dataset size."
        )

    expression_values = expression_values_df.to_numpy(copy=False)

    data: AlgorithmData = {
        "genes": genes,
        "samples": samples,
        "expression_values": expression_values,
        "variances": variances,
        "hierarchies": hierarchies,
        "cannot_clusters": cannot_clusters,
        "sample_count": sample_count,
        "node_count": node_count,
    }

    print(
        f"[Data] Loaded and validated {gene_count} genes x "
        f"{sample_count} samples "
        f"with {node_count} immutable tree nodes "
        f"(IDs 0..{max_tree_node_id}; expression dtype: "
        f"{expression_values.dtype}).", file=sys.stderr)

    return data


# ======================================================================
# Block 4: Candidate Data Model
# ======================================================================
@dataclass
class Antecedent:
    """Represents a mutable candidate antecedent built by an ant.

    Attributes:
        term_ids: Ordered term IDs in the candidate antecedent.
        covered_indices: Indices of samples satisfied by the antecedent.
        coverage_words: Optional packed ``uint64`` sample-coverage
            bitset.
        selectivity: Current selectivity score.
        quality: Current quality score.
        cannot_cluster_share: Share of covered samples belonging to
            cannot clusters.
        home_cluster_index: Matrix index of the assigned home cluster;
            in the immutable layout this is also the tree-node ID.
        home_cluster_id: Canonical decimal immutable node-ID string.
        ant_launch_order: Ant generation order used for tie-breaking.
    """

    term_ids: list[int] = field(default_factory=list)
    covered_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )
    coverage_words: np.ndarray | None = None
    selectivity: float = 0.0
    quality: float = 0.0
    cannot_cluster_share: float = 0.0
    home_cluster_index: int = -1
    home_cluster_id: str = ""
    ant_launch_order: int = 0


# ======================================================================
# Block 5: Shared Computational Utilities
# ======================================================================
_POPCOUNT_TABLE = (
    np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None],
        axis=1,
    )
    .sum(axis=1)
    .astype(np.uint8)
)

_OPERATOR_SYMBOLS = {
    "0": "==",
    "1": "<",
    "2": ">",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "=": "==",
    "==": "==",
}


def normalize_operator(
    op: Any,
    allowed: set[str] | None = None,
) -> str:
    """Normalize an operator token to a standard comparison symbol.

    Args:
        op: Operator token, such as ``1``, ``"1.0"``, ``"<"``, or
            ``">="``.
        allowed: Optional set of permitted normalized symbols.

    Returns:
        The normalized comparison symbol.

    Raises:
        ValueError: If the operator is missing, unsupported, or not
            permitted.
    """
    token = str(op).strip().strip('"').strip("'")
    if not token:
        raise ValueError("Operator cannot be empty.")

    try:
        numeric_value = float(token)
        if numeric_value.is_integer():
            token = str(int(numeric_value))
    except ValueError:
        pass

    symbol = _OPERATOR_SYMBOLS.get(token)
    if symbol is None:
        raise ValueError(f"Unsupported operator: {op!r}")

    if allowed is not None and symbol not in allowed:
        raise ValueError(
            f"Operator {symbol!r} is not permitted here; "
            f"expected one of {sorted(allowed)}."
        )

    return symbol


def _validate_sample_indices(
    indices: np.ndarray,
    sample_count: int,
    name: str = "covered_indices",
) -> np.ndarray:
    """Validates, deduplicates, and sorts sample indices.

    Args:
        indices: Candidate sample indices.
        sample_count: Total number of samples.
        name: Name used in validation messages.

    Returns:
        Unique sorted indices using NumPy's platform index dtype.

    Raises:
        TypeError: If the input does not contain integers.
        ValueError: If the input is not one-dimensional.
        IndexError: If any index is outside ``[0, sample_count)``.
    """
    array = np.asarray(indices)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if array.size == 0:
        return np.empty(0, dtype=np.intp)

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices.")

    array = array.astype(np.intp, copy=False)

    invalid = array[(array < 0) | (array >= sample_count)]

    if invalid.size:
        raise IndexError(
            f"{name} contains out-of-range sample indices: "
            f"{sorted(np.unique(invalid).tolist())}."
        )

    return np.unique(array)


def _pack_sample_mask(mask: np.ndarray, sample_count: int) -> np.ndarray:
    """Pack a Boolean sample mask into unsigned 64-bit words.

    Args:
        mask: Boolean sample mask in matrix-column order.
        sample_count: Expected number of samples.

    Returns:
        Packed words with unused tail bits cleared.

    Raises:
        ValueError: If the sample mask has an invalid shape.
    """
    boolean_mask = np.asarray(mask, dtype=np.bool_)
    if boolean_mask.shape != (sample_count,):
        raise ValueError(
            f"Sample mask must have shape ({sample_count},), "
            f"not {boolean_mask.shape}."
        )

    packed_bytes = np.packbits(boolean_mask, bitorder="little")
    word_count = (sample_count + 63) // 64
    padded_bytes = np.zeros(word_count * 8, dtype=np.uint8)
    padded_bytes[: packed_bytes.size] = packed_bytes
    return padded_bytes.view(np.uint64)


def _coverage_words_to_indices(
    coverage_words: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Return sample indices represented by packed coverage words.

    Args:
        coverage_words: Packed unsigned 64-bit sample-membership words.
        sample_count: Number of meaningful sample bits.

    Returns:
        Covered sample indices in ascending matrix-column order.
    """
    words = np.asarray(coverage_words, dtype=np.uint64)
    if words.ndim != 1:
        raise ValueError("coverage_words must be one-dimensional.")

    unpacked = np.unpackbits(words.view(np.uint8), bitorder="little")
    return np.flatnonzero(unpacked[:sample_count]).astype(np.intp, copy=False)


def _count_coverage_words(coverage_words: np.ndarray) -> int:
    """Count sample bits without unpacking a Boolean vector."""
    words = np.asarray(coverage_words, dtype=np.uint64)
    if words.ndim != 1:
        raise ValueError("coverage_words must be one-dimensional.")

    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(words).sum(dtype=np.uint64))

    return int(_POPCOUNT_TABLE[words.view(np.uint8)].sum(dtype=np.uint64))


def _is_sparse_matrix(matrix: Any) -> bool:
    """Return whether matrix is a SciPy sparse matrix."""
    return scipy_sparse is not None and scipy_sparse.issparse(matrix)


def _require_2d_matrix(matrix: Any, *, name: str) -> tuple[int, int]:
    """Validate a matrix-like object and returns its dimensions.

    Args:
        matrix: Dense or sparse matrix-like object.
        name: Name used in validation messages.

    Returns:
        The matrix row and column counts.

    Raises:
        ValueError: If the object does not expose a two-dimensional
            shape.
    """
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be two-dimensional.")

    return int(shape[0]), int(shape[1])


def _require_boolean_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a Boolean mask and its expected sample count.

    Args:
        mask: Candidate Boolean mask.
        name: Name used in validation messages.

    Returns:
        The validated NumPy array.

    Raises:
        ValueError: If the mask is not one-dimensional or Boolean.
    """
    array = np.asarray(mask)
    if array.ndim != 1 or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a one-dimensional boolean array.")

    return array


def _sum_cluster_memberships(
    binary_matrix: Any,
    covered_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Sum cluster memberships for selected samples.

    Args:
        binary_matrix: Dense or sparse sample-by-cluster membership
            matrix.
        covered_indices: Optional sample subset. All samples are used
            when omitted.

    Returns:
        One integer membership count per cluster.

    Raises:
        TypeError: If sample indices are not integers.
        ValueError: If the matrix or sample-index shape is invalid.
        IndexError: If a sample index is out of range.
    """
    sample_count, _ = _require_2d_matrix(
        binary_matrix,
        name="binary_matrix",
    )

    if covered_indices is not None:
        indices = _validate_sample_indices(
            covered_indices,
            sample_count,
        )
        selected = binary_matrix[indices]
    else:
        selected = binary_matrix

    if _is_sparse_matrix(selected):
        counts = np.asarray(selected.sum(axis=0, dtype=np.int64)).ravel()
    else:
        counts = np.asarray(selected).sum(axis=0, dtype=np.int64)

    return np.asarray(counts, dtype=np.int64).ravel()


def compute_selectivity(
    binary_matrix: Any,
    covered_indices: np.ndarray,
    global_msd: float,
    cluster_sample_counts: np.ndarray | None = None,
) -> float:
    """Compute selectivity from cluster-membership statistics.

    For a binary cluster feature, population variance is ``p * (1 -
    p)``. Per-cluster counts are therefore sufficient; no uncovered
    matrix is built.

    Args:
        binary_matrix: Dense or sparse sample-by-cluster membership
            matrix.
        covered_indices: Samples covered by the candidate antecedent.
        global_msd: Mean squared distance for the complete dataset.
        cluster_sample_counts: Optional cached total membership counts.

    Returns:
        The candidate's selectivity score.

    Raises:
        ValueError: If matrix dimensions or cached counts are invalid.
        RuntimeError: If the computed score is non-finite.
    """
    sample_count, cluster_count = _require_2d_matrix(
        binary_matrix,
        name="binary_matrix",
    )
    if sample_count == 0:
        return 0.0

    covered_indices = _validate_sample_indices(
        covered_indices,
        sample_count,
    )
    covered_count = int(covered_indices.size)
    uncovered_count = sample_count - covered_count

    if covered_count == 0 or uncovered_count == 0:
        return 0.0

    if cluster_sample_counts is None:
        total_counts = _sum_cluster_memberships(binary_matrix)
    else:
        total_counts = np.asarray(cluster_sample_counts, dtype=np.int64)
        if total_counts.shape != (cluster_count,):
            raise ValueError(
                "cluster_sample_counts must contain one value per cluster."
            )

    covered_counts = _sum_cluster_memberships(
        binary_matrix,
        covered_indices,
    )
    uncovered_counts = total_counts - covered_counts

    covered_proportions = covered_counts.astype(np.float64) / covered_count
    uncovered_proportions = (
        uncovered_counts.astype(np.float64) / uncovered_count
    )

    covered_msd = float(
        np.sum(covered_proportions * (1.0 - covered_proportions))
    )
    uncovered_msd = float(
        np.sum(uncovered_proportions * (1.0 - uncovered_proportions))
    )

    score = global_msd - (
        (covered_count / sample_count) * covered_msd
        + (uncovered_count / sample_count) * uncovered_msd
    )

    if not np.isfinite(score):
        raise RuntimeError(
            "Selectivity calculation produced a non-finite value: " f"{score}."
        )

    return float(score)


def antecedent_priority_key(
    antecedent: Antecedent,
) -> tuple[float, float, int, int]:
    """Build the common four-criterion antecedent ranking key.

    Args:
        antecedent: Scored antecedent with coverage and contamination.

    Returns:
        Selectivity, negated cannot share, negated length, and coverage.
        Larger keys have higher priority.
    """
    return (
        antecedent.selectivity,  # Primary: Maximize selectivity
        # Tie 1: Minimize cannot-cluster share.
        -antecedent.cannot_cluster_share,
        -len(antecedent.term_ids),  # Tie 2: Minimize antecedent length
        len(
            antecedent.covered_indices
        ),  # Tie 3: Maximize overall sample coverage
    )


# ======================================================================
# Block 6: Compose Binary Vectors
# ======================================================================
def compose_binary_vectors(data: AlgorithmData) -> None:
    """Build canonical sample-by-immutable-node membership matrix.

    Build the canonical sample-by-immutable-node membership matrix.

    For ``N`` samples, the fully immutable tree has exactly ``2N - 1``
    nodes. Matrix columns are fixed in canonical numeric order
    ``0..2N-2``; therefore a matrix column index is also the immutable
    tree-node ID. This removes the old dependence on lexicographic
    string sorting and allows downstream code to use IDs above ``N - 1``
    without remapping ambiguity.

    The computational representation is a CSR matrix when SciPy is
    available and a dense ``uint8`` array otherwise.

    Args:
        data: Validated immutable source data. The function stores
            ``binary_matrix`` and canonical ``cluster_ids`` in derived
            state.

    Raises:
        ValueError: If sample/hierarchy identifiers or immutable-node
            coverage are inconsistent with the ``2N - 1`` contract.
    """
    hierarchies = data["hierarchies"]
    samples = data["samples"]
    sample_count = int(data["sample_count"])
    node_count = int(data["node_count"])

    if sample_count != len(samples):
        raise ValueError(
            f"Inconsistent sample count: sample_count={sample_count}, "
            f"but len(samples)={len(samples)}."
        )
    if len(set(samples)) != len(samples):
        raise ValueError("Duplicate sample IDs detected.")

    expected_node_count = _tree_node_count(sample_count)
    if node_count != expected_node_count:
        raise ValueError(
            f"Inconsistent immutable node count: node_count={node_count}, "
            f"but {sample_count} samples require {expected_node_count}."
        )

    sample_ids = set(samples)
    hierarchy_ids = set(hierarchies)
    missing_ids = sample_ids - hierarchy_ids
    extra_ids = hierarchy_ids - sample_ids
    if missing_ids or extra_ids:
        raise ValueError(
            "Samples and hierarchy entries are inconsistent. "
            f"Missing entries: {sorted(missing_ids)}; "
            f"unexpected entries: {sorted(extra_ids)}."
        )

    expected_clusters = {str(node_id) for node_id in range(node_count)}
    all_clusters: set[str] = set()
    for sample_id in samples:
        clusters = hierarchies[sample_id]
        if clusters is None:
            raise ValueError(
                f"Hierarchy entry for sample '{sample_id}' is None."
            )
        if "0" not in clusters:
            raise ValueError(
                f"Hierarchy entry for sample '{sample_id}' is missing root 0."
            )
        all_clusters.update(clusters)

    missing_clusters = expected_clusters - all_clusters
    unexpected_clusters = all_clusters - expected_clusters
    if missing_clusters or unexpected_clusters:
        raise ValueError(
            "Hierarchy does not match the complete immutable node domain. "
            f"Missing nodes: {sorted(missing_clusters, key=int)}; "
            f"unexpected nodes: {sorted(unexpected_clusters, key=int)}."
        )

    sorted_clusters = [str(node_id) for node_id in range(node_count)]
    cluster_to_idx = {
        cluster: index for index, cluster in enumerate(sorted_clusters)
    }

    row_indices: list[int] = []
    column_indices: list[int] = []
    for sample_idx, sample_id in enumerate(samples):
        for cluster in hierarchies[sample_id]:
            row_indices.append(sample_idx)
            column_indices.append(cluster_to_idx[cluster])

    if scipy_sparse is not None:
        membership_values = np.ones(len(row_indices), dtype=np.uint8)
        binary_matrix = scipy_sparse.csr_matrix(
            (membership_values, (row_indices, column_indices)),
            shape=(sample_count, node_count),
            dtype=np.uint8,
        )
        binary_matrix.sum_duplicates()
        binary_matrix.sort_indices()
        backend = "CSR sparse"
    else:
        binary_matrix = np.zeros(
            (sample_count, node_count),
            dtype=np.uint8,
        )
        if row_indices:
            binary_matrix[
                np.asarray(row_indices, dtype=np.intp),
                np.asarray(column_indices, dtype=np.intp),
            ] = 1
        backend = "dense uint8 fallback"

    data["binary_matrix"] = binary_matrix
    data["cluster_ids"] = sorted_clusters

    density = (
        len(row_indices) / (sample_count * node_count)
        if sample_count and node_count
        else 0.0
    )
    print(
        f"[Clusters] Built {backend} immutable membership matrix "
        f"{binary_matrix.shape} with node IDs 0..{node_count - 1} "
        f"(density: {density:.2%})."
    )


def _binary_vectors_to_dataframe(data: AlgorithmData) -> pd.DataFrame:
    """Build a DataFrame view of the cached membership matrix.

    Args:
        data: Algorithm state containing ``binary_matrix``, ``samples``,
            and ``cluster_ids``.

    Returns:
        A sparse or dense DataFrame matching the matrix backend.

    Raises:
        ValueError: If the membership matrix or cluster names are
            unavailable.
    """
    binary_matrix = data.get("binary_matrix")
    cluster_ids = data.get("cluster_ids")
    if binary_matrix is None or cluster_ids is None:
        raise ValueError(
            "Binary vectors are unavailable. Run compose_binary_vectors first."
        )

    if _is_sparse_matrix(binary_matrix):
        return pd.DataFrame.sparse.from_spmatrix(
            binary_matrix,
            index=data["samples"],
            columns=cluster_ids,
        )

    return pd.DataFrame(
        np.asarray(binary_matrix),
        index=data["samples"],
        columns=cluster_ids,
        dtype=np.uint8,
    )


def inspect_binary_vectors(data: AlgorithmData) -> None:
    """Offer interactive inspection of the cached binary vectors.

    The DataFrame view is constructed lazily only when this function is
    called.

    Args:
        data: Algorithm state containing the cached membership matrix.
    """
    binary_df = _binary_vectors_to_dataframe(data)
    if binary_df.empty:
        print("  - No binary vectors found to inspect.")
        return

    while True:
        print("\nHow would you like to inspect the binary vectors DataFrame?")
        print("  1) Inspect a single sample's vector")
        print("  2) Save the entire DataFrame to a CSV file")
        print("  3) Preview the first 10 rows (wide output is truncated)")
        print("  4) Continue without inspection")
        choice = prompt_input("Enter your choice (1-4): ").strip()
        if choice == "1":
            sample_id = (
                prompt_input("Enter the sample ID to inspect: ").strip()
            )
            if sample_id in binary_df.index:
                print(
                    f"\nVector for sample '{sample_id}':\n"
                    f"{binary_df.loc[sample_id]}"
                )
            else:
                print(f"  - Error: Sample ID '{sample_id}' not found.",
                    file=sys.stderr)
        elif choice == "2":
            output_path = (
                prompt_input((
                    'Enter output CSV path [binary_vectors.csv]: '
                )).strip()
                or BINARY_VECTORS_FILE
            )
            try:
                binary_df.to_csv(output_path, encoding="utf-8")
                print(
                    "  - Successfully saved vectors for "
                    f"{len(binary_df)} samples to '{output_path}'."
                )
            except OSError as exc:
                print(f"  - Error saving file: {exc}")
            break
        elif choice == "3":
            print("\nBinary-vector preview (up to 10 rows):")
            with pd.option_context(
                "display.max_columns",
                20,
                "display.width",
                160,
            ):
                print(binary_df.head(10))
            break
        elif choice == "4":
            print("Continuing without vector inspection.")
            break
        else:
            print((
                '  - Invalid choice. Please enter a number '
                'between 1 and 4.'
            ), file=sys.stderr)


# ======================================================================
# Block 7: Generate All Possible Terms
# ======================================================================
def generate_all_possible_terms(
    data: AlgorithmData,
    params: RunParameters,
) -> tuple[list[TermRecord], dict[str, list[int]]]:
    """Generate all discretized terms from validated expression data.

    The variances table is indexed once by ``(gene, value)`` so the
    inner gene/value loop does not repeatedly filter and sort a
    DataFrame.

    Args:
        data: Validated algorithm payload.
        params: User parameters containing ``max_bins``.

    Returns:
        A flat term list and a gene-to-term-ID mapping.

    Raises:
        ValueError: If the max_bins or required variance data is
            invalid.
    """
    genes: list[str] = data["genes"]
    expression_values = np.asarray(data["expression_values"])
    variances: pd.DataFrame = data["variances"].copy()

    max_bins = int(params["max_bins"])
    if max_bins <= 0:
        raise ValueError("Discretization bins must be a positive integer.")

    required_columns = ["gene", "value", "operator", "variance"]
    missing_columns = set(required_columns) - set(variances.columns)
    if missing_columns:
        raise ValueError(
            "Variances data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if expression_values.ndim != 2 or expression_values.shape[0] != len(genes):
        raise ValueError(
            "expression_values must be a two-dimensional "
            "gene-by-sample matrix "
            "aligned with data['genes']."
        )

    variances = variances.loc[:, required_columns].copy()
    variances["gene"] = variances["gene"].astype(str)
    variances["value"] = pd.to_numeric(variances["value"], errors="coerce")
    variances["variance"] = pd.to_numeric(
        variances["variance"],
        errors="coerce",
    )
    variances["operator"] = variances["operator"].apply(
        lambda op: normalize_operator(op, allowed={"<", ">"})
    )

    if variances[["value", "variance"]].isna().any().any():
        raise ValueError("Variances data contains non-numeric values.")

    variances = variances.sort_values(
        by=["gene", "value", "operator", "variance"],
        kind="mergesort",
    )

    variance_rows_by_key: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for row in variances.itertuples(index=False):
        key = (str(row.gene), int(row.value))
        variance_rows_by_key.setdefault(key, []).append(
            (str(row.operator), float(row.variance))
        )

    def choose_representative_integer(
        candidate_ints: np.ndarray,
        midpoint: float,
    ) -> int:
        """Choose nearest observed integer, breaking ties downward.

        Choose the nearest observed integer, breaking ties downward.

        Args:
            candidate_ints: Nonempty observed values within one bin.
            midpoint: Mathematical midpoint of that bin.

        Returns:
            Nearest observed integer; the smaller value wins a distance
            tie.
        """
        distances = np.abs(candidate_ints.astype(np.float64) - midpoint)
        best_distance = float(np.min(distances))
        tied = candidate_ints[
            np.isclose(
                distances,
                best_distance,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        return int(np.min(tied))

    all_terms: list[TermRecord] = []
    terms_by_gene: dict[str, list[int]] = {}
    seen_term_keys: set[tuple[str, str, int]] = set()
    term_id_counter = 0

    for gene_index, gene in enumerate(genes):
        gene_vals = expression_values[gene_index]
        observed_values = np.unique(gene_vals)
        if observed_values.size == 0:
            continue

        min_v = float(observed_values[0])
        max_v = float(observed_values[-1])

        if observed_values.size <= max_bins:
            candidate_values = observed_values.astype(np.int64).tolist()
        else:
            width = (max_v - min_v) / max_bins
            candidate_values: list[int] = []

            for k in range(1, max_bins + 1):
                left = min_v + (k - 1) * width
                right = max_v if k == max_bins else min_v + k * width

                start = int(
                    np.searchsorted(observed_values, left, side="left")
                )
                stop = (
                    observed_values.size
                    if k == max_bins
                    else int(
                        np.searchsorted(
                            observed_values,
                            right,
                            side="left",
                        )
                    )
                )
                bin_values = observed_values[start:stop]
                if bin_values.size == 0:
                    continue

                midpoint = (left + right) / 2.0
                candidate_values.append(
                    choose_representative_integer(bin_values, midpoint)
                )

            candidate_values = list(dict.fromkeys(candidate_values))

        gene_terms: list[int] = []

        for term_value in candidate_values:
            rows = variance_rows_by_key.get((gene, int(term_value)), ())

            for op, variance in rows:
                term_val = int(term_value)

                if op == "<" and term_val <= min_v:
                    continue
                if op == ">" and term_val >= max_v:
                    continue

                key = (gene, op, term_val)
                if key in seen_term_keys:
                    continue
                seen_term_keys.add(key)

                all_terms.append(
                    {
                        "id": term_id_counter,
                        "gene": gene,
                        "gene_index": gene_index,
                        "operator": op,
                        "value": term_val,
                        "variance": variance,
                    }
                )
                gene_terms.append(term_id_counter)
                term_id_counter += 1

        if gene_terms:
            terms_by_gene[gene] = gene_terms

    print(
        f"[Terms] Generated {len(all_terms)} terms across "
        f"{len(terms_by_gene)} genes (bin limit: {max_bins}).",
        file=sys.stderr,
    )
    if not all_terms:
        print(
            "  - Warning: No terms were generated. Check term_variances.txt "
            "and the observed value ranges.", file=sys.stderr)

    data["terms"] = all_terms
    data["terms_by_gene"] = terms_by_gene
    data["term_count"] = len(all_terms)

    return all_terms, terms_by_gene


def precompute_term_coverage_matrix(data: AlgorithmData) -> None:
    """Precompute packed coverage and per-term statistics.

    One Boolean value per sample would consume one byte. Packing 64
    sample flags per ``uint64`` word reduces coverage storage by
    approximately eightfold while preserving vectorized bitwise
    intersections. Results are committed only after the full
    construction has passed validation.

    Args:
        data: Algorithm state containing expression values, generated
            terms, and optional binary cluster memberships.

    Raises:
        ValueError: If term IDs, operators, or sample coverage are
            invalid.
        RuntimeError: If any term result remains uninitialized or
            non-finite.
    """
    expression_values = np.asarray(data["expression_values"])
    terms = data["terms"]
    sample_count = int(data["sample_count"])
    term_count = int(data["term_count"])
    word_count = (sample_count + 63) // 64

    coverage_words = np.zeros(
        (term_count, word_count),
        dtype=np.uint64,
    )
    coverage_counts = np.full(
        term_count,
        -1,
        dtype=np.int64,
    )
    binary_matrix = data.get("binary_matrix")
    term_variances = (
        np.full(term_count, np.nan, dtype=np.float64)
        if binary_matrix is not None
        else None
    )
    initialized = np.zeros(term_count, dtype=np.bool_)

    empty_term_ids: list[int] = []

    for term in terms:
        term_id = int(term["id"])

        if not 0 <= term_id < term_count:
            raise ValueError(
                f"Term ID {term_id} is outside the expected range "
                f"0 through {term_count - 1}."
            )

        if initialized[term_id]:
            raise ValueError(f"Duplicate term ID detected: {term_id}.")

        gene_values = expression_values[int(term["gene_index"])]
        threshold = int(term["value"])
        op = term["operator"]

        if op == "<":
            covered_mask = gene_values < threshold
        elif op == ">":
            covered_mask = gene_values > threshold
        else:
            raise ValueError(
                f"Unsupported generated-term operator {op!r} "
                f"for term {term_id}."
            )

        covered_count = int(np.count_nonzero(covered_mask))

        if covered_count == 0:
            empty_term_ids.append(term_id)
            continue

        coverage_words[term_id] = _pack_sample_mask(
            covered_mask,
            sample_count,
        )
        coverage_counts[term_id] = covered_count

        if term_variances is not None:
            covered_indices = np.flatnonzero(covered_mask)
            covered_cluster_counts = _sum_cluster_memberships(
                binary_matrix,
                covered_indices,
            )
            proportions = (
                covered_cluster_counts.astype(np.float64) / covered_count
            )
            term_variances[term_id] = float(
                np.sum(proportions * (1.0 - proportions))
            )

        initialized[term_id] = True

    # Validate all locally constructed results before modifying data.
    if empty_term_ids:
        raise ValueError(
            "Generated terms with zero sample coverage detected: "
            f"{empty_term_ids}."
        )

    if not initialized.all():
        missing = np.flatnonzero(~initialized).tolist()
        raise RuntimeError(
            "Coverage precomputation did not initialize terms: " f"{missing}."
        )

    if np.any(coverage_counts <= 0):
        invalid_count_ids = np.flatnonzero(coverage_counts <= 0).tolist()
        raise RuntimeError(
            "Coverage precomputation produced nonpositive coverage "
            f"counts for terms: {invalid_count_ids}."
        )

    if term_variances is not None and not np.isfinite(term_variances).all():
        invalid_msd_ids = np.flatnonzero(~np.isfinite(term_variances)).tolist()
        raise RuntimeError(
            "Coverage precomputation produced non-finite MSD "
            f"values for terms: {invalid_msd_ids}."
        )

    # Commit only after the complete construction has been validated.
    data["coverage_words"] = coverage_words
    data["term_coverage_counts"] = coverage_counts

    if term_variances is not None:
        data["term_variances"] = term_variances
    else:
        data.pop("term_variances", None)

    dense_bytes = term_count * sample_count * np.dtype(np.bool_).itemsize
    packed_bytes = int(coverage_words.nbytes)

    if packed_bytes < dense_bytes:
        storage_note = (
            f"{dense_bytes / packed_bytes:.2f}x smaller than dense bool"
        )
    elif packed_bytes > dense_bytes:
        storage_note = (
            f"{packed_bytes / dense_bytes:.2f}x larger than dense bool "
            "for this small dataset"
        )
    else:
        storage_note = "the same size as dense bool"

    print(
        f"[Coverage] Packed uint64 matrix {coverage_words.shape} "
        f"({packed_bytes} bytes; "
        f"{storage_note}).", file=sys.stderr)


def inspect_terms(
    terms: list[TermRecord],
    terms_by_gene: dict[str, list[int]],
    max_examples: int = 10,
) -> None:
    """Print a summary and examples of generated terms.

    Args:
        terms: Flat generated-term list.
        terms_by_gene: Mapping from genes to their term IDs.
        max_examples: Maximum number of example terms to print.
    """
    if not terms:
        print("  - No terms to inspect.")
        return
    print("\n--- Term Generation Summary ---")
    print(f"Total terms: {len(terms)}")
    print(f"Genes with at least one term: {len(terms_by_gene)}")

    # Show some per-gene counts (first up to 5 genes)
    print("Per-gene term counts (first few):")
    for i, (gene, ids) in enumerate(terms_by_gene.items()):
        print(f"  {gene}: {len(ids)} term(s)")
        if i >= 4:
            break

    print(f"\nExample terms (up to {max_examples}):")
    for t in terms[:max_examples]:
        print(
            f"  ID={t['id']}: {t['gene']} {t['operator']} {t['value']}, "
            f"variance={t['variance']}"
        )
    print("--- End of Term Summary ---\n")


# ======================================================================
# Block 8: Compose Antecedent Construction Graph
# ======================================================================
def compose_antecedent_construction_graph(
    data: AlgorithmData, active_term_ids: set[int] | None = None
) -> GraphState:
    """Build the active antecedent construction graph.

    The graph is represented as a complete multipartite directed acyclic
    graph with implicit adjacency. Each part contains terms from one
    gene.

    Args:
        data: Algorithm state containing generated terms and the
            gene-to-term mapping.
        active_term_ids: Optional subset of term IDs to include. All
            terms are active when omitted.

    Returns:
        The explicit graph structure for the active terms.

    Raises:
        ValueError: If terms are unavailable, unknown IDs are requested,
            or no graph parts can be formed.
    """
    terms: list[TermRecord] = data.get("terms", [])
    terms_by_gene: dict[str, list[int]] = data.get("terms_by_gene", {})

    if not terms or not terms_by_gene:
        raise ValueError(
            "Cannot compose antecedent construction graph: no terms have been "
            "generated. Please run 'generate_all_possible_terms' first."
        )

    # Build a mapping from term ID to the term dictionary
    term_by_id: dict[int, TermRecord] = {t["id"]: t for t in terms}
    # Initialize active_term_ids as all term IDs if not provided
    if active_term_ids is None:
        active_term_ids = set(term_by_id.keys())
    else:
        active_term_ids = set(active_term_ids)

        unknown_term_ids = active_term_ids - term_by_id.keys()

        if unknown_term_ids:
            raise ValueError(
                "Unknown active term IDs: " f"{sorted(unknown_term_ids)}."
            )

    if not active_term_ids:
        raise ValueError(
            "Active term set is empty: no nodes available to compose the "
            "antecedent construction graph."
        )

    genes_in_graph: list[str] = []
    parts: list[list[int]] = []
    part_index_by_gene: dict[str, int] = {}
    term_to_part: dict[int, int] = {}

    for gene, term_ids in terms_by_gene.items():
        # Filter this gene's term IDs by the active_term_ids set
        active_ids_for_gene = [
            term_id for term_id in term_ids if term_id in active_term_ids
        ]
        if not active_ids_for_gene:
            continue
        part_idx = len(parts)
        genes_in_graph.append(gene)
        parts.append(active_ids_for_gene)
        part_index_by_gene[gene] = part_idx
        for term_id in active_ids_for_gene:
            term_to_part[term_id] = part_idx

    if not parts:
        raise ValueError(
            "No graph parts could be formed: check that active_term_ids and "
            "terms_by_gene are consistent."
        )

    unmapped_term_ids = active_term_ids - term_to_part.keys()
    if unmapped_term_ids:
        raise ValueError(
            "Active terms are missing graph-part assignments: "
            f"{sorted(unmapped_term_ids)}."
        )

    active_term_ids_array = np.fromiter(
        sorted(active_term_ids),
        dtype=np.intp,
        count=len(active_term_ids),
    )
    active_term_parts_array = np.fromiter(
        (term_to_part[int(term_id)] for term_id in active_term_ids_array),
        dtype=np.intp,
        count=active_term_ids_array.size,
    )

    graph: GraphState = {
        "term_by_id": term_by_id,
        "parts": parts,
        "genes_in_graph": genes_in_graph,
        "part_index_by_gene": part_index_by_gene,
        "term_to_part": term_to_part,
        "active_term_ids": active_term_ids,
        "active_term_ids_array": active_term_ids_array,
        "active_term_parts_array": active_term_parts_array,
        # Virtual start/stop nodes as labels (no explicit edges stored)
        "start_node": "__START__",
        "stop_node": "__STOP__",
        "term_count": len(active_term_ids),
        "part_count": len(parts),
    }

    return graph


def inspect_antecedent_graph(
    graph: GraphState, max_genes: int = 5, max_terms_per_gene: int = 5
) -> None:
    """Print a lightweight graph-structure summary.

    Args:
        graph: Antecedent construction graph to inspect.
        max_genes: Maximum number of gene parts to display.
        max_terms_per_gene: Maximum terms to display for each shown
            gene.
    """
    if not graph:
        print("  - No graph structure present to inspect.")
        return

    term_count = graph.get("term_count", 0)
    part_count = graph.get("part_count", 0)
    genes_in_graph: list[str] = graph.get("genes_in_graph", [])
    parts: list[list[int]] = graph.get("parts", [])
    term_by_id: dict[int, TermRecord] = graph.get("term_by_id", {})
    part_index_by_gene: dict[str, int] = graph.get("part_index_by_gene", {})

    print("\n--- Antecedent Construction Graph Summary ---")
    print(f"  - Active term nodes: {term_count}")
    print(f"  - Number of parts (genes): {part_count}")
    print("\n  - Example parts:")

    for gene in genes_in_graph[:max_genes]:
        part_idx: int | None = part_index_by_gene.get(gene)

        if part_idx is None:
            print(f"    Gene '{gene}' has no part_index mapping.")
            continue

        if not (0 <= part_idx < len(parts)):
            print(f"    Gene '{gene}' has invalid part index {part_idx}.")
            continue

        part_terms: list[int] = (
            parts[part_idx] if isinstance(parts[part_idx], list) else []
        )
        term_ids = part_terms[:max_terms_per_gene]

        print(
            f"    Gene '{gene}' (part {part_idx}) has "
            f"{len(part_terms)} term(s)."
        )

        for term_id in term_ids:
            t = term_by_id.get(term_id)
            if t is None:
                print(f"      term_id={term_id}: [Missing from index]")
            else:
                print(
                    f"      term_id={term_id}: {t.get('gene', '?')} "
                    f"{t.get('operator', '?')} "
                    f"{t.get('value', '?')} "
                    f"(variance={t.get('variance', '?')})"
                )

    print("\n  - Adjacency is implicit:")
    print(
        "      * Each term node is adjacent to all nodes not in its own part."
    )
    print("      * No adjacencies exist within the same part (same gene).")
    print(
        "      * Virtual Start/Stop nodes are adjacent to all term "
        "nodes but are not "
        "stored explicitly."
    )
    print("--- End of Graph Summary ---\n")


# ======================================================================
# Block 9: Initialize Heuristics and Pheromones
# ======================================================================
def initialize_heuristics_and_pheromones(
    data: AlgorithmData,
    graph: GraphState,
    eps: float = 1e-9,
) -> None:
    """Initialize active-term heuristics and pheromones.

    Per-term binary-vector MSD values are invariant across epochs and
    are cached in the algorithm state. Graph rebuilds recompute only the
    active-set-dependent heuristic transformation and reset pheromones
    to a uniform probability distribution.

    Args:
        data: Algorithm state containing terms, coverage, and cluster
            memberships.
        graph: Active antecedent construction graph.
        eps: Positive fallback heuristic used for degenerate values.

    Raises:
        KeyError: If an active term is absent from the term mapping.
        ValueError: If required preprocessing state or term statistics
            are invalid.
    """
    terms = data.get("terms")
    if not terms:
        raise ValueError(
            "No terms available in data['terms']; run term generation first."
        )
    if not graph or "active_term_ids" not in graph:
        raise ValueError(
            "Antecedent construction graph must contain active_term_ids."
        )

    binary_matrix = data.get("binary_matrix")
    coverage_words = data.get("coverage_words")
    if binary_matrix is None:
        raise ValueError(
            "binary_matrix is missing. Run compose_binary_vectors first."
        )
    if coverage_words is None:
        raise ValueError(
            "coverage_words is missing. Run precompute_term_coverage_matrix "
            "first."
        )

    active_ids: set[int] = set(graph["active_term_ids"])
    if not active_ids:
        raise ValueError("No active term nodes to initialize.")

    term_by_id: dict[int, TermRecord] = graph["term_by_id"]
    sorted_active_ids = sorted(active_ids)

    variances = np.empty(len(sorted_active_ids), dtype=np.float64)
    for position, term_id in enumerate(sorted_active_ids):
        term = term_by_id.get(term_id)
        if term is None:
            raise KeyError(
                f"Active term ID {term_id} is missing from "
                "graph['term_by_id']."
            )
        try:
            variance = float(term["variance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Active term {term_id} has an invalid variance value."
            ) from exc
        if not math.isfinite(variance):
            raise ValueError(
                f"Active term {term_id} has a non-finite variance: {variance}."
            )
        variances[position] = variance

    max_var = float(np.max(variances))
    min_var = float(np.min(variances))
    A = max_var + min_var
    invalid_A = not math.isfinite(A) or A <= 0.0

    term_variances = data.get("term_variances")
    if term_variances is None:
        term_variances = np.full(
            int(data["term_count"]), np.nan, dtype=np.float64
        )

        coverage_counts = np.asarray(
            data["term_coverage_counts"],
            dtype=np.int64,
        )
        sample_count = int(data["sample_count"])

        for term in terms:
            term_id = int(term["id"])
            covered_count = int(coverage_counts[term_id])
            if covered_count == 0:
                raise ValueError(f"Active term {term_id} covers no samples.")

            covered_indices = _coverage_words_to_indices(
                coverage_words[term_id],
                sample_count,
            )
            covered_counts = _sum_cluster_memberships(
                binary_matrix,
                covered_indices,
            )
            proportions = covered_counts.astype(np.float64) / covered_count
            term_variances[term_id] = float(
                np.sum(proportions * (1.0 - proportions))
            )

        data["term_variances"] = term_variances
    else:
        term_variances = np.asarray(term_variances, dtype=np.float64)
        if term_variances.shape != (int(data["term_count"]),):
            raise ValueError("Cached term_variances has an invalid shape.")

    tau0 = 1.0 / len(active_ids)
    heuristics: dict[int, float] = {}

    for term_id in sorted_active_ids:
        B = float(term_variances[term_id])
        if not math.isfinite(B):
            raise ValueError(f"Term {term_id} has an invalid cached MSD: {B}.")

        if invalid_A:
            eta = eps
        else:
            eta = (A - B) / A
            if not math.isfinite(eta) or eta <= 0.0:
                eta = eps

        heuristics[term_id] = float(eta)

    pheromones = {term_id: float(tau0) for term_id in sorted_active_ids}

    graph["heuristics"] = heuristics
    graph["pheromones"] = pheromones


# ======================================================================
# Block 10: Prepare Algorithm State
# ======================================================================
def prepare_algorithm_state(
    data: AlgorithmData,
    params: RunParameters,
    console_options: ConsoleOptions = DEFAULT_CONSOLE_OPTIONS,
    eps: float = 1e-9,
) -> tuple[PreparedAlgorithmData, GraphState]:
    """Prepare validated inputs for antecedent optimization.

    This function is the explicit staging boundary between validated
    source data and optimization-ready state. It prevents callers from
    having to reproduce the temporal ordering of matrix construction,
    term generation, coverage precomputation, global statistics,
    constraint masks, and graph weight initialization.

    Args:
        data: Validated source data with optional derived fields.
        params: Validated run parameters.
        console_options: Optional setup-output and inspection
            configuration.
        eps: Numerical stability constant used for heuristic
            initialization.

    Returns:
        The fully prepared algorithm state and initialized active graph.

    Raises:
        KeyError: If a required source or parameter field is missing.
        ValueError: If a preprocessing invariant is violated.
    """
    compose_binary_vectors(data)

    if console_options.inspect_binary_vectors:
        inspect_binary_vectors(data)

    all_terms, terms_by_gene = generate_all_possible_terms(data, params)
    if console_options.show_setup_details:
        inspect_terms(all_terms, terms_by_gene)

    graph = compose_antecedent_construction_graph(data)
    if console_options.show_setup_details:
        inspect_antecedent_graph(graph)

    print(
        f"[Graph] {graph['term_count']} active terms across "
        f"{graph['part_count']} gene partitions.", file=sys.stderr)

    precompute_term_coverage_matrix(data)
    precompute_global_msd(data)
    precompute_cannot_cluster_mask(data)

    cannot_sample_count = int(np.count_nonzero(data["cannot_sample_mask"]))
    if data.get("cannot_clusters"):
        cannot_limit = params["max_cannot_cluster_share"]
        if cannot_limit is None:
            raise ValueError(
                "A maximum cannot-cluster share is required when cannot "
                "clusters are configured."
            )
        print(
            f"[Constraints] {cannot_sample_count}/{data['sample_count']} "
            "samples belong to cannot clusters; "
            f"maximum covered share: {cannot_limit:.1%}.", file=sys.stderr)
    else:
        print((
            '[Constraints] Cannot-cluster restriction '
            'disabled.'
        ), file=sys.stderr)

    initialize_heuristics_and_pheromones(
        data=data,
        graph=graph,
        eps=eps,
    )

    return cast(PreparedAlgorithmData, data), graph


# ======================================================================
# Block 11: Compute the Probabilities of Selecting Term Nodes
# ======================================================================
def compute_iteration_weights(
    graph: GraphState,
) -> np.ndarray:
    """Compute pheromone-heuristic products for active terms.

    Args:
        graph: Initialized active graph.

    Returns:
        An array aligned with ``active_term_ids_array``.

    Raises:
        KeyError: If an active term lacks a pheromone or heuristic
            entry.
        ValueError: If any weight is negative or non-finite.
    """
    active_ids = np.asarray(
        graph["active_term_ids_array"],
        dtype=np.intp,
    )
    heuristics = graph.get("heuristics", {})
    pheromones = graph.get("pheromones", {})

    try:
        weights = np.fromiter(
            (
                float(pheromones[int(term_id)])
                * float(heuristics[int(term_id)])
                for term_id in active_ids
            ),
            dtype=np.float64,
            count=active_ids.size,
        )
    except KeyError as exc:
        raise KeyError(
            "Missing active-term heuristic or pheromone entry: "
            f"{exc.args[0]}."
        ) from exc

    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Iteration weights must be finite and non-negative.")

    return weights


def compute_term_probabilities(
    graph: GraphState,
    iteration_weights: np.ndarray,
    visited_parts_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute probabilities for currently eligible terms.

    Args:
        graph: Active graph with arrays aligned by term position.
        iteration_weights: Pheromone-heuristic products for active
            terms.
        visited_parts_mask: Boolean mask of gene partitions already
            visited by an ant.

    Returns:
        Eligible term IDs and their normalized selection probabilities.

    Raises:
        ValueError: If active-term arrays are not aligned.
    """
    weights = np.asarray(
        iteration_weights,
        dtype=np.float64,
    )
    active_ids = np.asarray(
        graph["active_term_ids_array"],
        dtype=np.intp,
    )
    active_parts = np.asarray(
        graph["active_term_parts_array"],
        dtype=np.intp,
    )

    if (
        weights.shape != active_ids.shape
        or active_parts.shape != active_ids.shape
    ):
        raise ValueError("Active-term probability arrays are not aligned.")

    eligible = ~visited_parts_mask[active_parts]
    valid_term_ids = active_ids[eligible]

    if valid_term_ids.size == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float64),
        )

    valid_weights = weights[eligible]
    sum_weights = float(np.sum(valid_weights))

    if sum_weights == 0.0:
        probabilities = np.full(
            valid_term_ids.size,
            1.0 / valid_term_ids.size,
            dtype=np.float64,
        )
    else:
        probabilities = valid_weights / sum_weights

    return valid_term_ids, probabilities


# ======================================================================
# Block 12: Launch Ants to Compose Antecedents
# ======================================================================
def launch_ants_to_compose_antecedents(
    data: PreparedAlgorithmData,
    graph: GraphState,
    ant_count: int,
    max_terms: int,
    rng: np.random.Generator,
) -> list[Antecedent]:
    """Launch ants and constructs candidate antecedents.

    Packed term-coverage bitsets are intersected as each ant selects at
    most one term from each gene partition.

    Args:
        data: Fully prepared algorithm state.
        graph: Initialized active construction graph.
        ant_count: Number of ants to launch.
        max_terms: Maximum terms in one antecedent.
        rng: NumPy random generator used for term selection.

    Returns:
        Candidate antecedents generated by the launched ants.
    """
    iteration_weights = compute_iteration_weights(graph)

    generated_antecedents: list[Antecedent] = []
    part_count = int(graph["part_count"])
    sample_count = int(data["sample_count"])
    construction_limit = min(int(max_terms), part_count)
    term_to_part = graph["term_to_part"]
    coverage_words = data["coverage_words"]

    for ant_idx in range(1, ant_count + 1):
        selected_term_ids: list[int] = []
        visited_parts_mask = np.zeros(part_count, dtype=bool)
        combined_words: np.ndarray | None = None

        while len(selected_term_ids) < construction_limit:
            valid_term_ids, probabilities = compute_term_probabilities(
                graph=graph,
                iteration_weights=iteration_weights,
                visited_parts_mask=visited_parts_mask,
            )
            if valid_term_ids.size == 0:
                break

            chosen_term_id = int(rng.choice(valid_term_ids, p=probabilities))
            chosen_part = int(term_to_part[chosen_term_id])

            selected_term_ids.append(chosen_term_id)
            visited_parts_mask[chosen_part] = True

            if combined_words is None:
                combined_words = coverage_words[chosen_term_id].copy()
            else:
                np.bitwise_and(
                    combined_words,
                    coverage_words[chosen_term_id],
                    out=combined_words,
                )

        if selected_term_ids and combined_words is not None:
            covered_indices = _coverage_words_to_indices(
                combined_words,
                sample_count,
            )
            generated_antecedents.append(
                Antecedent(
                    term_ids=selected_term_ids,
                    covered_indices=covered_indices,
                    coverage_words=combined_words,
                    ant_launch_order=ant_idx,
                )
            )

    return generated_antecedents


# ======================================================================
# Block 13: Compute Selectivity
# ======================================================================
def precompute_global_msd(data: AlgorithmData) -> None:
    """Cache cluster counts and global MSD.

    Args:
        data: Algorithm state containing a dense or sparse membership
            matrix.

    Raises:
        ValueError: If the membership matrix is missing, malformed, or
            empty.
    """
    binary_matrix = data.get("binary_matrix")
    if binary_matrix is None:
        raise ValueError("binary_matrix is missing.")
    sample_count, _ = _require_2d_matrix(
        binary_matrix,
        name="binary_matrix",
    )
    if sample_count == 0:
        raise ValueError("binary_matrix contains no samples.")

    cluster_sample_counts = _sum_cluster_memberships(binary_matrix)
    proportions = cluster_sample_counts.astype(np.float64) / sample_count

    data["cluster_sample_counts"] = cluster_sample_counts
    data["global_msd"] = float(np.sum(proportions * (1.0 - proportions)))


def annotate_selectivity(
    data: PreparedAlgorithmData,
    antecedent: Antecedent,
) -> float:
    """Compute and assigns one antecedent's selectivity.

    Args:
        data: Algorithm state containing the membership matrix, global
            MSD, and optional cached cluster counts.
        antecedent: Candidate antecedent to evaluate.

    Returns:
        The computed selectivity score.

    Raises:
        ValueError: If required precomputed state is unavailable.
    """
    binary_matrix = data.get("binary_matrix")
    global_msd = data.get("global_msd")

    if binary_matrix is None:
        raise ValueError(
            "binary_matrix is missing. Run compose_binary_vectors first."
        )
    if global_msd is None:
        raise ValueError(
            "global_msd is missing. Run precompute_global_msd first."
        )

    selectivity = compute_selectivity(
        binary_matrix,
        antecedent.covered_indices,
        global_msd,
        data.get("cluster_sample_counts"),
    )
    antecedent.selectivity = selectivity
    return selectivity


def annotate_selectivities(
    data: PreparedAlgorithmData,
    antecedents: list[Antecedent],
) -> list[Antecedent]:
    """Compute selectivity for each candidate antecedent.

    Args:
        data: Algorithm state containing precomputed selectivity inputs.
        antecedents: Candidate antecedents to score.

    Returns:
        The same list with each antecedent's selectivity populated.
    """
    for antecedent in antecedents:
        annotate_selectivity(data, antecedent)
    return antecedents


# ======================================================================
# Block 14: Prune Antecedents
# ======================================================================
def precompute_cannot_cluster_mask(data: AlgorithmData) -> None:
    """Store dense and packed membership in the cannot-cluster union.

    Args:
        data: Prepared state; sample masks are assigned in place.

    The union prevents double-counting samples in overlapping clusters.
    """
    sample_count = int(data["sample_count"])
    mask = np.zeros(sample_count, dtype=bool)
    cannot_clusters = data.get("cannot_clusters")

    if cannot_clusters:
        hierarchies = data["hierarchies"]
        samples = data["samples"]
        for i, sample_id in enumerate(samples):
            sample_clusters = hierarchies.get(sample_id, set())
            if sample_clusters.intersection(cannot_clusters):
                mask[i] = True
    data["cannot_sample_mask"] = mask
    data["cannot_sample_words"] = _pack_sample_mask(mask, sample_count)


def compute_cannot_cluster_share(
    data: PreparedAlgorithmData,
    covered_indices: np.ndarray,
    coverage_words: np.ndarray | None = None,
) -> float:
    """Compute the cannot-cluster share of covered samples.

    Args:
        data: Fully prepared algorithm state containing cannot-cluster
            masks.
        covered_indices: Samples covered by the antecedent.
        coverage_words: Optional packed coverage bitset used for a
            faster intersection.

    Returns:
        The share of covered samples belonging to cannot clusters.

    Raises:
        ValueError: If the fallback cannot-cluster mask is unavailable.
    """
    covered_count = int(covered_indices.size)
    if covered_count == 0:
        return 0.0

    cannot_words = data.get("cannot_sample_words")
    if coverage_words is not None and cannot_words is not None:
        covered_cannot_count = _count_coverage_words(
            np.bitwise_and(coverage_words, cannot_words)
        )
        return float(covered_cannot_count / covered_count)

    mask = data.get("cannot_sample_mask")
    if mask is None:
        raise ValueError(
            "Cannot-cluster mask missing. Run "
            "precompute_cannot_cluster_mask first."
        )

    covered_cannot_count = int(np.sum(mask[covered_indices]))
    return float(covered_cannot_count / covered_count)


def prune_antecedent(
    data: PreparedAlgorithmData,
    antecedent: Antecedent,
    max_cannot_cluster_share: float | None,
) -> Antecedent:
    """Prune redundant terms from one antecedent.

    Each possible one-term removal is evaluated using packed prefix and
    suffix intersections. Variants that exceed the optional maximum
    cannot-cluster share are infeasible and excluded. Among feasible
    variants, a removal is accepted when selectivity improves or remains
    numerically equal, with cannot-cluster share and coverage used as
    tie-breakers.

    Args:
        data: Fully prepared algorithm state.
        antecedent: Candidate antecedent to prune in place.
        max_cannot_cluster_share: Optional maximum allowed share of
            covered samples belonging to cannot clusters.

    Returns:
        The pruned antecedent.

    Raises:
        ValueError: If required coverage or selectivity state is
            unavailable.
    """
    binary_matrix = data.get("binary_matrix")
    global_msd = data.get("global_msd")
    coverage_words = data.get("coverage_words")
    cluster_sample_counts = data.get("cluster_sample_counts")
    sample_count = int(data["sample_count"])

    if binary_matrix is None or global_msd is None:
        raise ValueError(
            "Missing global matrix or MSD for selectivity calculations."
        )
    if coverage_words is None:
        raise ValueError("Packed coverage matrix is missing.")

    if antecedent.coverage_words is None and antecedent.term_ids:
        selected_words = coverage_words[
            np.asarray(antecedent.term_ids, dtype=np.intp)
        ]
        antecedent.coverage_words = np.bitwise_and.reduce(
            selected_words,
            axis=0,
        )

    antecedent.cannot_cluster_share = compute_cannot_cluster_share(
        data,
        antecedent.covered_indices,
        antecedent.coverage_words,
    )

    while len(antecedent.term_ids) > 1:
        current_selectivity = antecedent.selectivity
        term_ids = antecedent.term_ids
        term_word_rows = coverage_words[np.asarray(term_ids, dtype=np.intp)]

        prefix = np.bitwise_and.accumulate(term_word_rows, axis=0)
        suffix = np.bitwise_and.accumulate(
            term_word_rows[::-1],
            axis=0,
        )[::-1]
        best_variant = None

        for i in range(len(term_ids)):
            if i == 0:
                variant_words = suffix[1].copy()
            elif i == len(term_ids) - 1:
                variant_words = prefix[-2].copy()
            else:
                variant_words = np.bitwise_and(
                    prefix[i - 1],
                    suffix[i + 1],
                )

            variant_terms = term_ids[:i] + term_ids[i + 1 :]
            variant_covered_indices = _coverage_words_to_indices(
                variant_words,
                sample_count,
            )
            variant_selectivity = compute_selectivity(
                binary_matrix,
                variant_covered_indices,
                global_msd,
                cluster_sample_counts,
            )
            variant_cannot_cluster_share = compute_cannot_cluster_share(
                data,
                variant_covered_indices,
                variant_words,
            )

            if (
                max_cannot_cluster_share is not None
                and variant_cannot_cluster_share > max_cannot_cluster_share
            ):
                continue

            candidate = (
                variant_selectivity,
                -variant_cannot_cluster_share,
                int(variant_covered_indices.size),
                i,
                variant_terms,
                variant_covered_indices,
                variant_words,
            )
            if best_variant is None or candidate[:4] > best_variant[:4]:
                best_variant = candidate

        if best_variant is None:
            break

        best_selectivity = best_variant[0]
        if best_selectivity > current_selectivity or math.isclose(
            best_selectivity,
            current_selectivity,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            antecedent.term_ids = best_variant[4]
            antecedent.covered_indices = best_variant[5]
            antecedent.coverage_words = best_variant[6]
            antecedent.selectivity = best_variant[0]
            antecedent.cannot_cluster_share = -best_variant[1]
        else:
            break

    return antecedent


def prune_antecedents(
    data: PreparedAlgorithmData,
    antecedents: list[Antecedent],
    max_cannot_cluster_share: float | None,
) -> list[Antecedent]:
    """Prune each antecedent in a candidate batch.

    Args:
        data: Algorithm state containing coverage and selectivity
            inputs.
        antecedents: Candidate antecedents to prune.
        max_cannot_cluster_share: Optional maximum allowed share of
            covered samples belonging to cannot clusters.

    Returns:
        The same list with each antecedent pruned in place.
    """
    for antecedent in antecedents:
        prune_antecedent(data, antecedent, max_cannot_cluster_share)
    return antecedents


# ======================================================================
# Block 15: Compute Each Antecedent’s Quality + Home-Cluster Selection
# ======================================================================
def assign_home_cluster(
    data: PreparedAlgorithmData, antecedent: Antecedent
) -> None:
    """Assign the highest-quality home cluster to an antecedent.

    Quality is computed from vectorized sensitivity and specificity
    statistics. Ties are resolved by specificity, sensitivity, and then
    larger cluster size.

    Args:
        data: Fully prepared algorithm state.
        antecedent: Candidate antecedent to annotate in place.

    Raises:
        ValueError: If cluster memberships or names are unavailable.
    """
    binary_matrix = data.get("binary_matrix")
    cluster_ids = data.get("cluster_ids")
    cluster_sizes = data.get("cluster_sample_counts")

    if binary_matrix is None or cluster_ids is None:
        raise ValueError(
            "Missing binary_matrix or cluster_ids for quality calculations."
        )

    covered_indices = antecedent.covered_indices
    covered_count = int(covered_indices.size)
    sample_count = int(binary_matrix.shape[0])

    if covered_count == 0:
        antecedent.quality = 0.0
        antecedent.home_cluster_index = -1
        antecedent.home_cluster_id = ""
        return

    if cluster_sizes is None:
        cluster_sizes = _sum_cluster_memberships(binary_matrix)
    else:
        cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)

    true_positives = _sum_cluster_memberships(
        binary_matrix,
        covered_indices,
    )
    candidate_indices = np.flatnonzero(true_positives)

    candidate_tp = true_positives[candidate_indices]
    candidate_sizes = cluster_sizes[candidate_indices]
    false_positives = covered_count - candidate_tp
    true_negatives = sample_count - candidate_sizes - false_positives

    sensitivity = candidate_tp.astype(np.float64) / candidate_sizes
    outside_sizes = sample_count - candidate_sizes
    specificity = np.divide(
        true_negatives,
        outside_sizes,
        out=np.zeros(candidate_indices.size, dtype=np.float64),
        where=outside_sizes > 0,
    )
    qualities = sensitivity * specificity

    best_quality = -1.0
    best_E = -1.0
    best_F = -1.0
    best_size = -1
    best_cluster_index = -1
    for position, cluster_index in enumerate(candidate_indices):
        quality = float(qualities[position])
        E = float(sensitivity[position])
        F = float(specificity[position])
        cluster_size = int(candidate_sizes[position])

        is_better = False
        if quality > best_quality:
            is_better = True
        elif math.isclose(
            quality,
            best_quality,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            if F > best_F and not math.isclose(
                F,
                best_F,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                is_better = True
            elif math.isclose(
                F,
                best_F,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                if E > best_E and not math.isclose(
                    E,
                    best_E,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    is_better = True
                elif (
                    math.isclose(
                        E,
                        best_E,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                    and cluster_size > best_size
                ):
                    is_better = True

        if is_better:
            best_quality = quality
            best_E = E
            best_F = F
            best_size = cluster_size
            best_cluster_index = int(cluster_index)

    if best_cluster_index != -1:
        antecedent.quality = best_quality
        antecedent.home_cluster_index = best_cluster_index
        antecedent.home_cluster_id = cluster_ids[best_cluster_index]


def assign_home_clusters(
    data: PreparedAlgorithmData, antecedents: list[Antecedent]
) -> list[Antecedent]:
    """Assign home clusters and quality scores to antecedents.

    Args:
        data: The main data payload.
        antecedents: The list of pruned antecedents to evaluate.

    Returns:
        The updated list of antecedents, now annotated with home cluster
        data.
    """
    for antecedent in antecedents:
        assign_home_cluster(data, antecedent)
    return antecedents


# ======================================================================
# Block 16: Select the Best Antecedent in the Current Iteration
# (iteration_best)
# ======================================================================
def select_iteration_best(
    antecedents: list[Antecedent], params: RunParameters
) -> Antecedent | None:
    """Select the highest-ranked valid antecedent in an iteration.

    Candidates below the minimum sample coverage or above the optional
    maximum cannot-cluster share are discarded. Remaining candidates are
    ranked by the shared antecedent priority key and then by earlier ant
    launch order.

    Args:
        antecedents: Candidates generated in the current iteration.
        params: Run parameters containing the iteration-level validity
            limits.

    Returns:
        The highest-ranked valid candidate, or ``None`` when none
        qualify.
    """
    min_coverage = params.get("min_sample_coverage", 1)
    max_cannot_cluster_share = params.get("max_cannot_cluster_share")

    def iteration_best_key(
        antecedent: Antecedent,
    ) -> tuple[float, float, int, int]:
        # Append generation order (negative to prioritize earlier ants)
        """Return the common priority key for an eligible antecedent."""
        return antecedent_priority_key(antecedent) + (
            -antecedent.ant_launch_order,
        )

    # Filter antecedents based on constraints
    valid_antecedents = []
    for antecedent in antecedents:
        if len(antecedent.covered_indices) < min_coverage:
            continue

        if (
            max_cannot_cluster_share is not None
            and antecedent.cannot_cluster_share > max_cannot_cluster_share
        ):
            continue

        valid_antecedents.append(antecedent)

    # Skip iteration if all ants produced invalid antecedents
    if not valid_antecedents:
        return None

    # Select the iteration_best
    iteration_best = max(valid_antecedents, key=iteration_best_key)

    return iteration_best


# ======================================================================
# Block 17: Update Pheromones
# ======================================================================
def update_pheromones(
    graph: GraphState, iteration_best: Antecedent | None
) -> None:
    """Update pheromone levels using the iteration-best antecedent.

    The pheromone update consists of two phases:

    1. Deposition:
       For each term in ``iteration_best``, multiply its pheromone value
       by ``(1 + Q)``, where ``Q`` is the antecedent quality score.

    2. Evaporation via normalization:
       Normalize pheromone values across all active terms so their total
       sum equals 1.0.

    If ``iteration_best`` is None, the update is skipped.

    Args:
        graph: Antecedent Construction Graph containing:
            - 'pheromones': Mapping of term IDs to pheromone values.
            - 'active_term_ids': Iterable of active term IDs.
        iteration_best: Best antecedent found in the current iteration,
            or None if no valid antecedent was produced.

    Raises:
        ValueError:
            If 'pheromones' or 'active_term_ids' are missing from the
            graph, or if the new pheromone sum is non-positive.
        KeyError:
            If an active term is missing from the pheromone dictionary.
    """
    if iteration_best is None:
        return

    # Retrieve pheromone dictionary
    pheromones = graph.get("pheromones")

    if pheromones is None:
        raise ValueError("Pheromones dictionary not found in graph.")

    active_ids = set(graph.get("active_term_ids", set()))

    if not active_ids:
        raise ValueError(
            "No active terms are available for pheromone updating."
        )

    missing = active_ids - pheromones.keys()

    if missing:
        raise KeyError(f"Missing pheromone entries: {sorted(missing)}")

    term_ids = list(iteration_best.term_ids)

    if not term_ids:
        raise ValueError("iteration_best contains no terms.")

    selected_ids = set(term_ids)

    if len(selected_ids) != len(term_ids):
        raise ValueError("iteration_best contains duplicate term IDs.")

    invalid_selected_ids = selected_ids - active_ids

    if invalid_selected_ids:
        raise KeyError(
            "iteration_best contains inactive or unknown term IDs: "
            f"{sorted(invalid_selected_ids)}."
        )

    Q = float(iteration_best.quality)

    if not math.isfinite(Q) or not 0.0 <= Q <= 1.0:
        raise ValueError(
            "Iteration-best quality must be finite and within "
            f"[0, 1]; found {Q}."
        )

    updated: dict[int, float] = {}

    for term_id in active_ids:
        value = float(pheromones[term_id])

        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Invalid pheromone value for term {term_id}: {value}."
            )

        updated[term_id] = value

    for term_id in selected_ids:
        updated[term_id] *= 1.0 + Q

    new_sum = sum(updated.values())

    if not math.isfinite(new_sum) or new_sum <= 0.0:
        raise ValueError(
            f"Invalid pheromone sum: {new_sum}. Cannot normalize."
        )

    # Commit only after every validation has succeeded.
    for term_id in active_ids:
        pheromones[term_id] = updated[term_id] / new_sum

    # fsum adds the stored values exactly. Each division above rounds by
    # half an ulp, and new_sum carries the rounding of one addition per
    # term, so the exact total may miss 1.0 by about n machine epsilons.
    normalized_sum = math.fsum(
        pheromones[term_id] for term_id in active_ids
    )
    tolerance = max(1e-12, len(active_ids) * sys.float_info.epsilon)

    if not math.isclose(
        normalized_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            "Normalized pheromones do not sum to 1.0: " f"{normalized_sum}."
        )


# ======================================================================
# Block 18: Select Best antecedent in the Current Epoch (epoch_best)
# ======================================================================
def select_epoch_best(
    epoch_best: Antecedent | None,
    iteration_best: Antecedent | None,
) -> tuple[Antecedent | None, bool]:
    """Assign the best antecedent for the current epoch.

    Compares ``iteration_best`` against the current ``epoch_best`` using
    a centralized priority key and returns the higher-ranked antecedent.
    If both candidates have equal priority, the existing ``epoch_best``
    is preserved.

    Args:
        epoch_best: The best antecedent selected so far in the epoch.
        iteration_best: The best antecedent found in the current
            iteration.

    Returns:
        A tuple containing:
            - The updated epoch-best antecedent, or None if neither
              exists.
            - A boolean flag indicating whether the epoch-best changed.
    """
    if iteration_best is None:
        return epoch_best, False

    # 1. Initialize the flag to False so it exists regardless of the
    # outcome
    changed = False

    # 2. Compare using the centralized helper function A dead tie makes
    # the comparison false and preserves epoch_best.
    if epoch_best is None or antecedent_priority_key(
        iteration_best
    ) > antecedent_priority_key(epoch_best):
        epoch_best = iteration_best  # Update the tracker
        changed = True

    return epoch_best, changed


# ======================================================================
# Block 19: Epoch Iteration
# ======================================================================
def run_epoch(
    data: PreparedAlgorithmData,
    graph: GraphState,
    params: RunParameters,
    rng: np.random.Generator,
    console_options: ConsoleOptions = DEFAULT_CONSOLE_OPTIONS,
) -> tuple[Antecedent | None, bool, int, int]:
    """Execute the ACO inner loop for one epoch.

    Args:
        data: Algorithm state containing matrices, coverage, and masks.
        graph: Active construction graph with initialized weights.
        params: Validated run parameters.
        rng: NumPy random generator used for stochastic selections.
        console_options: Optional console-output configuration.

    Returns:
        A tuple containing the epoch-best antecedent, whether
        convergence was reached, the performed iteration count, and the
        generated- candidate count.
    """
    max_iterations = int(params["max_iterations"])
    convergence_window = int(params["convergence_window"])

    epoch_best: Antecedent | None = None
    unchanged_count = 0
    iteration_count = 0
    total_generated = 0

    for iteration in range(1, max_iterations + 1):
        antecedents = launch_ants_to_compose_antecedents(
            data=data,
            graph=graph,
            ant_count=params["ant_count"],
            max_terms=params["max_terms"],
            rng=rng,
        )
        total_generated += len(antecedents)

        annotate_selectivities(data, antecedents)
        prune_antecedents(
            data,
            antecedents,
            params.get("max_cannot_cluster_share"),
        )
        assign_home_clusters(data, antecedents)
        iteration_best = select_iteration_best(antecedents, params)
        iteration_count = iteration

        if iteration_best is None:
            if console_options.show_iteration_progress:
                print(
                    (
                        f'  Iteration {iteration:>3}/{max_iterations}: '
                        f'candidates={len(antecedents)}; no candidate met '
                        f'the coverage/cannot-cluster constraints.'
                    ), file=sys.stderr)
            continue

        update_pheromones(graph, iteration_best)
        epoch_best, changed = select_epoch_best(epoch_best, iteration_best)

        if changed:
            unchanged_count = 0
            progress = "new epoch best"
        else:
            unchanged_count += 1
            progress = f"no improvement {unchanged_count}/{convergence_window}"

        if console_options.show_iteration_progress:
            print(
                (
                    f'  Iteration {iteration:>3}/{max_iterations}: '
                    f'candidates={len(antecedents)}, '
                    f'quality={iteration_best.quality:.6f}, '
                    f'selectivity={iteration_best.selectivity:.6f}, '
                    f'coverage={len(iteration_best.covered_indices)}, '
                    f'terms={len(iteration_best.term_ids)}; '
                    f'{progress}.'
                ), file=sys.stderr)

        if unchanged_count >= convergence_window:
            return epoch_best, True, iteration_count, total_generated

    return epoch_best, False, iteration_count, total_generated


# ======================================================================
# Block 20: Complete a Signature
# ======================================================================
@dataclass(frozen=True)
class FinalizedAntecedent:
    """Immutable antecedent snapshot in a finalized signature.

    Attributes:
        term_ids: Ordered term IDs in the finalized antecedent.
        covered_indices: Read-only sample indices covered by the
            antecedent.
        coverage_words: Optional read-only packed coverage bitset.
        selectivity: Final selectivity score.
        quality: Final quality score.
        cannot_cluster_share: Final cannot-cluster share.
        home_cluster_index: Matrix index of the selected home cluster;
            in the immutable layout this is also the tree-node ID.
        home_cluster_id: Canonical decimal immutable node-ID string.
        ant_launch_order: Ant generation order used for tie-breaking.
    """

    term_ids: tuple[int, ...]
    covered_indices: np.ndarray
    coverage_words: np.ndarray | None
    selectivity: float
    quality: float
    cannot_cluster_share: float
    home_cluster_index: int
    home_cluster_id: str
    ant_launch_order: int


@dataclass(frozen=True)
class SignatureRecord:
    """Finalized, signature record produced from an epoch best.

    Finalized, immutable signature record produced from an epoch best.

    Attributes:
        signature_id: One-based signature number.
        termination_reason: Signature-level stopping reason.
        antecedent: Immutable snapshot of the selected antecedent.
        antecedent_text: Rendered IF section.
        consequent_text: Rendered cluster-coverage section.
        scaling_text: Optional scaling diagnostics.
    """

    signature_id: int
    termination_reason: str
    antecedent: FinalizedAntecedent
    antecedent_text: str
    consequent_text: str
    scaling_text: str


def _format_term_for_signature(term: TermRecord) -> str:
    """Format one term as text for a signature antecedent."""
    return f"{term['gene']} {term['operator']} {term['value']}"


def _format_antecedent_text(
    graph: GraphState,
    antecedent: Antecedent,
) -> str:
    """Render the IF clause of a finalized signature.

    Args:
        graph: Graph containing the term-ID lookup.
        antecedent: Finalized term selection in signature order.

    Returns:
        The exact ``IF ... THEN:`` line consumed by sample extraction.
    """
    term_by_id = graph["term_by_id"]
    term_texts = [
        _format_term_for_signature(term_by_id[term_id])
        for term_id in antecedent.term_ids
    ]

    if not term_texts:
        return "IF <empty> THEN:"

    if_text = " AND ".join(term_texts)
    return f"IF {if_text} THEN:"


def _compute_cluster_coverages(
    data: PreparedAlgorithmData,
    antecedent: Antecedent,
) -> list[tuple[str, float, bool]]:
    """Compute nonzero per-cluster coverage shares.

    Args:
        data: Fully prepared algorithm state.
        antecedent: Candidate antecedent whose covered samples are
            evaluated.

    Returns:
        Tuples containing cluster name, coverage share, and whether the
        cluster is a cannot cluster, sorted by descending share and then
        name.
    """
    binary_matrix = data["binary_matrix"]
    cluster_ids = data["cluster_ids"]
    cannot_clusters = data.get("cannot_clusters") or set()

    cluster_sizes = data.get("cluster_sample_counts")
    if cluster_sizes is None:
        cluster_sizes = _sum_cluster_memberships(binary_matrix)
    else:
        cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)

    covered_counts = _sum_cluster_memberships(
        binary_matrix,
        antecedent.covered_indices,
    )
    candidate_indices = np.flatnonzero(covered_counts)

    rows: list[tuple[str, float, bool]] = []
    for cluster_index in candidate_indices:
        cluster_size = int(cluster_sizes[cluster_index])
        if cluster_size <= 0:
            continue

        cluster_id = cluster_ids[int(cluster_index)]
        share = int(covered_counts[cluster_index]) / cluster_size
        rows.append(
            (
                cluster_id,
                float(share),
                cluster_id in cannot_clusters,
            )
        )

    rows.sort(key=lambda row: (-row[1], int(row[0])))
    return rows


def _format_consequent_text(
    data: PreparedAlgorithmData,
    antecedent: Antecedent,
) -> str:
    """Build the rendered THEN section of a signature.

    Args:
        data: Fully prepared algorithm state.
        antecedent: Final candidate antecedent.

    Returns:
        Human-readable can-cluster and cannot-cluster coverage text.
    """
    rows = _compute_cluster_coverages(data, antecedent)
    cannot_cluster_coverages = [row[:2] for row in rows if row[2]]
    can_cluster_coverages = [row[:2] for row in rows if not row[2]]

    lines: list[str] = []
    lines.append("Can clusters:")
    if can_cluster_coverages:
        for cluster_id, share in can_cluster_coverages:
            lines.append(f"{cluster_id}: {share * 100:.1f}% covered")
    else:
        lines.append("None")

    lines.append("Cannot clusters:")
    if cannot_cluster_coverages:
        for cluster_id, share in cannot_cluster_coverages:
            lines.append(f"{cluster_id}: {share * 100:.1f}% covered")
    else:
        lines.append("None")

    return "\n".join(lines)


def _format_scaling_text(
    data: PreparedAlgorithmData,
    graph: GraphState,
    antecedent: Antecedent,
    scale_values: tuple[int, ...] = tuple(range(10, 101, 10)),
) -> str:
    """Build scaling diagnostics section for finalized signature.

    Build the scaling diagnostics section for the finalized signature.

    Scaling preserves the original logical operator. For upregulated
    terms using ">", the threshold is divided by the scale factor. For
    downregulated terms using "<", the threshold is multiplied by the
    scale factor.

    Args:
        data: Main algorithm payload.
        graph: Current antecedent construction graph.
        antecedent: Finalized antecedent.
        scale_values: Scale factors to apply to each term value.

    Returns:
        A human-readable scaling diagnostics section, or an empty string
        when cannot clusters are not defined.
    """
    expression_values = data["expression_values"]
    term_by_id = graph["term_by_id"]
    cannot_mask = data.get("cannot_sample_mask")
    cannot_clusters = data.get("cannot_clusters")

    if cannot_mask is None or not cannot_clusters:
        return ""

    cannot_total = int(np.sum(cannot_mask))
    if cannot_total == 0:
        return ""

    lines: list[str] = ["----------"]

    for term_id in antecedent.term_ids:
        term = term_by_id[term_id]
        gene_index = int(term["gene_index"])
        op = str(term["operator"])
        base_value = float(term["value"])
        gene_name = str(term["gene"])

        if op not in {">", "<"}:
            continue

        for scale in scale_values:
            if op == ">":
                scaled_value = base_value / float(scale)
                covered = expression_values[gene_index, :] > scaled_value
            else:
                scaled_value = base_value * float(scale)
                covered = expression_values[gene_index, :] < scaled_value

            covered_in_cannot = int(np.sum(covered & cannot_mask))
            share = covered_in_cannot / cannot_total

            lines.append(
                f"Scale = {scale}, {gene_name} {op} {scaled_value}: "
                f"{share * 100:.1f}% covered in cannot clusters"
            )

    if len(lines) == 1:
        return ""

    lines.append("----------")
    return "\n".join(lines)


def complete_signature(
    data: PreparedAlgorithmData,
    graph: GraphState,
    epoch_best: Antecedent,
    signature_id: int,
    termination_reason: str,
) -> SignatureRecord:
    """Finalize the selected antecedent and its consequent.

    Args:
        data: Main algorithm payload.
        graph: Current antecedent construction graph.
        epoch_best: Best antecedent selected for the epoch.
        signature_id: 1-based signature number in the decision list.
        termination_reason: Either ``"convergence"`` or
            ``"max_iterations"``.

    Returns:
        A finalized signature record.
    """
    if epoch_best is None:
        raise ValueError("epoch_best is required to complete a signature.")

    if signature_id < 1:
        raise ValueError("signature_id must be at least 1.")

    if not graph:
        raise ValueError(
            "A non-empty antecedent construction graph is required."
        )

    required_graph_fields = {
        "term_by_id",
        "active_term_ids",
    }
    missing_graph_fields = required_graph_fields - graph.keys()

    if missing_graph_fields:
        raise ValueError(
            "Antecedent construction graph is missing required fields: "
            f"{sorted(missing_graph_fields)}."
        )

    allowed_termination_reasons = {
        "convergence",
        "max_iterations",
    }

    if termination_reason not in allowed_termination_reasons:
        raise ValueError(
            "Unsupported signature termination reason: "
            f"{termination_reason!r}."
        )

    if not epoch_best.term_ids:
        raise ValueError("Cannot finalize a signature with no terms.")

    if len(epoch_best.term_ids) != len(set(epoch_best.term_ids)):
        raise ValueError(
            "Cannot finalize a signature with duplicate term IDs."
        )

    term_by_id: dict[int, TermRecord] = graph["term_by_id"]

    unknown_term_ids = set(epoch_best.term_ids) - term_by_id.keys()

    if unknown_term_ids:
        raise KeyError(
            "Signature contains unknown term IDs: "
            f"{sorted(unknown_term_ids)}."
        )

    inactive_term_ids = set(epoch_best.term_ids) - set(
        graph["active_term_ids"]
    )

    if inactive_term_ids:
        raise ValueError(
            "Signature contains inactive term IDs: "
            f"{sorted(inactive_term_ids)}."
        )

    epoch_best.covered_indices = _validate_sample_indices(
        epoch_best.covered_indices,
        int(data["sample_count"]),
    )

    if epoch_best.covered_indices.size == 0:
        raise ValueError("Cannot finalize a signature that covers no samples.")

    if not math.isfinite(float(epoch_best.selectivity)):
        raise ValueError("Signature selectivity must be finite.")

    if not math.isfinite(float(epoch_best.quality)):
        raise ValueError("Signature quality must be finite.")

    if epoch_best.home_cluster_index < 0 or not epoch_best.home_cluster_id:
        raise ValueError("Signature has no valid home-cluster assignment.")

    antecedent_text = _format_antecedent_text(
        graph=graph,
        antecedent=epoch_best,
    )
    consequent_text = _format_consequent_text(
        data=data,
        antecedent=epoch_best,
    )
    scaling_text = _format_scaling_text(
        data=data,
        graph=graph,
        antecedent=epoch_best,
    )

    finalized_covered_indices = epoch_best.covered_indices.copy()
    finalized_covered_indices.setflags(write=False)

    finalized_coverage_words = None
    if epoch_best.coverage_words is not None:
        finalized_coverage_words = epoch_best.coverage_words.copy()
        finalized_coverage_words.setflags(write=False)

    finalized_antecedent = FinalizedAntecedent(
        term_ids=tuple(epoch_best.term_ids),
        covered_indices=finalized_covered_indices,
        coverage_words=finalized_coverage_words,
        selectivity=float(epoch_best.selectivity),
        quality=float(epoch_best.quality),
        cannot_cluster_share=float(epoch_best.cannot_cluster_share),
        home_cluster_index=int(epoch_best.home_cluster_index),
        home_cluster_id=str(epoch_best.home_cluster_id),
        ant_launch_order=int(epoch_best.ant_launch_order),
    )

    return SignatureRecord(
        signature_id=signature_id,
        termination_reason=termination_reason,
        antecedent=finalized_antecedent,
        antecedent_text=antecedent_text,
        consequent_text=consequent_text,
        scaling_text=scaling_text,
    )


# ======================================================================
# Block 21: Update the List of Signatures
# ======================================================================
def update_signature_list(
    signatures: list[SignatureRecord],
    covered_mask: np.ndarray,
    new_signature: SignatureRecord,
) -> tuple[list[SignatureRecord], np.ndarray]:
    """Append a signature and updates cumulative sample coverage.

    Args:
        signatures: Current finalized-signature list.
        covered_mask: Boolean mask of samples already covered.
        new_signature: Finalized signature to append.

    Returns:
        The mutated signature list and covered mask.

    Raises:
        ValueError: If the signature is missing or the mask is invalid.
        IndexError: If the signature contains an out-of-range sample
            index.
    """
    if new_signature is None:
        raise ValueError("new_signature is required.")

    covered_mask = _require_boolean_mask(
        covered_mask,
        name="covered_mask",
    )

    covered_indices = _validate_sample_indices(
        new_signature.antecedent.covered_indices,
        covered_mask.shape[0],
    )

    signatures.append(new_signature)

    # Update the covered mask with the samples covered by this signature
    covered_mask[covered_indices] = True

    return signatures, covered_mask


# ======================================================================
# Block 22: Remove the Used Terms
# ======================================================================
def remove_used_terms(
    data: PreparedAlgorithmData,
    graph: GraphState,
    signature_antecedent: FinalizedAntecedent,
    eps: float = 1e-9,
) -> GraphState:
    """Remove finalized terms and rebuilds the active graph.

    Args:
        data: Algorithm state containing the complete term collection
            and precomputed coverage data.
        graph: Current antecedent construction graph.
        signature_antecedent: Finalized antecedent whose terms must be
            removed.
        eps: Numerical stability constant used during heuristic
            initialization.

    Returns:
        The rebuilt graph, or an explicit empty graph when no terms
        remain.

    Raises:
        ValueError: If the graph or finalized term IDs are invalid.
    """
    if not graph:
        raise ValueError(
            "A current antecedent construction graph is required."
        )

    if "active_term_ids" not in graph:
        raise ValueError(
            "Antecedent construction graph is missing active_term_ids."
        )

    active_term_ids: set[int] = set(graph.get("active_term_ids", set()))
    if not active_term_ids:
        return graph

    used_term_ids = set(signature_antecedent.term_ids)

    if not used_term_ids:
        raise ValueError(
            "Finalized signature contains no terms; nothing removed."
        )

    invalid_used_term_ids = used_term_ids - active_term_ids

    if invalid_used_term_ids:
        raise ValueError(
            "Signature contains inactive or unknown term IDs: "
            f"{sorted(invalid_used_term_ids)}."
        )

    # Remove all signature terms from active processing.
    remaining_active_term_ids = active_term_ids - used_term_ids

    if not remaining_active_term_ids:
        empty_graph: GraphState = {
            "term_by_id": graph["term_by_id"],
            "parts": [],
            "genes_in_graph": [],
            "part_index_by_gene": {},
            "term_to_part": {},
            "active_term_ids": set(),
            "active_term_ids_array": np.empty(0, dtype=np.intp),
            "active_term_parts_array": np.empty(0, dtype=np.intp),
            "start_node": "__START__",
            "stop_node": "__STOP__",
            "term_count": 0,
            "part_count": 0,
            "heuristics": {},
            "pheromones": {},
        }

        return empty_graph

    # Rebuild graph with remaining active terms.
    new_graph = compose_antecedent_construction_graph(
        data=data, active_term_ids=remaining_active_term_ids
    )

    # Reinitialize heuristic and pheromone structures.
    initialize_heuristics_and_pheromones(data=data, graph=new_graph, eps=eps)

    return new_graph


# ======================================================================
# Block 23: Iterate Until Completion
# ======================================================================
@dataclass
class RunCompletionResult:
    """Stores the outcome of the full outer-loop Tjala run.

    Attributes:
        signatures: Finalized signatures generated during the run.
        covered_mask: Boolean mask indicating samples covered by at
            least one finalized signature.
        graph: The final antecedent construction graph after used-term
            removal.
        termination_reason: Reason the outer loop stopped.
        epochs_completed: Number of completed epochs.
        total_iterations: Total inner-loop iterations across all epochs.
        total_generated_antecedents: Total candidate antecedents
            generated by all ants across all epochs.
        final_uncovered_count: Number of samples not covered by any
            finalized signature at termination.
    """

    signatures: list[SignatureRecord]
    covered_mask: np.ndarray
    graph: GraphState
    termination_reason: str
    epochs_completed: int
    total_iterations: int
    total_generated_antecedents: int
    final_uncovered_count: int

    def summary(self) -> str:
        """Build a human-readable completion summary.

        Returns:
            A formatted string summarizing the full Tjala run.
        """
        return (
            "Run summary\n"
            "Termination: "
            f"{_format_run_termination_reason(self.termination_reason)} "
            f"({self.termination_reason})\n"
            f"Signatures generated: {len(self.signatures)}\n"
            f"Epochs completed: {self.epochs_completed}\n"
            f"Inner-loop iterations: {self.total_iterations}\n"
            f"Generated antecedents: {self.total_generated_antecedents}\n"
            f"Final uncovered samples: {self.final_uncovered_count}\n"
            f"Remaining active terms: {self.graph.get('term_count', 0)}"
        )


def _count_newly_covered_samples(
    covered_mask: np.ndarray,
    covered_indices: np.ndarray,
) -> int:
    """Count samples newly covered by a candidate signature.

    Args:
        covered_mask: Boolean mask indicating samples already covered by
            previously finalized signatures.
        covered_indices: Sample indices covered by the new signature.

    Returns:
        The number of valid covered indices that were not already
        covered.
    """
    covered_mask = _require_boolean_mask(
        covered_mask,
        name="covered_mask",
    )

    valid_indices = _validate_sample_indices(
        covered_indices,
        covered_mask.shape[0],
    )

    if valid_indices.size == 0:
        return 0

    return int(np.sum(~covered_mask[valid_indices]))


def count_uncovered_samples(covered_mask: np.ndarray) -> int:
    """Return the number of samples not yet covered by any signature."""
    return int(np.sum(~covered_mask))


def iterate_until_completion(
    data: PreparedAlgorithmData,
    graph: GraphState,
    params: RunParameters,
    rng: np.random.Generator,
    eps: float = 1e-9,
    console_options: ConsoleOptions = DEFAULT_CONSOLE_OPTIONS,
) -> RunCompletionResult:
    """Generate signatures until a documented stop condition.

    This function repeatedly executes one epoch, finalizes the
    epoch-best antecedent as a signature, appends the signature to the
    decision list, updates the cumulative covered-sample mask, and
    removes the finalized signature's terms from future processing.

    The loop stops when one of the following occurs:
        1. The number of uncovered samples is at or below the
           user-defined
           threshold.
        2. No valid epoch-best antecedent can be produced.
        3. The epoch-best antecedent fails the minimum sample coverage
           requirement.
        4. No active terms or graph parts remain.
        5. Used-term removal does not reduce the active term set.

    Args:
        data: Global algorithm payload containing matrices, terms,
            masks, and graph-dependent state.
        graph: Current antecedent construction graph.
        params: User-defined algorithm parameters.
        rng: NumPy random generator used by the ant colony.
        eps: Numerical stability constant for heuristic initialization.
            Defaults to 1e-9.
        console_options: Optional console-output configuration.

    Returns:
        A RunCompletionResult containing signatures and run-level
        metadata.

    Raises:
        ValueError: If required data or parameter fields are missing or
            invalid.
    """
    if "sample_count" not in data:
        raise ValueError(
            "data['sample_count'] is required before optimization."
        )
    if "max_uncovered_samples" not in params:
        raise ValueError(
            "params['max_uncovered_samples'] is required before "
            "optimization."
        )
    if "min_sample_coverage" not in params:
        raise ValueError(
            "params['min_sample_coverage'] is required before optimization."
        )

    sample_count = int(data["sample_count"])
    max_uncovered_samples = int(params["max_uncovered_samples"])
    min_coverage = int(params["min_sample_coverage"])

    if max_uncovered_samples < 0:
        raise ValueError(
            "Maximum uncovered-sample threshold cannot be negative."
        )
    if min_coverage < 1:
        raise ValueError("Minimum sample coverage must be at least 1.")

    signatures: list[SignatureRecord] = []
    covered_mask = np.zeros(sample_count, dtype=bool)
    epochs_completed = 0
    total_iterations = 0
    total_generated_antecedents = 0
    termination_reason = ""

    print(
        "[ACO] Starting optimization: "
        f"{graph.get('term_count', 0)} active terms across "
        f"{graph.get('part_count', 0)} genes; "
        f"target <= {max_uncovered_samples} uncovered samples.",
        file=sys.stderr,
    )

    while True:
        uncovered_before_epoch = count_uncovered_samples(covered_mask)
        active_terms = int(graph.get("term_count", 0)) if graph else 0
        active_parts = int(graph.get("part_count", 0)) if graph else 0

        if uncovered_before_epoch <= max_uncovered_samples:
            termination_reason = "uncovered_threshold_reached"
            break

        if active_terms == 0 or active_parts == 0:
            termination_reason = "no_active_terms_remaining"
            print(
                f"  - Insufficient antecedent coverage:\n"
                "    the run stopped before reaching the "
                "uncovered-sample threshold.\n"
                f"    {uncovered_before_epoch}/{sample_count} samples "
                "are uncovered.", file=sys.stderr)
            break

        epochs_completed += 1
        print(
            (
                f'\nEpoch {epochs_completed} | '
                f'uncovered={uncovered_before_epoch}/{sample_count} '
                f'| active terms={active_terms} | active '
                f'genes={active_parts}'
            ), file=sys.stderr)

        epoch_best, converged, iteration_count, generated_count = run_epoch(
            data=data,
            graph=graph,
            params=params,
            rng=rng,
            console_options=console_options,
        )
        total_iterations += iteration_count
        total_generated_antecedents += generated_count

        if epoch_best is None:
            termination_reason = "insufficient_antecedent_coverage"
            print(
                "[Stop] No candidate satisfied the minimum coverage and "
                "cannot-cluster constraints.", file=sys.stderr)
            break

        epoch_coverage = len(np.unique(epoch_best.covered_indices))
        if epoch_coverage < min_coverage:
            termination_reason = "insufficient_antecedent_coverage"
            print(
                "[Stop] Best candidate coverage "
                f"({epoch_coverage}) is below the required minimum "
                f"({min_coverage}).", file=sys.stderr)
            break

        signature_id = len(signatures) + 1
        signature_termination_reason = (
            "convergence" if converged else "max_iterations"
        )
        signature = complete_signature(
            data=data,
            graph=graph,
            epoch_best=epoch_best,
            signature_id=signature_id,
            termination_reason=signature_termination_reason,
        )

        newly_covered_count = _count_newly_covered_samples(
            covered_mask=covered_mask,
            covered_indices=signature.antecedent.covered_indices,
        )
        signatures, covered_mask = update_signature_list(
            signatures=signatures,
            covered_mask=covered_mask,
            new_signature=signature,
        )
        uncovered_after_signature = count_uncovered_samples(covered_mask)

        epoch_outcome = "converged" if converged else "iteration limit reached"
        print(
            (
                f'  Signature #{signature_id} finalized '
                f'({epoch_outcome} after {iteration_count} '
                f'iterations): {signature.antecedent_text}'
            ), file=sys.stderr)
        print(
            (
                f'    quality={epoch_best.quality:.6f}, '
                f'selectivity={epoch_best.selectivity:.6f}, home '
                f'cluster={epoch_best.home_cluster_id}, newly '
                f'covered={newly_covered_count}, '
                f'uncovered={uncovered_after_signature}/{sample_count}.'
            ), file=sys.stderr)

        active_terms_before_removal = active_terms
        graph = remove_used_terms(
            data=data,
            graph=graph,
            signature_antecedent=signature.antecedent,
            eps=eps,
        )
        active_terms_after_removal = int(graph.get("term_count", 0))
        print(
            f"    Active terms: {active_terms_before_removal} -> "
            f"{active_terms_after_removal}.", file=sys.stderr)

        if active_terms_after_removal >= active_terms_before_removal:
            termination_reason = "no_terms_removed"
            print(
                (
                    '[Stop] Used-term removal did not reduce the '
                    'active term set.'
                ), file=sys.stderr)
            break

        if uncovered_after_signature <= max_uncovered_samples:
            termination_reason = "uncovered_threshold_reached"
            print(
                (
                    f'[Stop] Uncovered-sample target reached: '
                    f'{uncovered_after_signature} <= {max_uncovered_samples}.'
                ), file=sys.stderr)
            break

    final_uncovered_count = count_uncovered_samples(covered_mask)
    result = RunCompletionResult(
        signatures=signatures,
        covered_mask=covered_mask,
        graph=graph,
        termination_reason=termination_reason,
        epochs_completed=epochs_completed,
        total_iterations=total_iterations,
        total_generated_antecedents=total_generated_antecedents,
        final_uncovered_count=final_uncovered_count,
    )

    return result


# ======================================================================
# Block 24: Generate Final Output File
# ======================================================================
def _format_signature_termination_reason(reason: str) -> str:
    """Format a signature-search termination reason.

    Args:
        reason: Internal termination reason, usually "convergence" or
            "max_iterations".

    Returns:
        A human-readable termination label.
    """
    labels = {
        "convergence": "convergence",
        "max_iterations": "max iterations",
    }
    return labels.get(str(reason), str(reason).replace("_", " "))


def _format_run_termination_reason(reason: str) -> str:
    """Describe the whole-run termination reason.

    Known termination codes are mapped to concise explanatory labels.
    Unknown codes are converted by replacing underscores with spaces.

    Args:
        reason: Machine-readable termination reason.

    Returns:
        A human-readable description of the termination reason.
    """
    labels = {
        "uncovered_threshold_reached": "uncovered-sample target reached",
        "no_active_terms_remaining": "no active terms remain",
        "insufficient_antecedent_coverage": (
            "No further valid signatures met the constraints"
        ),
        "no_terms_removed": "used-term removal made no progress",
    }
    return labels.get(str(reason), str(reason).replace("_", " "))


def _format_signature_for_output(signature: SignatureRecord) -> str:
    """Render a finalized signature in the report grammar.

    Args:
        signature: Finalized signature record.

    Returns:
        Human-readable text for one signature.
    """
    termination_text = _format_signature_termination_reason(
        signature.termination_reason
    )
    lines = [
        f"Signature #{signature.signature_id}:",
        f"({termination_text})",
        signature.antecedent_text,
        signature.consequent_text,
    ]

    if signature.scaling_text:
        lines.append(signature.scaling_text.rstrip())

    return "\n".join(lines)


def _write_text_atomically(
    output_path: str,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write complete text using atomic file replacement.

    The temporary file is created in the destination directory so that
    ``os.replace`` operates within the same filesystem. The existing
    output remains untouched unless the complete replacement file is
    written and synchronized successfully.

    Args:
        output_path: Final destination path.
        content: Complete text to write.
        encoding: Text encoding. Defaults to UTF-8.

    Raises:
        OSError: If the temporary file cannot be written, synchronized,
            or moved into place.
    """
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}"
        )

    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=output_dir,
            prefix=f".{output_name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # The temporary file is closed before replacement, which is
        # required on Windows.
        os.replace(temporary_path, output_path)
        temporary_path = None

    finally:
        # Remove an abandoned temporary file after any failed write or
        # replace.
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def generate_output_file(
    run_result: RunCompletionResult,
    output_dir: str,
    filename: str = TJALA_OUTPUT_FILE,
    extra_inputs: tuple[Path, ...] = (),
) -> str:
    """Write the final Tjala output using atomic file replacement.

    The complete output is first rendered in memory and written to a
    temporary file in the destination directory. The destination is
    replaced only after the temporary file has been completely written,
    flushed, and synchronized.

    Args:
        run_result: Result object returned by
            ``iterate_until_completion``.
        output_dir: Directory where the output file should be written.
        extra_inputs: Optional cannot-cluster file used by this run.
        filename: Output filename. Defaults to ``"tjala_output.txt"``.

    Returns:
        The absolute path to the generated output file.

    Raises:
        OSError: If the output cannot be written or atomically replaced.
    """
    output_path = os.path.abspath(os.path.join(output_dir, filename))

    sections: list[str] = []

    if run_result.signatures:
        sections.extend(
            _format_signature_for_output(signature)
            for signature in run_result.signatures
        )
    else:
        sections.append("No signatures were generated.")

    sections.append(run_result.summary())

    # Render the complete file before touching the destination.
    output_text = "\n\n".join(sections) + "\n"

    destination = Path(output_path)
    source_paths = [
        Path(output_dir) / name for name in (
            MATRIX_GENES_FILE, MATRIX_SAMPLES_FILE, MATRIX_VALUES_FILE,
            SAMPLE_LINEAGES_FILE, TERM_VARIANCES_FILE,
        )
    ]
    source_paths.extend(extra_inputs)
    record = {
        "stage": "signatures",
        "inputs": source_paths,
        "parents": {
            provenance_path(path) for path in source_paths
            if provenance_path(path).is_file()
        },
        "producer": Path(__file__),
    }
    manifest = provenance_path(destination)
    # An identical report changes nothing, so it keeps its dependents.
    if publication_unchanged(manifest, {destination: output_text},
                             **record):
        return output_path
    invalidate_dependents(
        [destination, manifest],
        [destination.parent, destination.parent.parent],
    )
    _write_text_atomically(
        output_path=output_path,
        content=output_text,
    )
    write_provenance(manifest, outputs=[destination], **record)
    return output_path


# ======================================================================
# Main Execution Block
# ======================================================================
if __name__ == "__main__":
    try:
        console_options = DEFAULT_CONSOLE_OPTIONS

        # Prompt user for parameters
        user_params = prompt_parameters()

        # Load and validate the source data files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_payload = load_algorithm_data(
            base_path=script_dir,
            params=user_params,
            cannot_clusters_path=user_params.get("cannot_clusters_path"),
        )

        # Build all derived state in the required order.
        prepared_data, antecedent_graph = prepare_algorithm_state(
            data=data_payload,
            params=user_params,
            console_options=console_options,
            eps=1e-9,
        )

        rng = np.random.default_rng()

        run_result = iterate_until_completion(
            data=prepared_data,
            graph=antecedent_graph,
            params=user_params,
            rng=rng,
            eps=1e-9,
            console_options=console_options,
        )

        output_path = generate_output_file(
            run_result=run_result,
            output_dir=script_dir,
            filename=TJALA_OUTPUT_FILE,
            extra_inputs=(Path(user_params["cannot_clusters_path"]),)
            if user_params["cannot_clusters_path"] else (),
        )

        covered_count = int(np.count_nonzero(run_result.covered_mask))
        sample_count = int(run_result.covered_mask.size)
        print("\nRun complete")
        print(
            "  Stop reason: "
            + _format_run_termination_reason(
                run_result.termination_reason
            )
            + "."
        )
        print(
            f"  Signatures={len(run_result.signatures)} | "
            f"epochs={run_result.epochs_completed} | "
            f"iterations={run_result.total_iterations} | "
            f"antecedents={run_result.total_generated_antecedents}"
        )
        print(
            f"  Covered={covered_count}/{sample_count} | "
            f"uncovered={run_result.final_uncovered_count}/{sample_count}"
        )
        print(f"Final output file written to: {output_path}")

    except (
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"Error: [signatures] {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
