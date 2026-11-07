"""Sample extraction for finalized Tjala signatures.

The signature antecedent determines which samples are extracted.
Immutable tree memberships from ``sample_lineages.txt`` are used to
validate Tjala's reported clusters and reproduce the existing
hierarchy-derived output ordering. Requires Python 3.10 or later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pipeline_cli import prompt_input
from pipeline_formats import (
    Signature,
    iter_matrix_rows,
    load_tree_states,
    read_sample_lineages,
    read_signature_report,
    tree_node_count,
)
from pipeline_io import (
    MATRIX_GENES_FILE,
    MATRIX_SAMPLES_FILE,
    MATRIX_VALUES_FILE,
    SAMPLE_LINEAGES_FILE,
    TJALA_OUTPUT_FILE,
    TREE_STATES_FILE,
    provenance_path,
    read_identifiers,
    verify_artifacts,
    write_provenance,
    write_text_atomically,
)

FloatVector = NDArray[np.float64]
BoolVector = NDArray[np.bool_]


# ======================================================================
# Block 1: Input Contracts
# ======================================================================


def ensure_required_files(base_path: Path) -> dict[str, Path]:
    """Return paths to all inputs after validating their existence.

    Return paths to all required inputs after validating their
    existence.

    Args:
        base_path: Directory containing the pipeline files.

    Returns:
        Mapping from logical input name to validated path.

    Raises:
        FileNotFoundError: If a required input file is missing.
    """
    paths = {
        "values": base_path / MATRIX_VALUES_FILE,
        "columns": base_path / MATRIX_SAMPLES_FILE,
        "genes": base_path / MATRIX_GENES_FILE,
        "tjala_output": base_path / TJALA_OUTPUT_FILE,
        "hierarchy": base_path / SAMPLE_LINEAGES_FILE,
    }

    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s): " + ", ".join(sorted(missing))
        )

    return paths


# ======================================================================
# Block 2: Signature Parsing
# ======================================================================


def prompt_for_signature(signatures: list[Signature]) -> Signature | None:
    """Prompt until the user chooses a valid signature or exits.

    Args:
        signatures: Parsed, consecutively numbered signatures.

    Returns:
        The selected signature, or ``None`` if the user exits.
    """
    by_number = {signature.number: signature for signature in signatures}
    highest = len(signatures)

    while True:
        try:
            response = prompt_input(
                f'Enter signature number to consider, from 1 to {highest} '
                '(or "Exit" (without quotes) to end): '
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program.")
            return None

        if response.lower() == "exit":
            print("Exiting program.")
            return None

        try:
            number = int(response)
        except ValueError:
            print((
                'Signature number is not an integer, please try '
                'again.'
            ), file=sys.stderr)
            continue

        signature = by_number.get(number)
        if signature is None:
            print((
                'Signature number out of range, please try again.'
            ), file=sys.stderr)
            continue

        return signature


def validate_signature_genes(
    signature: Signature,
    *,
    gene_to_index: dict[str, int],
) -> None:
    """Require each signature gene to occur in the dataset.

    Dataset uniqueness is validated while loading ``matrix_genes.txt``;
    this check therefore establishes complete signature-to-matrix
    identity mapping.

    Args:
        signature: Selected signature.
        gene_to_index: Unique gene-name to expression-row mapping.

    Raises:
        ValueError: If any required gene is absent.
    """
    missing = [
        condition.gene
        for condition in signature.terms
        if condition.gene not in gene_to_index
    ]
    if missing:
        raise ValueError(
            f"Signature #{signature.number} references gene(s) absent from "
            f"{MATRIX_GENES_FILE}: "
            + ", ".join(repr(gene) for gene in missing)
        )


# ======================================================================
# Block 3: Signature Evaluation
# ======================================================================


def evaluate_signature(
    signature: Signature,
    *,
    values_by_gene_index: dict[int, FloatVector],
    gene_to_index: dict[str, int],
    sample_count: int,
) -> BoolVector:
    """Evaluate all signature terms against every sample.

    Args:
        signature: Selected signature.
        values_by_gene_index: Retained expression rows keyed by original
            gene index.
        gene_to_index: Unique gene-name to expression-row mapping.
        sample_count: Number of dataset samples.

    Returns:
        One Boolean value per sample; ``True`` means all signature
        terms are satisfied.

    Raises:
        ValueError: If an unexpected operator reaches evaluation.
    """
    matches = np.ones(sample_count, dtype=np.bool_)

    for condition in signature.terms:
        gene_values = values_by_gene_index[gene_to_index[condition.gene]]
        if condition.operator == "<":
            matches &= gene_values < condition.threshold
        elif condition.operator == ">":
            matches &= gene_values > condition.threshold
        else:
            raise ValueError(
                f"Unsupported operator {condition.operator!r} reached "
                f"evaluation for signature #{signature.number}."
            )

    return matches


def validate_signature_cluster_ids(
    signature: Signature,
    *,
    sample_count: int,
) -> None:
    """Validate Tjala-reported clusters against node-ID domain.

    Validate Tjala-reported clusters against the immutable node-ID
    domain.

    Args:
        signature: Selected Tjala signature.
        sample_count: Number of terminal sample leaves.

    Raises:
        ValueError: If a reported cluster is not in ``0..2N-2``.
    """
    max_node_id = tree_node_count(sample_count) - 1
    invalid: list[str] = []
    for cluster_id in signature.cluster_ids:
        node_id = int(cluster_id)
        if node_id > max_node_id:
            invalid.append(cluster_id)

    if invalid:
        raise ValueError(
            f"Signature #{signature.number} contains immutable node ID(s) "
            f"outside 0..{max_node_id}: {invalid}."
        )


def validate_signature_cluster_consistency(
    signature: Signature,
    *,
    hierarchy_rows: tuple[frozenset[str], ...],
    covered_mask: BoolVector,
) -> None:
    """Compare reported clusters with recomputed covered samples.

    Current Tjala output lists every cluster having nonzero membership
    among the samples covered by the signature, split between Can and
    Cannot sections. This check detects stale/mismatched pipeline files
    and also guarantees that every listed cluster exists in the
    hierarchy.

    Args:
        signature: Selected signature.
        hierarchy_rows: Per-sample cluster memberships in sample order.
        covered_mask: Samples recomputed from the signature antecedent.

    Raises:
        ValueError: If listed and recomputed covered clusters differ.
    """
    listed_clusters = set(signature.cluster_ids)
    covered_clusters: set[str] = set()
    for sample_index in np.flatnonzero(covered_mask):
        covered_clusters.update(hierarchy_rows[int(sample_index)])

    missing_from_output = covered_clusters - listed_clusters
    unexpected_in_output = listed_clusters - covered_clusters
    if missing_from_output or unexpected_in_output:
        details: list[str] = []
        if missing_from_output:
            details.append(
                "covered cluster(s) missing from tjala_output.txt: "
                f"{sorted(missing_from_output)}"
            )
        if unexpected_in_output:
            details.append(
                "cluster(s) listed in tjala_output.txt but not represented by "
                f"recomputed covered samples: {sorted(unexpected_in_output)}"
            )
        raise ValueError(
            f"Signature #{signature.number} is inconsistent with the current "
            f"expression/hierarchy inputs: {'; '.join(details)}."
        )


def build_hierarchy_rank(
    signature: Signature,
    *,
    hierarchy_rows: tuple[frozenset[str], ...],
) -> dict[int, int]:
    """Build the existing hierarchy-derived sample ordering.

    Signature-associated clusters are processed in their Tjala output
    order. Within each cluster, samples are visited in physical
    hierarchy/ ``matrix_samples.txt`` order. A sample belonging to
    multiple
    clusters receives the rank of its first encounter.

    Args:
        signature: Selected signature with ordered associated clusters.
        hierarchy_rows: Per-sample hierarchy memberships.

    Returns:
        Mapping from zero-based sample index to hierarchy rank.

    Raises:
        ValueError: If a signature cluster is absent from the hierarchy.
    """
    cluster_to_sample_indices: dict[str, list[int]] = {}
    for sample_index, clusters in enumerate(hierarchy_rows):
        for cluster_id in clusters:
            cluster_to_sample_indices.setdefault(cluster_id, []).append(
                sample_index)

    missing_clusters = [
        cluster_id
        for cluster_id in signature.cluster_ids
        if cluster_id not in cluster_to_sample_indices
    ]
    if missing_clusters:
        raise ValueError(
            f"Signature #{signature.number} references cluster(s) absent from "
            f"{SAMPLE_LINEAGES_FILE}: {missing_clusters}."
        )

    ordered_sample_indices: list[int] = []
    seen_sample_indices: set[int] = set()

    for cluster_id in signature.cluster_ids:
        for sample_index in cluster_to_sample_indices[cluster_id]:
            if sample_index in seen_sample_indices:
                continue
            seen_sample_indices.add(sample_index)
            ordered_sample_indices.append(sample_index)

    return {
        sample_index: rank
        for rank, sample_index in enumerate(ordered_sample_indices)
    }


def order_covered_samples(
    *,
    signature: Signature,
    samples: list[str],
    covered_mask: BoolVector,
    hierarchy_rank: dict[int, int],
) -> list[str]:
    """Order all covered samples by reported hierarchy membership.

    Args:
        signature: Selected signature, used in diagnostics.
        samples: Sample identifiers in ``matrix_samples.txt`` order.
        covered_mask: One Boolean signature result per sample.
        hierarchy_rank: Sample-index to hierarchy-rank mapping.

    Returns:
        Covered sample identifiers in the required output order.

    Raises:
        ValueError: If any covered sample cannot be positioned by the
            hierarchy ordering contract.
    """
    covered_indices = [int(index) for index in np.flatnonzero(covered_mask)]
    missing_indices = [
        sample_index
        for sample_index in covered_indices
        if sample_index not in hierarchy_rank
    ]
    if missing_indices:
        missing_samples = [samples[index] for index in missing_indices]
        raise ValueError(
            f"Signature #{signature.number} covers sample(s) that cannot be "
            "positioned by its hierarchy-associated clusters: "
            + ", ".join(repr(sample) for sample in missing_samples)
        )

    covered_indices.sort(key=hierarchy_rank.__getitem__)
    return [samples[index] for index in covered_indices]


# ======================================================================
# Block 4: Output Publication
# ======================================================================


def write_results_atomically(
    output_path: Path,
    *,
    signature_number: int,
    ordered_samples: list[str],
) -> None:
    """Write the sample list using atomic file replacement.

    Args:
        output_path: Final ``signature_N.txt`` destination.
        signature_number: Selected signature number.
        ordered_samples: Covered sample identifiers in final order.

    Raises:
        OSError: If writing or replacement fails.
    """
    lines = [
        f"Signature #{signature_number}",
        "Samples that satisfy the signature:",
        *ordered_samples,
    ]
    write_text_atomically(output_path, "\n".join(lines) + "\n")


def run(base_path: Path) -> Path | None:
    """Run the interactive extraction workflow.

    Args:
        base_path: Directory containing all pipeline inputs and
            receiving the output file.

    Returns:
        Final output path, or ``None`` when the user elects to exit.

    Raises:
        OSError: If required I/O fails.
        ValueError: If any input violates the established pipeline
            contract.
    """
    paths = ensure_required_files(base_path)
    verify_artifacts(paths.values())

    signatures = list(read_signature_report(paths["tjala_output"]).signatures)
    if not signatures:
        raise ValueError(f"No signatures were found in {TJALA_OUTPUT_FILE}.")
    selected_signature = prompt_for_signature(signatures)
    if selected_signature is None:
        return None

    genes = read_identifiers(paths["genes"], label="gene name")
    samples = read_identifiers(paths["columns"], label="sample ID")

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    validate_signature_genes(
        selected_signature,
        gene_to_index=gene_to_index,
    )

    required_gene_indices = frozenset(
        gene_to_index[condition.gene]
        for condition in selected_signature.terms
    )
    # Every row is checked; only the rows the signature tests are kept.
    values_by_gene_index: dict[int, FloatVector] = {}
    row_count = 0
    for index, row in iter_matrix_rows(paths["values"], len(samples)):
        row_count = index + 1
        if index in required_gene_indices:
            values_by_gene_index[index] = row.astype(np.float64)
    if row_count != len(genes):
        raise ValueError(
            f"Row-count mismatch: {MATRIX_VALUES_FILE} has {row_count} "
            f"row(s), but {MATRIX_GENES_FILE} has {len(genes)} gene(s).")
    # The lineages must be the ones the history implies (task 2.9).
    history, _ = load_tree_states(base_path / TREE_STATES_FILE)
    lineages = read_sample_lineages(paths["hierarchy"], samples,
                                    history=history)
    hierarchy_rows = tuple(
        frozenset(str(node) for node in lineages[sample])
        for sample in samples
    )
    validate_signature_cluster_ids(
        selected_signature,
        sample_count=len(samples),
    )

    covered_mask = evaluate_signature(
        selected_signature,
        values_by_gene_index=values_by_gene_index,
        gene_to_index=gene_to_index,
        sample_count=len(samples),
    )

    validate_signature_cluster_consistency(
        selected_signature,
        hierarchy_rows=hierarchy_rows,
        covered_mask=covered_mask,
    )

    hierarchy_rank = build_hierarchy_rank(
        selected_signature,
        hierarchy_rows=hierarchy_rows,
    )
    ordered_samples = order_covered_samples(
        signature=selected_signature,
        samples=samples,
        covered_mask=covered_mask,
        hierarchy_rank=hierarchy_rank,
    )

    output_path = base_path / f"signature_{selected_signature.number}.txt"
    write_results_atomically(
        output_path,
        signature_number=selected_signature.number,
        ordered_samples=ordered_samples,
    )
    write_provenance(
        provenance_path(output_path), stage="signature_export",
        outputs=[output_path], inputs=paths.values(),
        parents={provenance_path(path) for path in paths.values()},
        producer=Path(__file__),
    )
    return output_path


def main() -> int:
    """Run the command-line workflow."""
    try:
        output_path = run(Path.cwd())
    except (OSError, ValueError) as exc:
        print(f"Error: [signature_export] {exc}", file=sys.stderr)
        return 1

    if output_path is not None:
        print(f"Done! Results written to {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
