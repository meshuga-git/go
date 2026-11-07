"""Sample extraction for finalized Tjala signatures.

The signature antecedent determines which samples are extracted.
Immutable tree memberships from ``sample_lineages.txt`` are used to
validate Tjala's reported clusters and reproduce the existing
hierarchy-derived output ordering. Requires Python 3.10 or later.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from compute_variances import (
    MATRIX_GENES_FILE,
    MATRIX_SAMPLES_FILE,
    MATRIX_VALUES_FILE,
    SAMPLE_LINEAGES_FILE,
    TJALA_OUTPUT_FILE,
    iter_utf8_lines,
    parse_canonical_int,
    prompt_input,
    provenance_path,
    read_identifiers,
    read_utf8_lines,
    verify_artifacts,
    write_provenance,
)



_SIGNATURE_HEADER_RE = re.compile(r"^Signature #(\d+):$")
_CLUSTER_COVERAGE_RE = re.compile(
    r"^(?P<cluster>.+):\s+"
    r"(?P<percentage>[+-]?(?:\d+(?:\.\d*)?|\.\d+))%\s+covered$"
)
_ALLOWED_OPERATORS = {"<", ">"}

FloatVector = NDArray[np.float64]
BoolVector = NDArray[np.bool_]


@dataclass(frozen=True)
class Term:
    """One gene-threshold condition in a Tjala signature.

    Attributes:
        gene: Literal gene identifier.
        operator: Strict comparison operator, either ``<`` or ``>``.
        threshold: Finite comparison threshold from the selected report.
    """

    gene: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class Signature:
    """A parsed Tjala signature and its hierarchy-ordering clusters.

    Attributes:
        number: One-based signature number in the report.
        terms: Ordered terms forming the antecedent
            conjunction.
        cluster_ids: Canonical covered-cluster IDs in report section
            order.
    """

    number: int
    terms: tuple[Term, ...]
    cluster_ids: tuple[str, ...]


# ======================================================================
# Block 1: Input Contracts
# ======================================================================


def normalize_cluster_id(cluster_id: str) -> str:
    """Validate and return a canonical immutable node-ID string.

    Args:
        cluster_id: Literal canonical decimal machine field.

    Returns:
        The unchanged validated nonnegative node-ID spelling.

    Raises:
        ValueError: If syntax or sign violates the node-ID grammar.
    """
    node_id = parse_canonical_int(cluster_id, source="cluster ID")
    if node_id < 0:
        raise ValueError("Immutable node ID must be nonnegative")
    return str(node_id)


def _tree_node_count(sample_count: int) -> int:
    """Return the node count for a full immutable binary hierarchy.

    Args:
        sample_count: Number of terminal sample leaves.

    Returns:
        Total immutable node count, ``2 * sample_count - 1``.

    Raises:
        ValueError: If ``sample_count`` is not positive.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    return 2 * sample_count - 1


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


def _read_utf8_lines(path: Path) -> list[str]:
    """Read report text with intentional separator lines preserved.

    Args:
        path: Human-readable signature report or lineage input.

    Returns:
        Literal UTF-8 records; schema-specific callers validate blanks.

    Raises:
        ValueError: If transport encoding or terminators are invalid.
        OSError: If the file cannot be read.
    """
    return read_utf8_lines(path, allow_blank=True)


def load_identifier_file(path: Path, *, label: str) -> list[str]:
    """Read unique literal identifiers in physical-file order.

    Args:
        path: Gene or sample identifier file.
        label: Identifier kind for diagnostics.

    Returns:
        Validated identifiers in physical order.

    Raises:
        ValueError: If identifiers are empty, invalid, or duplicated.
        OSError: If reading fails.
    """
    return read_identifiers(path, label=label)


def load_expression_values(
    path: Path,
    *,
    expected_genes: int,
    expected_samples: int,
    required_gene_indices: frozenset[int],
) -> dict[int, FloatVector]:
    """Validate the matrix and retain only selected gene rows.

    The file is streamed one physical row at a time. Every value is
    still validated, but only rows needed by the selected signature are
    retained in memory. An initial UTF-8 BOM is accepted.

    Args:
        path: ``matrix_values.txt`` path.
        expected_genes: Required number of matrix rows.
        expected_samples: Required number of values in each row.
        required_gene_indices: Zero-based gene rows needed for
            evaluation.

    Returns:
        Mapping from original zero-based gene row index to a finite
        ``float64`` sample vector.

    Raises:
        ValueError: If required indices are invalid or if the matrix is
            empty, malformed, nonnumeric, non-finite, or dimensionally
            inconsistent.
    """
    invalid_indices = sorted(
        index
        for index in required_gene_indices
        if index < 0 or index >= expected_genes
    )
    if invalid_indices:
        raise ValueError(
            f"Required gene index/indices outside the expected range: "
            f"{invalid_indices}."
        )

    selected_rows: dict[int, FloatVector] = {}
    actual_genes = 0

    try:
        values_file = iter_utf8_lines(path)
        for row_number, line in enumerate(values_file, start=1):
            actual_genes = row_number
            tokens = line.split()
            if len(tokens) != expected_samples:
                raise ValueError(
                    (
                        f'Column-count mismatch in {path.name} at row '
                        f'{row_number}: found {len(tokens)} value(s), '
                        f'expected {expected_samples} from '
                        f'{MATRIX_SAMPLES_FILE}.'
                    )
                )

            row_index = row_number - 1
            retained_row = (
                np.empty(expected_samples, dtype=np.float64)
                if row_index in required_gene_indices
                else None
            )

            for column_index, token in enumerate(tokens, start=1):
                try:
                    value = float(parse_canonical_int(
                        token, source=f"{path}:{row_number}:{column_index}"
                    ))
                except ValueError as exc:
                    raise ValueError(
                        f"Nonnumeric value in {path.name} at row "
                        f"{row_number}, column {column_index}: {token!r}."
                    ) from exc

                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value in {path.name} at row "
                        f"{row_number}, column {column_index}."
                    )

                if retained_row is not None:
                    retained_row[column_index - 1] = value

            if retained_row is not None:
                selected_rows[row_index] = retained_row
    except UnicodeError as exc:
        raise ValueError(f"{path.name} is not valid UTF-8.") from exc

    if actual_genes == 0:
        raise ValueError(f"{path.name} is empty.")

    if actual_genes != expected_genes:
        raise ValueError(
            f"Row-count mismatch: {path.name} has {actual_genes} row(s), "
            f"but {MATRIX_GENES_FILE} has {expected_genes} gene(s)."
        )

    missing_required_rows = (
        sorted(required_gene_indices - selected_rows.keys())
    )
    if missing_required_rows:
        raise ValueError(
            f"Required expression row(s) missing from {path.name}: "
            f"{missing_required_rows}."
        )

    return selected_rows


def load_sample_lineages(
    path: Path,
    *,
    samples: list[str],
) -> tuple[frozenset[str], ...]:
    """Load and validate sample memberships in the immutable tree.

    The immutable hierarchy format is ``sample_id count node_id...``.
    Rows may appear in any physical order; they are bound to sample IDs
    explicitly and returned in ``matrix_samples.txt`` order. For ``N``
    samples, valid immutable node IDs are exactly ``0..2N-2`` and every
    lineage begins at root node 0.

    Args:
        path: ``sample_lineages.txt`` path.
        samples: Unique sample IDs from ``matrix_samples.txt`` in output
            order.

    Returns:
        One immutable cluster-membership set per sample, aligned with
        ``samples``.

    Raises:
        ValueError: If rows, sample identities, lineage counts, node
            IDs, root membership, final leaves, or the global
            immutable-node domain are malformed or inconsistent.
    """
    sample_count = len(samples)
    node_count = _tree_node_count(sample_count)
    max_node_id = node_count - 1
    sample_set = set(samples)

    if len(sample_set) != sample_count:
        raise ValueError(
            f"{MATRIX_SAMPLES_FILE} contains duplicate sample IDs; immutable "
            "hierarchy rows cannot be matched unambiguously."
        )

    lines = _read_utf8_lines(path)
    if len(lines) != sample_count:
        raise ValueError(
            f"Hierarchy/sample mismatch: {path.name} has {len(lines)} "
            f"physical row(s), but {MATRIX_SAMPLES_FILE} has {sample_count} "
            "sample(s)."
        )

    hierarchy_by_sample: dict[str, frozenset[str]] = {}
    seen_nodes: set[int] = set()
    final_nodes: set[int] = set()

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            raise ValueError(
                f"Empty hierarchy row detected in {path.name} at line "
                f"{line_number}."
            )

        tokens = stripped.split()
        if len(tokens) < 3:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} must contain "
                "a sample ID, a positive lineage count, and root node 0."
            )

        sample_id = tokens[0]
        if sample_id not in sample_set:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} names unknown "
                f"sample {sample_id!r}."
            )
        if sample_id in hierarchy_by_sample:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} duplicates "
                f"sample {sample_id!r}."
            )

        try:
            declared_count = parse_canonical_int(
                tokens[1], source=f"{path}:{line_number}:2"
            )
        except ValueError as exc:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} has a "
                f"non-integer lineage count: {tokens[1]!r}."
            ) from exc

        if declared_count <= 0:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} must declare "
                "at least root node 0."
            )
        if declared_count > sample_count:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} declares "
                f"{declared_count} node(s), but an immutable root-to-leaf "
                f"lineage with {sample_count} leaves can contain at most "
                f"{sample_count} node(s)."
            )

        raw_node_ids = tokens[2:]
        if declared_count != len(raw_node_ids):
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} declares "
                f"{declared_count} node(s) but contains {len(raw_node_ids)}."
            )

        node_ids: list[int] = []
        for raw_node_id in raw_node_ids:
            try:
                canonical = normalize_cluster_id(raw_node_id)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid immutable node ID in {path.name} at hierarchy "
                    f"row {line_number}: {raw_node_id!r}."
                ) from exc
            if not canonical:
                raise ValueError(
                    f"Hierarchy row {line_number} in {path.name} contains an "
                    "empty immutable node ID."
                )
            node_id = int(canonical)
            if node_id > max_node_id:
                raise ValueError(
                    (
                        f'Immutable node ID {node_id} in {path.name} at '
                        f'row {line_number} is outside the valid range '
                        f'0..{max_node_id} for {sample_count} samples.'
                    )
                )
            node_ids.append(node_id)

        if len(node_ids) != len(set(node_ids)):
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} contains "
                "duplicate immutable node IDs."
            )
        if node_ids[0] != 0:
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} must begin with "
                "immutable root node 0."
            )
        if any(left >= right for left, right in zip(node_ids, node_ids[1:])):
            raise ValueError(
                f"Hierarchy row {line_number} in {path.name} must list its "
                "immutable lineage in strictly increasing node-ID order."
            )

        hierarchy_by_sample[sample_id] = (
            frozenset(str(node_id) for node_id in node_ids)
        )
        seen_nodes.update(node_ids)
        final_nodes.add(node_ids[-1])

    missing_samples = (
        [sample for sample in samples if sample not in hierarchy_by_sample]
    )
    if missing_samples:
        raise ValueError(
            f"{path.name} is missing hierarchy row(s) for sample(s): "
            + ", ".join(repr(sample) for sample in missing_samples)
        )

    expected_nodes = set(range(node_count))
    if seen_nodes != expected_nodes:
        missing_nodes = sorted(expected_nodes - seen_nodes)
        extra_nodes = sorted(seen_nodes - expected_nodes)
        details: list[str] = []
        if missing_nodes:
            details.append(f"missing node IDs {missing_nodes}")
        if extra_nodes:
            details.append(f"unexpected node IDs {extra_nodes}")
        raise ValueError(
            f"{path.name} does not represent the complete immutable tree "
            f"0..{max_node_id}: {'; '.join(details)}."
        )

    if len(final_nodes) != sample_count:
        raise ValueError(
            (
                f'{path.name} has {len(final_nodes)} distinct '
                f'final lineage node(s), but {sample_count} '
                f'singleton leaves are required.'
            )
        )

    return tuple(hierarchy_by_sample[sample] for sample in samples)


# ======================================================================
# Block 2: Signature Parsing
# ======================================================================


def parse_antecedent(line: str, *, signature_number: int) -> (
    tuple[Term, ...]
):
    """Parse one ``IF ... THEN:`` antecedent line.

    Args:
        line: Antecedent text emitted by Tjala.
        signature_number: Signature number used in diagnostics.

    Returns:
        Conditions in their Tjala order.

    Raises:
        ValueError: If the antecedent is malformed, repeats a gene, uses
            an unsupported operator, or contains a non-finite threshold.
    """
    stripped = line.strip()
    prefix = "IF "
    suffix = " THEN:"
    if not stripped.startswith(prefix) or not stripped.endswith(suffix):
        raise ValueError(
            f"Signature #{signature_number} has a malformed antecedent: "
            f"{stripped!r}."
        )

    body = stripped[len(prefix) : -len(suffix)].strip()
    if body == "<empty>":
        return ()
    if not body:
        raise ValueError((
            f'Signature #{signature_number} has an empty '
            f'antecedent.'
        ))

    terms: list[Term] = []
    seen_genes: set[str] = set()

    for term_number, term_text in enumerate(body.split(" AND "), start=1):
        parts = term_text.strip().rsplit(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(
                f"Malformed condition {term_number} in signature "
                f"#{signature_number}: {term_text!r}."
            )

        gene, operator, threshold_text = parts
        gene = gene.strip()
        if not gene:
            raise ValueError(
                f"Term {term_number} in signature #{signature_number} "
                "has an empty gene name."
            )

        if operator not in _ALLOWED_OPERATORS:
            raise ValueError(
                f"Unsupported operator {operator!r} in signature "
                f"#{signature_number}; expected only '<' or '>'."
            )

        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric threshold {threshold_text!r} in signature "
                f"#{signature_number}."
            ) from exc

        if not np.isfinite(threshold):
            raise ValueError(
                f"Non-finite threshold {threshold_text!r} in signature "
                f"#{signature_number}."
            )

        if gene in seen_genes:
            raise ValueError(
                f"Signature #{signature_number} contains more than one "
                f"condition for gene {gene!r}."
            )
        seen_genes.add(gene)
        terms.append(Term(gene, operator, threshold))

    return tuple(terms)


def _parse_cluster_section(
    lines: list[str],
    *,
    signature_number: int,
    section_name: str,
    allow_trailing_content: bool,
) -> list[str]:
    """Parse one rendered cluster-coverage section.

    Args:
        lines: Physical report lines after the section heading.
        signature_number: Signature number for diagnostics.
        section_name: Human-readable section heading.
        allow_trailing_content: Whether a separator ends this section.

    Returns:
        Canonical cluster IDs in their rendered order.

    Raises:
        ValueError: If lines, identifiers, or percentages are malformed.
    """
    clusters: list[str] = []
    saw_none = False

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            if allow_trailing_content:
                break
            raise ValueError(
                f"Unexpected blank line inside {section_name!r} for "
                f"signature #{signature_number}."
            )

        if allow_trailing_content and stripped == "----------":
            break

        if stripped == "None":
            if clusters or saw_none:
                raise ValueError(
                    f"Signature #{signature_number} has an invalid 'None' "
                    f"entry in {section_name!r}."
                )
            saw_none = True
            continue

        match = _CLUSTER_COVERAGE_RE.fullmatch(stripped)
        if match is None:
            raise ValueError(
                f"Malformed cluster coverage line in signature "
                f"#{signature_number} {section_name!r}: {stripped!r}."
            )

        if saw_none:
            raise ValueError(
                f"Signature #{signature_number} lists a cluster after 'None' "
                f"in {section_name!r}."
            )

        percentage = float(match.group("percentage"))
        if not np.isfinite(percentage) or not 0.0 <= percentage <= 100.0:
            raise ValueError(
                f"Invalid cluster coverage percentage {percentage!r} in "
                f"signature #{signature_number}."
            )

        cluster_id = normalize_cluster_id(match.group("cluster"))
        if not cluster_id:
            raise ValueError(
                f"Signature #{signature_number} contains an empty cluster ID."
            )
        clusters.append(cluster_id)

    return clusters


def parse_signature_block(
    *,
    signature_number: int,
    block_lines: list[str],
) -> Signature:
    """Parse one signature block from the Tjala report.

    Args:
        signature_number: Number from the ``Signature #N:`` header.
        block_lines: Lines after that header up to the next signature
            header.

    Returns:
        Parsed signature record.

    Raises:
        ValueError: If required antecedent or consequent structure is
            missing or malformed.
    """
    antecedent_positions = [
        index
        for index, line in enumerate(block_lines)
        if line.strip().startswith("IF ") and line.strip().endswith(" THEN:")
    ]
    if len(antecedent_positions) != 1:
        raise ValueError(
            f"Signature #{signature_number} must contain exactly one "
            f"'IF ... THEN:' antecedent; found {len(antecedent_positions)}."
        )

    antecedent_position = antecedent_positions[0]
    terms = parse_antecedent(
        block_lines[antecedent_position],
        signature_number=signature_number,
    )

    can_positions = [
        index
        for index, line in enumerate(block_lines)
        if line.strip() == "Can clusters:"
    ]
    cannot_positions = [
        index
        for index, line in enumerate(block_lines)
        if line.strip() == "Cannot clusters:"
    ]
    if len(can_positions) != 1 or len(cannot_positions) != 1:
        raise ValueError(
            f"Signature #{signature_number} must contain exactly one "
            "'Can clusters:' section and one 'Cannot clusters:' section."
        )

    can_position = can_positions[0]
    cannot_position = cannot_positions[0]
    if not antecedent_position < can_position < cannot_position:
        raise ValueError(
            f"Signature #{signature_number} has its antecedent/cluster "
            "sections in an unexpected order."
        )

    can_clusters = _parse_cluster_section(
        block_lines[can_position + 1 : cannot_position],
        signature_number=signature_number,
        section_name="Can clusters",
        allow_trailing_content=False,
    )
    cannot_clusters = _parse_cluster_section(
        block_lines[cannot_position + 1 :],
        signature_number=signature_number,
        section_name="Cannot clusters",
        allow_trailing_content=True,
    )

    cluster_ids = can_clusters + cannot_clusters
    seen_clusters: set[str] = set()
    duplicate_clusters: set[str] = set()
    for cluster_id in cluster_ids:
        if cluster_id in seen_clusters:
            duplicate_clusters.add(cluster_id)
        else:
            seen_clusters.add(cluster_id)

    if duplicate_clusters:
        raise ValueError(
            f"Signature #{signature_number} contains duplicate cluster IDs: "
            f"{sorted(duplicate_clusters)}."
        )

    return Signature(
        number=signature_number,
        terms=terms,
        cluster_ids=tuple(cluster_ids),
    )


def load_signatures(path: Path) -> list[Signature]:
    """Parse all signatures from the current Tjala output format.

    Args:
        path: ``tjala_output.txt`` path.

    Returns:
        Signatures in numeric order.

    Raises:
        ValueError: If the file is malformed, contains no signatures, or
            uses non-contiguous/duplicate signature numbers.
    """
    lines = _read_utf8_lines(path)

    headers: list[tuple[int, int]] = []
    for line_index, line in enumerate(lines):
        match = _SIGNATURE_HEADER_RE.fullmatch(line.strip())
        if match is not None:
            headers.append((line_index, int(match.group(1))))

    if not headers:
        raise ValueError(f"No signatures were found in {path.name}.")

    numbers = [number for _, number in headers]
    expected_numbers = list(range(1, len(numbers) + 1))
    if numbers != expected_numbers:
        raise ValueError(
            f"Unexpected signature numbering in {path.name}: found {numbers}, "
            f"expected {expected_numbers}."
        )

    signatures: list[Signature] = []
    for header_offset, (start_index, signature_number) in enumerate(headers):
        end_index = (
            headers[header_offset + 1][0]
            if header_offset + 1 < len(headers)
            else len(lines)
        )
        signatures.append(
            parse_signature_block(
                signature_number=signature_number,
                block_lines=lines[start_index + 1 : end_index],
            )
        )

    return signatures


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
    max_node_id = _tree_node_count(sample_count) - 1
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


def _existing_regular_file_mode(path: Path) -> int | None:
    """Return POSIX permission bits for an existing regular file.

    Symlinks are deliberately not followed because the final output
    operation must replace the symlink itself rather than its target.

    Args:
        path: Final output pathname.

    Returns:
        Existing POSIX permission bits, or ``None`` when they should not
        be preserved.
    """
    if os.name != "posix":
        return None

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None

    if not stat.S_ISREG(metadata.st_mode):
        return None
    return stat.S_IMODE(metadata.st_mode)


def _create_atomic_temp_file(output_path: Path) -> tuple[Path, TextIO]:
    """Create an exclusive temporary file beside the output.

    The exclusive ``x`` open prevents races. Keeping the handle open
    until the content is fully written also prevents another process
    from substituting a different filesystem object between creation and
    writing.

    Args:
        output_path: Final output pathname.

    Returns:
        Temporary pathname and its open text stream.

    Raises:
        FileExistsError: If unique-name creation repeatedly collides.
        OSError: If the temporary file cannot otherwise be created.
    """
    output_dir = output_path.parent
    for _ in range(100):
        candidate = output_dir / (
            f".{output_path.name}.{secrets.token_hex(16)}.tmp"
        )
        try:
            stream = candidate.open("x", encoding="utf-8", newline="\n")
        except FileExistsError:
            continue
        return candidate, stream

    raise FileExistsError(
        f"Could not create a unique temporary file in {output_dir}."
    )


def write_results_atomically(
    output_path: Path,
    *,
    signature_number: int,
    ordered_samples: list[str],
) -> None:
    """Write the sample list using atomic file replacement.

    The final pathname is never resolved through a symlink. An existing
    regular file keeps its POSIX permission bits; a new file uses the
    platform's normal file-creation permissions.

    Args:
        output_path: Final ``signature_N.txt`` destination.
        signature_number: Selected signature number.
        ordered_samples: Covered sample identifiers in final order.

    Raises:
        OSError: If writing, synchronization, permission adjustment, or
            replacement fails.
    """
    output_path = output_path.absolute()
    output_dir = output_path.parent
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}")

    lines = [
        f"Signature #{signature_number}",
        "Samples that satisfy the signature:",
        *ordered_samples,
    ]
    content = "\n".join(lines) + "\n"

    existing_mode = _existing_regular_file_mode(output_path)
    temporary_path: Path | None = None
    try:
        temporary_path, temporary_file = _create_atomic_temp_file(output_path)
        with temporary_file:
            if existing_mode is not None:
                os.chmod(temporary_path, existing_mode)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


# ======================================================================
# Block 5: Interactive Execution
# ======================================================================


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

    signatures = load_signatures(paths["tjala_output"])
    selected_signature = prompt_for_signature(signatures)
    if selected_signature is None:
        return None

    genes = load_identifier_file(paths["genes"], label="gene name")
    samples = load_identifier_file(paths["columns"], label="sample ID")

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    validate_signature_genes(
        selected_signature,
        gene_to_index=gene_to_index,
    )

    required_gene_indices = frozenset(
        gene_to_index[condition.gene]
        for condition in selected_signature.terms
    )
    values_by_gene_index = load_expression_values(
        paths["values"],
        expected_genes=len(genes),
        expected_samples=len(samples),
        required_gene_indices=required_gene_indices,
    )
    hierarchy_rows = load_sample_lineages(
        paths["hierarchy"],
        samples=samples,
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
