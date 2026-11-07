#!/usr/bin/env python3
"""Restartable preparation, build, and variance computation.

The pipeline has three restartable stages:

    prepare
        total_data.csv + per-sample expression CSVs
        -> matrix_genes.txt / matrix_samples.txt /
           sample_lineages.txt / matrix_values.txt

    build
        compute_variances.cpp -> native executable

    run
        generated inputs + native executable -> term_variances.txt

Running without a stage, or with ``all``, executes all three stages.
Durable state in ``compute_variances_state.json`` records the committed
input set and built executable so interrupted multi-file publication is
detected before a later run stage is allowed to proceed.

Requires Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, TypedDict, cast

import numpy as np
import pandas as pd

from pipeline_formats import (
    INT32_INFO,
    SampleLineages,
    load_tree_states,
    read_sample_expression,
    render_sample_lineages,
    sample_id,
    tree_node_count,
    write_matrix_values,
)
from pipeline_io import (
    INPUT_FILES,
    MATRIX_GENES_FILE,
    MATRIX_SAMPLES_FILE,
    MATRIX_VALUES_FILE,
    NATIVE_FILES_HEADER_FILE,
    PREPARE_MARKER_FILE,
    PREPARED_PROVENANCE_FILE,
    SAMPLE_LINEAGES_FILE,
    TERM_VARIANCES_FILE,
    TREE_STATES_FILE,
    TREE_TOPOLOGY_FILE,
    invalidate_dependents,
    parse_canonical_int,
    parse_decimal_float,
    provenance_path,
    read_identifiers,
    read_utf8_lines,
    render_identifiers,
    sha256_file,
    validate_identifier,
    verify_artifacts,
    write_provenance,
    write_text_atomically,
)
from pipeline_native import (
    EXE_SUFFIX,
    build_program,
    find_compiler,
    run_program,
)

LOGGER = logging.getLogger("compute_variances")

VARIANCE_SOURCE_FILE = "compute_variances.cpp"
VARIANCE_STATE_FILE = "compute_variances_state.json"
EXECUTABLE_STEM = "compute_variances"
STATE_SCHEMA_VERSION = 2


class BuildState(TypedDict):
    """Metadata for one committed native build.

    Attributes:
        compiler: Resolved compiler executable used by the build.
        compiler_family: Compiler driver family, either GNU-style or
            MSVC-style.
        source_sha256: Exact native source-file fingerprint.
        artifact_header_sha256: Exact shared filename-header
            fingerprint.
        executable: Committed executable basename in the working
            directory.
        executable_sha256: Exact committed executable fingerprint.
    """

    compiler: str
    compiler_family: str
    source_sha256: str
    artifact_header_sha256: str
    executable: str
    executable_sha256: str


class VarianceState(TypedDict):
    """Integrity metadata for a committed variance result.

    Attributes:
        sha256: Exact variance-output fingerprint.
        input_fingerprint: Fingerprint of the four prepared input-file
            hashes.
        executable_sha256: Exact committed executable fingerprint.
    """

    sha256: str
    input_fingerprint: str
    executable_sha256: str


class VarianceStageState(TypedDict, total=False):
    """Validated schema for the restartable pipeline state file.

    Attributes:
        schema_version: Version of the persisted state schema.
        inputs: Canonical prepared filenames mapped to SHA-256 hashes.
        input_fingerprint: Fingerprint of the four prepared input-file
            hashes.
        build: Committed build metadata, or None when no build is
            trusted.
        variance: Committed variance metadata, or None after
            invalidation.
    """

    schema_version: int
    inputs: dict[str, str]
    input_fingerprint: str
    build: BuildState | None
    variance: VarianceState | None


# ======================================================================
# Block 1: Small Utilities
# ======================================================================


def validate_name(name: str, *, what: str, source: str) -> None:
    """Validate a literal identifier for downstream text formats.

    Args:
        name: Gene or sample identifier after ingestion normalization.
        what: Identifier kind used in diagnostics.
        source: File and field location.

    Raises:
        ValueError: If the identifier contains whitespace or controls.
    """
    validate_identifier(name, source=f"{source} ({what})")


def resolve_sample_path(raw_path: str, data_dir: Path) -> Path:
    """Resolve one manifest entry to an existing sample CSV.

    The manifest may have been generated on another platform. The path
    is therefore tried literally first, then by basename under
    ``data_dir``.

    Args:
        raw_path: Path string stored in the manifest.
        data_dir: Fallback directory containing the per-sample CSVs.

    Returns:
        The resolved sample file path.

    Raises:
        FileNotFoundError: If neither candidate exists as a file.
    """
    literal = Path(raw_path)
    if literal.is_file():
        return literal

    candidate = data_dir / sample_id(raw_path)
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not locate sample file for manifest entry {raw_path!r}: "
        f"tried {literal} and {candidate}"
    )


def input_hashes(directory: Path) -> dict[str, str]:
    """Hash the four generated C++ input files.

    Args:
        directory: Directory containing the generated inputs.

    Returns:
        Mapping from canonical filename to SHA-256 digest.
    """
    return {name: sha256_file(directory / name) for name in INPUT_FILES}


def input_fingerprint(hashes: dict[str, str]) -> str:
    """Build one deterministic fingerprint for a generated input set.

    Args:
        hashes: Per-input SHA-256 mapping.

    Returns:
        SHA-256 digest covering all canonical filenames and their
        digests.
    """
    payload = (
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_state_record(
    value: object,
    *,
    field_name: str,
    required_string_fields: tuple[str, ...],
    path: Path,
) -> None:
    """Validate the fields of a persisted state record.

    Args:
        value: Deserialized field value.
        field_name: State-field name used in diagnostics.
        required_string_fields: Required keys whose values must be
            strings.
        path: State-file path used in diagnostics.

    Raises:
        ValueError: If the record does not match its expected schema.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(
            (
                f'Pipeline state field {field_name!r} in {path} '
                f'must be an object or null'
            )
        )

    invalid = [
        key for key in required_string_fields if not isinstance(value.get(
            key), str)
    ]
    if invalid:
        raise ValueError(
            (
                f'Pipeline state field {field_name!r} in {path} '
                f'has missing or invalid string field(s): '
                f'{invalid}'
            )
        )


def load_state(out_dir: Path) -> VarianceStageState:
    """Load pipeline state, returning an empty state if absent.

    Args:
        out_dir: Pipeline output directory.

    Returns:
        Validated state dictionary, or an empty dictionary if no state
        exists.

    Raises:
        ValueError: If the state file is malformed, has an unsupported
            schema, or does not match the expected state structure.
    """
    path = out_dir / VARIANCE_STATE_FILE
    if not path.exists():
        return {}

    try:
        loaded: object = (
            json.loads("\n".join(read_utf8_lines(path, allow_blank=True)))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read valid pipeline state from {path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"Pipeline state {path} must contain a JSON object")

    version = loaded.get("schema_version")
    if type(version) is not int or version != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pipeline state schema {version!r} in {path}; "
            f"expected {STATE_SCHEMA_VERSION}"
        )

    if "inputs" in loaded:
        inputs = loaded["inputs"]
        if not isinstance(inputs, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in inputs.items()
        ):
            raise ValueError(
                f"Pipeline state field 'inputs' in {path} must be an object "
                "mapping strings to strings"
            )

    if "input_fingerprint" in loaded and not isinstance(
        loaded["input_fingerprint"], str
    ):
        raise ValueError(
            (
                f"Pipeline state field 'input_fingerprint' in "
                f"{path} must be a string"
            )
        )

    _validate_state_record(
        loaded.get("build"),
        field_name="build",
        required_string_fields=(
            "compiler",
            "compiler_family",
            "source_sha256",
            "artifact_header_sha256",
            "executable",
            "executable_sha256",
        ),
        path=path,
    )
    _validate_state_record(
        loaded.get("variance"),
        field_name="variance",
        required_string_fields=(
            "sha256",
            "input_fingerprint",
            "executable_sha256",
        ),
        path=path,
    )

    # All fields consumed by the program have been checked at the JSON
    # boundary. Unknown fields are retained so a same-schema producer
    # can add nonessential metadata without this reader silently
    # discarding it.
    return cast(VarianceStageState, loaded)


def write_state_atomically(out_dir: Path, state: VarianceStageState) -> None:
    """Atomically replace the pipeline-state JSON file.

    Args:
        out_dir: Pipeline output directory.
        state: Complete state object to publish.

    Raises:
        OSError: If the temporary state cannot be written or replaced.
    """
    state_to_write: dict[str, object] = dict(state)
    state_to_write["schema_version"] = STATE_SCHEMA_VERSION
    payload = json.dumps(state_to_write, indent=2, sort_keys=True) + "\n"
    write_text_atomically(out_dir / VARIANCE_STATE_FILE, payload)


def invalidate_variance(
    out_dir: Path, state: VarianceStageState, *, reason: str
) -> None:
    """Invalidate variance output after its dependencies change.

    Args:
        out_dir: Pipeline output directory.
        state: Mutable pipeline state object.
        reason: Human-readable reason used in logging.
    """
    variance_path = out_dir / TERM_VARIANCES_FILE
    invalidate_dependents([variance_path], [out_dir, out_dir.parent])
    provenance_path(variance_path).unlink(missing_ok=True)
    if variance_path.exists():
        LOGGER.info("Removing stale %s (%s)", TERM_VARIANCES_FILE, reason)
        variance_path.unlink()
    state["variance"] = None


# ======================================================================
# Block 2: Gene Filters and Compressed History
# ======================================================================
def load_gene_filter(path: str | None) -> set[str] | None:
    """Load and validate an optional one-gene-per-line allowlist.

    ``None`` means no filtering, while an empty set means a supplied
    filter file contained no usable gene names. UTF-8 BOMs are accepted,
    matching the CSV readers. Blank or whitespace-only lines are
    ignored; whitespace inside an actual gene name is rejected rather
    than silently stripped.

    Args:
        path: Allowlist path, or ``None`` for no filter.

    Returns:
        A set of validated gene names, or ``None``.

    Raises:
        ValueError: If a nonblank gene name contains whitespace that
            downstream text formats cannot encode.
    """
    if path is None:
        return None

    genes: set[str] = set()
    for line_number, gene in enumerate(
        read_utf8_lines(Path(path), allow_blank=True, allow_empty=True), 1
    ):
        if not gene or gene.isspace():
            continue
        validate_name(gene, what="gene name", source=f"{path}:{line_number}")
        genes.add(gene)

    return genes


# ======================================================================
# Block 3: Expression Matrix
# ======================================================================


def build_matrix(raw_paths: Iterable[str], data_dir: Path) -> pd.DataFrame:
    """Assemble a deterministic genes-by-samples matrix.

    Missing genes introduced by cross-sample index alignment become
    zero. A missing or invalid expression value inside an actual source
    row is rejected before alignment, so it cannot be silently converted
    to zero.

    Args:
        raw_paths: Manifest sample paths in desired sample order.
        data_dir: Fallback directory containing sample CSVs.

    Returns:
        Dense floating-point expression matrix with sorted genes.
    """
    series_by_sample: dict[str, pd.Series] = {}
    folders: dict[str, list[str]] = {}

    for raw_path in raw_paths:
        sid = sample_id(raw_path)
        resolved = resolve_sample_path(raw_path, data_dir)
        series_by_sample[sid] = read_sample_expression(resolved, sid)
        folders.setdefault(str(resolved.parent), []).append(sid)

    matrix = pd.DataFrame(series_by_sample)
    matrix = matrix.reindex(sorted(matrix.index))
    # Zero-fill stays the rule; say how many genes it supplies where.
    for folder, members in folders.items():
        filled = int(matrix[members].isna().any(axis=1).sum())
        if filled:
            print(
                f"Warning: {filled} gene(s) absent from samples in "
                f"{folder} are set to 0 there.", file=sys.stderr,
            )
    matrix = matrix.fillna(0.0)
    return matrix


def filter_nonzero_genes(matrix: pd.DataFrame) -> pd.DataFrame:
    """Drop genes whose original expression is zero in every sample.

    Evaluate zero expression before integer truncation so fractional
    nonzero source values remain represented in the retained gene set.

    Args:
        matrix: Floating-point expression matrix.

    Returns:
        Matrix containing only genes with at least one nonzero
        expression.
    """
    return matrix.loc[matrix.any(axis=1)]


def apply_gene_filter(matrix: pd.DataFrame, genes: set[str] | None) -> (
    pd.DataFrame
):
    """Apply an optional gene allowlist.

    Args:
        matrix: Expression matrix.
        genes: Gene allowlist, or ``None`` to keep every gene.

    Returns:
        Filtered expression matrix.
    """
    if genes is None:
        return matrix
    return matrix.loc[matrix.index.isin(genes)]


def to_integer_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Truncate expressions toward zero using downstream 32-bit.

    Truncate expressions toward zero using the downstream 32-bit
    contract.

    The input DataFrame is never modified. Bounds are checked from
    scalar extrema before allocating the int32 destination, avoiding an
    additional full float64 truncation copy and full-size boolean mask
    on the normal path.

    Args:
        matrix: Floating-point expression matrix.

    Returns:
        Matrix of signed 32-bit integers.

    Raises:
        ValueError: If a value is non-finite or its truncated value
            cannot be represented by the 32-bit signed integer format
            accepted by the C++ reader on the supported toolchains.
    """
    values = matrix.to_numpy(dtype=np.float64, copy=False)

    if values.size == 0:
        integers = np.empty(values.shape, dtype=np.int32)
        return pd.DataFrame(
            integers,
            index=matrix.index,
            columns=matrix.columns,
            copy=False,
        )

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(
            "Non-finite value present in expression matrix before truncation"
        )

    truncated_minimum = float(np.trunc(minimum))
    truncated_maximum = float(np.trunc(maximum))
    if (
        truncated_minimum < INT32_INFO.min
        or truncated_maximum > INT32_INFO.max
    ):
        # Truncation toward zero is in range exactly when the original
        # finite value lies strictly between these two adjacent integer
        # thresholds.
        lower_invalid = float(INT32_INFO.min) - 1.0
        upper_invalid = float(INT32_INFO.max) + 1.0
        out_of_range = (values <= lower_invalid) | (values >= upper_invalid)
        flat_index = int(np.flatnonzero(out_of_range)[0])
        row_index, column_index = np.unravel_index(flat_index, values.shape)
        row = int(row_index)
        column = int(column_index)
        raise ValueError(
            "Expression value truncates outside the signed 32-bit range for "
            f"gene {matrix.index[row]!r}, sample {matrix.columns[column]!r}: "
            f"{values[row, column]!r}"
        )

    # NumPy float-to-integer conversion truncates toward zero.
    # Finiteness and post-truncation bounds were validated above, so
    # this direct conversion needs only the int32 destination rather
    # than another float64 matrix.
    integers = values.astype(np.int32, copy=True)
    return pd.DataFrame(
        integers,
        index=matrix.index,
        columns=matrix.columns,
        copy=False,
    )


# ======================================================================
# Block 4: Prepared Input Publication
# ======================================================================
def _write_staged_inputs(
    matrix: pd.DataFrame,
    hierarchy: SampleLineages,
    staging_dir: Path,
) -> None:
    """Write the prepared matrix and identifiers into staging.

    Args:
        matrix: Filtered floating-point expression matrix.
        hierarchy: Sample-to-node membership mapping.
        staging_dir: Empty staging directory.

    Raises:
        KeyError: If a matrix sample has no hierarchy entry.
        ValueError: If a gene name or hierarchy reference is invalid.
    """
    sample_names = list(matrix.columns)
    gene_names = [str(gene) for gene in matrix.index]

    missing = [sample for sample in sample_names if sample not in hierarchy]
    if missing:
        raise KeyError(f"No hierarchy entry for sample(s): {missing}")

    for gene in gene_names:
        validate_name(gene, what="gene name", source="expression matrix")

    node_count = tree_node_count(len(sample_names))
    max_node_id = node_count - 1

    for sample in sample_names:
        validate_name(sample, what="sample name", source="expression matrix")
        node_ids = hierarchy[sample]
        if not node_ids or node_ids[0] != 0:
            raise ValueError(
                (
                    f'SampleLineages for sample {sample!r} must include '
                    f'immutable root node 0'
                )
            )
        for node_id in node_ids:
            if node_id < 0 or node_id > max_node_id:
                raise ValueError(
                    (
                        f'SampleLineages for sample {sample!r} contains node '
                        f'ID {node_id}; valid immutable node IDs are '
                        f'0..{max_node_id}'
                    )
                )

    (staging_dir / MATRIX_GENES_FILE).write_text(
        render_identifiers(gene_names), encoding="utf-8", newline="\n"
    )
    (staging_dir / MATRIX_SAMPLES_FILE).write_text(
        render_identifiers(sample_names), encoding="utf-8", newline="\n"
    )
    (staging_dir / SAMPLE_LINEAGES_FILE).write_text(
        render_sample_lineages(sample_names, hierarchy),
        encoding="utf-8", newline="\n",
    )
    write_matrix_values(staging_dir / MATRIX_VALUES_FILE,
                        to_integer_matrix(matrix).to_numpy())


def publish_inputs(
    matrix: pd.DataFrame,
    hierarchy: SampleLineages,
    out_dir: Path,
    *,
    source_paths: Iterable[Path] = (),
    parent_manifests: Iterable[Path] = (),
) -> None:
    """Publish a complete prepared input set with restart detection.

    The four input files cannot be replaced as one filesystem
    transaction. Instead, a prepare marker is created before publication
    and the committed hashes are written atomically to
    ``compute_variances_state.json`` only after every replacement
    succeeds. The run stage refuses to proceed while the marker exists
    or the committed hashes do not match.

    A variance result is removed before publishing changed inputs so a
    failed prepare cannot leave an old ``term_variances.txt`` looking
    current.

    Args:
        matrix: Filtered floating-point expression matrix.
        hierarchy: Sample-to-node membership mapping.
        out_dir: Pipeline output directory.
        source_paths: Upstream history, sample data, and selected
            allowlist.
        parent_manifests: Verified provenance of generated source files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / PREPARE_MARKER_FILE

    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".compute_variances_prepare_"
    ) as staging_name:
        staging_dir = Path(staging_name)
        _write_staged_inputs(matrix, hierarchy, staging_dir)
        staged_hashes = input_hashes(staging_dir)
        staged_fingerprint = input_fingerprint(staged_hashes)

        state = load_state(out_dir)
        old_fingerprint = state.get("input_fingerprint")
        prepared_manifest = out_dir / PREPARED_PROVENANCE_FILE
        old_commit_hash = (
            sha256_file(prepared_manifest)
            if prepared_manifest.is_file() else None
        )
        inputs_changed = old_fingerprint != staged_fingerprint

        if not inputs_changed:
            _invalidate_untrusted_variance(out_dir, state)

        # From this point onward, failure may leave some final inputs
        # replaced. Keep the marker until the new state has been
        # committed successfully.
        marker.write_text("prepare in progress\n", encoding="utf-8")

        if inputs_changed:
            invalidate_dependents(
                [out_dir / name for name in INPUT_FILES],
                [out_dir, out_dir.parent],
            )
            invalidate_variance(out_dir, state,
                reason="generated inputs changed")
        (out_dir / PREPARED_PROVENANCE_FILE).unlink(missing_ok=True)

        # Individual replacements are atomic. The marker + committed
        # state make an interrupted multi-file sequence detectable on
        # restart.
        for name in INPUT_FILES:
            os.replace(staging_dir / name, out_dir / name)

        state["inputs"] = staged_hashes
        state["input_fingerprint"] = staged_fingerprint
        if "build" not in state:
            state["build"] = None
        if "variance" not in state:
            state["variance"] = None

        write_state_atomically(out_dir, state)
        write_provenance(
            out_dir / PREPARED_PROVENANCE_FILE, stage="prepare",
            outputs=[out_dir / name for name in INPUT_FILES],
            inputs=source_paths, parents=parent_manifests,
            producer=Path(__file__),
        )
        if old_commit_hash != sha256_file(prepared_manifest):
            invalidate_dependents(
                [prepared_manifest], [out_dir, out_dir.parent]
            )
            state["variance"] = None
            write_state_atomically(out_dir, state)

    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            (
                f'Prepared inputs and state were committed, but '
                f'the prepare marker {marker} could not be '
                f'removed. Fix the filesystem problem and rerun '
                f'the prepare stage before running variances.'
            )
        ) from exc


def ensure_committed_inputs(out_dir: Path) -> VarianceStageState:
    """Verify inputs against the last successful preparation.

    Args:
        out_dir: Pipeline output directory.

    Returns:
        Current pipeline state.

    Raises:
        RuntimeError: If preparation was interrupted, no committed
            prepare exists, an input file is missing, or an input hash
            has changed.
    """
    marker = out_dir / PREPARE_MARKER_FILE
    if marker.exists():
        raise RuntimeError(
            f"Detected interrupted prepare stage ({marker.name} exists). "
            "Rerun the prepare stage before computing variances."
        )

    verify_artifacts(out_dir / name for name in INPUT_FILES)
    state = load_state(out_dir)
    committed = state.get("inputs")
    fingerprint = state.get("input_fingerprint")
    if committed is None or fingerprint is None:
        raise RuntimeError(
            (
                'No committed prepare stage is recorded. Run the '
                'prepare stage first.'
            )
        )

    actual_hashes: dict[str, str] = {}
    for name in INPUT_FILES:
        expected = committed.get(name)
        if expected is None:
            raise RuntimeError(
                f"Pipeline state is missing the committed hash for {name}"
            )

        path = out_dir / name
        if not path.is_file():
            raise RuntimeError(f"Committed generated input is missing: {path}")

        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Generated input {name} does not match the last successful "
                "prepare stage. Rerun prepare before continuing."
            )
        actual_hashes[name] = actual

    if input_fingerprint(actual_hashes) != fingerprint:
        raise RuntimeError(
            (
                'Generated input fingerprint does not match the '
                'committed prepare stage'
            )
        )

    return state


def ensure_committed_variance(
    out_dir: Path, state: VarianceStageState | None = None
) -> Path:
    """Verify variance output against its committed input and build.

    The recorded variance hash is authoritative: when a variance record
    exists, the output file must exist, match that hash, and refer to
    the currently committed input fingerprint and native executable
    hash.

    Args:
        out_dir: Pipeline output directory.
        state: Already loaded pipeline state, or ``None`` to load it
            here.

    Returns:
        Path to the verified committed variance output.

    Raises:
        RuntimeError: If no committed variance exists or if its file,
            hash, or provenance does not match the committed pipeline
            state.
    """
    out_dir = out_dir.resolve()
    verify_artifacts(
        out_dir / name for name in (*INPUT_FILES, TERM_VARIANCES_FILE)
    )
    current_state = load_state(out_dir) if state is None else state

    variance = current_state.get("variance")
    if variance is None:
        raise RuntimeError("No committed variance result is recorded")

    fingerprint = current_state.get("input_fingerprint")
    if fingerprint is None:
        raise RuntimeError(
            "Committed variance exists without a committed input fingerprint"
        )
    if variance["input_fingerprint"] != fingerprint:
        raise RuntimeError(
            (
                'Committed variance input fingerprint does not '
                'match pipeline state'
            )
        )

    build = current_state.get("build")
    if build is None:
        raise RuntimeError((
            'Committed variance exists without a committed '
            'native build'
        ))
    if variance["executable_sha256"] != build["executable_sha256"]:
        raise RuntimeError(
            "Committed variance executable hash does not match pipeline state"
        )

    variance_path = out_dir / TERM_VARIANCES_FILE
    if not variance_path.is_file():
        raise RuntimeError((
            f'Committed variance output is missing: '
            f'{variance_path}'
        ))

    actual_hash = sha256_file(variance_path)
    if actual_hash != variance["sha256"]:
        raise RuntimeError(
            (
                f'Committed variance output {variance_path} does '
                f'not match its stored hash'
            )
        )

    return variance_path


def _invalidate_untrusted_variance(out_dir: Path,
    state: VarianceStageState) -> None:
    """Discard variance output that fails integrity checks.

    Args:
        out_dir: Pipeline output directory.
        state: Mutable pipeline state object.
    """
    if state.get("variance") is None:
        return

    try:
        ensure_committed_variance(out_dir, state)
    except (RuntimeError, OSError, ValueError) as exc:
        LOGGER.warning("Invalidating untrusted variance result: %s", exc)
        invalidate_variance(out_dir, state,
            reason="variance integrity check failed")
        write_state_atomically(out_dir, state)


# ======================================================================
# Block 5: Native Build and Execution
# ======================================================================
def build_native_binary(
    source: Path,
    out_dir: Path,
    *,
    explicit_cxx: str | None = None,
) -> Path:
    """Build the native variance executable and publish it.

    pipeline_native builds the program, or takes it from its cache in
    the per-user cache folder, with the compiler that ``explicit_cxx``,
    ``CXX`` or discovery selects. A copy is published in ``out_dir``,
    where the variance result's provenance names it, unless the copy
    there already matches.

    Args:
        source: C++ source file.
        out_dir: Directory for the executable and pipeline state.
        explicit_cxx: Optional compiler executable from ``--cxx``.

    Returns:
        Published executable path.

    Raises:
        FileNotFoundError: If the C++ source is missing.
        RuntimeError: If no supported compiler can be found or
            compilation fails.
    """
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"C++ source not found: {source}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    compiler = find_compiler(explicit_cxx)
    exe_name = EXECUTABLE_STEM + EXE_SUFFIX
    exe_path = out_dir / exe_name
    source_hash = sha256_file(source)
    artifact_header = source.with_name(NATIVE_FILES_HEADER_FILE)
    if not artifact_header.is_file():
        raise FileNotFoundError(
            f"Shared native filename header not found: {artifact_header}"
        )
    artifact_header_hash = sha256_file(artifact_header)
    built = build_program(source, compiler)
    built_hash = sha256_file(built)
    state = load_state(out_dir)
    existing_build = state.get("build")

    if (
        existing_build is not None
        and existing_build["compiler"] == compiler.path
        and existing_build["compiler_family"] == compiler.family
        and existing_build["source_sha256"] == source_hash
        and existing_build["artifact_header_sha256"] == artifact_header_hash
        and existing_build["executable"] == exe_name
        and existing_build["executable_sha256"] == built_hash
        and exe_path.is_file()
    ):
        actual_executable_hash = sha256_file(exe_path)
        if actual_executable_hash == existing_build["executable_sha256"]:
            _invalidate_untrusted_variance(out_dir, state)
            LOGGER.info("Using existing verified build %s", exe_path)
            return exe_path
        LOGGER.warning(
            (
                'Existing executable does not match committed '
                'build metadata; republishing'
            )
        )

    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".compute_variances_build_"
    ) as build_name:
        build_dir = Path(build_name)
        temporary_exe = build_dir / exe_name
        shutil.copy(built, temporary_exe)

        new_executable_hash = sha256_file(temporary_exe)
        # Reload state in case another trusted process has updated the
        # pipeline metadata meanwhile.
        state = load_state(out_dir)
        old_build = state.get("build")
        old_executable_hash = (
            old_build["executable_sha256"] if old_build is not None else None
        )

        _invalidate_untrusted_variance(out_dir, state)

        new_build: BuildState = {
            "compiler": compiler.path,
            "compiler_family": compiler.family,
            "source_sha256": source_hash,
            "artifact_header_sha256": artifact_header_hash,
            "executable": exe_name,
            "executable_sha256": new_executable_hash,
        }

        if old_executable_hash == new_executable_hash:
            # Replacing identical bytes cannot invalidate a committed
            # variance.
            os.replace(temporary_exe, exe_path)
            state["build"] = new_build
            state.setdefault("variance", None)
            write_state_atomically(out_dir, state)
            return exe_path

        # Preserve the last committed executable and variance until the
        # changed build and its state have both been published. If
        # publication fails, restore the previous canonical files so a
        # good result is not lost.
        previous_exe = build_dir / f"previous_{exe_name}"
        previous_variance = build_dir / f"previous_{TERM_VARIANCES_FILE}"
        had_previous_exe = exe_path.is_file()
        variance_path = out_dir / TERM_VARIANCES_FILE
        had_previous_variance = variance_path.is_file()

        try:
            if had_previous_exe:
                os.replace(exe_path, previous_exe)
            if had_previous_variance:
                os.replace(variance_path, previous_variance)

            os.replace(temporary_exe, exe_path)
            state["build"] = new_build
            state["variance"] = None
            write_state_atomically(out_dir, state)
        except Exception:
            exe_path.unlink(missing_ok=True)
            variance_path.unlink(missing_ok=True)
            if had_previous_exe and previous_exe.exists():
                os.replace(previous_exe, exe_path)
            if had_previous_variance and previous_variance.exists():
                os.replace(previous_variance, variance_path)
            raise

        invalidate_dependents(
            [exe_path, variance_path], [out_dir, out_dir.parent]
        )
        provenance_path(variance_path).unlink(missing_ok=True)

    return exe_path


def _copy_committed_inputs(source_dir: Path, run_dir: Path) -> None:
    """Copy verified native inputs into an isolated run directory.

    The native executable uses fixed relative filenames. Running it in a
    private directory prevents a failed or incomplete invocation from
    modifying the canonical variance output. Copies are used rather than
    links so the native process cannot accidentally mutate the committed
    input files through a link.

    Args:
        source_dir: Directory containing the already-verified committed
            inputs.
        run_dir: Empty temporary directory used as the native working
            directory.
    """
    for name in INPUT_FILES:
        shutil.copyfile(source_dir / name, run_dir / name)
    write_provenance(
        run_dir / PREPARED_PROVENANCE_FILE, stage="prepare",
        outputs=[run_dir / name for name in INPUT_FILES],
        inputs=[source_dir / name for name in INPUT_FILES],
        parents=[source_dir / PREPARED_PROVENANCE_FILE],
        producer=Path(__file__),
    )


def run_native_binary(out_dir: Path) -> Path:
    """Run committed native executable and atomically publish its.

    Run the committed native executable and atomically publish its
    output.

    The native engine is executed in an isolated temporary working
    directory containing copies of the committed inputs. This guarantees
    that a stale pre-existing ``term_variances.txt`` cannot be mistaken
    for fresh output and that a failed native process cannot leave a
    partial canonical result. Only a successfully produced staged
    variance file is atomically moved into place.

    Args:
        out_dir: Pipeline working/output directory.

    Returns:
        Path to the newly validated ``term_variances.txt``.

    Raises:
        RuntimeError: If input or executable provenance is
            missing/mismatched, or if a successful native process does
            not create the expected output.
        NativeProgramError: If the native program exits nonzero.
    """
    out_dir = out_dir.resolve()
    state = ensure_committed_inputs(out_dir)
    build = state.get("build")
    if build is None:
        raise RuntimeError(
            (
                'No committed native build is recorded. Run the '
                'build stage before the run stage.'
            )
        )

    executable_name = build["executable"]
    expected_hash = build["executable_sha256"]

    exe_path = out_dir / executable_name
    if not exe_path.is_file():
        raise RuntimeError((
            f'Committed native executable is missing: '
            f'{exe_path}'
        ))

    actual_hash = sha256_file(exe_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Native executable does not match the committed build. Rerun the "
            "build stage."
        )

    fingerprint = state.get("input_fingerprint")
    if fingerprint is None:
        raise RuntimeError(
            (
                'Committed input fingerprint disappeared from '
                'pipeline state during run'
            )
        )

    variance_path = out_dir / TERM_VARIANCES_FILE
    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".compute_variances_run_"
    ) as run_name:
        run_dir = Path(run_name)
        _copy_committed_inputs(out_dir, run_dir)

        run_program(exe_path, cwd=run_dir)

        staged_variance = run_dir / TERM_VARIANCES_FILE
        if not staged_variance.is_file():
            raise RuntimeError(
                "Native stage completed without producing expected output "
                f"{TERM_VARIANCES_FILE}"
            )

        verify_artifacts([staged_variance])
        validate_variance_output(staged_variance, out_dir)
        variance_hash = sha256_file(staged_variance)
        invalidate_dependents([variance_path], [out_dir, out_dir.parent])
        provenance_path(variance_path).unlink(missing_ok=True)
        os.replace(staged_variance, variance_path)

    state["variance"] = {
        "sha256": variance_hash,
        "input_fingerprint": fingerprint,
        "executable_sha256": actual_hash,
    }
    write_state_atomically(out_dir, state)
    write_provenance(
        provenance_path(variance_path), stage="variances",
        outputs=[variance_path],
        inputs=[out_dir / name for name in INPUT_FILES] + [exe_path],
        parents=[out_dir / PREPARED_PROVENANCE_FILE],
        producer=Path(__file__),
    )
    return variance_path


# ======================================================================
# Block 6: Command-Line Execution
# ======================================================================
def _normalize_gene_argument(value: str | None) -> str | None:
    """Normalize legacy gene-filter syntax.

    A positional ``None`` disables gene filtering. The ``--genes``
    option selects an allowlist through the same normalization rule.

    Args:
        value: Raw CLI value, or ``None`` when no filter was supplied.

    Returns:
        ``None`` for no filtering, otherwise the original path string.
    """
    if value is None or value.lower() == "none":
        return None
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse modern and backward-compatible command-line arguments.

    Supported invocation styles include::

        python compute_variances.py genes.txt
        python compute_variances.py None
        python compute_variances.py
        python compute_variances.py all --genes genes.txt
        python compute_variances.py prepare genes.txt
        python compute_variances.py build
        python compute_variances.py run

    A single positional value that is not a stage name is interpreted
    exactly as the original script interpreted ``sys.argv[1]``: as the
    optional gene allowlist path.  In that compatibility form the
    complete pipeline runs.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Parsed arguments with normalized ``stage`` and ``genes``
        attributes.
    """
    stages = ("all", "prepare", "build", "run")
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, build, and run the compute_variances pipeline. "
            "For backward compatibility, a lone positional PATH is treated "
            "as the gene allowlist and executes all stages."
        )
    )
    parser.add_argument(
        "command_or_genes",
        nargs="?",
        metavar="STAGE|GENES",
        help=(
            "Pipeline stage (all/prepare/build/run), or a legacy positional "
            "gene allowlist path. Use None for no gene filter."
        ),
    )
    parser.add_argument(
        "stage_genes",
        nargs="?",
        metavar="GENES",
        help=(
            "Optional positional gene allowlist after STAGE; mainly provided "
            "for convenient backward-compatible use with prepare/all."
        ),
    )
    parser.add_argument(
        "--genes",
        default=None,
        metavar="PATH",
        help=(
            (
                'Gene allowlist, one name per line. Omit it (or '
                'pass None) to keep all genes.'
            )
        ),
    )
    parser.add_argument(
        "--tree-states",
        default=TREE_STATES_FILE,
        metavar="PATH",
        help=f"Compressed hierarchy history (default: {TREE_STATES_FILE}).",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        metavar="DIR",
        help=(
            "Fallback directory for per-sample CSVs when sample paths are "
            "not directly openable (default: current directory)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        metavar="DIR",
        help="Pipeline working/output directory (default: current directory).",
    )
    parser.add_argument(
        "--cpp-source",
        default=VARIANCE_SOURCE_FILE,
        metavar="PATH",
        help=f"C++ source to compile (default: {VARIANCE_SOURCE_FILE}).",
    )
    parser.add_argument(
        "--cxx",
        default=None,
        metavar="COMPILER",
        help=(
            "C++ compiler executable. Supports cl/clang-cl, g++/MinGW, and "
            "clang++. Overrides CXX when supplied."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include a traceback when the pipeline fails.",
    )

    args = parser.parse_args(argv)
    first = args.command_or_genes
    positional_genes = args.stage_genes
    option_genes = args.genes

    if first in stages:
        args.stage = first
        if positional_genes is not None and option_genes is not None:
            parser.error((
                'gene filter specified both positionally and with '
                '--genes'
            ))
        selected_genes = (
            positional_genes if positional_genes is not None else option_genes
        )
        if args.stage in {"build", "run"} and positional_genes is not None:
            parser.error(
                (
                    f"stage '{args.stage}' does not prepare inputs, "
                    f"so a positional gene filter would have no effect"
                )
            )
    else:
        # Legacy interface: `compute_variances.py genes.txt` or `...
        # None`.
        if positional_genes is not None:
            parser.error(
                "too many positional arguments: use STAGE GENES or a single "
                "legacy GENES argument"
            )
        if first is not None and option_genes is not None:
            parser.error((
                'gene filter specified both positionally and with '
                '--genes'
            ))
        args.stage = "all"
        selected_genes = first if first is not None else option_genes

    args.genes = _normalize_gene_argument(selected_genes)
    del args.command_or_genes
    del args.stage_genes
    return args


def validate_variance_output(path: Path, out_dir: Path) -> None:
    """Validate native variance records before their publication.

    Args:
        path: Newly generated variance text.
        out_dir: Committed input directory containing gene identifiers.

    Raises:
        ValueError: If field counts, terms, or finite variances are
            invalid.
        OSError: If an input cannot be read.
    """
    genes = set(read_identifiers(out_dir / MATRIX_GENES_FILE, label="gene"))
    seen: set[tuple[str, int, int]] = set()
    for number, line in enumerate(read_utf8_lines(path), 1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{number}: expected four variance fields")
        gene, threshold, operator, variance = fields
        value = parse_canonical_int(threshold, source=f"{path}:{number}:2")
        code = parse_canonical_int(operator, source=f"{path}:{number}:3")
        statistic = parse_decimal_float(variance, source=f"{path}:{number}:4")
        key = (gene, value, code)
        if gene not in genes or code not in {0, 1, 2} or statistic < 0:
            raise ValueError(f"{path}:{number}: invalid variance term")
        if key in seen:
            raise ValueError(f"{path}:{number}: duplicate variance term")
        seen.add(key)


def _prepare_stage(args: argparse.Namespace, out_dir: Path) -> None:
    """Execute the prepare stage.

    Args:
        args: Parsed command-line arguments.
        out_dir: Resolved pipeline output directory.
    """
    genes = load_gene_filter(args.genes)
    data_dir = Path(args.data_dir).resolve()
    tree_states_path = Path(args.tree_states).resolve()

    verify_artifacts([tree_states_path, tree_states_path.with_name(
        TREE_TOPOLOGY_FILE)])
    hierarchy, raw_paths = load_tree_states(tree_states_path)
    sample_paths = [resolve_sample_path(raw, data_dir) for raw in raw_paths]
    verify_artifacts(sample_paths)
    matrix = build_matrix(raw_paths, data_dir)
    matrix = filter_nonzero_genes(matrix)
    matrix = apply_gene_filter(matrix, genes)

    if matrix.empty:
        raise ValueError("No genes remain after filtering; nothing to write")

    sources = [
        tree_states_path, tree_states_path.with_name(TREE_TOPOLOGY_FILE)
    ]
    sources.extend(sample_paths)
    if args.genes is not None:
        sources.append(Path(args.genes).resolve())
    parents = (
        {provenance_path(item) for item in sources if item != Path(
            args.genes or "")}
    )
    parents = {item for item in parents if item.is_file()}
    publish_inputs(
        matrix, hierarchy, out_dir, source_paths=sources,
        parent_manifests=parents,
    )
    LOGGER.info(
        "Prepared %d genes x %d samples in %s",
        matrix.shape[0],
        matrix.shape[1],
        out_dir,
    )


def main(argv: list[str]) -> int:
    """Run the command-line workflow.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Process exit status: 0 on success, 1 on failure.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.stage in {"all", "prepare"}:
            _prepare_stage(args, out_dir)

        if args.stage in {"all", "build"}:
            exe_path = build_native_binary(
                Path(args.cpp_source),
                out_dir,
                explicit_cxx=args.cxx,
            )
            LOGGER.info("Built %s", exe_path)

        if args.stage in {"all", "run"}:
            variance_path = run_native_binary(out_dir)
            LOGGER.info("Wrote %s", variance_path)

    except Exception as exc:  # CLI boundary: report and return nonzero.
        if args.debug:
            LOGGER.exception("compute_variances failed")
        else:
            LOGGER.error("Error: [compute_variances] %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
