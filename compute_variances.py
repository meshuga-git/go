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
import csv
import hashlib
import io
import json
import logging
import math
import ntpath
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, TypedDict, cast

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("compute_variances")

TREE_STATES_FILE = "total_data.csv"
VARIANCE_SOURCE_FILE = "compute_variances.cpp"
MATRIX_GENES_FILE = "matrix_genes.txt"
MATRIX_SAMPLES_FILE = "matrix_samples.txt"
MATRIX_VALUES_FILE = "matrix_values.txt"
SAMPLE_LINEAGES_FILE = "sample_lineages.txt"
TERM_VARIANCES_FILE = "term_variances.txt"
VARIANCE_STATE_FILE = "compute_variances_state.json"
TREE_TOPOLOGY_FILE = "tree_splits.txt"
SAMPLE_PATHS_FILE = "filenames.txt"
TJALA_OUTPUT_FILE = "tjala_output.txt"
DENDROGRAM_FILE = "dendrogram.html"
BINARY_VECTORS_FILE = "binary_vectors.csv"
CANNOT_LINKS_FILE = "cannot_links.txt"
MUST_LINKS_FILE = "must_links.txt"
NO_LINKS_FILE = "no_links.txt"
CLUSTERING_SOURCE_FILE = "run_clustering.cpp"
RECONSTRUCTION_SOURCE_FILE = "reconstruct_hierarchy.cpp"
PREPARED_PROVENANCE_FILE = "prepared.provenance"
CLUSTERING_PROVENANCE_FILE = "clustering.provenance"
PARSED_DATA_PROVENANCE_FILE = "parsed_data.provenance"
NATIVE_FILES_HEADER_FILE = "pipeline_files.hpp"
PREPARE_MARKER_FILE = ".compute_variances_prepare_in_progress"
EXECUTABLE_STEM = "compute_variances"
STATE_SCHEMA_VERSION = 2

INPUT_FILES = (
    MATRIX_GENES_FILE, MATRIX_SAMPLES_FILE,
    SAMPLE_LINEAGES_FILE, MATRIX_VALUES_FILE,
)
INT32_INFO = np.iinfo(np.int32)

SampleLineages = dict[str, tuple[int, ...]]


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
# Block 0: Text Contracts and Artifact Provenance
# ======================================================================
PROVENANCE_HEADER = "TAVNIT-PROVENANCE-1"
_CANONICAL_INT_RE = re.compile(r"(?:0|-?[1-9][0-9]*)\Z", re.ASCII)
_DECIMAL_FLOAT_RE = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?\Z", re.ASCII
)
_STAGE_NAMES = {
    "ingest", "configuration", "sample_manifest", "clustering",
    "reconstruct", "prepare", "variances", "signatures", "cluster_export",
    "signature_export", "dendrogram",
}


def iter_utf8_lines(path: Path) -> Iterator[str]:
    """Yield validated physical UTF-8 records incrementally.

    Args:
        path: Nonempty machine-data text file.

    Yields:
        Literal nonblank records in physical order.

    Raises:
        ValueError: If encoding, terminators, or blank records are
            invalid.
        OSError: If the file cannot be read.
    """
    count = 0
    with path.open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            encoding = "utf-8-sig" if count == 1 else "utf-8"
            try:
                line = raw.decode(encoding, errors="strict")
            except UnicodeError as exc:
                raise ValueError(f"{path}:{count}: invalid UTF-8") from exc
            if line.endswith("\n"):
                line = line[:-1]
                if line.endswith("\r"):
                    line = line[:-1]
            if "\r" in line or "\x00" in line or "\ufeff" in line:
                raise ValueError(f"{path}:{count}: invalid text transport")
            if not line or line.isspace():
                raise ValueError(f"{path}:{count}: blank record")
            yield line
    if count == 0:
        raise ValueError(f"{path}: empty file")


def read_utf8_lines(
    path: Path, *, allow_blank: bool = False, allow_empty: bool = False
) -> list[str]:
    """Read physical UTF-8 records without inferring missing values.

    Accept one initial BOM and LF or CRLF terminators. Reject interior
    BOMs, lone carriage returns, invalid UTF-8, and prohibited blanks.

    Args:
        path: Input text file.
        allow_blank: Whether the schema permits blank records.
        allow_empty: Whether a zero-record file is meaningful.

    Returns:
        Literal records with only transport terminators removed.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If encoding, terminators, or records are invalid.
    """
    try:
        text = path.read_bytes().decode("utf-8-sig", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8") from exc
    if "\ufeff" in text:
        raise ValueError(f"{path}: repeated or interior UTF-8 BOM")
    text = text.replace("\r\n", "\n")
    if "\r" in text or "\x00" in text:
        raise ValueError(f"{path}: invalid record terminator or NUL")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines and not allow_empty:
        raise ValueError(f"{path}: file is empty")
    if not allow_blank:
        for number, line in enumerate(lines, 1):
            if not line or line.isspace():
                raise ValueError(f"{path}:{number}: blank record")
    return lines


def validate_identifier(value: str, *, source: str) -> None:
    """Validate a literal identifier without normalization.

    Args:
        value: Case-sensitive identifier after ingestion normalization.
        source: File and record location for diagnostics.

    Raises:
        ValueError: If the identifier is empty or has
            whitespace/control.
    """
    if not value or any(
        char.isspace() or unicodedata.category(char) == "Cc"
        or char == "\ufeff" for char in value
    ):
        raise ValueError(f"{source}: invalid identifier {value!r}")


def parse_canonical_int(token: str, *, source: str) -> int:
    """Parse one canonical ASCII decimal integer without narrowing.

    Args:
        token: Literal machine-format field.
        source: File and field location for diagnostics.

    Returns:
        Exact Python integer; callers enforce semantic range bounds.

    Raises:
        ValueError: If the spelling is not canonical ASCII decimal.
    """
    if _CANONICAL_INT_RE.fullmatch(token) is None:
        raise ValueError(f"{source}: noncanonical integer {token!r}")
    return int(token, 10)


def parse_decimal_float(token: str, *, source: str) -> float:
    """Parse a complete, finite decimal floating-point field.

    Args:
        token: Literal field; a leading sign and exponent are permitted.
        source: File and field location for diagnostics.

    Returns:
        Finite floating-point value, with signed zero normalized.

    Raises:
        ValueError: If syntax, finiteness, or nonzero range is invalid.
    """
    if _DECIMAL_FLOAT_RE.fullmatch(token) is None:
        raise ValueError(f"{source}: invalid floating-point field {token!r}")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"{source}: nonfinite floating-point value")
    if value == 0.0 and Decimal(token) != 0:
        raise ValueError(f"{source}: nonzero value underflows to zero")
    return 0.0 if value == 0.0 else value


def parse_csv_record(line: str, *, source: str) -> list[str]:
    """Decode one strictly quoted, single-line CSV record.

    Args:
        line: One physical record without its line terminator.
        source: File and record location for diagnostics.

    Returns:
        Literal decoded fields, preserving empty and quoted contents.

    Raises:
        ValueError: If a quote is misplaced or a field is unterminated.
    """
    fields: list[str] = []
    pos = 0
    while True:
        value: list[str] = []
        if pos < len(line) and line[pos] == '"':
            pos += 1
            while True:
                if pos >= len(line):
                    raise ValueError(f"{source}: unterminated CSV quote")
                char = line[pos]
                pos += 1
                if char == '"':
                    if pos < len(line) and line[pos] == '"':
                        value.append('"')
                        pos += 1
                    else:
                        break
                else:
                    value.append(char)
            if pos < len(line) and line[pos] != ",":
                raise ValueError(f"{source}: content after closing CSV quote")
        else:
            while pos < len(line) and line[pos] != ",":
                if line[pos] == '"':
                    raise ValueError(f"{source}: quote inside unquoted field")
                value.append(line[pos])
                pos += 1
        fields.append("".join(value))
        if pos == len(line):
            return fields
        pos += 1


def read_csv_rows(path: Path) -> list[list[str]]:
    """Read strict single-line CSV records with literal empty fields.

    Args:
        path: CSV file using the pipeline dialect.

    Returns:
        Parsed records in physical order, including any header.

    Raises:
        OSError: If reading fails.
        ValueError: If encoding, blank records, or CSV syntax is
            invalid.
    """
    return [
        parse_csv_record(line, source=f"{path}:{number}")
        for number, line in enumerate(read_utf8_lines(path), 1)
    ]


def read_identifiers(path: Path, *, label: str) -> list[str]:
    """Read unique literal identifiers, one per physical line.

    Args:
        path: Identifier-list file.
        label: Identifier kind used in diagnostics.

    Returns:
        Identifiers in physical order, without CSV interpretation.

    Raises:
        ValueError: If any identifier is invalid or duplicated.
        OSError: If reading fails.
    """
    values = read_utf8_lines(path)
    seen: set[str] = set()
    for number, value in enumerate(values, 1):
        validate_identifier(value, source=f"{path}:{number} ({label})")
        if value in seen:
            raise ValueError(f"{path}:{number}: duplicate {label} {value!r}")
        seen.add(value)
    return values


def prompt_input(prompt: str = "") -> str:
    """Display an interactive prompt on stderr and read one response.

    Args:
        prompt: Prompt text; no newline is added.

    Returns:
        User response without the input line terminator.

    Raises:
        EOFError: If standard input is exhausted.
        KeyboardInterrupt: If the user interrupts input.
    """
    print(prompt, end="", file=sys.stderr, flush=True)
    return input()


def write_text_atomically(path: Path, content: str) -> None:
    """Flush, close, and replace one UTF-8/LF output file.

    Args:
        path: Final output path; its parent must already exist.
        content: Complete text, already serialized for its schema.

    Raises:
        OSError: If writing, flushing, closing, or replacement fails.
    """
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Warning: temporary cleanup failed: {exc}", file=sys.stderr)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Verified structure of a stage-local commit manifest.

    Attributes:
        stage: Producing stage identifier.
        producer: SHA-256 of the producer source, for traceability.
        inputs: Absolute input paths and their exact byte hashes.
        parents: Upstream commit manifests and their exact byte hashes.
        outputs: Local output basenames and their exact byte hashes.
        categories: Sample IDs mapped to absolute source-directory
            paths.
    """

    stage: str
    producer: str
    inputs: dict[str, str]
    parents: dict[str, str]
    outputs: dict[str, str]
    categories: dict[str, str]


def provenance_path(path: Path) -> Path:
    """Return the commit-manifest path owning one generated artifact.

    Args:
        path: Generated artifact path.

    Returns:
        Expected stage-local provenance path.
    """
    if path.name in INPUT_FILES:
        return path.with_name(PREPARED_PROVENANCE_FILE)
    if path.name in {SAMPLE_PATHS_FILE, TREE_TOPOLOGY_FILE} or re.fullmatch(
        r"iter[1-9][0-9]*\.txt", path.name
    ):
        return path.with_name(CLUSTERING_PROVENANCE_FILE)
    if path.suffix == ".csv" and path.parent.name.endswith("_parsed_data"):
        return path.with_name(PARSED_DATA_PROVENANCE_FILE)
    return path.with_name(path.name + ".provenance")


def _decode_provenance_field(token: str, *, source: str) -> str:
    """Decode one canonical hexadecimal UTF-8 provenance field.

    Args:
        token: Lowercase hexadecimal byte pairs, without separators.
        source: Manifest and record location for diagnostics.

    Returns:
        Decoded literal text.

    Raises:
        ValueError: If hexadecimal, UTF-8, or path transport is invalid.
    """
    if re.fullmatch(r"(?:[0-9a-f]{2})*", token) is None:
        raise ValueError(f"{source}: noncanonical hexadecimal field")
    try:
        value = bytes.fromhex(token).decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{source}: invalid encoded UTF-8") from exc
    if any(char in value for char in ("\x00", "\r", "\n", "\ufeff")):
        raise ValueError(f"{source}: invalid encoded text transport")
    return value


def read_provenance(path: Path) -> ProvenanceRecord:
    """Read and validate the structure of a commit manifest.

    Args:
        path: Provenance file in the producer's output directory.

    Returns:
        Parsed manifest with validated paths and record fields.

    Raises:
        ValueError: If records, hashes, or output basenames are
            malformed.
        OSError: If the manifest cannot be read.
    """
    lines = read_utf8_lines(path)
    if len(lines) < 4 or lines[0] != PROVENANCE_HEADER:
        raise ValueError(f"{path}: invalid provenance header")
    if not lines[1].startswith("stage\t") or lines[1][6:] not in _STAGE_NAMES:
        raise ValueError(f"{path}: invalid provenance stage")
    if not lines[2].startswith("producer\t"):
        raise ValueError(f"{path}: missing producer fingerprint")
    producer = lines[2][9:]
    if re.fullmatch(r"[0-9a-f]{64}", producer) is None:
        raise ValueError(f"{path}: invalid producer fingerprint")
    maps: dict[str, dict[str, str]] = {k: {} for k in ("I", "P", "O", "C")}
    for number, line in enumerate(lines[3:], 4):
        fields = line.split("\t")
        if len(fields) != 3 or fields[0] not in maps:
            raise ValueError(f"{path}:{number}: invalid provenance record")
        kind, digest, encoded = fields
        value = _decode_provenance_field(
            encoded, source=f"{path}:{number}"
        )
        if kind == "C":
            key = _decode_provenance_field(
                digest, source=f"{path}:{number}"
            )
            validate_identifier(key, source=f"{path}:{number}")
            if not Path(value).is_absolute():
                raise ValueError(f"{path}:{number}: relative category path")
            if key in maps[kind]:
                raise ValueError(f"{path}:{number}: duplicate category ID")
            maps[kind][key] = value
            continue
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{path}:{number}: invalid SHA-256")
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{path}:{number}: invalid path")
        if kind in {"I", "P"} and not Path(value).is_absolute():
            raise ValueError(f"{path}:{number}: relative dependency path")
        if kind == "O" and (
            Path(value).name != value or "/" in value or "\\" in value
            or value in {".", ".."}
        ):
            raise ValueError(f"{path}:{number}: output must be a basename")
        if value in maps[kind]:
            raise ValueError(f"{path}:{number}: duplicate provenance path")
        maps[kind][value] = digest
    if not maps["O"]:
        raise ValueError(f"{path}: provenance has no outputs")
    return ProvenanceRecord(
        lines[1][6:], producer, maps["I"], maps["P"], maps["O"], maps["C"]
    )


def verify_provenance(
    path: Path, *, _verified: dict[Path, ProvenanceRecord] | None = None,
    _visiting: set[Path] | None = None,
) -> ProvenanceRecord:
    """Verify committed bytes and upstream dependency hashes.

    Args:
        path: Commit manifest to verify before consuming its outputs.
        _verified: Internal cache for manifests already checked this
            call.
        _visiting: Internal cycle-detection set.

    Returns:
        Validated provenance record.

    Raises:
        ValueError: If a dependency is changed, incomplete, or cyclic.
        OSError: If an input or commit manifest is missing/unreadable.
    """
    path = path.resolve()
    verified = {} if _verified is None else _verified
    visiting = set() if _visiting is None else _visiting
    if path in verified:
        return verified[path]
    record = read_provenance(path)
    if path in visiting:
        raise ValueError(f"{path}: cyclic artifact provenance")
    visiting.add(path)
    for raw, expected in record.parents.items():
        parent = Path(raw)
        if sha256_file(parent) != expected:
            raise ValueError(f"{path}: changed upstream commit {parent}")
        verify_provenance(parent, _verified=verified, _visiting=visiting)
    pairs = list(record.inputs.items()) + [
        (str(path.parent / name), digest)
        for name, digest in record.outputs.items()
    ]
    for raw, expected in pairs:
        if sha256_file(Path(raw)) != expected:
            raise ValueError(f"{path}: changed artifact {raw}; rerun producer")
    visiting.remove(path)
    verified[path] = record
    return record


def verify_artifacts(paths: Iterable[Path]) -> None:
    """Verify provenance for every required generated input.

    Args:
        paths: Generated artifacts required by a stage.

    Raises:
        OSError: If any artifact or its provenance is unavailable.
        ValueError: If provenance or dependency hashes do not match.
    """
    verified: dict[Path, ProvenanceRecord] = {}
    for path in paths:
        path = Path(path).resolve()
        owner = provenance_path(path)
        if owner.name == PREPARED_PROVENANCE_FILE and (
            path.parent / PREPARE_MARKER_FILE
        ).exists():
            raise ValueError(f"{path.parent}: interrupted prepare stage")
        record = verify_provenance(owner, _verified=verified)
        if path.name not in record.outputs:
            raise ValueError(f"{owner}: does not own required {path.name}")


def render_provenance(
    path: Path, *, stage: str, outputs: Iterable[Path],
    inputs: Iterable[Path] = (), parents: Iterable[Path] = (),
    categories: dict[str, str] | None = None,
    producer: Path | None = None,
) -> str:
    """Return the record that ``write_provenance`` would write.

    Args:
        path: Stage-local manifest destination.
        stage: Recognized producing-stage name.
        outputs: Complete owned output set, all beside the manifest.
        inputs: Explicit input files; raw external data need no parent.
        parents: Verified upstream manifests for generated inputs.
        categories: Optional sample-to-source-folder mapping for
            plotting.
        producer: Producing source file; defaults to this module.

    Returns:
        The record's text.

    Raises:
        OSError: If hashing fails.
        ValueError: If stage, ownership, or dependency validation fails.
    """
    if stage not in _STAGE_NAMES:
        raise ValueError(f"Unknown provenance stage: {stage}")
    path = path.resolve()
    producer = Path(__file__) if producer is None else producer
    lines = [PROVENANCE_HEADER, f"stage\t{stage}",
             f"producer\t{sha256_file(producer)}"]
    for kind, objects in (("I", inputs), ("P", parents), ("O", outputs)):
        unique = sorted({Path(obj).resolve() for obj in objects}, key=str)
        for item in unique:
            if kind == "P":
                verify_provenance(item)
            if kind == "O" and item.parent != path.parent:
                raise ValueError(f"{item}: output is outside manifest folder")
            name = item.name if kind == "O" else str(item)
            lines.append(
                f"{kind}\t{sha256_file(item)}\t{name.encode('utf-8').hex()}"
            )
    for sid, folder in sorted((categories or {}).items()):
        lines.append(
            f"C\t{sid.encode('utf-8').hex()}\t{folder.encode('utf-8').hex()}"
        )
    return "\n".join(lines) + "\n"


def write_provenance(
    path: Path, *, stage: str, outputs: Iterable[Path],
    inputs: Iterable[Path] = (), parents: Iterable[Path] = (),
    categories: dict[str, str] | None = None,
    producer: Path | None = None,
) -> None:
    """Commit stage output hashes and their dependencies.

    Call only after successful semantic validation and output
    publication. The stage is uncommitted while this record is missing
    or mismatched.

    Args:
        path: Stage-local manifest destination.
        stage: Recognized producing-stage name.
        outputs: Complete owned output set, all beside the manifest.
        inputs: Explicit input files; raw external data need no parent.
        parents: Verified upstream manifests for generated inputs.
        categories: Optional sample-to-source-folder mapping for
            plotting.
        producer: Producing source file; defaults to this module.

    Raises:
        OSError: If hashing or publication fails.
        ValueError: If stage, ownership, or dependency validation fails.
    """
    record = render_provenance(
        path, stage=stage, outputs=outputs, inputs=inputs,
        parents=parents, categories=categories, producer=producer,
    )
    write_text_atomically(path.resolve(), record)


def publication_unchanged(
    manifest: Path, contents: dict[Path, str], **record: Any
) -> bool:
    """Tell whether publishing ``contents`` would change nothing.

    An identical result must not be rewritten or invalidate anything
    downstream. That holds when every output already holds exactly the
    new text and the manifest already holds exactly the record that
    would be written for it.

    Args:
        manifest: The outputs' provenance record.
        contents: New text by output path.
        **record: The other ``write_provenance`` arguments, without
            ``outputs``.

    Returns:
        True when the outputs and their record are already current.
    """
    try:
        for path, text in contents.items():
            if Path(path).read_bytes() != text.encode("utf-8"):
                return False
        current = Path(manifest).read_text(encoding="utf-8")
        return current == render_provenance(
            manifest, outputs=list(contents), **record)
    except (OSError, ValueError):
        return False


def invalidate_dependents(
    changed: Iterable[Path], directories: Iterable[Path]
) -> None:
    """Invalidate manifest-owned downstream outputs.

    The supplied directories are inspected nonrecursively. Input files,
    unknown manifests, and unrelated files are never cleanup targets.

    Args:
        changed: Artifacts about to change or already invalidated.
        directories: Existing result directories and their workflow
            parent.

    Raises:
        OSError: If a recognized downstream output cannot be
            invalidated.
        ValueError: If a recognized manifest contains unsafe ownership.
    """
    changed_paths = {Path(item).resolve() for item in changed}
    candidates: dict[Path, ProvenanceRecord] = {}
    for directory in {Path(item).resolve() for item in directories}:
        if not directory.is_dir():
            continue
        for candidate in directory.glob("*.provenance"):
            if candidate.is_symlink():
                continue
            try:
                record = read_provenance(candidate)
            except (OSError, ValueError):
                continue
            candidates[candidate] = record
    # A changed member also changes the identity of its owning commit.
    for manifest, record in candidates.items():
        owned = {(manifest.parent / name).resolve()
                 for name in record.outputs}
        if owned.intersection(changed_paths):
            changed_paths.add(manifest.resolve())
    while True:
        removed: list[Path] = []
        for manifest, record in candidates.items():
            dependencies = {
                Path(raw).resolve()
                for raw in (*record.inputs, *record.parents)
            }
            if not dependencies.intersection(changed_paths):
                continue
            manifest.unlink(missing_ok=True)
            changed_paths.add(manifest.resolve())
            for name in record.outputs:
                output = manifest.parent / name
                # unlink removes a symbolic link itself, not its target.
                output.unlink(missing_ok=True)
                changed_paths.add(output.resolve())
            removed.append(manifest)
        if not removed:
            return
        for manifest in removed:
            del candidates[manifest]


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
    node_count = _tree_node_count(sample_count)
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
    node_count = _tree_node_count(count)
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


# ======================================================================
# Block 1: Small Utilities
# ======================================================================
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


# Digests already computed by this process, keyed by resolved path and
# checked against the file's size, modification time and file ID.
_DIGEST_MEMO: dict[str, tuple[tuple[int, int, int, int], str]] = {}


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file.

    A file whose size, modification time and file ID are unchanged
    since this process last hashed it is not read again. Replacing a
    file, as every atomic write does, gives it a new file ID.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    path = Path(path)
    status = path.stat()
    identity = (status.st_size, status.st_mtime_ns, status.st_ino,
                status.st_dev)
    key = str(path.resolve())
    known = _DIGEST_MEMO.get(key)
    if known is not None and known[0] == identity:
        return known[1]
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    _DIGEST_MEMO[key] = (identity, digest.hexdigest())
    return digest.hexdigest()


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

    fd, temporary_name = tempfile.mkstemp(
        dir=out_dir,
        prefix=f".{VARIANCE_STATE_FILE}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary_path, out_dir / VARIANCE_STATE_FILE)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


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


def _parse_hierarchy_id(
    value: object, *, row_number: int, column_number: int
) -> int:
    """Parse a canonical nonnegative hierarchy-state identifier.

    Args:
        value: Decoded field, without automatic normalization.
        row_number: One-based physical record.
        column_number: One-based physical field position.

    Returns:
        Exact nonnegative node ID.

    Raises:
        ValueError: If spelling or sign violates the state schema.
    """
    node_id = parse_canonical_int(
        str(value), source=f"total_data.csv:{row_number}:{column_number}"
    )
    if node_id < 0:
        raise ValueError("SampleLineages node IDs must be nonnegative")
    return node_id


def _tree_node_count(sample_count: int) -> int:
    """Return number of nodes in a full binary tree with N leaves.

    Return the number of nodes in a full binary tree with N leaves.

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


# ======================================================================
# Block 3: Expression Matrix
# ======================================================================
def _read_sample_expression(resolved: Path, sid: str) -> pd.Series:
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
    if rows[0] != ["gene", "expression"] or len(rows) < 2:
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
        series_by_sample[sid] = _read_sample_expression(resolved, sid)
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

    node_count = _tree_node_count(len(sample_names))
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
        "\n".join(gene_names) + "\n", encoding="utf-8", newline="\n"
    )
    (staging_dir / MATRIX_SAMPLES_FILE).write_text(
        "\n".join(sample_names) + "\n", encoding="utf-8", newline="\n"
    )
    (staging_dir / SAMPLE_LINEAGES_FILE).write_text(
        "\n".join(
            f"{sample} {len(hierarchy[sample])} "
            + " ".join(str(node_id) for node_id in hierarchy[sample])
            for sample in sample_names
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    integer_matrix = to_integer_matrix(matrix)
    integer_matrix.to_csv(
        staging_dir / MATRIX_VALUES_FILE,
        header=False,
        index=False,
        sep=" ",
        lineterminator="\n",
    )


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
def _resolve_compiler(explicit_cxx: str | None) -> str:
    """Resolve requested or automatically discovered C++ compiler.

    Resolve the requested or automatically discovered C++ compiler.

    Args:
        explicit_cxx: Compiler executable supplied by ``--cxx``.

    Returns:
        Resolved compiler executable path/name.

    Raises:
        RuntimeError: If no supported compiler executable can be found.
    """
    requested = explicit_cxx or os.environ.get("CXX")
    if requested:
        resolved = shutil.which(requested)
        if resolved is not None:
            return resolved
        candidate = Path(requested)
        if candidate.is_file():
            if os.name != "nt" and not os.access(candidate, os.X_OK):
                raise RuntimeError(
                    f"Requested C++ compiler is not executable: {requested}"
                )
            return str(candidate.resolve())
        raise RuntimeError((
            f'Requested C++ compiler was not found: '
            f'{requested}'
        ))

    candidates = (
        ("cl", "clang-cl", "g++", "clang++", "c++")
        if os.name == "nt"
        else ("c++", "g++", "clang++")
    )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

    if os.name == "nt":
        raise RuntimeError(
            "No supported C++ compiler found. Use a Visual Studio Developer "
            "Command Prompt for cl.exe, install MinGW/LLVM, or pass --cxx."
        )
    raise RuntimeError(
        (
            'No C++ compiler found. Install g++ or clang++, '
            'set CXX, or pass --cxx.'
        )
    )


def _compiler_family(cxx: str) -> str:
    """Classify a compiler driver as GNU-style or MSVC-style.

    Args:
        cxx: Resolved compiler executable.

    Returns:
        ``"msvc"`` for ``cl``/``clang-cl`` and ``"gnu"`` otherwise.
    """
    name = Path(cxx).name.lower()
    if name in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}:
        return "msvc"
    return "gnu"


def build_native_binary(
    source: Path,
    out_dir: Path,
    *,
    explicit_cxx: str | None = None,
) -> Path:
    """Compile and publish the native variance executable.

    GCC/MinGW and ``clang++`` use GNU-style flags. MSVC ``cl.exe`` and
    ``clang-cl`` use MSVC-style flags.

    Args:
        source: C++ source file.
        out_dir: Directory for the executable and pipeline state.
        explicit_cxx: Optional compiler executable from ``--cxx``.

    Returns:
        Published executable path.

    Raises:
        FileNotFoundError: If the C++ source is missing.
        RuntimeError: If no supported compiler can be found.
        subprocess.CalledProcessError: If compilation fails.
    """
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"C++ source not found: {source}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cxx = _resolve_compiler(explicit_cxx)
    family = _compiler_family(cxx)
    exe_name = EXECUTABLE_STEM + (".exe" if os.name == "nt" else "")
    exe_path = out_dir / exe_name
    source_hash = sha256_file(source)
    artifact_header = source.with_name(NATIVE_FILES_HEADER_FILE)
    if not artifact_header.is_file():
        raise FileNotFoundError(
            f"Shared native filename header not found: {artifact_header}"
        )
    artifact_header_hash = sha256_file(artifact_header)
    state = load_state(out_dir)
    existing_build = state.get("build")

    if (
        existing_build is not None
        and existing_build["compiler"] == cxx
        and existing_build["compiler_family"] == family
        and existing_build["source_sha256"] == source_hash
        and existing_build["artifact_header_sha256"] == artifact_header_hash
        and existing_build["executable"] == exe_name
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
                'build metadata; rebuilding'
            )
        )

    LOGGER.info("Compiling %s with %s", source, cxx)

    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".compute_variances_build_"
    ) as build_name:
        build_dir = Path(build_name)
        temporary_exe = build_dir / exe_name

        if family == "msvc":
            object_path = build_dir / f"{EXECUTABLE_STEM}.obj"
            command = [
                cxx,
                "/nologo",
                "/O2",
                "/std:c++20",
                "/EHsc",
                str(source),
                f"/Fe:{temporary_exe}",
                f"/Fo:{object_path}",
            ]
        else:
            # -ffp-contract=off: fused multiply-adds, which Apple Clang
            # uses by default on arm64, would change the results there.
            command = [
                cxx,
                "-O2",
                "-std=c++20",
                "-ffp-contract=off",
                "-o",
                str(temporary_exe),
                str(source),
            ]

        subprocess.run(command, check=True, cwd=out_dir)
        if not temporary_exe.is_file():
            raise RuntimeError(
                f"Compiler reported success but did not create {temporary_exe}"
            )

        new_executable_hash = sha256_file(temporary_exe)
        # Reload state because compilation can be long-running and
        # another trusted process may have updated the pipeline metadata
        # meanwhile.
        state = load_state(out_dir)
        old_build = state.get("build")
        old_executable_hash = (
            old_build["executable_sha256"] if old_build is not None else None
        )

        _invalidate_untrusted_variance(out_dir, state)

        new_build: BuildState = {
            "compiler": cxx,
            "compiler_family": family,
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
        subprocess.CalledProcessError: If the native program exits
        nonzero.
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

        subprocess.run([str(exe_path)], check=True, cwd=run_dir)

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
