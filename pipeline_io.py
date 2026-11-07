"""Shared input and output for every pipeline stage.

Requires Python 3.10 or newer; uses only the standard library.

The module holds the pieces every stage shares and that do not need
NumPy: the names of the pipeline's artifacts, UTF-8 and CSV transport,
the machine grammar for identifiers and numbers, the per-process hash
memo, the provenance records that link each output to its inputs, the
one atomic writer, and the prompt that reads one answer from the
console. Format readers and writers live in pipeline_formats.py.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

# Names of the pipeline's artifacts.
TREE_STATES_FILE = "total_data.csv"
MATRIX_GENES_FILE = "matrix_genes.txt"
MATRIX_SAMPLES_FILE = "matrix_samples.txt"
MATRIX_VALUES_FILE = "matrix_values.txt"
SAMPLE_LINEAGES_FILE = "sample_lineages.txt"
TERM_VARIANCES_FILE = "term_variances.txt"
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
# The four files the variance preparation commits together.
INPUT_FILES = (
    MATRIX_GENES_FILE, MATRIX_SAMPLES_FILE,
    SAMPLE_LINEAGES_FILE, MATRIX_VALUES_FILE,
)
# The folder the clustering stage writes and Part 2 works in.
RESULTS_DIR = "clustering_results"
# Ingestion writes <folder>_parsed_data/ and <folder>_parsed_files.txt
# beside each input folder.
PARSED_DATA_SUFFIX = "_parsed_data"
PARSED_FILES_SUFFIX = "_parsed_files.txt"
# A manifest is named after its output plus this suffix.
PROVENANCE_SUFFIX = ".provenance"
# Snapshot k, the sample assignment after split k, is iter<k>.txt.
SNAPSHOT_FILE_RE = re.compile(r"iter([1-9][0-9]*)\.txt\Z", re.ASCII)


def snapshot_file(split: int) -> str:
    """Return the file name of the snapshot after one split.

    Args:
        split: One-based split number.

    Returns:
        ``iter<split>.txt``.
    """
    return f"iter{split}.txt"


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


def is_blank(record: str) -> bool:
    """Tell whether a record holds nothing but ASCII spaces and tabs.

    Other whitespace, such as a no-break space, is content, which the
    format's own rules then reject.

    Args:
        record: One record, without its terminator.

    Returns:
        Whether the record is blank.
    """
    return not record.strip(" \t")


def _check_record(record: str, *, source: str, allow_blank: bool) -> None:
    """Apply the rules every text record follows.

    Args:
        record: One record, without its terminator.
        source: File and record location for diagnostics.
        allow_blank: Whether the format permits blank records.

    Raises:
        ValueError: If the record is a prohibited blank, or holds only
            control characters and spaces.
    """
    if is_blank(record):
        if not allow_blank:
            raise ValueError(f"{source}: blank record")
    elif all(char == " " or unicodedata.category(char) == "Cc"
             for char in record):
        raise ValueError(f"{source}: record holds only control characters")


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
            _check_record(line, source=f"{path}:{count}", allow_blank=False)
            yield line
    if count == 0:
        raise ValueError(f"{path}: empty file")


def read_utf8_lines(
    path: Path, *, allow_blank: bool = False, allow_empty: bool = False
) -> list[str]:
    """Read physical UTF-8 records without inferring missing values.

    Accept one initial BOM and LF or CRLF terminators. Reject interior
    BOMs, lone carriage returns, invalid UTF-8, prohibited blanks and
    records that hold only control characters.

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
    for number, line in enumerate(lines, 1):
        _check_record(line, source=f"{path}:{number}", allow_blank=allow_blank)
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


def render_identifiers(values: Iterable[str]) -> str:
    """Return the text of an identifier list, one per line.

    Args:
        values: Identifiers, in order.

    Returns:
        The identifiers, each followed by a newline.
    """
    return "".join(f"{value}\n" for value in values)


TEMPORARY_SUFFIX = ".tmp"


def _new_file_mode() -> int:
    """Return the permission bits a new file gets under the umask.

    Returns:
        ``0o666`` with the bits of the current umask cleared.
    """
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def write_text_atomically(path: Path, content: str) -> None:
    """Flush, close, and replace one UTF-8/LF output file.

    Args:
        path: Final output path; its parent must already exist.
        content: Complete text, already serialized for its schema.

    Raises:
        OSError: If writing, flushing, closing, or replacement fails.
    """
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                                suffix=TEMPORARY_SUFFIX)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp creates owner-only files; give the output the
        # permissions any new file would get under the user's umask.
        os.chmod(temporary, _new_file_mode())
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
    if (path.name in {SAMPLE_PATHS_FILE, TREE_TOPOLOGY_FILE}
            or SNAPSHOT_FILE_RE.match(path.name)):
        return path.with_name(CLUSTERING_PROVENANCE_FILE)
    if path.suffix == ".csv" and path.parent.name.endswith(PARSED_DATA_SUFFIX):
        return path.with_name(PARSED_DATA_PROVENANCE_FILE)
    return path.with_name(path.name + PROVENANCE_SUFFIX)


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
    producer: Path,
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
        producer: The stage script that produced the outputs.

    Returns:
        The record's text.

    Raises:
        OSError: If hashing fails.
        ValueError: If stage, ownership, or dependency validation fails.
    """
    if stage not in _STAGE_NAMES:
        raise ValueError(f"Unknown provenance stage: {stage}")
    path = path.resolve()
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
    producer: Path,
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
        producer: The stage script that produced the outputs.

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
        for candidate in directory.glob("*" + PROVENANCE_SUFFIX):
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
