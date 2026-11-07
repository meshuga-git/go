"""Expression-data preparation and constrained divisive clustering.

Requires Python 3.10 or later. Inputs and outputs follow the documented
working-directory contract; generated inputs require valid provenance.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from compute_variances import (
    CANNOT_LINKS_FILE,
    CLUSTERING_PROVENANCE_FILE,
    CLUSTERING_SOURCE_FILE,
    MUST_LINKS_FILE,
    NO_LINKS_FILE,
    PARSED_DATA_PROVENANCE_FILE,
    SAMPLE_PATHS_FILE,
    TREE_TOPOLOGY_FILE,
    TreeTopology,
    invalidate_dependents,
    parse_canonical_int,
    parse_decimal_float,
    prompt_input,
    provenance_path,
    read_csv_rows,
    read_provenance,
    read_tree_topology,
    read_utf8_lines,
    sha256_file,
    validate_identifier,
    verify_artifacts,
    write_provenance,
    write_text_atomically,
)

DISTANCE_METRICS: dict[str, str] = {
    "1": "euclidean",
    "2": "pearson",
    "3": "spearman",
    "4": "manhattan",
}

OUTPUT_EXTENSION = ".csv"

_ITERATION_FILE_RE = re.compile(r"^iter([1-9]\d*)\.txt$")

Reader = Callable[[str], pd.DataFrame]


# ======================================================================
# Block 1: User Configuration
# ======================================================================


def prompt_distance_metric() -> str:
    """Prompt the user to choose a supported distance metric."""
    print("Choose a distance metric:", file=sys.stderr)
    for key, name in DISTANCE_METRICS.items():
        print(f"  {key} - {name}", file=sys.stderr)

    while True:
        choice = (
            prompt_input("Enter the number of the distance metric: ").strip()
        )
        if choice in DISTANCE_METRICS:
            return choice
        print(
            "Invalid choice. Please enter one of "
            f"{', '.join(DISTANCE_METRICS)}.\n", file=sys.stderr)


def _split_paths(value: str) -> list[str]:
    """Split prompt paths while preserving literal backslashes.

    Args:
        value: Prompt text containing optional single or double quotes.

    Returns:
        Path tokens, with only prompt-level surrounding quotes removed.

    Raises:
        ValueError: If a quoted token is unterminated.
    """
    if not value:
        return []

    paths: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for char in value:
        if quote is None:
            if char in {'"', "'"}:
                quote = char
            elif char.isspace():
                if current:
                    paths.append("".join(current))
                    current.clear()
            else:
                current.append(char)
        elif char == quote:
            quote = None
        else:
            current.append(char)

    if quote is not None:
        raise ValueError("Unterminated quoted path.")

    if current:
        paths.append("".join(current))

    return paths


def prompt_folder_paths(prompt_text: str, expected_count: int) -> list[str]:
    """Prompt for an exact number of existing input folders.

    Args:
        prompt_text: Explanation of the requested constraint category.
        expected_count: Required count when the category is enabled.

    Returns:
        Valid directory paths, or an empty list for ``None``.

    Raises:
        EOFError: If standard input ends before a valid response.
        KeyboardInterrupt: If the user interrupts the prompt.
    """
    while True:
        print(prompt_text, file=sys.stderr)
        value = prompt_input().strip()

        if value.lower() == "none":
            return []

        try:
            paths = _split_paths(value)
        except ValueError as exc:
            print(f"Invalid path input: {exc}\n", file=sys.stderr)
            continue

        if len(paths) != expected_count:
            print(
                f"Invalid input. Expected exactly {expected_count} paths "
                'or "None" (without quotes).\n', file=sys.stderr)
            continue

        missing = [p for p in paths if not Path(p).is_dir()]
        if missing:
            print(
                "Invalid input. The following are not existing "
                f"directories: {', '.join(missing)}\n", file=sys.stderr)
            continue

        return paths


# ======================================================================
# Block 2: Input Readers
# ======================================================================


def _read_por_wrapper(filepath: str) -> pd.DataFrame:
    """Read a POR file and return its DataFrame."""
    import pyreadstat

    return pyreadstat.read_por(filepath)[0]


def _read_tab_delimited(filepath: str) -> pd.DataFrame:
    """Read a headered TSV/counts matrix with literal identifiers.

    Args:
        filepath: Tab-delimited source matrix.

    Returns:
        DataFrame preserving literal identifiers and empty fields.

    Raises:
        ValueError: If records or field counts are invalid.
        OSError: If reading fails.
    """
    rows = [line.split("\t") for line in read_utf8_lines(Path(filepath))]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{filepath}: inconsistent TSV records")
    return pd.DataFrame(rows[1:], columns=rows[0])


def _read_csv(filepath: str) -> pd.DataFrame:
    """Read a headered input matrix without missing-value inference.

    Args:
        filepath: Raw CSV matrix path, not a generated per-sample file.

    Returns:
        Literal gene and expression fields with original sample headers.

    Raises:
        ValueError: If rows are blank, ragged, or improperly quoted.
        OSError: If the file cannot be read.
    """
    rows = read_csv_rows(Path(filepath))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"{filepath}: expected sample headers and data")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError(f"{filepath}: inconsistent field count")
    return pd.DataFrame(rows[1:], columns=rows[0])


def _read_dat(filepath: str) -> pd.DataFrame:
    """Read a headered whitespace-delimited input matrix.

    Args:
        filepath: Raw DAT source file.

    Returns:
        Literal field table without automatic missing-value inference.

    Raises:
        ValueError: If records have inconsistent field counts.
        OSError: If the file cannot be read.
    """
    rows = [line.split() for line in read_utf8_lines(Path(filepath))]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{filepath}: inconsistent DAT records")
    return pd.DataFrame(rows[1:], columns=rows[0])


def _read_excel(filepath: str) -> pd.DataFrame:
    """Read an Excel file, treating row 1 as the sample-name header.

    Excel is treated like CSV/TSV/DAT input: row 1 is a column-header
    row of sample names, not a data row.
    """
    return pd.read_excel(filepath, header=0, dtype=str, keep_default_na=False)


def _read_json(filepath: str) -> pd.DataFrame:
    """Read a JSON file containing records."""
    return pd.read_json(filepath, orient="records")


def _read_stata(filepath: str) -> pd.DataFrame:
    """Read a Stata (.dta) file.

    The wrapper exposes the same eager ``Reader`` contract as the
    other supported formats.
    """
    return pd.read_stata(filepath)


def _read_spss(filepath: str) -> pd.DataFrame:
    """Read an SPSS file as a DataFrame."""
    return pd.read_spss(filepath)


def _read_hdf_file(filepath: str) -> pd.DataFrame:
    """Read one HDF5 expression table.

    Args:
        filepath: HDF5 source containing a single DataFrame.

    Returns:
        The source expression DataFrame.

    Raises:
        TypeError: If the HDF5 object is not a DataFrame.
        OSError: If the source cannot be read.
    """
    result = pd.read_hdf(filepath)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"{filepath}: expected a DataFrame from HDF5 store, got "
            f"{type(result).__name__}."
        )
    return result


def _read_parquet(filepath: str) -> pd.DataFrame:
    """Read a Parquet file as a DataFrame."""
    return pd.read_parquet(filepath)


def _read_orc(filepath: str) -> pd.DataFrame:
    """Read an ORC file as a DataFrame."""
    return pd.read_orc(filepath)


FILE_READERS: dict[str, Reader] = {
    ".counts": _read_tab_delimited,
    ".tsv": _read_tab_delimited,
    ".csv": _read_csv,
    ".xls": _read_excel,
    ".xlsx": _read_excel,
    ".dat": _read_dat,
    ".json": _read_json,
    ".dta": _read_stata,
    ".sav": _read_spss,
    ".por": _read_por_wrapper,
    ".h5": _read_hdf_file,
    ".parquet": _read_parquet,
    ".orc": _read_orc,
}


def _load_dataframe(file_path: str, reader: Reader) -> pd.DataFrame:
    """Load an expression table with contextual reader diagnostics.

    Args:
        file_path: Original input-table path.
        reader: Format-specific callable returning a DataFrame.

    Returns:
        The loaded source table, before ingestion normalization.

    Raises:
        RuntimeError: If a dependency is missing or the reader fails.
    """
    try:
        return reader(file_path)
    except ImportError as exc:
        raise RuntimeError(
            (
                f'Missing required library for file type '
                f'{Path(file_path).suffix}: {exc}. Please install '
                f'the missing dependency (e.g., pyreadstat, '
                f'tables, pyarrow).'
            )
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Error reading file {file_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ======================================================================
# Block 3: Input Validation and Sample Publication
# ======================================================================


def _validate_input_dataframe(dataframe: pd.DataFrame, file_path: str) -> None:
    """Validate source identifiers and finite numerical expressions.

    Args:
        dataframe: Headered genes-by-samples source table.
        file_path: Source pathname used in diagnostics.

    Raises:
        ValueError: If the table, labels, or expressions are malformed.
    """
    if dataframe.empty or dataframe.shape[1] < 2:
        raise ValueError(f"{file_path}: expected genes and sample columns")
    labels = [str(label) for label in dataframe.columns[1:]]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{file_path}: duplicate sample-column headers")
    for column, label in enumerate(labels, 2):
        validate_identifier(label, source=f"{file_path}:1:{column}")
    for row_number, row in enumerate(
        dataframe.itertuples(index=False, name=None), 2
    ):
        if pd.isna(row[0]):
            raise ValueError(f"{file_path}:{row_number}: missing gene")
        validate_identifier(str(row[0]), source=f"{file_path}:{row_number}:1")
        for column, value in enumerate(row[1:], 2):
            parse_decimal_float(str(value),
                source=f"{file_path}:{row_number}:{column}")


def _normalize_expression_columns(
    dataframe: pd.DataFrame, file_path: str
) -> list[str]:
    """Normalize gene versions and validate resulting identifiers.

    Args:
        dataframe: Source table, modified in place with normalized
            genes.
        file_path: Original source path for diagnostic coordinates.

    Returns:
        Sample-column labels in original order.

    Raises:
        ValueError: If normalization creates empty or duplicate genes.
    """
    _validate_input_dataframe(dataframe, file_path)
    expression_columns = [str(col) for col in dataframe.columns[1:]]
    normalized = (
        [str(value).split(".", 1)[0] for value in dataframe.iloc[:, 0]]
    )
    seen: set[str] = set()
    for number, gene in enumerate(normalized, 2):
        validate_identifier(gene, source=f"{file_path}:{number}:1")
        if gene in seen:
            raise ValueError((
                f'{file_path}:{number}: duplicate normalized gene '
                f'{gene!r}'
            ))
        seen.add(gene)
    dataframe.columns = pd.Index(["gene_raw", *expression_columns])
    # Keep the normalized gene column separate from named sample
    # columns.
    if "gene" in expression_columns or "gene_raw" in expression_columns:
        raise ValueError((
            f'{file_path}: sample header conflicts with gene '
            f'columns'
        ))
    dataframe["gene"] = normalized
    return expression_columns


def _build_output_name(
    prefix: str, file_path: str, expression_column: str
) -> str:
    """Assign a sample ID as the generated CSV basename at ingestion.

    Args:
        prefix: Source-file or immediate-folder prefix.
        file_path: Original matrix path, stored separately in
            provenance.
        expression_column: Literal sample label from the matrix header.

    Returns:
        A portable CSV basename; global uniqueness is checked by caller.

    Raises:
        ValueError: If the generated ID cannot be represented safely.
    """
    suffix = expression_column
    if expression_column == "expression":
        suffix = Path(file_path).stem.removeprefix(prefix)
    name = f"{prefix}_{suffix}{OUTPUT_EXTENSION}"
    validate_identifier(name,
        source=f"{file_path} sample {expression_column!r}")
    if (
        any(char in name for char in '<>:\"/\\|?*') or name.endswith((" ",
            "."))
    ):
        raise ValueError(f"{file_path}: invalid sample filename {name!r}")
    return name


def _clean_output_files(output_path: Path) -> None:
    """Remove only CSV outputs explicitly owned by ingestion.

    Remove only CSV outputs explicitly owned by the ingestion manifest.

    Legacy sample-file manifests identify exact files on the first
    rerun. Other files, including unlisted CSVs, are never cleanup
    targets.

    Args:
        output_path: Existing sibling ``*_parsed_data`` directory.

    Raises:
        OSError: If an owned file cannot be removed.
        ValueError: If manifest ownership escapes the output directory.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    owned: list[Path] = []
    manifest = output_path / PARSED_DATA_PROVENANCE_FILE
    if manifest.exists():
        record = read_provenance(manifest)
        owned = [output_path / name for name in record.outputs]
    else:
        legacy = output_path.with_name(
            output_path.name.removesuffix("_parsed_data") + "_parsed_files.txt"
        )
        if legacy.is_file():
            for raw in read_utf8_lines(legacy, allow_empty=True):
                candidate = Path(raw)
                if candidate.parent.resolve() != output_path.resolve():
                    raise ValueError(
                        f"{legacy}: output is outside owned directory")
                owned.append(candidate)
    for item in owned:
        if item.suffix != OUTPUT_EXTENSION or item.is_symlink():
            raise ValueError(f"Refusing to remove unowned output: {item}")
    manifest.unlink(missing_ok=True)
    for item in owned:
        item.unlink(missing_ok=True)


def write_sample_csvs(
    all_files: dict[str, list[str]], output_dir: str,
    known_sample_ids: set[str] | None = None,
) -> list[str]:
    """Publish normalized per-sample CSVs and their provenance.

    Source tables keep their declared order. Gene versions are
    normalized once; sample IDs are basenames and must be unique across
    all groups.

    Args:
        all_files: Source paths grouped by deterministic output prefix.
        output_dir: Existing-layout parsed-data destination.
        known_sample_ids: Run-wide sample IDs, updated in place.

    Returns:
        Absolute generated sample paths in deterministic ingestion
        order.

    Raises:
        ValueError: If input schemas or global sample identities
            conflict.
        OSError: If source reading, cleanup, or publication fails.
    """
    output_path = Path(output_dir).resolve()
    _clean_output_files(output_path)
    generated_files: list[str] = []
    seen = set() if known_sample_ids is None else known_sample_ids
    sources = (
        sorted({Path(raw).resolve() for paths in all_files.values()
            for raw in paths})
    )
    initial_hashes = {path: sha256_file(path) for path in sources}
    categories: dict[str, str] = {}
    for prefix in sorted(all_files):
        for file_path in sorted(all_files[prefix]):
            reader = FILE_READERS.get(Path(file_path).suffix.lower())
            if reader is None:
                raise ValueError(f"Unsupported format: {file_path}")
            dataframe = _load_dataframe(file_path, reader)
            labels = _normalize_expression_columns(dataframe, file_path)
            for expression_column in labels:
                name = _build_output_name(prefix, file_path, expression_column)
                if name in seen:
                    raise ValueError((
                        f'Duplicate global sample ID {name!r}; rename its '
                        f'source'
                    ))
                seen.add(name)
                output = output_path / name
                if output.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite unowned file: {output}")
                buffer = io.StringIO(newline="")
                writer = csv.writer(buffer, lineterminator="\n")
                writer.writerow(["gene", "expression"])
                for number, (gene, value) in enumerate(
                    zip(dataframe["gene"], dataframe[expression_column]), 2
                ):
                    number_value = parse_decimal_float(
                        str(value), source=f"{file_path}:{number}"
                    )
                    writer.writerow([gene, format(number_value, ".17g")])
                write_text_atomically(output, buffer.getvalue())
                generated_files.append(str(output))
                categories[name] = str(Path(file_path).resolve().parent)
                print(f"Sample {len(seen)} parsed to CSV: {name}",
                    file=sys.stderr)
    if not generated_files:
        raise ValueError(f"{output_path}: no visible input samples")
    if (
        any(sha256_file(path) != digest for path,
            digest in initial_hashes.items())
    ):
        raise ValueError("A source file changed during ingestion")
    write_provenance(
        output_path / PARSED_DATA_PROVENANCE_FILE, stage="ingest",
        outputs=map(Path, generated_files), inputs=sources,
        categories=categories, producer=Path(__file__),
    )
    return generated_files


def discover_input_files(mypath: str) -> dict[str, list[str]]:
    """Discover visible source files at the supported folder depths.

    Args:
        mypath: Source folder containing files and immediate subfolders.

    Returns:
        Deterministically ordered file paths grouped by output prefix.

    Raises:
        NotADirectoryError: If the source is not a directory.
        RuntimeError: If directory enumeration fails.

    Dot-prefixed files and folders are ignored at both depths. Visible
    unsupported file types are retained for an explicit ingestion error.
    """
    path = Path(mypath)
    if not path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {mypath}")

    all_files: defaultdict[str, list[str]] = defaultdict(list)

    try:
        for entry in sorted(path.iterdir(), key=lambda item: item.name):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                for child in sorted(
                    entry.iterdir(), key=lambda item: item.name
                ):
                    if child.is_file() and not child.name.startswith("."):
                        all_files[entry.name].append(str(child))
            elif entry.is_file() and not entry.name.startswith("."):
                all_files[entry.stem].append(str(entry))
    except OSError as exc:
        raise RuntimeError(
            f"Error parsing data from path {mypath}: {exc}"
        ) from exc

    return {
        prefix: sorted(file_list)
        for prefix, file_list in sorted(all_files.items())
    }


# ======================================================================
# Block 4: Native Execution and Output Validation
# ======================================================================


def compile_cpp_code(cpp_file: str, output_base: str) -> str:
    """Compile the native clustering target with GNU-style options.

    Args:
        cpp_file: Source path, relative to the current directory if
            needed.
        output_base: Executable stem, without its platform suffix.

    Returns:
        Executable filename ending in ``.exe`` or ``.out``.

    Raises:
        RuntimeError: If the selected compiler is missing or fails.

    ``CXX`` selects the compiler executable; otherwise ``g++`` is used.
    The target is compiled in C++20 mode and written in the current
    folder.
    """
    output_file = (
        f"{output_base}.exe"
        if platform.system() == "Windows"
        else f"{output_base}.out"
    )

    try:
        result = subprocess.run(
            # -ffp-contract=off: fused multiply-adds, which Apple Clang
            # uses by default on arm64, would change the results there.
            [os.environ.get("CXX", "g++"), "-std=c++20", cpp_file, "-O2",
                "-ffp-contract=off", "-o", output_file],
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "A C++ compiler not found. "
            "Is a C++ compiler installed and on PATH?"
        ) from exc

    if result.returncode != 0 or not os.path.isfile(output_file):
        message = result.stderr.strip() or (
            f"Compiler exited with code {result.returncode}"
        )
        raise RuntimeError(f"Error compiling C++ code: {message}")

    print(f"C++ code compiled successfully to {output_file}")
    return output_file


def _write_paths_file(filename: Path, paths: list[str]) -> None:
    """Publish constraint-configuration paths with provenance.

    Args:
        filename: Configuration destination in the current directory.
        paths: Optional source folders, in constraint order.

    Raises:
        OSError: If publication fails.
    """
    content = "\n".join(paths) + ("\n" if paths else "")
    write_text_atomically(filename, content)
    write_provenance(
        provenance_path(filename), stage="configuration", outputs=[filename],
        producer=Path(__file__),
    )


def _process_input_folder(folder: str, known_sample_ids: set[str]) -> None:
    """Publish one source group and its sample-file manifest.

    Args:
        folder: Constraint-group source folder.
        known_sample_ids: Run-wide IDs used to reject cross-group
            collisions.

    Raises:
        ValueError: If data or global sample identity is invalid.
        OSError: If discovery, reading, or publication fails.
    """
    folder_path = Path(folder).resolve()
    all_files = discover_input_files(str(folder_path))
    parsed_dir = folder_path.with_name(folder_path.name + "_parsed_data")
    generated = write_sample_csvs(all_files, str(parsed_dir), known_sample_ids)
    manifest = folder_path.with_name(folder_path.name + "_parsed_files.txt")
    write_text_atomically(manifest, "\n".join(generated) + "\n")
    write_provenance(
        provenance_path(manifest), stage="sample_manifest", outputs=[manifest],
        inputs=map(Path, generated), parents=[
            parsed_dir / PARSED_DATA_PROVENANCE_FILE],
        producer=Path(__file__),
    )


def _clean_clustering_results(output_dir: Path) -> None:
    """Clean exact clustering outputs and invalidate dependents.

    Args:
        output_dir: Existing-layout clustering-results directory.

    Raises:
        OSError: If an owned output cannot be removed.
        ValueError: If manifest ownership is malformed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    owned = [
        path for path in output_dir.iterdir()
        if path.name in {SAMPLE_PATHS_FILE, TREE_TOPOLOGY_FILE}
        or _ITERATION_FILE_RE.fullmatch(path.name)
        or path.name in {SAMPLE_PATHS_FILE + ".tmp",
            TREE_TOPOLOGY_FILE + ".tmp"}
        or re.fullmatch(r"iter[1-9][0-9]*\.txt\.tmp", path.name)
    ]
    manifest = output_dir / CLUSTERING_PROVENANCE_FILE
    invalidate_dependents(owned + [manifest], [output_dir, output_dir.parent])
    manifest.unlink(missing_ok=True)
    for path in owned:
        # unlink removes a symbolic link itself, not its target.
        if not path.is_symlink() and not path.is_file():
            raise ValueError(f"Refusing nonregular clustering output: {path}")
        path.unlink()


def _read_required_lines(path: Path, *, description: str) -> list[str]:
    """Read a literal text output with common transport rules.

    Read a required literal text output with the common transport rules.

    Args:
        path: Required output file.
        description: Human-readable role for the error context.

    Returns:
        Nonblank physical records in file order.

    Raises:
        ValueError: If the required output is empty or malformed.
        OSError: If the file is unavailable.
    """
    return read_utf8_lines(path)


def _load_tree_topology(path: Path, sample_count: int) -> TreeTopology:
    """Return the common validated immutable topology.

    Args:
        path: Authoritative topology file.
        sample_count: Number of ordered sample paths.

    Returns:
        The same ``TreeTopology`` model used by every Python consumer.

    Raises:
        ValueError: If topology records violate the immutable contract.
        OSError: If reading fails.
    """
    return read_tree_topology(path, sample_count)


def _read_iteration_assignments(
    path: Path,
    *,
    sample_count: int,
    max_node_id: int,
) -> list[int]:
    """Read one immutable ``iterK.txt`` assignment vector.

    Args:
        path: Iteration output file.
        sample_count: Required number of assignments.
        max_node_id: Largest valid immutable node ID.

    Returns:
        One immutable node ID per sample.

    Raises:
        RuntimeError: If the vector is unreadable, malformed, or out of
            range.
    """
    if not path.is_file():
        raise RuntimeError((
            f'Missing required clustering iteration file: '
            f'{path}'
        ))
    try:
        tokens = " ".join(read_utf8_lines(path)).split()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError((
            f'Could not read clustering iteration file '
            f'{path}: {exc}'
        )) from exc

    if len(tokens) != sample_count:
        raise RuntimeError(
            (
                f'{path.name} must contain exactly {sample_count} '
                f'sample assignment(s); found {len(tokens)}.'
            )
        )

    assignments: list[int] = []
    for position, token in enumerate(tokens, start=1):
        try:
            node_id = parse_canonical_int(token, source=f"{path}:1:{position}")
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name} contains non-integer node ID {token!r} at "
                f"sample position {position}."
            ) from exc
        if not 0 <= node_id <= max_node_id:
            raise RuntimeError(
                (
                    f'{path.name} contains out-of-range immutable '
                    f'node ID {node_id} at sample position '
                    f'{position}; valid IDs are 0..{max_node_id}.'
                )
            )
        assignments.append(node_id)
    return assignments


def _validate_clustering_outputs(output_dir: Path) -> (
    tuple[int, int]
):
    """Validate the complete immutable clustering output contract.

    This is intentionally stronger than checking the executable's return
    code. It rejects stale/partial output sets and cross-checks every
    ``iterK.txt`` transition against the authoritative
    ``tree_splits.txt`` record.

    Args:
        output_dir: Directory populated by the clustering executable.

    Returns:
        ``(sample_count, node_count)`` for status reporting.

    Raises:
        RuntimeError: If any required artifact or immutable-tree
            invariant fails.
    """
    filename_lines = _read_required_lines(
        output_dir / SAMPLE_PATHS_FILE,
        description=SAMPLE_PATHS_FILE,
    )
    sample_count = len(filename_lines)
    topology = _load_tree_topology(
        output_dir / TREE_TOPOLOGY_FILE,
        sample_count,
    )

    expected_iterations = set(range(1, sample_count))
    actual_iterations: set[int] = set()
    unexpected_iteration_files: list[str] = []
    for entry in output_dir.iterdir():
        if not entry.is_file() or not entry.name.startswith("iter"):
            continue
        match = _ITERATION_FILE_RE.fullmatch(entry.name)
        if match is None:
            continue
        actual_iterations.add(int(match.group(1)))

    if unexpected_iteration_files:
        raise RuntimeError(
            "Unexpected iteration-like output file(s) produced: "
            f"{sorted(unexpected_iteration_files)}."
        )
    if actual_iterations != expected_iterations:
        missing = sorted(expected_iterations - actual_iterations)
        extra = sorted(actual_iterations - expected_iterations)
        raise RuntimeError(
            (
                f'Immutable clustering output has an incomplete '
                f'iteration-file set; missing={missing}, '
                f'extra={extra}.'
            )
        )

    node_count = 2 * sample_count - 1
    max_node_id = node_count - 1
    previous = [0] * sample_count
    active_nodes = {0}

    for split in topology.splits:
        iteration, parent = split.iteration, split.parent
        retained_child = split.retained_child
        extracted_child = split.extracted_child
        current = _read_iteration_assignments(
            output_dir / f"iter{iteration}.txt",
            sample_count=sample_count,
            max_node_id=max_node_id,
        )
        retained_count = 0
        extracted_count = 0
        for sample_position, (before, after) in enumerate(
            zip(previous, current, strict=True),
            start=1,
        ):
            if before == parent:
                if after == retained_child:
                    retained_count += 1
                elif after == extracted_child:
                    extracted_count += 1
                else:
                    raise RuntimeError(
                        (
                            f'iter{iteration}.txt sample position '
                            f'{sample_position} belongs to split parent '
                            f'{parent} but moved to node {after}; expected '
                            f'child {retained_child} or {extracted_child}.'
                        )
                    )
            elif after != before:
                raise RuntimeError(
                    (
                        f'iter{iteration}.txt sample position '
                        f'{sample_position} changed from unrelated active '
                        f'node {before} to {after}; only parent {parent} '
                        f'may change during split {iteration}.'
                    )
                )

        if retained_count == 0 or extracted_count == 0:
            raise RuntimeError(
                f"Split {iteration} produced an empty immutable child: "
                f"node {retained_child} has {retained_count} sample(s), "
                f"node {extracted_child} has {extracted_count} sample(s)."
            )

        active_nodes.remove(parent)
        active_nodes.update((retained_child, extracted_child))
        actual_active = set(current)
        if actual_active != active_nodes:
            raise RuntimeError(
                (
                    f'iter{iteration}.txt active-node set does not '
                    f'match topology; expected '
                    f'{sorted(active_nodes)}, found '
                    f'{sorted(actual_active)}.'
                )
            )
        previous = current

    final_assignments = previous
    if len(set(final_assignments)) != sample_count:
        raise RuntimeError(
            "Final immutable clustering state must assign every sample to a "
            "distinct terminal leaf."
        )
    if set(final_assignments) != active_nodes:
        raise RuntimeError(
            "Final iteration leaf IDs do not match the active leaves declared "
            f"by {TREE_TOPOLOGY_FILE}."
        )

    return sample_count, node_count


def _collect_link_folders() -> list[str]:
    """Prompt for cannot-, must-, and no-constraint folders.

    Each set of paths is written to its own manifest file as it is
    collected and accumulated into the combined folder list returned to
    the caller.
    """
    folders_all: list[str] = []

    cannot_paths = prompt_folder_paths(
        'Write paths to 2 folders with cannot-links (separated by '
        'space) or "None" (without quotes):',
        expected_count=2,
    )
    cannot_paths = [str(Path(p)) for p in cannot_paths]
    folders_all.extend(cannot_paths)
    _write_paths_file(Path(CANNOT_LINKS_FILE), cannot_paths)

    must_paths = prompt_folder_paths(
        'Write paths to the folder with must-links or "None" '
        '(without quotes):',
        expected_count=1,
    )
    must_paths = [str(Path(p)) for p in must_paths]
    folders_all.extend(must_paths)
    _write_paths_file(Path(MUST_LINKS_FILE), must_paths)

    no_paths = prompt_folder_paths(
        'Write paths to the folder with no constraints or "None" '
        '(without quotes):',
        expected_count=1,
    )
    no_paths = [str(Path(p)) for p in no_paths]
    folders_all.extend(no_paths)
    _write_paths_file(Path(NO_LINKS_FILE), no_paths)

    return folders_all


def main() -> int:
    """Run the command-line workflow."""
    cpp_source = Path(CLUSTERING_SOURCE_FILE)
    if not cpp_source.is_file():
        raise FileNotFoundError(
            f"Required source file '{cpp_source}' not found in the "
            "current directory. Please run this script from the "
            "directory containing the C++ source."
        )

    distance_metric = prompt_distance_metric()

    # Validate that at least one folder was provided before any
    # processing begins. The three manifest files (cannot_links,
    # must_links, no_links) are still written as each set of paths is
    # collected, since they should always reflect the most recent prompt
    # answers regardless of outcome -- but no further work (parsing,
    # compiling, running) happens until this check passes.
    folders_all = _collect_link_folders()
    if not folders_all:
        raise ValueError(
            "You must provide at least one folder (cannot-links, "
            "must-links, or no-links)."
        )

    known_sample_ids: set[str] = set()
    for folder in folders_all:
        _process_input_folder(folder, known_sample_ids)

    exe_path = compile_cpp_code(str(cpp_source), "run_clustering")

    clustering_results_dir = Path("clustering_results")
    _clean_clustering_results(clustering_results_dir)

    if platform.system() == "Windows":
        executable = f".\\{exe_path}"
    else:
        executable = f"./{exe_path}"

    print(f"Running clustering executable: {executable}...", file=sys.stderr)
    # Build the results-dir argument with explicit str(...) for
    # consistency with the rest of the file's Path-to-str conversions,
    # to preserve the executable's prefix-path interface.
    results_dir_arg = str(clustering_results_dir) + os.sep
    result = subprocess.run(
        [executable, results_dir_arg, distance_metric],
        stdout=None,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Error running clustering executable:\n"
            + (result.stderr.strip() or f"Exit code {result.returncode}")
        )

    verify_artifacts([
        clustering_results_dir / SAMPLE_PATHS_FILE,
        clustering_results_dir / TREE_TOPOLOGY_FILE,
    ])
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    sample_count, node_count = _validate_clustering_outputs(
        clustering_results_dir
    )
    print(
        "Clustering completed successfully with a validated immutable tree: "
        f"{sample_count} sample(s), {node_count} node(s) "
        f"(IDs 0..{node_count - 1})."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Error: [clustering] {exc}", file=sys.stderr)
        raise SystemExit(1)
