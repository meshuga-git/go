"""Self-test: no pipeline format is parsed outside the shared core.

Requires Python 3.10 or newer, with pytest installed.

Every read of text and every token parser in a stage script must sit
in a function listed below, with the reason it is not a pipeline
format. Pipeline formats are read only through pipeline_io and
pipeline_formats. Phase 2's exit gate rests on this check.
"""

from __future__ import annotations

import ast

from unit_support import REPO

CORE = {"pipeline_io.py", "pipeline_formats.py", "pipeline_native.py"}
# Calls that read files or parse machine tokens.
READERS = {
    "read_utf8_lines", "iter_utf8_lines", "read_csv_rows",
    "parse_csv_record", "parse_canonical_int", "parse_decimal_float",
    "open", "read_text", "read_bytes", "load", "loads", "loadtxt",
    "genfromtxt", "fromfile", "read_csv", "read_table", "read_fwf",
    "read_excel", "read_json", "read_stata", "read_spss", "read_hdf",
    "read_parquet", "read_orc", "read_sas", "read_pickle",
}
# (script, function): why the input it reads is not a pipeline format.
ALLOWED = {
    ("build_dendrogram.py", "load_folder_categories"):
        "folder lists, which the user may also write (--config)",
    ("clustering.py", "_clean_output_files"):
        "sample lists of older runs, read to clean up (task 5.4)",
    ("clustering.py", "_read_csv"): "the user's input data",
    ("clustering.py", "_read_dat"): "the user's input data",
    ("clustering.py", "_read_excel"): "the user's input data",
    ("clustering.py", "_read_hdf_file"): "the user's input data",
    ("clustering.py", "_read_json"): "the user's input data",
    ("clustering.py", "_read_orc"): "the user's input data",
    ("clustering.py", "_read_parquet"): "the user's input data",
    ("clustering.py", "_read_spss"): "the user's input data",
    ("clustering.py", "_read_stata"): "the user's input data",
    ("clustering.py", "_read_tab_delimited"): "the user's input data",
    ("clustering.py", "_validate_input_dataframe"):
        "values of the user's input data",
    ("clustering.py", "write_sample_csvs"):
        "values of the user's input data",
    ("compute_variances.py", "load_gene_filter"): "the user's gene list",
    ("compute_variances.py", "load_state"):
        "the variance stage's private state (task 4.2)",
}


def _reads(source: str) -> set[tuple[str, str]]:
    """List the top-level functions that call a reader, with the call.

    Args:
        source: A script's source code.

    Returns:
        ``(function, reader)`` pairs; module-level calls count as
        ``<module>``.
    """
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and owner == "<module>"):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call):
                function = child.func
                name = (function.id if isinstance(function, ast.Name)
                        else getattr(function, "attr", None))
                if name in READERS:
                    found.add((owner, name))
            visit(child, owner)

    visit(ast.parse(source), "<module>")
    return found


def test_no_pipeline_format_is_parsed_outside_the_core() -> None:
    """Scripts read pipeline formats only through the core."""
    scripts = sorted(path for path in REPO.glob("*.py")
                     if path.name not in CORE)
    assert {path.name for path in scripts} >= {
        "clustering.py", "build_dendrogram.py", "compute_variances.py",
        "samples_by_cluster.py", "tjala.py", "samples_by_signature.py"}
    unexpected = sorted(
        (path.name, function, reader)
        for path in scripts
        for function, reader in _reads(path.read_text(encoding="utf-8"))
        if (path.name, function) not in ALLOWED
    )
    assert not unexpected


def test_the_allowed_readers_still_exist() -> None:
    """Every exemption names a function that still reads something."""
    present = {
        (path.name, function)
        for path in REPO.glob("*.py") if path.name not in CORE
        for function, _ in _reads(path.read_text(encoding="utf-8"))
    }
    assert set(ALLOWED) <= present
