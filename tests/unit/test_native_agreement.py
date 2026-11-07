"""Unit tests that hold the Python core and the C++ programs together.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed, and a C++20 compiler (see unit_support.py).

Each C++ program carries its own copy of the machine grammar. A small
harness includes a program's source and feeds its integer and decimal
parsers the same tokens as pipeline_io's; both sides must accept the
same tokens and read the same values. The Python history writer must
reproduce reconstruct_hierarchy's output byte for byte.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from unit_support import REPO, compiler_name, make_tree

import pipeline_formats as formats
import pipeline_io
import pipeline_native

PROGRAMS = ("run_clustering", "reconstruct_hierarchy", "compute_variances")
INTEGERS = [
    "0", "7", "-7", "10", "2147483647", "-2147483648",
    "9223372036854775807", "-9223372036854775808", "9223372036854775808",
    "-0", "+7", "07", "00", "7.0", "1e3", " 7", "7 ", "", "-", "0x7",
    "7_0", "٧", "７",
]
DECIMALS = [
    "0", "1", "-1", "+1.5", ".5", "5.", "1e5", "1E-5", "1e+5", "-0.0",
    "0.1", "123456789012345678901234567890", "2.5e-310",
    "4.9406564584124654e-324", "2.4703282292062328e-324",
    "2.4703282292062327e-324", "1e-400", "1e400", "1.7976931348623157e308",
    "inf", "nan", "0x1p3", "1_0", " 1", "1 ", "", ".", "e5", "1e", "+",
    "١",
]
# The harness reads "kind<TAB>hex-encoded token" lines and answers
# "ok <value>" or "error" for each.
HARNESS = """#define main tavnit_program_main
#include "{program}.cpp"
#undef main
#include <iostream>
#include <string>

int main() {{
    std::string line;
    while (std::getline(std::cin, line)) {{
        if (!line.empty() && line.back() == '\\r') line.pop_back();
        const auto tab = line.find('\\t');
        const std::string kind = line.substr(0, tab);
        std::string token;
        for (auto i = tab + 1; i + 1 < line.size(); i += 2) {{
            token.push_back(static_cast<char>(
                std::stoi(line.substr(i, 2), nullptr, 16)));
        }}
        try {{
            // Parse before printing, so that a rejection prints nothing
            // but "error".
            if (kind == "int") {{
                const auto value =
                    pipeline_io::parse_integer<long long>(token, "t");
                std::cout << "ok " << value << "\\n";
            }} else {{
                const auto value =
                    pipeline_io::parse_decimal_float(token, "t");
                std::cout << std::format("ok {{:.17g}}\\n", value);
            }}
        }} catch (const std::exception&) {{
            std::cout << "error\\n";
        }}
    }}
}}
"""


@pytest.fixture(scope="module")
def compiler() -> pipeline_native.Compiler:
    """Return the compiler of the toolchain under test."""
    return pipeline_native.find_compiler(compiler_name())


def _python_answer(kind: str, token: str) -> str:
    """Return what the Python grammar makes of one token.

    Args:
        kind: ``int`` or ``decimal``.
        token: The token.

    Returns:
        ``ok <value>`` or ``error``, as the harness would print it; an
        integer outside int64 counts as an error, as in C++.
    """
    try:
        if kind == "int":
            value = pipeline_io.parse_canonical_int(token, source="t")
            if not -2**63 <= value < 2**63:
                return "error"
            return f"ok {value}"
        return f"ok {pipeline_io.parse_decimal_float(token, source='t')!r}"
    except ValueError:
        return "error"


def _same(python: str, native: str, kind: str) -> bool:
    """Tell whether two answers agree, comparing values numerically."""
    if python == "error" or native == "error":
        return python == native
    convert = int if kind == "int" else float
    return convert(python[3:]) == convert(native[3:])


@pytest.mark.parametrize("program", PROGRAMS)
def test_cpp_grammar_agrees_with_python(
        program: str, compiler: pipeline_native.Compiler,
        tmp_path: Path) -> None:
    """Integers and decimals: same tokens accepted, same values read."""
    for name in (f"{program}.cpp", pipeline_io.NATIVE_FILES_HEADER_FILE):
        shutil.copyfile(REPO / name, tmp_path / name)
    harness = tmp_path / f"grammar_{program}.cpp"
    harness.write_text(HARNESS.format(program=program), encoding="utf-8")
    executable = pipeline_native.build_program(harness, compiler)
    cases = [("int", token) for token in INTEGERS] + [
        ("decimal", token) for token in DECIMALS]
    done = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=True,
        input="".join(f"{kind}\t{token.encode().hex()}\n"
                      for kind, token in cases),
    )
    answers = done.stdout.splitlines()
    assert len(answers) == len(cases)
    disagreements = [
        (kind, token, _python_answer(kind, token), native)
        for (kind, token), native in zip(cases, answers, strict=True)
        if not _same(_python_answer(kind, token), native, kind)
    ]
    assert not disagreements


@pytest.mark.parametrize("samples", [2, 3, 64, 300])
def test_history_writer_matches_reconstruct_hierarchy(
        samples: int, compiler: pipeline_native.Compiler,
        tmp_path: Path) -> None:
    """render_tree_states writes reconstruct_hierarchy's exact bytes."""
    results = make_tree(tmp_path, samples, seed=100 + samples)
    executable = pipeline_native.build_program(
        REPO / pipeline_io.RECONSTRUCTION_SOURCE_FILE, compiler)
    pipeline_native.run_program(executable, [str(results)], cwd=tmp_path)
    paths, states = formats.read_snapshots(results)
    written = (results / pipeline_io.TREE_STATES_FILE).read_bytes()
    assert formats.render_tree_states(paths, states).encode() == written
