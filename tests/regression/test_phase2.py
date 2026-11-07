"""Regression tests for the Phase 2 shared core, one or more per task.

Requires Python 3.10 or newer, with requirements.txt and pytest
installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import golden, toolchain

REPO = golden.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pipeline_native  # noqa: E402 - needs the package on sys.path

# A program whose exit code comes from a local header, so that the
# header's content decides the result.
PROGRAM = """#include <cstdio>
#include "tiny.hpp"

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::printf("%s%s", i > 1 ? " " : "", argv[i]);
    }
    std::printf("\\n");
    return EXIT_CODE;
}
"""


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the per-user cache folder into the test's folder.

    Args:
        tmp_path: Pytest's per-test temporary folder.
        monkeypatch: Pytest's environment patcher.

    Returns:
        The build cache folder.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "local"))
    if sys.platform == "darwin":
        monkeypatch.setenv("HOME", str(home))
    folder = pipeline_native.cache_folder()
    assert home in folder.parents
    return folder


@pytest.fixture
def compiler(monkeypatch: pytest.MonkeyPatch) -> pipeline_native.Compiler:
    """Return the compiler of this machine's golden toolchain.

    Args:
        monkeypatch: Pytest's environment patcher.

    Returns:
        The compiler, found through ``CXX``.
    """
    monkeypatch.setenv("CXX", golden.TOOLCHAINS[toolchain()][0])
    return pipeline_native.find_compiler()


def _sources(folder: Path, exit_code: int = 3) -> Path:
    """Write the test program and its header.

    Args:
        folder: Folder to write them in.
        exit_code: Exit code the header sets.

    Returns:
        The program's source.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tiny.hpp").write_text(f"#define EXIT_CODE {exit_code}\n",
                                     encoding="utf-8")
    source = folder / "tiny.cpp"
    source.write_text(PROGRAM, encoding="utf-8")
    return source


def _run(executable: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a built program and capture its output."""
    return subprocess.run([str(executable), *args], capture_output=True,
                          text=True, check=False)


def test_2_3_programs_are_built_once_into_the_user_cache(
        cache: Path, compiler: pipeline_native.Compiler, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Task 2.3: builds go to the per-user cache and are reused."""
    source = _sources(tmp_path / "code")
    executable = pipeline_native.build_program(source, compiler)
    assert cache in executable.parents
    assert executable.name == "tiny" + (".exe" if os.name == "nt" else "")
    assert sorted(p.name for p in source.parent.iterdir()) == [
        "tiny.cpp", "tiny.hpp"]
    assert "Compiling tiny.cpp" in capsys.readouterr().err
    done = _run(executable, "a", "b")
    assert (done.returncode, done.stdout.strip()) == (3, "a b")

    assert pipeline_native.build_program(source, compiler) == executable
    assert "Using the cached build" in capsys.readouterr().err


def test_2_3_the_cache_key_covers_included_headers(
        cache: Path, compiler: pipeline_native.Compiler,
        tmp_path: Path) -> None:
    """Task 2.3 (A9): a changed header gives a new build."""
    first = pipeline_native.build_program(_sources(tmp_path / "one", 3),
                                          compiler)
    second = pipeline_native.build_program(_sources(tmp_path / "two", 4),
                                           compiler)
    assert first.parent != second.parent
    assert _run(first).returncode == 3
    assert _run(second).returncode == 4


def test_2_3_a_damaged_cached_program_is_rebuilt(
        cache: Path, compiler: pipeline_native.Compiler, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Task 2.3: a cached program that has changed is not run."""
    source = _sources(tmp_path / "code")
    executable = pipeline_native.build_program(source, compiler)
    executable.write_bytes(b"damaged")
    capsys.readouterr()
    assert pipeline_native.build_program(source, compiler) == executable
    assert "Compiling tiny.cpp" in capsys.readouterr().err
    assert _run(executable).returncode == 3


def _command_line(cache: Path, tmp_path: Path, *args: str,
                  source: str = PROGRAM) -> subprocess.CompletedProcess:
    """Run ``pipeline_native.py`` as a script next to a test source.

    Args:
        cache: The test's build cache, to keep the run inside it.
        tmp_path: Pytest's per-test temporary folder.
        *args: The script's arguments.
        source: Content of ``tiny.cpp``.

    Returns:
        The finished process, with text output.
    """
    folder = tmp_path / "script"
    _sources(folder)
    (folder / "tiny.cpp").write_text(source, encoding="utf-8")
    shutil.copyfile(REPO / "pipeline_native.py", folder / "pipeline_native.py")
    done = subprocess.run(
        [sys.executable, "pipeline_native.py", *args], cwd=folder,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False,
    )
    assert not list(folder.glob("tiny*.exe")) and not (
        folder / "tiny").exists()
    assert cache.is_dir()
    return done


def test_2_3_run_passes_arguments_and_the_exit_code(
        cache: Path, compiler: pipeline_native.Compiler,
        tmp_path: Path) -> None:
    """Task 2.3: ``run`` builds, then runs with the given arguments."""
    done = _command_line(cache, tmp_path, "run", "tiny", "x", "y z")
    assert done.returncode == 3, done.stderr
    assert done.stdout.strip() == "x y z"


def test_2_3_compiler_errors_reach_the_console(
        cache: Path, compiler: pipeline_native.Compiler,
        tmp_path: Path) -> None:
    """Task 2.3: the compiler's own diagnostics are shown."""
    done = _command_line(cache, tmp_path, "build", "tiny",
                         source="int main() { return missing_name; }\n")
    assert done.returncode == 1
    assert "missing_name" in done.stdout + done.stderr
    assert "Error: [pipeline_native] Compiling tiny.cpp" in done.stderr


def test_2_3_an_explicit_compiler_comes_before_cxx(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 2.3: --cxx, then CXX, decide before discovery."""
    name = golden.TOOLCHAINS[toolchain()][0]
    monkeypatch.setenv("CXX", "no-such-compiler")
    assert pipeline_native.find_compiler(name).path == shutil.which(name)
    with pytest.raises(RuntimeError, match="given by CXX"):
        pipeline_native.find_compiler()


def test_2_3_a_missing_compiler_lists_what_was_tried(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 2.3: discovery reports every place it looked."""
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("VSCMD_VER", raising=False)
    monkeypatch.delenv("VCINSTALLDIR", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    with pytest.raises(RuntimeError) as caught:
        pipeline_native.find_compiler()
    message = str(caught.value)
    assert "g++ (not on PATH)" in message
    assert "clang++ (not on PATH)" in message
    if os.name == "nt":
        assert "vswhere" in message


@pytest.mark.skipif(os.name != "nt", reason="Visual Studio is Windows-only")
def test_2_3_windows_finds_visual_studio_without_a_developer_prompt(
        cache: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 2.3 (G7): vswhere finds MSVC and sets up its environment."""
    root = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    if not (Path(root) / "Microsoft Visual Studio" / "Installer"
            / "vswhere.exe").is_file():
        pytest.skip("vswhere is not installed")
    for name in ("CXX", "VSCMD_VER", "VCINSTALLDIR"):
        monkeypatch.delenv(name, raising=False)
    compiler = pipeline_native.find_compiler()
    assert compiler.family == "msvc"
    assert compiler.env is not None
    executable = pipeline_native.build_program(_sources(tmp_path / "code"),
                                               compiler)
    assert _run(executable, "ok").returncode == 3


def test_2_5_every_manifest_names_its_stage_script(workflow: Path) -> None:
    """Task 2.5 (A8): the producer is the stage script, never the core.

    Since the manifest writer moved into pipeline_io, a default of "this
    module" would name pipeline_io.py; the producer is now required.
    """
    import hashlib
    import inspect

    import pipeline_io

    for function in (pipeline_io.write_provenance,
                     pipeline_io.render_provenance):
        producer = inspect.signature(function).parameters["producer"]
        assert producer.default is inspect.Parameter.empty
    results = workflow / "clustering_results"
    script = hashlib.sha256(
        (results / "compute_variances.py").read_bytes()).hexdigest()
    for name in ("prepared.provenance", "term_variances.txt.provenance"):
        record = pipeline_io.read_provenance(results / name)
        assert record.producer == script, name
