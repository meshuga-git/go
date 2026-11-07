"""Find a C++ compiler, build the native programs and run them.

Requires Python 3.10 or newer; uses only the standard library.

Every native program of the pipeline is built here, the same way:

* The compiler comes from an explicit choice (``--cxx``), then the
  ``CXX`` environment variable, then discovery: on Windows ``cl`` in
  an active Developer prompt, then Visual Studio located with vswhere;
  everywhere ``g++``, then ``clang++``, on PATH.
* Builds are cached in the per-user cache folder, never in the code
  folder: ``%LOCALAPPDATA%`` on Windows, ``~/Library/Caches`` on macOS
  and ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere, each followed by
  ``tavnit/build``. The cache key covers the source and every local
  header it includes, the compiler's path and version, and the flags;
  the compiler reads copies of exactly the files that were hashed.
* Executables end in ``.exe`` on Windows and have no suffix elsewhere.
* The compiler's and the program's output reach the console as they
  are written.

Run as a script, the module builds the program whose C++ source of
that name sits next to it, or builds it if needed and runs it in the
current folder with the remaining arguments:

    python pipeline_native.py build reconstruct_hierarchy
    python pipeline_native.py run reconstruct_hierarchy DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Suffix of an executable on this platform.
EXE_SUFFIX = ".exe" if os.name == "nt" else ""

#: Compiler options by command-line style, after the compiler itself.
#: ``-ffp-contract=off`` keeps Apple Clang on arm64 from fusing
#: multiply-adds, which would change the results there.
FLAGS = {
    "gnu": ("-std=c++20", "-O2", "-ffp-contract=off"),
    "msvc": ("/nologo", "/std:c++20", "/O2", "/EHsc", "/utf-8"),
}

# Bumped whenever the layout of a cache entry changes.
_CACHE_FORMAT = 1
_MSVC_NAMES = {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}
_LOCAL_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"',
                               re.MULTILINE)
_PROGRAM_RE = re.compile(r"[A-Za-z0-9_]+\Z")
# vcvarsall.bat argument and Visual Studio component by host machine.
_MSVC_HOSTS = {
    "AMD64": ("x64", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"),
    "X86": ("x86", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"),
    "ARM64": ("arm64", "Microsoft.VisualStudio.Component.VC.Tools.ARM64"),
}


@dataclass(frozen=True)
class Compiler:
    """A C++ compiler ready to run.

    Attributes:
        path: The compiler executable.
        family: ``"msvc"`` for cl and clang-cl, ``"gnu"`` otherwise;
            it selects the command-line style.
        identity: The compiler's version text, part of the cache key.
        env: Environment to run it in, or None for the current one; a
            Visual Studio found through vswhere needs its own.
    """

    path: str
    family: str
    identity: str
    env: Mapping[str, str] | None = None


class NativeProgramError(RuntimeError):
    """A native program exited with a nonzero code.

    Attributes:
        returncode: The program's exit code.
    """

    returncode: int

    def __init__(self, name: str, returncode: int) -> None:
        """Describe the failure.

        Args:
            name: The program's file name.
            returncode: Its exit code.
        """
        super().__init__(f"{name} exited with code {returncode}")
        self.returncode = returncode


def cache_folder() -> Path:
    """Return the per-user folder that holds the compiled programs.

    Returns:
        ``tavnit/build`` in the platform's per-user cache folder.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        base = os.environ.get("XDG_CACHE_HOME", "")
        root = Path(base) if os.path.isabs(base) else Path.home() / ".cache"
    return root / "tavnit" / "build"


def _family(path: str) -> str:
    """Return the command-line style of a compiler executable.

    Args:
        path: The compiler executable.

    Returns:
        ``"msvc"`` for cl and clang-cl, ``"gnu"`` otherwise.
    """
    return "msvc" if Path(path).name.lower() in _MSVC_NAMES else "gnu"


def _describe(path: str, env: Mapping[str, str] | None = None) -> Compiler:
    """Ask a compiler for its version and describe it.

    Args:
        path: The compiler executable.
        env: Environment to run it in, or None for the current one.

    Returns:
        The compiler.

    Raises:
        RuntimeError: If the compiler cannot be started.
    """
    family = _family(path)
    # cl prints its version banner on stderr when given no arguments;
    # clang-cl, like the GNU-style drivers, answers --version.
    plain_cl = Path(path).name.lower() in {"cl", "cl.exe"}
    command = [path] if plain_cl else [path, "--version"]
    try:
        done = subprocess.run(command, capture_output=True, env=env,
                              check=False)
    except OSError as exc:
        raise RuntimeError(f"Cannot start the C++ compiler {path}: "
                           f"{exc}") from exc
    banner = done.stderr if plain_cl else done.stdout or done.stderr
    identity = banner.decode("utf-8", "replace").replace("\r\n", "\n")
    return Compiler(path, family, identity.strip(), env)


def _locate(requested: str) -> str | None:
    """Resolve a compiler named on PATH or by a path.

    Args:
        requested: A command name or a path to an executable.

    Returns:
        The executable's path, or None if there is none.
    """
    found = shutil.which(requested)
    if found is not None:
        return found
    candidate = Path(requested)
    if candidate.is_file() and (os.name == "nt"
                                or os.access(candidate, os.X_OK)):
        return str(candidate.resolve())
    return None


def _lookup(env: Mapping[str, str], name: str) -> str:
    """Return an environment variable, ignoring the case of its name.

    Args:
        env: The environment.
        name: The variable's name.

    Returns:
        Its value, or an empty string if it is not set.
    """
    for key, value in env.items():
        if key.upper() == name.upper():
            return value
    return ""


def _msvc_environment(vcvarsall: Path, host: str) -> dict[str, str]:
    """Return the environment that vcvarsall.bat sets up.

    Args:
        vcvarsall: The Visual Studio batch file.
        host: Its architecture argument, such as ``x64``.

    Returns:
        The environment variables after the batch file has run.

    Raises:
        RuntimeError: If the batch file fails.
    """
    # cmd /u writes the output of its own commands, here "set", in
    # UTF-16, so no character of the environment is lost.
    command = f'cmd.exe /d /u /s /c ""{vcvarsall}" {host} >nul && set"'
    done = subprocess.run(command, capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{vcvarsall} {host} failed with exit code "
                           f"{done.returncode}")
    env: dict[str, str] = {}
    for line in done.stdout.decode("utf-16-le", "replace").splitlines():
        name, sep, value = line.partition("=")
        if sep and name:
            env[name] = value
    return env


def _msvc_from_vswhere(tried: list[str]) -> Compiler | None:
    """Find cl in the newest Visual Studio that has the C++ tools.

    Args:
        tried: What discovery has tried so far; this adds its steps.

    Returns:
        The compiler with its environment, or None if there is none.

    Raises:
        RuntimeError: If Visual Studio's setup script fails.
    """
    root = (os.environ.get("ProgramFiles(x86)")
            or os.environ.get("ProgramFiles") or r"C:\Program Files (x86)")
    vswhere = Path(root) / "Microsoft Visual Studio" / "Installer" / (
        "vswhere.exe")
    if not vswhere.is_file():
        tried.append(f"vswhere ({vswhere} does not exist)")
        return None
    host, component = _MSVC_HOSTS.get(platform.machine().upper(),
                                      _MSVC_HOSTS["AMD64"])
    done = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires", component,
         "-property", "installationPath", "-utf8"],
        capture_output=True, check=False,
    )
    lines = done.stdout.decode("utf-8", "replace").splitlines()
    if done.returncode != 0 or not lines or not lines[0].strip():
        tried.append("vswhere (no Visual Studio with the C++ tools)")
        return None
    vcvarsall = (Path(lines[0].strip()) / "VC" / "Auxiliary" / "Build"
                 / "vcvarsall.bat")
    if not vcvarsall.is_file():
        tried.append(f"vswhere ({vcvarsall} does not exist)")
        return None
    env = _msvc_environment(vcvarsall, host)
    path = shutil.which("cl", path=_lookup(env, "PATH"))
    if path is None:
        tried.append(f"vswhere ({vcvarsall} put no cl on PATH)")
        return None
    return _describe(path, env)


def find_compiler(explicit: str | None = None) -> Compiler:
    """Choose the C++ compiler for the native programs.

    The order is: ``explicit``, the ``CXX`` environment variable, then,
    on Windows, cl in an active Developer prompt and the newest Visual
    Studio that vswhere finds, then ``g++`` and ``clang++`` on PATH.

    Args:
        explicit: A compiler named by the user (``--cxx``), if any.

    Returns:
        The compiler.

    Raises:
        RuntimeError: If a named compiler does not exist, or discovery
            finds none; the message lists what was tried.
    """
    for requested, origin in ((explicit, "--cxx"),
                              (os.environ.get("CXX"), "CXX")):
        if requested:
            path = _locate(requested)
            if path is None:
                raise RuntimeError(
                    f"The C++ compiler {requested!r} given by {origin} "
                    "was not found")
            return _describe(path)
    tried = ["CXX (not set)"]
    if os.name == "nt":
        cl = shutil.which("cl")
        if cl is not None and (os.environ.get("VSCMD_VER")
                               or os.environ.get("VCINSTALLDIR")):
            return _describe(cl)
        tried.append("cl in an active Developer prompt (none)")
        compiler = _msvc_from_vswhere(tried)
        if compiler is not None:
            return compiler
    for name in ("g++", "clang++"):
        path = shutil.which(name)
        if path is not None:
            return _describe(path)
        tried.append(f"{name} (not on PATH)")
    raise RuntimeError(
        "No C++ compiler found. Tried: " + "; ".join(tried)
        + ". Install one, or name it with CXX.")


def _source_files(source: Path) -> dict[str, bytes]:
    """Read a source file and the local headers it includes.

    Local headers are those included with quotes, searched next to the
    file that includes them; headers that do not exist are left to the
    compiler to report.

    Args:
        source: The C++ source file.

    Returns:
        Content by path relative to the source's folder, the source
        first.

    Raises:
        RuntimeError: If a local header lies outside that folder.
        OSError: If the source cannot be read.
    """
    base = source.parent
    files: dict[str, bytes] = {}
    pending = [source.name]
    while pending:
        name = pending.pop(0)
        if name in files:
            continue
        if name.startswith("../") or Path(name).is_absolute():
            raise RuntimeError(f"{source.name} includes {name!r}, which "
                               "lies outside its folder")
        path = base / name
        if name != source.name and not path.is_file():
            continue
        data = path.read_bytes()
        files[name] = data
        folder = Path(name).parent
        for match in _LOCAL_INCLUDE_RE.finditer(
                data.decode("utf-8", "replace")):
            pending.append(posixpath.normpath(
                (folder / match[1]).as_posix()))
    return files


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        Its digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _compile_command(compiler: Compiler, source: str,
                     executable: str) -> list[str]:
    """Return the command line that compiles one source file.

    Args:
        compiler: The compiler.
        source: The source's file name, in the build folder.
        executable: The executable's file name, in the build folder.

    Returns:
        The command line, to run in the build folder.
    """
    if compiler.family == "msvc":
        stem = Path(executable).stem
        return [compiler.path, *FLAGS["msvc"], source, f"/Fe:{executable}",
                f"/Fo:{stem}.obj"]
    return [compiler.path, *FLAGS["gnu"], "-o", executable, source]


def _cached_build(entry: Path, key: str, executable: str) -> Path | None:
    """Return a cached executable if its entry is intact.

    Args:
        entry: The cache entry's folder.
        key: The cache key the entry must record.
        executable: The executable's file name.

    Returns:
        The executable, or None if the entry is missing, records
        another key, or holds an executable that has changed.
    """
    try:
        record = json.loads((entry / "build.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    path = entry / executable
    if (not isinstance(record, dict) or record.get("key") != key
            or not path.is_file()
            or _sha256_file(path) != record.get("executable_sha256")):
        return None
    return path


def build_program(source: Path, compiler: Compiler | None = None) -> Path:
    """Build a native program unless a current build is cached.

    The source and its local headers are hashed, copied into a
    temporary folder in the cache and compiled there, so the cached
    executable comes from exactly the hashed files. A finished build
    is published by renaming its folder, so concurrent builds of the
    same program cannot leave a partial entry.

    Args:
        source: The program's C++ source; the executable takes its
            stem.
        compiler: The compiler, or None to call ``find_compiler()``.

    Returns:
        The executable, in the cache.

    Raises:
        RuntimeError: If no compiler is found or the compiler fails.
        OSError: If the source cannot be read or the cache written.
    """
    compiler = compiler or find_compiler()
    source = Path(source).resolve()
    files = _source_files(source)
    executable = source.stem + EXE_SUFFIX
    description = {
        "format": _CACHE_FORMAT,
        "program": source.stem,
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(files.items())},
        "compiler": compiler.path,
        "identity": compiler.identity,
        # MSVC finds its headers and the Windows SDK through INCLUDE.
        "include": _lookup(compiler.env or os.environ, "INCLUDE")
        if compiler.family == "msvc" else "",
        "flags": list(FLAGS[compiler.family]),
        "platform": [sys.platform, platform.machine()],
    }
    key = hashlib.sha256(
        json.dumps(description, sort_keys=True).encode("utf-8")
    ).hexdigest()
    folder = cache_folder()
    entry = folder / f"{source.stem}-{key[:16]}"
    cached = _cached_build(entry, key, executable)
    if cached is not None:
        print(f"Using the cached build of {source.name}: {cached}",
              file=sys.stderr, flush=True)
        return cached

    folder.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{source.stem}-", dir=folder))
    try:
        for name, data in files.items():
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_bytes(data)
        print(f"Compiling {source.name} with {compiler.path}",
              file=sys.stderr, flush=True)
        done = subprocess.run(
            _compile_command(compiler, source.name, executable),
            cwd=work, env=compiler.env, check=False,
        )
        if done.returncode != 0 or not (work / executable).is_file():
            raise RuntimeError(
                f"Compiling {source.name} with {compiler.path} failed "
                f"(exit code {done.returncode})")
        staged = work / "entry"
        staged.mkdir()
        os.replace(work / executable, staged / executable)
        record = dict(description, key=key, executable=executable,
                      executable_sha256=_sha256_file(staged / executable))
        (staged / "build.json").write_text(
            json.dumps(record, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        if entry.exists():
            # A damaged entry under the same key: set it aside.
            os.replace(entry, work / "stale")
        try:
            os.replace(staged, entry)
        except OSError:
            # Another process has just published the same build.
            cached = _cached_build(entry, key, executable)
            if cached is None:
                raise
            return cached
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"Compiled {source.name} into {entry / executable}",
          file=sys.stderr, flush=True)
    return entry / executable


def run_program(executable: Path, args: Sequence[str] = (), *,
                cwd: Path | None = None) -> None:
    """Run a native program, its output reaching the console live.

    Args:
        executable: The program.
        args: Its arguments.
        cwd: Folder to run it in, or None for the current one.

    Raises:
        NativeProgramError: If it exits with a nonzero code.
        OSError: If it cannot be started.
    """
    done = subprocess.run([str(executable), *args], cwd=cwd, check=False)
    if done.returncode != 0:
        raise NativeProgramError(Path(executable).name, done.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a program, or build it if needed and run it.

    Args:
        argv: Arguments without the program name, or None for the
            command line.

    Returns:
        0 after a build, the program's exit code after a run, or 1 if
        the build fails.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline_native.py",
        description=("Build a native program of the pipeline from the C++ "
                     "source of that name next to this script, or build it "
                     "if needed and run it."),
    )
    parser.add_argument(
        "--cxx", metavar="COMPILER",
        help="C++ compiler to use; overrides CXX and discovery.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="build the program and print its path")
    build.add_argument("program", help="name of the C++ source, without .cpp")
    run = commands.add_parser(
        "run", help="build the program if needed, then run it here")
    run.add_argument("program", help="name of the C++ source, without .cpp")
    run.add_argument("args", nargs=argparse.REMAINDER,
                     help="arguments passed to the program")
    args = parser.parse_args(argv)

    source = Path(__file__).resolve().with_name(f"{args.program}.cpp")
    if not _PROGRAM_RE.match(args.program) or not source.is_file():
        print(f"Error: [pipeline_native] There is no C++ source "
              f"{args.program}.cpp next to {Path(__file__).name}",
              file=sys.stderr)
        return 1
    try:
        executable = build_program(source, find_compiler(args.cxx))
    except (RuntimeError, OSError) as exc:
        print(f"Error: [pipeline_native] {exc}", file=sys.stderr)
        return 1
    if args.command == "build":
        print(executable)
        return 0
    try:
        run_program(executable, args.args)
    except NativeProgramError as exc:
        return exc.returncode
    except OSError as exc:
        print(f"Error: [pipeline_native] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
