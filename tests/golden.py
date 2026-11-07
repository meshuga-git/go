"""Run the pipeline on the golden datasets and check what it writes.

Requires Python 3.10 or newer.

Each scenario copies the package into a fresh work folder, drives every
script non-interactively by feeding its answers on stdin, and records
the SHA-256 of every file the scripts write. Tjala runs under a seed
fixed by this harness; the program itself stays unseeded.

The golden file of a toolchain, ``tests/golden/<toolchain>-min.json``,
is recorded with the minimal versions: Python 3.10 and the oldest
library releases that requirements.txt allows. A run in that
environment must reproduce it byte for byte. A run with any other
versions is checked with ``--compat``: every stage that passes in the
golden file must pass, and every artifact must match, except
``dendrogram.html`` and its provenance record, which embed the Plotly
library itself. Where the golden run failed a stage that passes in the
new environment, the artifacts that stage would have written are not
compared.

Content that depends on the machine rather than on the code is
normalized before hashing:

* The work folder's absolute path becomes ``<ROOT>`` wherever it
  appears, including the hex-encoded paths in provenance records.
* A SHA-256 written inside an artifact that matches a file in the work
  folder becomes ``nh:`` plus that file's normalized hash, or
  ``<BINARY>`` for a compiled executable. Recorded hashes are still
  checked, but no longer depend on the folder or the compiler build.
* Fields that identify code rather than results become placeholders:
  the producer line of a provenance record, which hashes the script
  that wrote it, becomes ``<PRODUCER>``, and the compiler path and
  source hashes in ``compute_variances_state.json`` become ``<CODE>``.
  They change with every edit of the code, while the outputs they
  vouch for are compared in full.
* The random element id Plotly writes into ``dendrogram.html`` becomes
  ``<UUID>``.

Child Python processes run with ``PYTHONIOENCODING=utf-8``, as an
interactive console would give them, and without bytecode caches.

Usage:
    python tests/golden.py --toolchain linux-gcc            # compare
    python tests/golden.py --toolchain linux-gcc --compat   # other libs
    python tests/golden.py --toolchain linux-gcc --record   # record
    python tests/golden.py --toolchain linux-gcc --measure FILE

Exit codes: 0 when every scenario matches, or after recording or
measuring; 1 on a difference; 2 when no golden file exists for the
key, unless --allow-missing is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
GOLDEN_DIR = TESTS / "golden"
DEFAULT_WORK = REPO / ".golden-work"

PACKAGE = (
    "README.md", "build_dendrogram.py", "clustering.py",
    "compute_variances.cpp", "compute_variances.py", "pipeline_files.hpp",
    "pipeline_formats.py", "pipeline_io.py", "reconstruct_hierarchy.cpp",
    "run_clustering.cpp",
    "samples_by_cluster.py", "samples_by_signature.py", "tjala.py",
)
PART2 = (
    "compute_variances.py", "compute_variances.cpp", "pipeline_files.hpp",
    "pipeline_formats.py", "pipeline_io.py", "samples_by_cluster.py",
    "tjala.py",
    "samples_by_signature.py",
)
BINARY_NAMES = {
    "run_clustering.out", "run_clustering.exe", "reconstruct_hierarchy",
    "reconstruct_hierarchy.exe", "compute_variances",
    "compute_variances.exe",
}
BINARY_SUFFIXES = {".obj", ".o", ".pdb", ".ilk", ".exp", ".lib"}
EXE = ".exe" if os.name == "nt" else ""
PACKAGES = ("numpy", "pandas", "scipy", "plotly")
# Artifacts that embed a library, not results; --compat skips them.
VERSION_DEPENDENT = {"dendrogram.html", "dendrogram.html.provenance"}
TJALA_SEED = 20261106
STAGE_TIMEOUT = 3600

# name: (compiler for GNU-style builds, native build style).
TOOLCHAINS = {
    "linux-gcc": ("g++", "gnu"),
    "linux-clang": ("clang++", "gnu"),
    "macos-clang": ("clang++", "gnu"),
    "windows-mingw": ("g++", "gnu"),
    # Today's clustering.py drives GNU-style compilers only, so this run
    # builds run_clustering with MinGW g++ and the other programs with
    # cl, as on a Windows machine with both installed.
    "windows-msvc": ("g++", "msvc"),
}

CANNOT = "data/cancer data/noncancer"
SMALL_TJALA = "3\n2\n10\n10\n10\n3\n2\ncluster_1.txt\n0.05\n"
DEFAULT_TJALA = "\n" * 7 + "\n"
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
CODE_FIELDS = re.compile(
    r'("(compiler|source_sha256|artifact_header_sha256)"\s*:\s*)'
    r'"(?:[^"\\]|\\.)*"'
)
# The script's own --seed option fixes its otherwise random colors.
DENDROGRAM_SEED = "7"


@dataclass
class Stage:
    """One step of a scenario.

    Attributes:
        name: Stage name used in reports.
        cwd: Working folder, relative to the scenario root.
        kind: ``py``, ``py-seeded``, ``compile``, ``reconstruct`` or
            ``copy-part2``.
        args: Script name and arguments for Python stages.
        stdin: Answers fed to the prompts.
    """

    name: str
    cwd: str
    kind: str
    args: list[str] = field(default_factory=list)
    stdin: str = ""


@dataclass
class Scenario:
    """A dataset and the stages run on it.

    Attributes:
        name: Scenario name.
        stages: Stages in order.
        cohorts: ``make_cohorts.py`` arguments, or None.
        tree: ``(samples, seed)`` for ``make_trees.py``, or None.
        genes: Gene list placed as ``clustering_results/genes.txt``.
        slow: Whether the scenario is the timing set.
    """

    name: str
    stages: list[Stage]
    cohorts: list[str] | None = None
    tree: tuple[int, int] | None = None
    genes: str | None = None
    slow: bool = False


DENDROGRAM = Stage("build_dendrogram", ".", "py",
                   ["build_dendrogram.py", "--seed", DENDROGRAM_SEED])


def part1(metric: str, cannot: str, must: str, nolink: str) -> list[Stage]:
    """Return the clustering stages run from the package folder.

    Args:
        metric: Distance-metric number.
        cannot: Cannot-link folders, or ``None``.
        must: Must-link folder, or ``None``.
        nolink: No-link folder, or ``None``.

    Returns:
        The stages of Part 1 of the README.
    """
    answers = f"{metric}\n{cannot}\n{must}\n{nolink}\n"
    return [
        Stage("clustering", ".", "py", ["clustering.py"], answers),
        Stage("build_hierarchy", ".", "compile"),
        Stage("reconstruct_hierarchy", ".", "reconstruct"),
        DENDROGRAM,
    ]


def part2(genes: str = "None", tjala: str = SMALL_TJALA) -> list[Stage]:
    """Return the signature stages run from ``clustering_results``.

    Args:
        genes: Gene-list argument for ``compute_variances.py``.
        tjala: Answers to Tjala's prompts.

    Returns:
        The stages of Part 2 of the README.
    """
    where = "clustering_results"
    return [
        Stage("copy_part2", where, "copy-part2"),
        Stage("compute_variances", where, "py",
              ["compute_variances.py", genes, "--cxx", "{native_cxx}"]),
        Stage("samples_by_cluster", where, "py", ["samples_by_cluster.py"],
              "1\n1\nN\nExit\n"),
        Stage("tjala", where, "py-seeded", ["tjala.py"], tjala),
        Stage("samples_by_signature", where, "py",
              ["samples_by_signature.py"], "1\nExit\n"),
    ]


def build_scenarios() -> dict[str, Scenario]:
    """Define every golden scenario.

    Returns:
        Scenarios by name, in run order.
    """
    cohorts22 = ["--cancer", "12", "--noncancer", "10", "--genes", "400",
                 "--seed", "7"]
    found = [
        Scenario("pipeline22", part1("2", CANNOT, "None", "None") + part2(),
                 cohorts=cohorts22),
        Scenario("euclidean22", part1("1", CANNOT, "None", "None"),
                 cohorts=cohorts22),
        Scenario("spearman22",
                 part1("3", "None", "data/cancer", "data/noncancer"),
                 cohorts=cohorts22),
        Scenario("manhattan22", part1("4", "None", "None", "data/noncancer"),
                 cohorts=cohorts22),
        Scenario("genes22",
                 part1("2", CANNOT, "None", "None") + part2("genes.txt")[:2],
                 cohorts=cohorts22, genes="genes22.txt"),
    ]
    for index, samples in enumerate((2, 3, 64, 300, 1181), start=1):
        where = "clustering_results"
        found.append(Scenario(
            f"tree{index}",
            [
                Stage("build_hierarchy", ".", "compile"),
                Stage("reconstruct_hierarchy", ".", "reconstruct"),
                DENDROGRAM,
                Stage("copy_part2", where, "copy-part2"),
                Stage("samples_by_cluster", where, "py",
                      ["samples_by_cluster.py"], "0\n1\nN\nExit\n"),
            ],
            tree=(samples, index),
        ))
    found.append(Scenario(
        "timing300",
        part1("2", CANNOT, "None", "None") + part2(tjala=DEFAULT_TJALA),
        cohorts=["--cancer", "150", "--noncancer", "150", "--genes", "5000",
                 "--seed", "3"],
        slow=True,
    ))
    return {scenario.name: scenario for scenario in found}


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def is_binary(rel: str) -> bool:
    """Return whether ``rel`` names a compiled program or object."""
    name = rel.rsplit("/", 1)[-1]
    return name in BINARY_NAMES or Path(name).suffix in BINARY_SUFFIXES


def snapshot(root: Path) -> dict[str, str]:
    """Hash every file below ``root`` except bytecode caches.

    Args:
        root: Folder to scan.

    Returns:
        Raw SHA-256 by POSIX-style relative path.
    """
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = path.relative_to(root).as_posix()
            found[rel] = sha256(path.read_bytes())
    return found


class Normalizer:
    """Hash artifacts with machine-dependent content normalized."""

    def __init__(self, root: Path, files: dict[str, str]) -> None:
        """Index the files of one finished scenario.

        Args:
            root: Scenario root folder.
            files: Raw SHA-256 by relative path, for every file.
        """
        self.root = root
        self.files = files
        self.by_raw = {raw: rel for rel, raw in files.items()}
        variants = set()
        for base in (str(root), str(root.resolve())):
            for text in (base, base.replace("\\", "/")):
                variants.update({text, text.replace("\\", "\\\\")})
                if len(text) > 1 and text[1] == ":":
                    variants.add(text[0].swapcase() + text[1:])
        self.roots = sorted(variants, key=len, reverse=True)
        self.memo: dict[str, str] = {}

    def digest(self, rel: str) -> str:
        """Return the normalized hash of one file.

        Args:
            rel: Relative path of the file.

        Returns:
            Hex SHA-256 of the normalized content.
        """
        if rel not in self.memo:
            self.memo[rel] = "<CYCLE>"
            data = (self.root / rel).read_bytes()
            self.memo[rel] = sha256(self.normalize(rel, data))
        return self.memo[rel]

    def _hash_token(self, match: re.Match[str]) -> str:
        rel = self.by_raw.get(match.group(0))
        if rel is None:
            return match.group(0)
        if is_binary(rel):
            return "<BINARY>"
        return "nh:" + self.digest(rel)

    def normalize(self, rel: str, data: bytes) -> bytes:
        """Return ``data`` with machine-dependent content replaced.

        Args:
            rel: Relative path of the file.
            data: Raw file content.

        Returns:
            Normalized content; binary content is returned unchanged.
        """
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        name = rel.rsplit("/", 1)[-1]
        if name.endswith(".provenance"):
            text = self._decode_provenance(text)
        if name == "compute_variances_state.json":
            text = CODE_FIELDS.sub(r'\1"<CODE>"', text)
        if name.endswith(".html"):
            text = UUID.sub("<UUID>", text)
        text = HEX64.sub(self._hash_token, text)
        for root in self.roots:
            text = text.replace(root, "<ROOT>")
        return text.encode("utf-8")

    @staticmethod
    def _decode_provenance(text: str) -> str:
        """Spell out the hex-encoded names in a provenance record."""
        lines = []
        for line in text.split("\n"):
            fields = line.split("\t")
            if fields[0] == "producer" and len(fields) == 2:
                fields[1] = "<PRODUCER>"
            elif fields[0] in ("I", "O", "P") and len(fields) == 3:
                fields[2] = "path:" + _unhex(fields[2])
            elif fields[0] == "C" and len(fields) == 3:
                fields[1] = "name:" + _unhex(fields[1])
                fields[2] = "path:" + _unhex(fields[2])
            lines.append("\t".join(fields))
        return "\n".join(lines)


def _unhex(value: str) -> str:
    """Decode a hex-encoded UTF-8 string, or return it unchanged."""
    try:
        return bytes.fromhex(value).decode("utf-8")
    except ValueError:
        return value


class Runner:
    """Run scenarios for one toolchain."""

    def __init__(self, toolchain: str, work: Path, measure: bool) -> None:
        """Prepare the runner.

        Args:
            toolchain: Key of ``TOOLCHAINS``.
            work: Parent folder of the scenario folders.
            measure: Whether to record time, memory and hash counts.
        """
        self.cxx, self.native = TOOLCHAINS[toolchain]
        self.native_cxx = "cl" if self.native == "msvc" else self.cxx
        self.work = work
        self.measure = measure
        self.env = dict(os.environ, CXX=self.cxx, PYTHONIOENCODING="utf-8",
                        PYTHONDONTWRITEBYTECODE="1")

    def setup(self, scenario: Scenario, root: Path) -> dict[str, str]:
        """Create the scenario folder and its inputs.

        Args:
            scenario: Scenario to prepare.
            root: Scenario root folder, recreated empty.

        Returns:
            Raw hashes of every file the harness placed.
        """
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for name in PACKAGE:
            shutil.copyfile(REPO / name, root / name)
        if scenario.cohorts is not None:
            subprocess.run(
                [sys.executable, str(TESTS / "data" / "make_cohorts.py"),
                 str(root / "data"), *scenario.cohorts],
                check=True,
            )
        if scenario.tree is not None:
            samples, seed = scenario.tree
            subprocess.run(
                [sys.executable, str(TESTS / "data" / "make_trees.py"),
                 str(root), str(root / "clustering_results"), str(samples),
                 str(seed)],
                check=True, env=self.env,
            )
        return snapshot(root)

    def argv(self, stage: Stage, root: Path, logs: Path) -> list[str]:
        """Build the command line of one stage.

        Args:
            stage: Stage to run.
            root: Scenario root folder.
            logs: Folder for measurement files.

        Returns:
            The command line.
        """
        if stage.kind == "compile":
            target = "reconstruct_hierarchy"
            source = target + ".cpp"
            if self.native == "msvc":
                return ["cl", "/nologo", "/std:c++20", "/O2", "/EHsc", source,
                        f"/Fe:{target}.exe"]
            return [self.cxx, "-std=c++20", "-O2", source, "-o",
                    target + EXE]
        if stage.kind == "reconstruct":
            return [str(root / f"reconstruct_hierarchy{EXE}"),
                    "clustering_results"]
        args = [a.replace("{native_cxx}", self.native_cxx) for a in stage.args]
        hooks = []
        if stage.kind == "py-seeded":
            hooks += ["--seed", str(TJALA_SEED)]
        if self.measure:
            hooks += ["--hashlog", str(logs / f"{stage.name}.hashes.json")]
        if hooks:
            return [sys.executable, str(TESTS / "_bootstrap.py"), *hooks,
                    *args]
        return [sys.executable, *args]

    def run(self, scenario: Scenario) -> dict:
        """Run one scenario and hash what it wrote.

        Args:
            scenario: Scenario to run.

        Returns:
            Stage results, artifact hashes and, when measuring, the
            measurements.
        """
        root = self.work / scenario.name
        logs = self.work / "_logs" / scenario.name
        if logs.exists():
            shutil.rmtree(logs)
        logs.mkdir(parents=True)
        placed = self.setup(scenario, root)
        result: dict = {"stages": {}, "artifacts": {}}
        measured: dict = {}
        for stage in scenario.stages:
            cwd = root / stage.cwd
            if stage.kind == "copy-part2":
                for name in PART2:
                    shutil.copyfile(REPO / name, cwd / name)
                    placed[f"{stage.cwd}/{name}"] = sha256(
                        (cwd / name).read_bytes())
                if scenario.genes is not None:
                    shutil.copyfile(TESTS / "data" / scenario.genes,
                                    cwd / "genes.txt")
                    placed[f"{stage.cwd}/genes.txt"] = sha256(
                        (cwd / "genes.txt").read_bytes())
                continue
            command = self.argv(stage, root, logs)
            if self.measure:
                command = [sys.executable, str(TESTS / "_measure.py"),
                           str(logs / f"{stage.name}.measure.json"), *command]
            started = time.perf_counter()
            done = subprocess.run(
                command, cwd=cwd, input=stage.stdin.encode("utf-8"),
                capture_output=True, env=self.env, timeout=STAGE_TIMEOUT,
            )
            seconds = time.perf_counter() - started
            (logs / f"{stage.name}.stdout").write_bytes(done.stdout)
            (logs / f"{stage.name}.stderr").write_bytes(done.stderr)
            result["stages"][stage.name] = done.returncode
            print(f"  {stage.name:<22} exit {done.returncode:>3}"
                  f"  {seconds:8.2f} s", flush=True)
            if self.measure:
                measured[stage.name] = self._read_measurements(
                    logs, stage.name, root)
            if done.returncode != 0:
                tail = done.stderr.decode("utf-8", "replace")[-3000:]
                print(f"    stderr tail:\n{tail}", flush=True)
                result["failure"] = {"stage": stage.name, "stderr": tail}
                break
        files = snapshot(root)
        normalizer = Normalizer(root, files)
        for rel, raw in files.items():
            if placed.get(rel) == raw or is_binary(rel):
                continue
            result["artifacts"][rel] = normalizer.digest(rel)
        if self.measure:
            result["measurements"] = measured
        return result

    @staticmethod
    def _read_measurements(logs: Path, stage: str, root: Path) -> dict:
        """Collect one stage's measurement files.

        Args:
            logs: Measurement folder.
            stage: Stage name.
            root: Scenario root, for relative paths in hash counts.

        Returns:
            Wall time, peak memory and hash counts per file.
        """
        found = json.loads(
            (logs / f"{stage}.measure.json").read_text(encoding="utf-8"))
        hashes = logs / f"{stage}.hashes.json"
        if hashes.exists():
            counts = json.loads(hashes.read_text(encoding="utf-8"))
            resolved = str(root.resolve())
            found["hashes"] = {
                path.replace(resolved, "<ROOT>").replace("\\", "/"): count
                for path, count in counts.items()
            }
        return found

    def environment(self) -> dict:
        """Describe the interpreter, packages and compilers in use.

        Returns:
            Versions by component.
        """
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        for package in PACKAGES:
            try:
                module = __import__(package)
                info[package] = module.__version__
            except ImportError:
                info[package] = None
        for compiler in sorted({self.cxx, self.native_cxx}):
            command = [compiler] if compiler == "cl" else [compiler,
                                                           "--version"]
            try:
                done = subprocess.run(command, capture_output=True,
                                      text=True, errors="replace")
                # cl prints its banner on stderr and usage on stdout.
                text = (done.stderr if compiler == "cl"
                        else done.stdout or done.stderr).strip()
                info[compiler] = text.splitlines()[0] if text else None
            except OSError:
                info[compiler] = None
        return info


def compare(expected: dict, actual: dict,
            compat: bool = False) -> tuple[list[str], list[str]]:
    """List the differences between two scenario results.

    Args:
        expected: Recorded result.
        actual: New result.
        compat: Whether to apply the rules for other library versions
            (see the module docstring).

    Returns:
        Differences that count as failures, and notes that do not.
    """
    problems: list[str] = []
    notes: list[str] = []
    want, got = expected["artifacts"], actual["artifacts"]
    allow_extra = False
    if expected["stages"] != actual["stages"]:
        message = (f"stage exit codes {actual['stages']} "
                   f"!= {expected['stages']}")
        newly_passing = [
            stage for stage, code in expected["stages"].items()
            if code != 0 and actual["stages"].get(stage) == 0
        ]
        newly_failing = [
            stage for stage, code in expected["stages"].items()
            if code == 0 and actual["stages"].get(stage, 1) != 0
        ]
        if compat and newly_passing and not newly_failing:
            notes.append(f"passes {', '.join(newly_passing)}, which "
                         "fails in the golden run")
            allow_extra = True
        else:
            problems.append(message)
    for rel in sorted(set(want) | set(got)):
        if compat and rel.rsplit("/", 1)[-1] in VERSION_DEPENDENT:
            continue
        if rel not in got:
            problems.append(f"missing  {rel}")
        elif rel not in want:
            if not allow_extra:
                problems.append(f"extra    {rel}")
        elif want[rel] != got[rel]:
            problems.append(f"changed  {rel}")
    return problems, notes


def main(argv: list[str] | None = None) -> int:
    """Run the selected scenarios and compare, record or measure them.

    Args:
        argv: Command-line arguments, without the program name.

    Returns:
        Process exit code.
    """
    scenarios = build_scenarios()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toolchain", required=True, choices=TOOLCHAINS)
    parser.add_argument("--only", nargs="+", choices=scenarios,
                        help="run these scenarios only")
    parser.add_argument("--skip-slow", action="store_true",
                        help="leave out the timing set")
    parser.add_argument("--record", action="store_true",
                        help="write the results as the new golden file")
    parser.add_argument("--compat", action="store_true",
                        help="compare as for library versions other than "
                             "the minimal ones")
    parser.add_argument("--allow-missing", action="store_true",
                        help="exit with 0 when no golden file exists yet")
    parser.add_argument("--measure", type=Path, metavar="FILE",
                        help="write timing, memory and hash counts here")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK,
                        help="parent folder for the scenario folders")
    parser.add_argument("--out", type=Path, metavar="FILE",
                        help="also write the results here, in the format "
                             "of a golden file plus the stderr of any "
                             "failed stage")
    args = parser.parse_args(argv)

    key = f"{args.toolchain}-min"
    selected = [s for s in scenarios.values()
                if (args.only is None or s.name in args.only)
                and not (args.skip_slow and s.slow)]
    golden_path = GOLDEN_DIR / f"{key}.json"
    golden = None
    if golden_path.exists():
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    comparing = golden is not None and not (args.record or args.measure)
    if golden is None and not (args.record or args.measure):
        print(f"No golden file for {key} yet.")

    runner = Runner(args.toolchain, args.work.resolve(), bool(args.measure))
    environment = runner.environment()
    if comparing and not args.compat:
        recorded_env = golden.get("environment", {})
        changed = [name for name in PACKAGES
                   if environment.get(name) != recorded_env.get(name)]
        minor = str(recorded_env.get("python", "")).split(".")[:2]
        if environment["python"].split(".")[:2] != minor:
            changed.insert(0, "python")
        if changed:
            print("Note: this environment differs from the golden run in "
                  f"{', '.join(changed)}; use --compat unless the "
                  "minimal versions are installed.")
    outcomes: dict[str, dict] = {}
    differing = 0
    for scenario in selected:
        print(f"{scenario.name}:", flush=True)
        outcome = runner.run(scenario)
        outcomes[scenario.name] = outcome
        if not comparing:
            continue
        recorded = golden["scenarios"].get(scenario.name)
        if recorded is None:
            print("  no recorded result for this scenario")
            differing += 1
            continue
        problems, notes = compare(recorded, outcome, args.compat)
        differing += bool(problems)
        for line in notes + (problems or ["matches the golden result"]):
            print(f"  {line}")

    stored = {
        "key": key,
        "environment": environment,
        "scenarios": {name: {"stages": outcome["stages"],
                             "artifacts": outcome["artifacts"]}
                      for name, outcome in outcomes.items()},
    }
    if args.out:
        # Failure logs help diagnose a run on another machine, but
        # depend on its paths, so golden files leave them out.
        report = json.loads(json.dumps(stored))
        report["compat"] = args.compat
        for name, outcome in outcomes.items():
            if "failure" in outcome:
                report["scenarios"][name]["failure"] = outcome["failure"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n",
                            encoding="utf-8")
    if args.measure:
        lines = {name: len((REPO / name).read_bytes().splitlines())
                 for name in PACKAGE
                 if name.endswith((".py", ".cpp", ".hpp"))}
        baseline = {
            "key": key,
            "environment": environment,
            "lines": lines,
            "scenarios": {name: outcome["measurements"]
                          for name, outcome in outcomes.items()},
        }
        args.measure.parent.mkdir(parents=True, exist_ok=True)
        args.measure.write_text(json.dumps(baseline, indent=1) + "\n",
                                encoding="utf-8")
        print(f"Wrote {args.measure}")
        return 0
    if args.record:
        if golden is not None and args.only:
            merged = dict(golden)
            merged["environment"] = environment
            merged["scenarios"] = {**golden["scenarios"],
                                   **stored["scenarios"]}
            stored = merged
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(stored, indent=1) + "\n",
                               encoding="utf-8")
        print(f"Recorded {golden_path.relative_to(REPO)}")
        return 0
    if golden is None:
        return 0 if args.allow_missing else 2
    print("All scenarios match." if not differing
          else f"{differing} scenario(s) differ.")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
