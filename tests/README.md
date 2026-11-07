# Tests

## Golden tests

`tests/golden.py` runs every pipeline script on the golden datasets and
compares the SHA-256 of every file the scripts write with the hashes in
`tests/golden/<toolchain>-min.json`. Any change that is not backed by a
decision record or a plan task must leave them identical.

The golden files are recorded with the minimal versions: Python 3.10
and the oldest library releases `requirements.txt` allows. To compare
byte for byte, create that environment:

```text
python3.10 -m venv .venv
.venv/bin/python -m pip install uv                       # POSIX
.venv/bin/python -m uv pip install --resolution lowest-direct --only-binary :all: -r requirements.txt
.venv\Scripts\python -m pip install uv                   # Windows
.venv\Scripts\python -m uv pip install --resolution lowest-direct --only-binary :all: -r requirements.txt
```

With any other Python or library versions, add `--compat`: every stage
that passes in the golden run must pass, and every artifact must match
except `dendrogram.html` and its provenance record, which embed the
Plotly library itself.

`python tests/random_env.py SEED` installs randomly chosen releases
newer than the minimums into the current environment; CI uses it,
under a randomly chosen newer Python, for the second run of each
toolchain.

Run the scenarios with the toolchain of the machine. The harness names
the toolchain's compiler through `CXX` and `--cxx`, and every program
is built through `pipeline_native.py` into the per-user cache folder:

| Toolchain | Compiler it uses |
| --- | --- |
| `linux-gcc` | `g++` |
| `linux-clang` | `clang++` |
| `macos-clang` | `clang++` |
| `windows-mingw` | MinGW-w64 `g++` |
| `windows-msvc` | `cl`; run it from a Developer PowerShell. |

An executable that a stage leaves in a work folder counts as an
artifact, so a build that lands in the code folder shows up as a
difference. The variance stage's own copy of its program is the
exception: its provenance names that copy.

```text
python tests/golden.py --toolchain linux-gcc             # compare
python tests/golden.py --toolchain linux-gcc --compat    # other versions
python tests/golden.py --toolchain linux-gcc --record    # record anew
python tests/golden.py --toolchain linux-gcc --only pipeline22 tree3
python tests/golden.py --toolchain linux-gcc --skip-slow # no timing set
```

Scenarios run in `.golden-work/`; stage output goes to
`.golden-work/_logs/`.

| Scenario | What it covers |
| --- | --- |
| `pipeline22` | The whole README procedure on the 22-sample set: Pearson distance, cannot-links between the cohorts, then variances, cluster export, Tjala and signature export. |
| `euclidean22`, `spearman22`, `manhattan22` | Part 1 with the other metrics, and with must-link and no-link folders. |
| `genes22` | Part 1 and the variance stage with a gene list (`data/genes22.txt`). |
| `tree1` … `tree5` | The hierarchy builder, dendrogram and cluster export on random trees of 2, 3, 64, 300 and 1,181 samples whose paths contain commas, quotes and non-ASCII. |
| `timing300` | The whole procedure on 300 samples × 5,006 genes with Tjala's default settings. |

The 22-sample set has three genes only in each cohort, one gene that is
zero everywhere, and two identical samples. `data/make_cohorts.py` and
`data/make_trees.py` generate every dataset from fixed seeds.

## Regression and unit tests

```text
python -m pytest -p no:cacheprovider tests/regression tests/unit
```

`tests/regression/` holds one or more tests per plan task; they build
the 22-sample workflow with the machine's toolchain, or the one that
`TAVNIT_TOOLCHAIN` names. `tests/unit/` tests every shared format
(each writer against its reader), holds the C++ programs' integer and
decimal grammar to pipeline_io's, compares the Python history writer
with `reconstruct_hierarchy` byte for byte, and checks that no stage
script parses a pipeline format outside `pipeline_io.py` and
`pipeline_formats.py` (`test_core_boundary.py`, Phase 2's exit gate).

## Baselines

`python tests/golden.py --toolchain linux-gcc --measure FILE` records the
lines per source file, and per stage the wall time, the peak memory and
how often each file is hashed (task 0.5). The files in
`tests/baselines/` were measured in the cloud container used for the
refactoring; only their proportions carry over to other machines.

## Style report

`python tests/check_style.py` (with `requirements-dev.txt` installed)
reports ruff, pydoclint, vermin and clang-format findings. It reports
only, until the plan makes the checks binding.
