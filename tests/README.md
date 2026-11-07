# Tests

## Golden tests

`tests/golden.py` runs every pipeline script on the golden datasets and
compares the SHA-256 of every file the scripts write with the hashes
recorded in `tests/golden/<toolchain>-py<major><minor>.json`. Any change
that is not backed by a decision record must leave them identical.

Set up an environment with the pinned packages first:

```text
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt        # POSIX
.venv\Scripts\python -m pip install -r requirements.txt    # Windows
```

Then run the scenarios with the toolchain of the machine:

| Toolchain | Compilers it uses |
| --- | --- |
| `linux-gcc` | `g++` |
| `linux-clang` | `clang++` |
| `macos-clang` | `clang++` |
| `windows-mingw` | MinGW-w64 `g++` |
| `windows-msvc` | `cl` for the hierarchy and variance programs, and MinGW-w64 `g++` for `run_clustering`, because today's `clustering.py` drives GNU-style compilers only. Run it from a Developer PowerShell with MinGW-w64 `g++` on `PATH`. |

```text
python tests/golden.py --toolchain windows-msvc          # compare
python tests/golden.py --toolchain windows-msvc --record # record anew
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
