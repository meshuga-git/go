# Refactoring progress

The plan in `docs/TAVNIT_Architecture_and_Refactoring_Plan.docx` is
authoritative. This file records where the work stands, what was
agreed in conversation that the plan does not spell out, and what the
work has found. Each phase is one branch, `phase-<n>`, and one pull
request into `main`; each task is one commit.

## Baseline

- Commit `4175c43` on `main` is the unmodified package (task 0.1). The
  tag `baseline` exists locally but could not be pushed from the
  working session; create it with
  `git tag -a baseline 4175c43 -m "Baseline" && git push origin baseline`.
- `.gitattributes` keeps every text file LF on every platform, so
  sources hash the same everywhere.

## Agreed in conversation

- Versions: the package runs with any Python, library and compiler
  version that supports its functionality; the versions installed on
  any one machine, and its system language, play no part.
  `requirements.txt` states the minimal versions. CI tests the minimal
  versions, or the closest newer ones available, and randomly chosen
  newer versions. The plan's constraints and task 0.1 now say so.
- Every defect is fixed as soon as possible. The macOS build failure
  found in Phase 0 is task 1.12 in Phase 1.
- Python 3.10 is a ceiling: the code uses nothing newer. Each module
  docstring states the minimum version vermin reports for it, raised to
  what the pinned packages need where that is higher. The plan's
  constraint now says so.
- The five smaller issues found in review are not separate tasks; the
  plan removes the code each lives in:
  - `--genes` ignored by the build and run stages: task 4.2 deletes
    those stages.
  - The relative gene-list path in parent manifests: Phase 5 replaces
    the manifests.
  - compute_variances exiting with 2 on argument errors while
    documenting 0 and 1: Phase 3 unifies exit codes.
  - A failed temporary-file cleanup replacing the original error: the
    private writers give way to the one atomic writer (2.1, 4.2, 5.4).
  - The dendrogram printing "saved" before its provenance is written:
    Phase 5's commit.
  Two checks keep them from coming back: a failed cleanup never
  replaces the original error, and a stage reports success only after
  its commit.

## Findings

- Phase 0: on the 300-sample timing set under Python 3.10 (NumPy
  2.2.6), Tjala stops with "Normalized pheromones do not sum to 1.0:
  0.9999999999986108" once 839,930 terms are active. Under Python 3.14
  (NumPy 2.5.4) the same check passes and Tjala finishes with two
  signatures: whether a run survives depends on how the NumPy version
  rounds the sum. The golden files record both outcomes as today's
  behavior; task 1.3 replaces the exact check, and that commit
  re-records the timing set.
- Phase 0: every other scenario gives byte-identical results with GCC
  and Clang under both Python 3.10 and 3.14.
- Phase 0: on macOS, Apple Clang 17 rejects floating-point
  `std::from_chars` ("call to deleted function"), so today's
  `run_clustering.cpp` (line 290) and `reconstruct_hierarchy.cpp` (line
  287) do not compile and the pipeline cannot run there. The macOS
  golden files record those build failures as today's behavior; task
  1.12 fixes them.
- Phase 0: Windows (MSVC and MinGW-w64 GCC 16) runs every scenario.
  Its results differ from Linux only in files that hold sample paths
  and in the provenance records that hash them; the tree, iteration,
  matrix, variance, Tjala and dendrogram outputs are byte-identical
  across Linux and Windows. MSVC and MinGW differ only in
  `compute_variances_state.json`, which names the compiler family.
- Phase 0 baseline (`tests/baselines/linux-gcc-py310.json`): the
  shared SHA-256 helper runs 1,024 times in the 22-sample variance
  stage and 10,754 times on the timing set, up to 28 times for one
  file; Phase 5's gate allows 3 per file per run. On the timing set
  the variance stage takes 109 s and Tjala 151 s with a 1.2 GiB peak.
- Phase 0: `build_dendrogram.py` picks colors at random unless given
  `--seed`; the golden harness passes `--seed 7`.
- Phase 0: today's code reports 21 ruff findings, 68 pydoclint
  findings and 1,421 clang-format findings against `.clang-format`;
  vermin puts every module at Python 3.9 or 3.10.

## Tasks

### Phase 0: Safety net

| Task | Summary | Status |
| --- | --- | --- |
| 0.1 | Tag the current code as the baseline. Pin the package versions in use in requirements.txt, and record the... | Done |
| 0.2 | Golden datasets in tests/data/: the 22-sample synthetic set, five random trees whose paths contain commas,... | Done |
| 0.3 | Golden outputs: the SHA-256 of every artifact the current code writes, per toolchain (MSVC, MinGW GCC,... | Done |
| 0.4 | tests/golden.py drives every script non-interactively, feeding its answers on stdin, and compares the hashes. | Done |
| 0.5 | Baselines: lines per file, wall time and peak memory per stage, and hashes per file per run, using the... | Done |
| 0.6 | Configure ruff, pydoclint, vermin and clang-format 19.1.7 in report-only mode, pinned in requirements-dev.txt. | Done |

### Phase 1: Stop the harm

| Task | Summary | Status |
| --- | --- | --- |
| 1.1 | Remove the deletion that clustering.py runs before its first prompt (lines 1036–1040). Results are... | Not started |
| 1.2 | Compare hashes before replacing an output. An identical result is not rewritten and invalidates nothing... | Not started |
| 1.3 | Check the pheromone sum with math.fsum against a tolerance that grows with the number of active terms (n·ε). | Not started |
| 1.4 | Remove the symlink refusal from cleanup. | Not started |
| 1.5 | Add a per-process hash memo keyed on path, size, modification time and file ID. | Not started |
| 1.6 | Correct the stage name "terms" to "signatures" in compute_variances.cpp. | Not started |
| 1.8 | Warn with the number of zero-filled genes per cohort, and name the zero-variance samples under Pearson or... | Not started |
| 1.9 | Strip only the .version suffix from gene IDs, so _PAR_Y IDs stay distinct from their chrX twins. | Not started |
| 1.10 | Cannot-cluster list: accept 017 and +17. With a manifest, check that it was exported from the current... | Not started |
| 1.11 | Read JSON input with type inference off, so identifiers such as "001" stay text. | Not started |
| 1.12 | Build on macOS: replace floating-point std::from_chars, which Apple Clang's library rejects, so that... | Not started |
| 1.12 | Reject a duplicate sample header in any input format before pandas renames it, naming the file and the header. | Not started |

### Phase 2: Shared core

| Task | Summary | Status |
| --- | --- | --- |
| 2.1 | Create pipeline_io.py from Block 0 of compute_variances.py: constants and artifact owners, UTF-8 and CSV... | Not started |
| 2.2 | Create pipeline_formats.py: topology, snapshots, history reader and validator, lineages, the matrix... | Not started |
| 2.3 | Create pipeline_native.py from the MSVC-capable helper in compute_variances.py, adding vswhere, MinGW and... | Not started |
| 2.4 | Switch every script to the core and delete what it replaces: five writers, four duplicate readers, three... | Not started |
| 2.5 | Make producer a required argument of the manifest writer. | Not started |
| 2.6 | Replace repeated literals with the constants: clustering_results, the parsed-data suffixes, .provenance,... | Not started |
| 2.7 | Delete the 10 unused imports and the if True: blocks. | Not started |
| 2.8 | Unit tests for every shared format and grammar: report round trip, Python–C++ grammar agreement, and the... | Not started |
| 2.9 | Strengthen the format checks: each lineage must follow real parent–child links in the topology and end at... | Not started |
| 2.10 | Make writer and readers agree: each format splits on the exact ASCII separator it names and rejects other... | Not started |

### Phase 3: Command-line layer

| Task | Summary | Status |
| --- | --- | --- |
| 3.1 | Create pipeline_cli.py: one prompt helper that reproduces today's prompts, defaults and sentinels, plus... | Not started |
| 3.2 | Switch the four prompting scripts to the shared helper, with no change to wording, defaults or sentinels.... | Not started |
| 3.4 | Add the error classes and run_main() to every script: one tagged line for expected errors, a traceback... | Not started |
| 3.5 | Console API: prompts, progress, warnings and errors on stderr with the [stage] prefix, README labels and... | Not started |
| 3.6 | One exception type per condition (for example, unknown term IDs), and integrity failures raise... | Not started |
| 3.7 | Transcript tests for every script, checked against today's prompts: defaults, sentinels, invalid answers,... | Not started |

### Phase 4: One native program

| Task | Summary | Status |
| --- | --- | --- |
| 4.1 | The clustering stage builds total_data.csv with the Python history writer, writing rows as it goes, and... | Not started |
| 4.2 | Apply DR1 and drop term_variances.txt: Tjala derives A from its own term variances. Delete... | Not started |
| 4.3 | Make run_clustering pure compute while keeping its two arguments (output prefix and distance metric), its... | Not started |
| 4.4 | Pass 1 reads gene names only; values are parsed once, with from_chars. | Not started |
| 4.5 | Error messages name the sample file, line and column; distance errors name both samples. | Not started |
| 4.6 | One line reader and one integer parser, std::format throughout. Delete the if (true) blocks, and move the... | Not started |
| 4.7 | Set .clang-format to the code's style (T&amp; x, short ifs allowed) and format the file. One Doxygen... | Not started |
| 4.8 | Compile warning-free with MSVC /W4 and with GCC and Clang -Wall -Wextra -Wpedantic. | Not started |
| 4.9 | Test the Pearson and Spearman distances on extreme-scale values. If a finite but wrong distance appears,... | Not started |

### Phase 5: Provenance v2 and workspace

| Task | Summary | Status |
| --- | --- | --- |
| 5.1 | Manifest v2 in pipeline_io: the JSON schema, the naming rule, the artifact-owner table and relative paths.... | Not started |
| 5.2 | Every stage verifies its direct inputs: each hash against the owning manifest, then the format's content... | Not started |
| 5.3 | commit_outputs() runs the commit protocol from Interfaces in every stage: the folder locks, outputs... | Not started |
| 5.4 | Delete invalidate_dependents, every start-of-run cleanup and the signature writer's private temporary-file... | Not started |
| 5.5 | Scripts are no longer copied. Part 2 stages, Tjala included, work in the current folder and run from... | Not started |
| 5.6 | Optional inputs are always recorded with their hash, and verified against their manifest when they have one. | Not started |
| 5.7 | Every text output goes through the one writer with \n line endings. | Not started |
| 5.8 | Upgrade command: fully verify a version 1 results folder, then write equivalent version 2 manifests marked... | Not started |
| 5.9 | relocate command for moves of separate parts or across drives: verify each file at its new location by... | Not started |
| 5.10 | A result counts as current only if its inputs, parameters, producing code (stage script, core modules and... | Not started |

### Phase 6: Stage internals

| Task | Summary | Status |
| --- | --- | --- |
| 6.1 | Store Tjala's term table as NumPy columns with an active mask, keep pheromones and heuristics as arrays,... | Not started |
| 6.2 | Use dataclasses for records. Drop the TypedDict layers, the presence checks after prepare_algorithm_state,... | Not started |
| 6.3 | Remove re-checks of data just built (compose_binary_vectors, the scoring loop's re-sort and range checks),... | Not started |
| 6.4 | Keep the sparse membership matrix only and delete the dense fallback, which makes SciPy required for Tjala... | Not started |
| 6.5 | The pheromone-sum check stays an explicit check with the n·ε tolerance and raises InternalError when it fails. | Not started |
| 6.6 | Delete Tjala's inspection options and the console options that gate them, about 210 lines, including the... | Not started |
| 6.7 | Choose a term in two stages with one random draw per step. Once per iteration, sum each gene's term... | Not started |
| 6.8 | Dendrogram: colour leaves from the ingestion records only (decision 13), and hold the JavaScript as one... | Not started |
| 6.9 | Define and test Tjala's empty outcomes: no candidate terms, a report with no signatures, max_terms above... | Not started |

### Phase 7: Runner and environment

| Task | Summary | Status |
| --- | --- | --- |
| 7.1 | Bootstrap: check for Python 3.10 or later, offer to create .venv and install the pinned requirements... | Not started |
| 7.2 | Phases as menu entries and as subcommands: cluster (ingest, clustering, dendrogram), explore, signatures... | Not started |
| 7.3 | Run a phase's stages in order in the runner's console, with Part 2 stages inside the results folder. Each... | Not started |
| 7.4 | status shows each stage as current, stale, missing or interrupted, by the rule in task 5.10, and asks... | Not started |
| 7.5 | environment reports Python and package versions, the compilers found (MSVC through vswhere) and the build... | Not started |
| 7.6 | Each phase writes a log to runs/&lt;UTC time&gt;.json: stages, exit codes, durations. | Not started |
| 7.7 | Platform details: sys.executable with argument lists and no shell; children share the console; on Ctrl-C... | Not started |
| 7.8 | selftest runs the golden datasets end to end, plus the style checks. | Not started |

### Phase 8: Consistency sweep

| Task | Summary | Status |
| --- | --- | --- |
| 8.1 | Run ruff format with double quotes. Ordinary wrapping replaces the 33 doubly parenthesised strings, and... | Not started |
| 8.2 | Names follow the README glossary: its count names, threshold, operator symbols, lineage and snapshot.... | Not started |
| 8.3 | Typing: collections.abc, NDArray[...], dataclasses, pathlib and Path parameters throughout. | Not started |
| 8.4 | Docstrings: fix the 17 repeated summaries, 10 non-imperative summaries, 6 missing Raises sections, the... | Not started |
| 8.5 | Module docstrings follow the template below. Every runnable script gets a shebang; core modules get none. | Not started |
| 8.6 | Python comments: remove history narration and stale names. | Not started |
| 8.7 | Switch every style check from report to enforce in selftest. | Not started |

### Phase 9: Documentation

| Task | Summary | Status |
| --- | --- | --- |
| 9.1 | Rewrite the README: setup through the runner, running by phase or by script, every prompt and argument... | Not started |
| 9.2 | Write docs/FORMATS.md: each artifact's grammar, producer, consumers and manifest. It records that sample... | Not started |
| 9.3 | CHARD and Tjala: move the input/output contracts to FORMATS.md, and keep implementation choices and... | Not started |
| 9.4 | Fix Tjala3f: drop the operator-code list along with term_variances.txt, and renumber the second remark... | Not started |
| 9.5 | Add a CHANGELOG listing every behavioural change, renamed file and new default. Each removed behaviour... | Not started |
| 9.6 | Add a glossary (cluster, node, split, snapshot, lineage, term, signature, cannot-cluster) that code and... | Not started |
