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
  found in Phase 0 is task 1.13 in Phase 1.
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

## Phase 2 working notes

- Order: 2.7, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.9, 2.10, 2.8; one commit
  per task, golden check (`--skip-slow` while iterating, full before
  pushing) and regression tests before each commit.
- `pipeline_io.py` imports only the standard library: constants and
  artifact owners, UTF-8 and CSV transport, machine grammar, the hash
  memo, today's provenance code and one atomic writer. Since task 2.4
  every script imports from the core modules directly; `tjala.py`
  still imports `validate_variance_output` from `compute_variances.py`
  until task 4.2 removes it. The console prompt moved into
  `pipeline_io.py` in 2.4, so that no script depends on
  `compute_variances.py` for it; task 3.1 moves prompting into
  `pipeline_cli.py`.
- Task 2.4 routes the five format writers through pipeline_formats
  (the lineages, matrix and identifier lists of the variance
  preparation, Tjala's report, the cannot-cluster list and the
  per-sample CSVs) and deletes the duplicate readers (the lineage and
  matrix readers of Tjala and of the signature export, the export's
  report parser, Tjala's list reader and clustering's snapshot
  reader), the extra node-count checks of Tjala and the signature
  export, the pass-through wrappers, `_decode_tree_states`, the
  linkage checks that the validated topology already guarantees
  (SciPy's `is_valid_linkage` stays) and `_parse_hierarchy_id`. Tjala,
  the signature export and the variance state now write through
  pipeline_io's atomic writer, which also removes the signature
  export's private temporary-file code that task 5.4 names; their
  outputs get permissions from the umask, like every other output.
  Readers now apply the producer's rules: matrix values outside int32,
  lineages out of node order and malformed reports are rejected.
- `pipeline_formats.py` may import NumPy and pandas: topology, tree
  states (history), snapshots, lineages, matrix, signature report,
  cannot-cluster list, per-sample CSVs, and the Python history writer
  from the review's prototype (`histbench/np_reconstruct.py`).
- `pipeline_native.py` (task 2.3): the compiler comes from `--cxx`,
  then `CXX`, then discovery: on Windows `cl` in an active Developer
  prompt, then the newest Visual Studio with the C++ tools through
  vswhere (with the environment its `vcvarsall.bat` sets up), then
  everywhere `g++` and `clang++` on PATH. The plan's risk register
  lists `CXX` after vswhere; it stays ahead of discovery because the
  scripts keep their configuration mechanisms, and the harness's
  MinGW runs rely on it. Builds go to
  `<user cache>/tavnit/build/<program>-<key>`, keyed on the source and
  the local headers it includes, the compiler's path and version
  (with `INCLUDE` for MSVC) and the flags; the compiler reads copies
  of exactly the hashed files, and an entry whose program has changed
  is rebuilt. GNU-style builds use `-std=c++20 -O2 -ffp-contract=off`,
  MSVC `/nologo /std:c++20 /O2 /EHsc /utf-8`. Compiler and program
  output reach the console live. `reconstruct_hierarchy` is built and
  run with `python pipeline_native.py run reconstruct_hierarchy
  clustering_results` until task 4.1 removes it. compute_variances
  keeps publishing a copy of its program into the results folder,
  because the variance result's provenance and the state JSON name
  it; task 4.2 deletes both. The golden harness now counts any other
  executable in a work folder as an artifact.
- Format inventory for 2.2 and 2.4 (where each reader and writer lives
  today; 2.2 moves or builds the canonical one in pipeline_formats,
  2.4 switches the callers and deletes the rest):
  - topology: `read_tree_topology` (moved); wrappers
    `build_dendrogram.load_tree_topology`,
    `samples_by_cluster.load_tree_topology`,
    `clustering._load_tree_topology`.
  - history: `read_tree_states`, `validate_tree_states`,
    `load_tree_states` (moved); `samples_by_cluster.load_tree_states`
    and `validate_history_against_topology`,
    `build_dendrogram._decode_tree_states`,
    `compute_variances._parse_hierarchy_id`; writer
    `render_tree_states` (new, matches reconstruct_hierarchy on every
    golden case).
  - snapshots: `read_snapshots` (new); `clustering._read_iteration_assignments`
    and `_validate_clustering_outputs`.
  - lineages: writer in `compute_variances._write_staged_inputs`
    ("sample count id..." in root-to-leaf order); readers
    `samples_by_signature.load_sample_lineages`, inline in
    `tjala.load_algorithm_data`.
  - matrix: writer `_write_staged_inputs` (int32, space-separated, one
    row per gene); readers `samples_by_signature.load_expression_values`
    (streams, keeps selected rows), inline in `tjala.load_algorithm_data`.
  - signature report: writer `tjala._format_*` and `_write_text_atomically`;
    parser `samples_by_signature.parse_antecedent`,
    `_parse_cluster_section`, `parse_signature_block`, `load_signatures`;
    canonical `render_signature_report` and `read_signature_report`
    (new; every golden report reads back and re-renders byte for
    byte).
  - cannot-cluster list: writer `samples_by_cluster._write_leaf_cluster_file`;
    reader inline in `tjala.load_algorithm_data`; canonical
    `render_cluster_list` and `read_cluster_list` (new, with the
    hand-written forms task 1.10 accepts).
  - per-sample CSV: writer inside `clustering.write_sample_csvs`; reader
    `read_sample_expression` (moved); canonical writer
    `render_sample_expression` (new, reproduces every golden CSV).
- The report parser accepts exactly what the writer produces: the
  signatures with their stop reason, conditions, both cluster sections
  and the optional scaling section, then the run summary, whose
  signature count must agree. Lines are compared without stripping,
  thresholds are canonical integers within int32 (they are observed
  matrix values), percentages have one decimal and lie in 0.0..100.0,
  and scaled thresholds are written as Python writes floats. The
  writer parses its own text back before returning it. So the report
  meets task 2.10's rules from the start; they take effect when 2.4
  moves `samples_by_signature.py` onto this parser.
- Task 2.9: Tjala and samples_by_signature.py derive each sample's
  lineage from `total_data.csv` and `tree_splits.txt` (through
  `load_tree_states`) and require `sample_lineages.txt` to match it,
  which tests the parent-child links and the sample-leaf binding. Both
  files sit beside the prepared files in the results folder and are
  already inputs of the prepare stage, so the manifests do not change.
  The history reader checks that rows are sorted by their node history.
- Part 2 runs from copies in `clustering_results`, so the README's copy
  list and the golden harness's `PACKAGE` and `PART2` lists gain the
  core modules.

## Findings

- Phase 2, task 2.5: from task 2.1 until this task, the variance
  stage's three manifests named pipeline_io.py as their producer, as
  finding A8 predicted, because the moved manifest writer defaulted to
  its own module. The producer is now a required argument.

- Phase 0: on the 300-sample timing set under Python 3.10, Tjala stops
  with "Normalized pheromones do not sum to 1.0: 0.9999999999986108"
  once 839,930 terms are active. Under Python 3.14 the same check passes
  and Tjala finishes with two signatures. The cause is Python's built-in
  `sum()`, which compensates rounding only from Python 3.12 on, as the
  register's C12 says; the NumPy version plays no part. The golden
  files record the failure as today's behavior; task 1.3 replaces the
  check and re-records the timing set.
- Phase 0: every other scenario gives byte-identical results with GCC
  and Clang under both Python 3.10 and 3.14.
- Phase 0, versions: the oldest releases that work on Python 3.10 are
  NumPy 1.21.2 and SciPy 1.7.2 (the first with wheels for 3.10), pandas
  1.5.0 (`to_csv(lineterminator=)`) and Plotly 4.0.0
  (`plotly.graph_objects`). With them, every quick scenario passes and
  every artifact matches the results of the newest releases, except
  `dendrogram.html`, which embeds Plotly's own library. The newest
  releases (Python 3.14, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, Plotly
  7.1.0) also pass. A headless browser loads the dendrogram built with
  Plotly 4.0.0 without errors.
- Phase 0: on macOS, Apple Clang 17 rejects floating-point
  `std::from_chars` ("call to deleted function"), so today's
  `run_clustering.cpp` (line 290) and `reconstruct_hierarchy.cpp` (line
  287) do not compile and the pipeline cannot run there. The macOS
  golden files record those build failures as today's behavior; task
  1.13 fixes them.
- Phase 0: Windows (MSVC and MinGW-w64 GCC 16) runs every scenario.
  Its results differ from Linux only in files that hold sample paths
  and in the provenance records that hash them; the tree, iteration,
  matrix, variance, Tjala and dendrogram outputs are byte-identical
  across Linux and Windows. MSVC and MinGW differ only in
  `compute_variances_state.json`, which names the compiler family.
- Phase 1, task 1.2: the Python exports (cluster files and Tjala's
  report) now skip an identical result. The C++ engines and the
  variance preparation still rewrite their outputs; Phase 5's commit
  (task 5.3) applies the same rule to every stage.
- Phase 1, task 1.13: once the C++ programs built on macOS, Apple
  Clang's default fused multiply-adds on arm64 changed the last digits
  of `term_variances.txt`; GNU-style builds now pass
  `-ffp-contract=off`. Since then macOS (arm64), Windows and Linux
  (x86-64) give byte-identical results in every scenario except where
  sample paths appear.
- Phase 1: provenance records name absolute paths, so a results folder
  that is copied or moved loses the links between its outputs and their
  parents; invalidation then misses them. Phase 5's relative paths and
  `relocate` command address this.
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
| 0.1 | Tag the current code as the baseline. State in requirements.txt the minimal version of each library... | Done |
| 0.2 | Golden datasets in tests/data/: the 22-sample synthetic set, five random trees whose paths contain commas,... | Done |
| 0.3 | Golden outputs: the SHA-256 of every artifact the current code writes, per toolchain (MSVC, MinGW GCC,... | Done |
| 0.4 | tests/golden.py drives every script non-interactively, feeding its answers on stdin, and compares the hashes. | Done |
| 0.5 | Baselines: lines per file, wall time and peak memory per stage, and hashes per file per run, using the... | Done |
| 0.6 | Configure ruff, pydoclint, vermin and clang-format 19.1.7 in report-only mode, pinned in requirements-dev.txt. | Done |

### Phase 1: Stop the harm

| Task | Summary | Status |
| --- | --- | --- |
| 1.1 | Remove the deletion that clustering.py runs before its first prompt (lines 1036–1040). Results are... | Done |
| 1.2 | Compare hashes before replacing an output. An identical result is not rewritten and invalidates nothing... | Done |
| 1.3 | Check the pheromone sum with math.fsum against a tolerance that grows with the number of active terms (n·ε). | Done |
| 1.4 | Remove the symlink refusal from cleanup. | Done |
| 1.5 | Add a per-process hash memo keyed on path, size, modification time and file ID. | Done |
| 1.6 | Correct the stage name "terms" to "signatures" in compute_variances.cpp. | Done |
| 1.8 | Warn with the number of zero-filled genes per cohort, and name the zero-variance samples under Pearson or... | Done |
| 1.9 | Strip only the .version suffix from gene IDs, so _PAR_Y IDs stay distinct from their chrX twins. | Done |
| 1.10 | Cannot-cluster list: accept 017 and +17. With a manifest, check that it was exported from the current... | Done |
| 1.11 | Read JSON input with type inference off, so identifiers such as "001" stay text. | Done |
| 1.12 | Reject a duplicate sample header in any input format before pandas renames it, naming the file and the header. | Done |
| 1.13 | Build on macOS: replace floating-point std::from_chars, which Apple Clang's library rejects, so that... | Done |

### Phase 2: Shared core

| Task | Summary | Status |
| --- | --- | --- |
| 2.1 | Create pipeline_io.py from Block 0 of compute_variances.py: constants and artifact owners, UTF-8 and CSV... | Done |
| 2.2 | Create pipeline_formats.py: topology, snapshots, history reader and validator, lineages, the matrix... | Done |
| 2.3 | Create pipeline_native.py from the MSVC-capable helper in compute_variances.py, adding vswhere, MinGW and... | Done |
| 2.4 | Switch every script to the core and delete what it replaces: five writers, four duplicate readers, three... | Done |
| 2.5 | Make producer a required argument of the manifest writer. | Done |
| 2.6 | Replace repeated literals with the constants: clustering_results, the parsed-data suffixes, .provenance,... | Done |
| 2.7 | Delete the 10 unused imports and the if True: blocks. | Done |
| 2.8 | Unit tests for every shared format and grammar: report round trip, Python–C++ grammar agreement, and the... | Not started |
| 2.9 | Strengthen the format checks: each lineage must follow real parent–child links in the topology and end at... | Done |
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
