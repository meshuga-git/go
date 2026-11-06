# TAVNIT package
## Constrained Divisive Hierarchical Clustering and Signature Extraction

This package performs hierarchical clustering of samples using transcriptomic (or other numerical) data and then extracts signatures of the resulting clusters.
Clustering can be performed on the complete dataset, while cluster signatures can be extracted using either the complete data or a specified subset of genes. For example, cancer and noncancer samples can be supplied as cannot-link groups and clustered using transcriptomic data for all genes. Subsequently, cancer signatures can be extracted using only genes encoding the surfaceome to identify potential targets for CAR-T cells.
The package implements a constrained divisive clustering algorithm and the Tjala algorithm (an Ant Colony Optimization algorithm). It is written in C++ and Python and is operated via a Command Line Interface.

An input dataset is accessible at <https://data.mendeley.com/datasets/rdhxwbhmvn/1>.

## Supported input formats

| Type | Extensions |
| --- | --- |
| Delimited text | `.tsv`, `.csv`, `.dat`, `.counts` |
| Spreadsheets | `.xlsx`, `.xls` |
| Data exchange | `.json` |
| Statistical formats | `.dta`, `.sav`, `.por` |
| Data storage formats | `.h5`, `.parquet`, `.orc` |

For delimited-text and spreadsheet inputs, the first row must contain column headers (sample names). In every input table, the first column contains gene identifiers and each remaining column contains expression values for one sample. Version suffixes in gene identifiers (everything from the first `.`, as in Ensembl IDs) are removed.

## Requirements

- A C++20-compliant compiler. `clustering.py` invokes the compiler specified by `CXX`, or `g++` by default, using GNU-style command-line options.
- Python 3.10 or higher with `pandas` and `numpy`. Dendrogram generation additionally requires `scipy` and `plotly`.
- Depending on the input formats used, pandas may require optional reader libraries such as `openpyxl` (`.xlsx`), `xlrd` (`.xls`), `pyreadstat` (`.sav`/`.por`), `tables` (`.h5`), and `pyarrow` or `fastparquet` (`.parquet`), and `pyarrow` (`.orc`).

The pipeline is intended to work on Windows and POSIX systems (e.g., Linux and macOS).

## Getting started

1. Place the following files in one directory:

   ```text
   README.md
   clustering.py
   compute_variances.cpp
   pipeline_files.hpp
   compute_variances.py
   samples_by_cluster.py
   reconstruct_hierarchy.cpp
   run_clustering.cpp
   samples_by_signature.py
   tjala.py
   build_dendrogram.py
   genes.txt            (optional)
   ```

2. Organize the input data into folders. Each supplied folder may contain data files directly or in one level of immediate subdirectories. Up to four folders can be supplied:
   - two **cannot-link** folders: samples from the two folders are treated as cannot-link groups during constrained splitting;
   - one **must-link** folder: samples in this folder are moved together when the must-link group is first separated from other samples, although later divisive splits can separate them as the algorithm builds the complete binary tree;
   - one **no-link** folder.

   Each group is optional and can be skipped by entering `None` when prompted, but at least one input folder must be supplied. The cannot-link group, when used, requires exactly two folder paths.

## Part 1: Clustering

1. Navigate to the folder containing the scripts listed above and run:

   ```text
   python clustering.py
   ```

   The script first prompts you to select a distance metric:

   | Number | Metric |
   | --- | --- |
   | 1 | Euclidean distance |
   | 2 | Pearson distance (`1 - correlation`) |
   | 3 | Spearman distance (`1 - rank correlation`) |
   | 4 | Manhattan distance |

> NB: Among standard metrics, the Pearson metric is likely the most suitable for analyzing transcriptomic big data.

   The script then requests the cannot-link, must-link, and no-link input folders and parses the input data. It creates:
   - for each supplied input folder, a sibling folder named `<folder>_parsed_data` containing one two-column CSV file (`gene`, `expression`) per sample;
   - for each supplied input folder, a file `<folder>_parsed_files.txt` listing the generated sample CSV files;
   - `cannot_links.txt`, `must_links.txt`, and `no_links.txt`, which record the selected input folders;
   - `clustering_results` folder

   `clustering.py` automatically compiles `run_clustering.cpp` and runs the resulting executable. The clustering engine computes all pairwise distances and repeatedly splits one active cluster into two until every sample is a terminal leaf.

   It writes the following files to `clustering_results`:
   - `filenames.txt`: the parsed sample files in downstream order (cannot-link folder 1, cannot-link folder 2, must-link folder, then no-link folder);
   - `iter1.txt`, `iter2.txt`, and so on through `iter(N-1).txt`: after each split, the immutable active tree-node ID assigned to every sample, in `filenames.txt` order;
   - `tree_splits.txt`: the authoritative immutable tree topology. It begins with `root 0`; each subsequent record has the form `split <iteration> <parent> <retained_child> <extracted_child>`. Split `k` retires its parent and creates fresh child IDs `2k-1` and `2k`.

2. Reconstruct the cluster hierarchy:
2a. Compile `reconstruct_hierarchy.cpp` in C++20 mode, for example:

   ```text
   g++ -std=c++20 -O2 reconstruct_hierarchy.cpp -o reconstruct_hierarchy
   ```

   On Windows, use the corresponding `.exe` output name. The variance
   wrapper also compiles its native source in C++20 mode. 
2b. Reconstruct the hierarchy:
   ```text
   ./reconstruct_hierarchy clustering_results        # POSIX
   reconstruct_hierarchy.exe clustering_results      # Windows
   ```

   The program reads `filenames.txt`, `tree_splits.txt`, and every `iter*.txt` file and creates `clustering_results/total_data.csv`.

   For `N` samples, `total_data.csv` contains `N` rows and `N + 1` columns. The first column is the sample path. The remaining `N` columns contain the immutable hierarchy states: explicit root state 0, followed by the state after each of the `N - 1` splits. Samples are sorted lexicographically by their complete immutable-node history so that each cluster occupies consecutive rows. Within each state column, repeated node IDs in adjacent rows are run-length-compressed as blank fields; downstream scripts forward-fill these blanks when decoding the hierarchy.

3. (Optional) Visualize the clustering results as an interactive dendrogram:

   ```text
   python build_dendrogram.py
   ```

   By default, the script reads `clustering_results/total_data.csv`, uses `tree_splits.txt` beside that file, reads `cannot_links.txt`, `must_links.txt`, and `no_links.txt` for source-folder colors, and writes `dendrogram.html` in the current directory. Branches are colored by configured input folder when a folder can be matched. The vertical geometry represents split/merge order, not a measured distance.

## Part 2: Signature extraction

1. Copy the following files into `clustering_results` and work from that folder:

   ```text
   compute_variances.py
   compute_variances.cpp
   pipeline_files.hpp
   samples_by_cluster.py
   tjala.py
   samples_by_signature.py
   genes.txt            (optional)
   ```

2. Compute the variance inputs used for signature generation, optionally restricting the analysis to a gene allowlist:

   ```text
   python compute_variances.py None         # all genes
   python compute_variances.py genes.txt    # a subset
   ```

   Running `python compute_variances.py` with no positional argument is also equivalent to using all genes. The script also supports restartable `prepare`, `build`, and `run` stages; running without a stage (or with `all`) executes all three.

   `genes.txt` must be in UTF-8 format with one gene identifier per line. Blank lines are ignored. Genes whose original expression is zero in every sample are dropped before integer conversion, and a supplied gene list is then used as an allowlist. If a gene is absent from an individual sample file, cross-sample alignment fills that absent gene with `0`; however, a missing, nonnumeric, or non-finite expression value in an actual source row is rejected rather than silently replaced. Expression values are truncated toward zero to signed 32-bit integers, so the input should be on a scale where integer values are meaningful (for example, counts).

   The full command prepares the inputs, compiles `compute_variances.cpp`, runs the native variance engine, and creates:
   - `matrix_genes.txt`: the retained genes, one per line;
   - `matrix_samples.txt`: the sample identifiers, one per line, in matrix-column order;
   - `sample_lineages.txt`: each sample identifier, its membership count, and its immutable hierarchy-node memberships;
   - `matrix_values.txt`: a space-delimited genes-by-samples integer expression matrix;
   - `term_variances.txt`: a space-delimited file with `gene threshold operator_code variance` on each line (`0` = equality, `1` = `<`, `2` = `>`);
   - `compute_variances_state.json`: hashes and metadata used to validate the prepared inputs, compiled executable, and variance output before restartable stages proceed;
   - the compiled `compute_variances` executable.

3. Generate a list of terminal descendant clusters and inspect the hierarchy below a selected immutable node:

   ```text
   python samples_by_cluster.py
   ```

   The script requests an immutable cluster node ID and, for a nonterminal node, a tree hierarchy level below that node. It writes:
   - `cluster_<n>.txt`: the terminal descendant leaf-node IDs of cluster `n`, separated by commas. This file can be passed to `tjala.py` as a cannot-cluster list;
   - `cluster_<n>_hier_<level>.txt`: the samples of cluster `n`, grouped by the tree frontier exactly `<level>` edges below the selected node (level 1 is its immediate children).

   Enter `Exit` (case-insensitive) at the cluster prompt to quit.

4. Generate signatures:

   ```text
   python tjala.py
   ```

   The script requests the following settings (press Enter to accept a default):

   | Prompt | Meaning | Default |
   | --- | --- | ---: |
   | Maximum number of terms per signature | Maximum number of gene-threshold conditions in one signature | 5 |
   | Minimum number of samples that must be covered by a signature | Minimum signature coverage | 10 |
   | Ant colony size (number of ants) | Number of ants generated per iteration | 50 |
   | Number of terms (discretization bins) per gene | Maximum number of representative expression-value bins used to generate candidate thresholds for each gene | 100 |
   | Maximum number of iterations per epoch | Iteration limit for one signature search | 100 |
   | Number of iterations yielding the same best signature for identifying signature convergence | Convergence window | 10 |
   | Maximum number of uncovered samples allowed before termination | The outer search stops successfully when the number of samples not covered by any finalized signature is at or below this value | 10 |

   The script then asks for the path to a **cannot-cluster file**; press Enter to skip this constraint. The file may contain immutable node IDs separated by commas, semicolons, or whitespace. `cluster_<n>.txt` from `samples_by_cluster.py` is directly usable.

   If cannot-clusters are supplied, the script asks for the **share of covered samples that may belong to cannot clusters**, between 0 and 1 (default `0.05`). This limit applies to the combined set of samples belonging to any configured cannot-cluster: a candidate is rejected when the fraction of its covered samples that belong to that combined set exceeds the threshold.

   The search is stochastic, so results may differ between runs.

   Results are written to `tjala_output.txt`. Tjala uses threshold terms of the form `gene < value` and `gene > value`. Each signature is a conjunction of one or more such terms. For each signature, the output reports:
   - whether the epoch ended by convergence or by reaching its iteration limit;
   - **Can clusters** and **Cannot clusters** with nonzero coverage, where each percentage is the fraction of that cluster's samples covered by the signature;
   - when cannot-clusters are configured, a threshold-scaling diagnostic for each term. For scale factors 10, 20, ..., 100, a `>` threshold is divided by the factor and a `<` threshold is multiplied by the factor; the reported percentage is the fraction of all samples belonging to the configured cannot-clusters that satisfy that scaled single term.

5. Obtain a list of samples that satisfy a selected signature:

   ```text
   python samples_by_signature.py
   ```

   Enter the signature number when prompted (or `Exit` to quit). The script re-evaluates the selected signature against `matrix_values.txt`, validates its reported immutable-cluster memberships against `sample_lineages.txt`, and writes `signature_<n>.txt` with the sample identifiers that satisfy all terms of the signature, in hierarchy-derived order.


## Revision: changes1

This revision implements the changes selected in `changes1.docx`. The
clustering and Tjala mathematical objectives, split rules, tie-breaking,
heuristic fallback, search defaults, and threshold conversion are unchanged.
No additional Python helper script or persistent directory hierarchy is
introduced. Shared Python I/O and topology utilities live in the existing
`compute_variances.py`; keep that file beside every dependent Python script.
The native files contain matching local I/O utilities and require C++20.
Their artifact filenames are defined once in `pipeline_files.hpp`; keep that
header beside every C++ source when compiling.

### Upgrade and regeneration

Replace the six Python files, three C++ files, shared C++ header, this README,
and the two algorithm documents as one matched revision. Recompile the native programs.
The delivered `build_dendrogram.py` is the canonical name of the uploaded
`build_dendrogram(1).py`.

Existing data without the new stage provenance are **not silently trusted**.
Rerun the documented workflow from the original source matrices, beginning
with `clustering.py`, then reconstruction, preparation, variance calculation,
and the required exports. Do not manufacture metadata for unknown results.
Existing `compute_variances_state.json` still controls restartable variance
stages; prepare/build/run must agree with both that state and the sidecars.

No supplied scientific input, intermediate file, or sample output is rewritten
by installing this revision. Retain original inputs and backups before
regenerating results. Run only one pipeline operation at a time in the same
working/result directories; concurrent writers are not supported.

### Artifact ownership, provenance, and invalidation

Stage-local commit manifests are generated **in the existing directories**:

| Output set | Commit manifest |
| --- | --- |
| Parsed sample CSVs in each `<folder>_parsed_data` | `parsed_data.provenance` |
| Constraint configuration or sample-file manifest | `<filename>.provenance` |
| `filenames.txt`, `tree_splits.txt`, and required `iterK.txt` | `clustering.provenance` |
| Compressed hierarchy history | `total_data.csv.provenance` |
| Four prepared signature-input files | `prepared.provenance` |
| Variance table | `term_variances.txt.provenance` |
| Dendrogram, Tjala report, or extraction output | `<filename>.provenance` |

A manifest begins with `TAVNIT-PROVENANCE-1`, followed by a stage and producer
fingerprint. `I`, `P`, and `O` records describe direct inputs, upstream commit
manifests, and local owned output basenames. Each record includes a SHA-256
hash. Paths are UTF-8 encoded and hex-escaped to preserve spaces and literal
characters. Optional `C` records map sample IDs to source-directory paths.
The producer fingerprint identifies the source/executable used; it does not
require that the currently installed producer still has the same bytes.

Consumers verify dependency and output hashes before reading generated data,
then validate their semantic contracts. A successful exit from a native
process alone is insufficient. Missing or mismatched metadata requires rerun
of the responsible producer. These records provide accidental-change/staleness
checks, not cryptographic authentication of an untrusted workspace.

Before changing an artifact, the relevant stage follows recorded dependencies
in the existing working/result folders and removes invalidated downstream
outputs **only when an explicit manifest establishes ownership**. Parsed-CSV
cleanup uses its ownership manifest, with a same-directory legacy sample-file
manifest as a migration source. Clustering snapshot names use an exact
`iter[1-9][0-9]*.txt` rule. Files such as `iteration_notes.md`, unrelated CSVs,
and extensionless notes are not selected by broad prefixes or extensions.

Native writers check writing, flushing, and closing before replacement. The
HTML writer uses checked temporary-file replacement and reports expected
output failures through its CLI error handler. Related flat-directory files
are not claimed to be one filesystem transaction: publication metadata is
written last, and mismatched or interrupted sets are rejected. The existing
prepare-in-progress marker remains part of variance preparation.

Provenance records contain absolute source paths. Moving or changing source
files can invalidate dependencies; regenerate through the producing stage
rather than editing a sidecar or accepting mismatched hashes.

### Text and numeric serialization

Per-sample CSVs have exactly `gene,expression` as the parsed header and exactly
two fields per data record. Use comma delimiters, double-quoted fields, and
doubled embedded quotes; embedded newlines and malformed quotes are rejected.
A quoted comma remains part of its field. Empty records, missing values,
repeated headers, duplicate normalized genes, and nonfinite expressions fail.
Decimal and scientific notation may include a leading plus. Fields must be
consumed completely. Absent genes align to zero; invalid existing rows do not.

Identifiers are literal, case-sensitive, nonempty text without whitespace or
control characters. `NA`, `NULL`, and `None` are not missing-value tokens.
Gene suffix normalization (remove from the first period) happens once at
initial ingestion; uniqueness is checked after normalization. A generated
CSV basename is its sample ID, and IDs are checked globally across all input
groups. Full paths and source-category identities are recorded separately.
A collision is reported rather than silently renamed or merged.

Machine-generated integers use ASCII decimal digits with no leading plus,
redundant leading zeros, negative zero, grouping separators, decimal points,
or exponents. Floating-point fields use a period and round-trip precision;
negative zero is emitted as zero. Existing integer-range and scientific
conversion rules are retained. Line-oriented identifier lists are not CSV;
variance records are exactly four whitespace-delimited fields.

Writers emit UTF-8 without a BOM and LF endings. Readers accept a single
initial UTF-8 BOM and LF or CRLF, but reject invalid UTF-8, interior/repeated
BOMs, and lone carriage returns. Blank machine-data records, including an
extra trailing blank record, fail. A single final LF does not add a record.
Human configuration may ignore blank lines where documented; an empty gene
allowlist remains an empty selection, not a request for all genes. Report
blank separators are part of the existing human-readable report grammar,
not empty matrix records.

In `total_data.csv`, only an empty hierarchy-state field repeats the preceding
row in the same column. The first row is explicit. Whitespace-only fields,
`NA`, `NaN`, and `null` do not mean repetition. Filenames are never filled.
The common Python history reader validates transitions against the topology.

### Discovery, display, and diagnostics

Discovery is deterministic, includes direct files and one level of immediate
subdirectories, and skips dot-prefixed files and directories at both levels.
A visible unsupported file causes an error. Dendrogram categories retain full
directory identity, even when two directories have the same basename.
Ambiguous fallback name matches remain unmatched rather than being conflated.

Errors, warnings, progress, and prompts go to stderr. Ordinary completion and
requested result summaries go to stdout. Physical lines and columns are
one-based; immutable root ID 0, history state 0, and split iteration 1 retain
their semantic meanings. Numeric IDs in `tjala_output.txt` are unchanged.
Use `Cluster #17` for a general human label and `C17` for an explained compact
plot label. Conceptual examples are not serialized report snippets.

### Documentation and source conventions

The suite baseline is Python 3.10+ and C++20. Python functions use imperative
Google-style summaries; classes, modules, and descriptive sections use
natural descriptive language. Nontrivial contracts document inputs, outputs,
ordering, side effects, and relevant failures. Dataclasses document public
fields under `Attributes:`. Inline literals use double backticks, builtin
generics and union annotations are preferred, and comments describe current
behavior rather than review history. C++ contracts use Doxygen.

Source lines target 79 characters; docstrings and comments target 72.
Use four-space indentation, UTF-8/LF, US English prose, and no trailing
whitespace. Substantial workflow modules use numbered block banners; those
numbers are organizational, not implied mathematical algorithm-step numbers.
`.clang-format` pins the C++ style settings for clang-format 19.1.7. The tool
version is a development-tool pin, not a compiler/runtime dependency.

Use these artifact terms consistently: constraint configuration (`*_links.txt`),
sample-file manifest (`*_parsed_files.txt`), ordered sample-path list
(`filenames.txt`), tree topology (`tree_splits.txt`), sample-assignment snapshot
(`iterK.txt`), compressed hierarchy history (`total_data.csv`), sample lineage
file (`sample_lineages.txt`), and commit manifest (publication metadata).
Native variance calculation evaluates single-gene comparison **terms**;
Tjala combines terms into antecedents and finalized signatures.


## Naming revision (after the CR-trimming fix)

This revision applies `change_names.docx` to the CR-trimming-fixed package.
It changes names and references, not the mathematical algorithms, numerical
comparisons, default parameters, sampling order, or the Tjala report grammar.
The previous single-boundary CRLF normalization remains in place.

### Canonical artifact names

| Previous name | Current name |
| --- | --- |
| `genes_new.txt` | `matrix_genes.txt` |
| `columns_new.txt` | `matrix_samples.txt` |
| `values_new.txt` | `matrix_values.txt` |
| `hierarchy_new.txt` | `sample_lineages.txt` |
| `variances_new.txt` | `term_variances.txt` |

`genes.txt` is still the optional gene allowlist; it is not a generated matrix
file. `total_data.csv`, `tree_splits.txt`, `filenames.txt`, the `iterK.txt`
snapshots, `tjala_output.txt`, and extraction-output basenames are unchanged.

Both configurable hierarchy-history consumers use **`--tree-states`**:

```text
python compute_variances.py prepare --tree-states total_data.csv
python build_dendrogram.py --tree-states clustering_results/total_data.csv
```

The former `--manifest` and `--data` switches are not aliases. Update stored
commands accordingly. The `all`, `prepare`, `build`, and `run` variance-stage
names and legacy optional gene-allowlist positional argument are unchanged.

### Internal naming conventions

Python artifact constants use `<ARTIFACT>_FILE` and are defined in
`compute_variances.py`, then imported by consumers. Native artifact constants
are defined once in `pipeline_files.hpp` and use `kArtifactFile`, matching the
requested C++ `k`-prefix convention. Keep the header beside all native sources,
including copies of `compute_variances.cpp` in `clustering_results`.

Descriptive counts use `sample_count`, `node_count`, `term_count`, `part_count`,
and `ant_count`. Tree-node IDs and SciPy linkage IDs are named separately;
`tree_to_linkage` maps the former to the latter. Interactive readers use
`prompt_*`; formatted-text helpers use `_format_*`; atomic-write helpers use
`*_atomically`. Internal Python API names were renamed without old-name aliases.

Tjala parameters now use `max_terms`, `max_bins`, `convergence_window`,
`max_uncovered_samples`, and `ant_count`. Console-only settings are represented
by `ConsoleOptions` and passed as `console_options`. The algorithm-step function
names and A/B/E/F/Q mathematical locals remain unchanged except for the
single/batch and loader renames explicitly requested in the naming document.

### Upgrade safely

Install this revision as a matched set and rebuild the native executables.
A new result directory is the recommended upgrade path: copy or regenerate the
verified upstream clustering/history inputs together with their provenance,
then rerun `prepare`, `build`, and `run`, Tjala, and the required exports.
Alternatively, rerun the complete workflow from the original source matrices
as described above. Keep backups of existing runs.

Do not merely rename old intermediate files or edit their provenance hashes.
Prepared-input fingerprints include filenames, and the variance sidecar is now
`term_variances.txt.provenance`. Existing outputs are not auto-migrated. Do not
mix old and new prepared artifacts in one working directory.

`compute_variances_state.json` now uses schema version 2. An old version-1 state
is rejected; move the old state and its generated-result set to a backup rather
than modifying it to look current. Fresh prepare/build/run stages create the
new state. Build metadata now records `artifact_header_sha256` as well as the
native source and executable hashes, so a changed `pipeline_files.hpp` cannot
silently reuse the old verified executable.
