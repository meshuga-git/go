/**
 * @file pipeline_files.hpp
 * @brief Canonical artifact basenames shared by native pipeline stages.
 *
 * Requires C++20. These names match the Python catalog in
 * compute_variances.py. Only filenames are defined here; scientific
 * algorithms and file-format grammars are unchanged.
 */
#ifndef TAVNIT_PIPELINE_FILES_HPP
#define TAVNIT_PIPELINE_FILES_HPP

namespace pipeline_files {
inline constexpr char kTreeStatesFile[] = "total_data.csv";
inline constexpr char kMatrixGenesFile[] = "matrix_genes.txt";
inline constexpr char kMatrixSamplesFile[] = "matrix_samples.txt";
inline constexpr char kMatrixValuesFile[] = "matrix_values.txt";
inline constexpr char kSampleLineagesFile[] = "sample_lineages.txt";
inline constexpr char kTermVariancesFile[] = "term_variances.txt";
inline constexpr char kVarianceStateFile[] = "compute_variances_state.json";
inline constexpr char kTreeTopologyFile[] = "tree_splits.txt";
inline constexpr char kSamplePathsFile[] = "filenames.txt";
inline constexpr char kTjalaOutputFile[] = "tjala_output.txt";
inline constexpr char kDendrogramFile[] = "dendrogram.html";
inline constexpr char kBinaryVectorsFile[] = "binary_vectors.csv";
inline constexpr char kCannotLinksFile[] = "cannot_links.txt";
inline constexpr char kMustLinksFile[] = "must_links.txt";
inline constexpr char kNoLinksFile[] = "no_links.txt";
inline constexpr char kPreparedProvenanceFile[] = "prepared.provenance";
inline constexpr char kClusteringProvenanceFile[] = "clustering.provenance";
inline constexpr char kParsedDataProvenanceFile[] = "parsed_data.provenance";
inline constexpr char kPrepareMarkerFile[] =
    ".compute_variances_prepare_in_progress";
}  // namespace pipeline_files

#endif  // TAVNIT_PIPELINE_FILES_HPP
