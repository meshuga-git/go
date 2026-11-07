/**
 * @file run_clustering.cpp
 * @note Requires C++20.
 * @brief Constrained divisive clustering of aligned expression samples.
 *
 * Read constraint configuration and per-sample CSVs. Publish
 *     filenames.txt, tree_splits.txt, iterK.txt, and
 *     clustering.provenance. All data files use UTF-8 and LF. Requires
 *     C++20.
 */
#include "pipeline_files.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <version>
#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
using pipeline_files::kCannotLinksFile;
using pipeline_files::kClusteringProvenanceFile;
using pipeline_files::kMustLinksFile;
using pipeline_files::kNoLinksFile;
using pipeline_files::kTreeTopologyFile;

using pipeline_files::kSamplePathsFile;

namespace pipeline_io {
namespace fs = std::filesystem;
/**
 * @brief Decode valid UTF-8 and reject interior byte-order marks.
 * @param text UTF-8 record without its line terminator.
 * @return Unicode scalar values in record order.
 * @throws std::runtime_error On malformed UTF-8 or an interior BOM.
 */
std::vector<std::uint32_t> unicode_values(std::string_view text) {
    std::vector<std::uint32_t> result;
    for (std::size_t pos = 0; pos < text.size();) {
        const auto first = static_cast<unsigned char>(text[pos++]);
        std::uint32_t value = first;
        std::size_t extra = 0;
        std::uint32_t minimum = 0;
        if (first < 0x80) {
            extra = 0;
        } else if (first >= 0xc2 && first <= 0xdf) {
            value = first & 0x1f;
            extra = 1;
            minimum = 0x80;
        } else if (first >= 0xe0 && first <= 0xef) {
            value = first & 0x0f;
            extra = 2;
            minimum = 0x800;
        } else if (first >= 0xf0 && first <= 0xf4) {
            value = first & 7;
            extra = 3;
            minimum = 0x10000;
        } else {
            throw std::runtime_error("Invalid UTF-8 lead byte");
        }
        if (text.size() - pos < extra) {
            throw std::runtime_error("Truncated UTF-8 sequence");
        }
        for (std::size_t i = 0; i < extra; ++i) {
            const auto next = static_cast<unsigned char>(text[pos++]);
            if ((next & 0xc0) != 0x80) {
                throw std::runtime_error("Invalid UTF-8 continuation byte");
            }
            value = (value << 6) | (next & 0x3f);
        }
        if (value < minimum || value > 0x10ffff ||
            (value >= 0xd800 && value <= 0xdfff) || value == 0xfeff ||
            value == 0) {
            throw std::runtime_error("Invalid Unicode scalar or interior BOM");
        }
        result.push_back(value);
    }
    return result;
}
/**
 * @brief Test whitespace and control scalars prohibited in identifiers.
 * @param value Unicode scalar value.
 * @return Whether the scalar is whitespace or a control character.
 */
bool invalid_name_scalar(std::uint32_t value) {
    return value <= 0x20 || (value >= 0x7f && value <= 0x9f) ||
    value == 0xa0 || value == 0x1680 ||
    (value >= 0x2000 && value <= 0x200a) || value == 0x2028 ||
    value == 0x2029 || value == 0x202f || value == 0x205f ||
    value == 0x3000;
}
/**
 * @brief Validate one literal gene or sample identifier.
 * @param name Identifier after ingestion normalization.
 * @param source File and field context.
 * @throws std::runtime_error On empty, whitespace, or control contents.
 */
void validate_identifier(std::string_view name, std::string_view source) {
    const auto scalars = unicode_values(name);
    if (scalars.empty() || std::any_of(
        scalars.begin(), scalars.end(), invalid_name_scalar)) {
        throw std::runtime_error(std::string(source) + ": invalid identifier");
    }
}
/**
 * @brief Read literal UTF-8 records, accepting one BOM and CRLF.
 * @param path Text input file.
 * @param allow_blank Whether the schema permits blank records.
 * @param allow_empty Whether an empty file is meaningful.
 * @return Physical records without transport terminators.
 * @throws std::runtime_error On I/O, encoding, or record errors.
 */
std::vector<std::string> read_lines(
    const fs::path& path, bool allow_blank = false, bool allow_empty = false) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!input.eof() && !line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (lines.empty() && line.starts_with("\xef\xbb\xbf")) {
            line.erase(0, 3);
        }
        if (line.find('\r') != std::string::npos) {
            throw std::runtime_error(path.string() + ": lone carriage return");
        }
        const auto scalars = unicode_values(line);
        if (!allow_blank && (line.empty() || std::all_of(
            scalars.begin(), scalars.end(), invalid_name_scalar))) {
            throw std::runtime_error(std::format("{}:{}: blank record",
                path.string(), lines.size() + 1));
        }
        lines.push_back(line);
    }
    if (!input.eof()) throw std::runtime_error("Read failed: " +
        path.string());
    if (lines.empty() && !allow_empty) {
        throw std::runtime_error("Empty input: " + path.string());
    }
    return lines;
}
/**
 * @brief Check canonical ASCII decimal integer spelling.
 * @param token Complete serialized integer field.
 * @return Whether spelling is canonical, independently of its range.
 */
bool canonical_integer(std::string_view token) {
    if (token == "0") return true;
    std::size_t pos = token.starts_with('-') ? 1 : 0;
    if (pos == token.size() || token[pos] < '1' || token[pos] > '9') {
        return false;
    }
    return std::all_of(token.begin() + pos + 1, token.end(),
        [](char value) {
            return value >= '0' && value <= '9';
        });
}
/**
 * @brief Parse one canonical decimal integer into a checked target
 *     type.
 * @tparam Integer Destination integral type.
 * @param token Complete input field.
 * @param source File and field context.
 * @return Exact, representable integer value.
 * @throws std::runtime_error On malformed or out-of-range contents.
 */
template <class Integer>
Integer parse_integer(std::string_view token, std::string_view source) {
    if (!canonical_integer(token)) {
        throw std::runtime_error(std::format("{}: noncanonical integer {}",
            std::string(source), std::string(token)));
    }
    Integer result{};
    const auto [end, error] = std::from_chars(
        token.data(), token.data() + token.size(), result);
    if (error != std::errc{} || end != token.data() + token.size()) {
        throw std::runtime_error(std::string(source)
            + ": integer out of range");
    }
    return result;
}
/**
 * @brief Split a whitespace-delimited machine record into fields.
 * @param line One physical record.
 * @return Literal tokens; no quotation or missing-value inference.
 */
std::vector<std::string> tokens(std::string_view line) {
    std::istringstream stream{std::string(line)};
    stream.imbue(std::locale::classic());
    std::vector<std::string> fields;
    for (std::string value; stream >> value;) fields.push_back(value);
    return fields;
}
/**
 * @brief Decode the pipeline's strict, single-line CSV dialect.
 * @param line Physical record without terminators.
 * @param source File and line context for diagnostics.
 * @return Decoded literal fields, including empty fields.
 * @throws std::runtime_error On misplaced or unterminated quotes.
 */
std::vector<std::string> csv_fields(
    std::string_view line, std::string_view source) {
    std::vector<std::string> fields;
    std::size_t pos = 0;
    while (true) {
        std::string value;
        if (pos < line.size() && line[pos] == '"') {
            ++pos;
            while (true) {
                if (pos == line.size()) {
                    throw std::runtime_error(std::string(source) +
                        ": unterminated CSV quote");
                }
                const char next = line[pos++];
                if (next != '"') {
                    value.push_back(next);
                    continue;
                }
                if (pos < line.size() && line[pos] == '"') {
                    value.push_back('"');
                    ++pos;
                } else break;
            }
            if (pos < line.size() && line[pos] != ',') {
                throw std::runtime_error(std::string(source) +
                    ": content after closing CSV quote");
            }
        } else {
            while (pos < line.size() && line[pos] != ',') {
                if (line[pos] == '"') {
                    throw std::runtime_error(std::string(source) +
                        ": quote inside unquoted field");
                }
                value.push_back(line[pos++]);
            }
        }
        fields.push_back(std::move(value));
        if (pos == line.size()) return fields;
        ++pos;
    }
}
/**
 * @brief Parse a complete finite decimal/scientific expression field.
 * @param token Literal decoded CSV field; a leading plus is accepted.
 * @param source File and field context.
 * @return Finite binary64-compatible expression, with zero normalized.
 * @throws std::runtime_error On syntax, overflow, or underflow errors.
 */
double parse_decimal_float(std::string_view token, std::string_view source) {
    static const std::regex GRAMMAR(
        R"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    if (!std::regex_match(token.begin(), token.end(), GRAMMAR)) {
        throw std::runtime_error(std::string(source) + ": invalid decimal");
    }
    if (token.starts_with('+')) token.remove_prefix(1);
    double result{};
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    const auto [end, error] = std::from_chars(
        token.data(), token.data() + token.size(), result);
    const bool parsed =
        error == std::errc{} && end == token.data() + token.size();
#else
    // Apple Clang's library has no floating-point from_chars. strtod
    // reads the same validated decimal syntax in the default "C" locale
    // and rounds correctly too; like from_chars, it rejects only values
    // that overflow or underflow to zero, not subnormal ones.
    const std::string text(token);
    char* end = nullptr;
    errno = 0;
    result = std::strtod(text.c_str(), &end);
    const bool out_of_range =
        errno == ERANGE && (result == 0.0 || std::isinf(result));
    const bool parsed =
        !out_of_range && end == text.c_str() + text.size();
#endif
    if (!parsed || !std::isfinite(result)) {
        throw std::runtime_error(std::string(source) +
            ": nonfinite or unrepresentable expression");
    }
    return result == 0 ? 0.0 : result;
}
/** @brief Incremental SHA-256 state for artifact-integrity checks. */
class Sha256 {
public:
    /**
     * @brief Append bytes to the hash without retaining the input.
     * @param bytes Bytes to append.
     */
    void update(std::string_view bytes) {
        for (const unsigned char value : bytes) {
            buffer_[used_++] = value;
            ++length_;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    /** @brief Finalize the hash and return lowercase hexadecimal. */
    std::string finish() {
        const std::uint64_t bits = length_ * 8;
        buffer_[used_++] = 0x80;
        if (used_ > 56) {
            std::fill(buffer_.begin() + used_, buffer_.end(), 0);
            compress();
            used_ = 0;
        }
        std::fill(buffer_.begin() + used_, buffer_.begin() + 56, 0);
        for (unsigned i = 0; i < 8; ++i) {
            buffer_[63 - i] = static_cast<unsigned char>(bits >> (8 * i));
        }
        compress();
        std::ostringstream result;
        result << std::hex << std::setfill('0');
        for (const auto value : state_) result << std::setw(8) << value;
        return result.str();
    }
private:
    std::array<std::uint32_t, 8> state_{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> buffer_{};
    std::size_t used_ = 0;
    std::uint64_t length_ = 0;
    /** @brief Fold the buffered 512-bit block into the digest state. */
    void compress() {
        static constexpr std::array<std::uint32_t, 64> kRoundConstants{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                w[i] = (w[i] << 8) | buffer_[4 * i + j];
            }
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const auto x = w[i - 15], y = w[i - 2];
            const auto s0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3);
            const auto s1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (std::size_t i = 0; i < 64; ++i) {
            const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e,
                25);
            const auto choice = (e & f) ^ (~e & g);
            const auto t1 = h + s1 + choice + kRoundConstants[i] + w[i];
            const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a,
                22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d+t1;
            d = c;
            c = b;
            b = a;
            a = t1+t2;
        }
        const std::array<std::uint32_t, 8> values{a, b, c, d, e, f, g, h};
        for (std::size_t i = 0; i < 8; ++i) state_[i] += values[i];
    }
};
/**
 * @brief Hash exact file bytes using SHA-256.
 * @param path File to hash.
 * @return Lowercase 64-character hexadecimal digest.
 * @throws std::runtime_error If the file cannot be read completely.
 */
std::string sha256_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot hash " + path.string());
    Sha256 hash;
    std::array<char, 65536> buffer{};
    while (input) {
        input.read(buffer.data(), buffer.size());
        hash.update(std::string_view(buffer.data(), input.gcount()));
    }
    if (!input.eof()) throw std::runtime_error("Hash read failed: " +
        path.string());
    return hash.finish();
}
/**
 * @brief Encode arbitrary UTF-8 bytes without whitespace delimiters.
 * @param value Input byte string.
 * @return Lowercase hexadecimal bytes.
 */
std::string hex_encode(std::string_view value) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string result;
    for (const unsigned char byte : value) {
        result.push_back(kHexDigits[byte >> 4]);
        result.push_back(kHexDigits[byte & 15]);
    }
    return result;
}
/**
 * @brief Decode a hexadecimal UTF-8 field.
 * @param value Hexadecimal bytes from a provenance record.
 * @return Decoded UTF-8 text.
 * @throws std::runtime_error If spelling or UTF-8 is invalid.
 */
std::string hex_decode(std::string_view value) {
    if (value.size() % 2 || value.find_first_not_of(
        "0123456789abcdef") != std::string_view::npos) {
        throw std::runtime_error("Invalid hexadecimal provenance field");
    }
    std::string result;
    for (std::size_t pos = 0; pos < value.size(); pos += 2) {
        unsigned byte{};
        const auto [end, error] = std::from_chars(
            value.data() + pos, value.data() + pos + 2, byte, 16);
        if (error != std::errc{} || end != value.data() + pos + 2) {
            throw std::runtime_error("Invalid hex path");
        }
        result.push_back(static_cast<char>(byte));
    }
    unicode_values(result);
    return result;
}
/**
 * @brief Decode a UTF-8 path using C++20 filesystem semantics.
 * @param value UTF-8 encoded pathname.
 * @return Filesystem path, without locale-dependent byte decoding.
 */
fs::path path_from_utf8(std::string_view value) {
    const auto* begin = reinterpret_cast<const char8_t*>(value.data());
    return fs::path(std::u8string_view(begin, value.size()));
}
/**
 * @brief Convert a filesystem path to UTF-8 independent of locale.
 * @param path Filesystem path.
 * @return UTF-8 path bytes.
 */
std::string utf8_path(const fs::path& path) {
    const auto value = path.u8string();
    return std::string(reinterpret_cast<const char*>(value.data()),
        value.size());
}
/**
 * @brief Write, check flush/close, and replace one binary-text output.
 * @param destination Final file path.
 * @param body Callback writing the complete file to the supplied
 *     stream.
 * @throws std::runtime_error On any write, flush, close, or replace
 *     error.
 */
void checked_write(const fs::path& destination,
    const std::function<void(std::ofstream&)>& body) {
    const fs::path temporary = destination.string() + ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.imbue(std::locale::classic());
        if (!output) throw std::runtime_error("Cannot write " +
            temporary.string());
        body(output);
        output.flush();
        if (!output) throw std::runtime_error("Flush failed: " +
            temporary.string());
        output.close();
        if (output.fail()) throw std::runtime_error("Close failed: " +
            temporary.string());
#ifdef _WIN32
        if (!MoveFileExW(temporary.wstring().c_str(),
            destination.wstring().c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("Replace failed: " +
                destination.string());
        }
#else
        if (std::rename(temporary.c_str(), destination.c_str()) != 0) {
            throw std::runtime_error("Replace failed: " +
                destination.string());
        }
#endif
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}
/** @brief Parsed stage-local artifact-provenance record. */
struct ProvenanceRecord {
    std::string stage; ///< Producing stage.
    /// SHA-256 of the producing source/executable.
    std::string producer;
    std::map<std::string, std::string> inputs; ///< Input-path hashes.
    /// Parent-commit hashes.
    std::map<std::string, std::string> parents;
    /// Local output hashes.
    std::map<std::string, std::string> outputs;
    /// Sample to folder.
    std::map<std::string, std::string> categories;
};
/**
 * @brief Parse stage metadata and validate local output ownership.
 * @param path Commit manifest path.
 * @return Record whose fields are syntactically valid but unverified.
 * @throws std::runtime_error On invalid records or unsafe output paths.
 */
ProvenanceRecord read_provenance(const fs::path& path) {
    const auto lines = read_lines(path);
    if (lines.size() < 4 || lines[0] != "TAVNIT-PROVENANCE-1" ||
        !lines[1].starts_with("stage\t") ||
        !lines[2].starts_with("producer\t")) {
        throw std::runtime_error("Invalid provenance: " + path.string());
    }
    static const std::set<std::string> STAGES{
        "ingest", "configuration", "sample_manifest", "clustering",
        "reconstruct", "prepare", "variances", "signatures",
        "cluster_export", "signature_export", "dendrogram"};
    static const std::regex HASH("[0-9a-f]{64}");
    ProvenanceRecord result;
    result.stage = lines[1].substr(6);
    result.producer = lines[2].substr(9);
    if (!STAGES.contains(result.stage) ||
        !std::regex_match(result.producer, HASH)) {
        throw std::runtime_error("Invalid provenance header");
    }
    for (std::size_t i = 3; i < lines.size(); ++i) {
        const auto& line = lines[i];
        const auto first = line.find('\t');
        const auto last = line.find('\t', first + 1);
        if (first == std::string::npos || last == std::string::npos ||
            line.find('\t', last + 1) != std::string::npos) {
            throw std::runtime_error("Expected three tab-separated fields");
        }
        const auto kind = line.substr(0, first);
        const auto digest = line.substr(first + 1, last - first - 1);
        const auto value = hex_decode(line.substr(last + 1));
        if (value.empty() || value.find_first_of("\r\n") !=
            std::string::npos) {
            throw std::runtime_error("Invalid provenance path");
        }
        if (kind == "C") {
            const auto key = hex_decode(digest);
            validate_identifier(key, path.string());
            if (!path_from_utf8(value).is_absolute() ||
                !result.categories.emplace(key, value).second) {
                throw std::runtime_error("Invalid source category record");
            }
            continue;
        }
        if (!std::regex_match(digest, HASH)) {
            throw std::runtime_error("Invalid provenance SHA-256");
        }
        auto* mapping = &result.inputs;
        if (kind == "P") {
            mapping = &result.parents;
        } else if (kind == "O") {
            const auto output = path_from_utf8(value);
            if (output.filename() != output ||
                value.find_first_of("/\\") != std::string::npos ||
                value == "." || value == "..") {
                throw std::runtime_error("Output must be a local basename");
            }
            mapping = &result.outputs;
        } else if (kind != "I") {
            throw std::runtime_error("Unknown provenance record");
        }
        if ((kind == "I" || kind == "P") &&
            !path_from_utf8(value).is_absolute()) {
            throw std::runtime_error("Relative provenance dependency");
        }
        if (!mapping->emplace(value, digest).second) {
            throw std::runtime_error("Duplicate provenance path");
        }
    }
    if (result.outputs.empty()) {
        throw std::runtime_error("No committed outputs");
    }
    return result;
}
/**
 * @brief Verify dependency hashes recursively with cycle detection.
 * @param path Commit manifest to verify.
 * @param verified Manifests already verified in this operation.
 * @param visiting Current recursion stack.
 * @return Verified manifest record.
 * @throws std::runtime_error On missing, changed, or cyclic
 *     dependencies.
 */
ProvenanceRecord verify_provenance_impl(
    const fs::path& path, std::set<fs::path>& verified,
    std::set<fs::path>& visiting) {
    const auto key = fs::weakly_canonical(path);
    const auto record = read_provenance(key);
    if (verified.contains(key)) return record;
    if (!visiting.insert(key).second) throw
    std::runtime_error("Provenance cycle");
    for (const auto& [raw, digest] : record.parents) {
        const auto parent = pipeline_io::path_from_utf8(raw);
        if (sha256_file(parent) != digest) {
            throw std::runtime_error("Changed upstream commit: " + raw);
        }
        verify_provenance_impl(parent, verified, visiting);
    }
    for (const auto& [raw, digest] : record.inputs) {
        if (sha256_file(pipeline_io::path_from_utf8(raw)) != digest) {
            throw std::runtime_error("Changed input: " + raw);
        }
    }
    for (const auto& [name, digest] : record.outputs) {
        if (sha256_file(key.parent_path() / pipeline_io::path_from_utf8(name))
            != digest) {
            throw std::runtime_error("Changed output: " + name);
        }
    }
    visiting.erase(key);
    verified.insert(key);
    return record;
}
/**
 * @brief Require intact artifact provenance before consumption.
 * @param path Commit manifest owning the required artifacts.
 * @return Verified commit record.
 * @throws std::runtime_error On invalid or mismatched provenance.
 */
ProvenanceRecord verify_provenance(const fs::path& path) {
    std::set<fs::path> verified, visiting;
    return verify_provenance_impl(path, verified, visiting);
}
/**
 * @brief Commit stage output hashes after successful validation.
 * @param path Destination manifest beside all outputs.
 * @param stage Producing stage identifier.
 * @param outputs Complete output set in this directory.
 * @param inputs Direct input files.
 * @param parents Upstream manifests, verified before publication.
 * @param producer Executable used to generate the outputs.
 * @throws std::runtime_error On invalid dependencies or failed
 *     publication.
 */
void write_provenance(const fs::path& path, const std::string& stage,
    const std::vector<fs::path>& outputs, const std::vector<fs::path>& inputs,
    const std::vector<fs::path>& parents, const fs::path& producer) {
    std::ostringstream text;
    text << "TAVNIT-PROVENANCE-1\nstage\t" << stage << "\nproducer\t"
    << sha256_file(producer) << '\n';
    const auto append = [&text](char kind, const fs::path& item) {
        const auto key = kind == 'O' ? item.filename() :
        fs::weakly_canonical(item);
        text << kind << '\t' << sha256_file(item) << '\t'
        << hex_encode(utf8_path(key)) << '\n';
    };
    for (const auto& item : inputs) append('I', item);
    std::set<fs::path> seen_parents;
    for (const auto& item : parents) {
        if (!seen_parents.insert(fs::weakly_canonical(item)).second) continue;
        verify_provenance(item);
        append('P', item);
    }
    for (const auto& item : outputs) append('O', item);
    checked_write(path, [&text](std::ofstream& output) {
            output << text.str();
        });
}
/**
 * @brief Invalidate only owned outputs whose dependencies change.
 * @param changed Artifacts about to be replaced.
 * @param directories Existing workspace locations inspected
 *     nonrecursively.
 * @throws std::runtime_error On unsafe ownership or failed removal.
 */
void invalidate_dependents(const std::vector<fs::path>& changed,
    const std::vector<fs::path>& directories) {
    std::set<fs::path> dirty;
    for (const auto& item : changed) dirty.insert(fs::weakly_canonical(item));
    std::map<fs::path, ProvenanceRecord> records;
    for (const auto& directory : directories) {
        if (!fs::is_directory(directory)) continue;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_symlink() || !entry.is_regular_file() ||
                entry.path().extension() != ".provenance") continue;
            try {
                records.emplace(entry.path(), read_provenance(entry.path()));
            } catch (const std::exception&) {
                // Unrecognized metadata never establishes cleanup
                // ownership.
            }
        }
    }
    // A changed output also invalidates references to its owning
    // commit.
    for (const auto& [manifest, record] : records) {
        for (const auto& [name, digest] : record.outputs) {
            const auto output = fs::weakly_canonical(
                manifest.parent_path() / pipeline_io::path_from_utf8(name));
            if (dirty.contains(output)) {
                dirty.insert(fs::weakly_canonical(manifest));
            }
        }
    }
    bool progress = true;
    while (progress) {
        progress = false;
        for (auto it = records.begin(); it != records.end();) {
            bool dependent = false;
            for (const auto* mapping : {&it->second.inputs,
                    &it->second.parents}) {
                for (const auto& [raw, digest] : *mapping) {
                    const auto dependency = fs::weakly_canonical(
                        pipeline_io::path_from_utf8(raw));
                    if (dirty.contains(dependency)) {
                        dependent = true;
                    }
                }
            }
            if (!dependent) {
                ++it;
                continue;
            }
            dirty.insert(fs::weakly_canonical(it->first));
            fs::remove(it->first);
            for (const auto& [name, digest] : it->second.outputs) {
                const auto output = it->first.parent_path() /
                pipeline_io::path_from_utf8(name);
                dirty.insert(fs::weakly_canonical(output));
                // fs::remove deletes a symbolic link, not its target.
                fs::remove(output);
            }
            it = records.erase(it);
            progress = true;
        }
    }
}
} // namespace pipeline_io
namespace fs = std::filesystem;
namespace {
// Supported distance metric selection.
/** @brief Supported pairwise dissimilarity choices. */
enum class DistanceMetric {
    Euclidean = 1,
    Pearson = 2,
    Spearman = 3,
    Manhattan = 4
};
/**
 * @brief Represents the gene expression dataset and associated
 *     metadata.
 *
 * This class handles the ingestion of gene expression data and
 *     constraints. It reads configuration files and parses input
 *     datasets in two passes (one for vocabulary discovery, one for
 *     population) to maintain per-file temporary parsing storage. The
 *     resulting data is flattened into a contiguous, dense row-major
 *     matrix for cache-friendly access.
 */
class ExpressionDataset {
public:
    static constexpr std::size_t kMaxGenes = 65'000;
    /**
     * @brief Load constraint groups and verified sample-file manifests.
     * @throws std::runtime_error If configuration, provenance, or IDs
     *     are invalid.
     */
    void load_dataset_config() {
        sample_paths_.clear();
        sample_counts_by_group_.fill(0);
        const auto cannot_links = read_constraints_file(kCannotLinksFile);
        if (cannot_links.size() != 0 && cannot_links.size() != 2) {
            throw std::runtime_error(std::format(
                "cannot_links.txt must list 0 or 2 folders "
                "(found {}).", cannot_links.size()));
        }
        if (cannot_links.size() == 2) {
            read_sample_manifest(cannot_links[0] + "_parsed_files.txt", 0);
            read_sample_manifest(cannot_links[1] + "_parsed_files.txt", 1);
        }
        const auto must_links = read_constraints_file(kMustLinksFile);
        if (must_links.size() > 1) {
            throw std::runtime_error(std::format(
                "must_links.txt must list 0 or 1 folder (found "
                "{}).", must_links.size()));
        }
        if (must_links.size() == 1) {
            read_sample_manifest(must_links[0] + "_parsed_files.txt", 2);
        }
        const auto no_links = read_constraints_file(kNoLinksFile);
        if (no_links.size() > 1) {
            throw std::runtime_error(std::format(
                "no_links.txt must list 0 or 1 folder (found "
                "{}).", no_links.size()));
        }
        if (no_links.size() == 1) {
            read_sample_manifest(no_links[0] + "_parsed_files.txt", 3);
        }
        if (sample_paths_.empty()) {
            throw
            std::runtime_error("No input files were found across constraint "
                "sets.");
        }
        std::unordered_set<std::string> sample_ids;
        for (const auto& path : sample_paths_) {
            const auto pos = path.find_last_of("/\\");
            const auto id = path.substr(pos == std::string::npos ? 0 : pos +
                1);
            pipeline_io::validate_identifier(id, path);
            if (!sample_ids.insert(id).second) {
                throw
                std::runtime_error(std::format("Duplicate sample ID: {}",
                    id));
            }
        }
    }
    /**
     * @brief Validate sample CSVs and populate zero-aligned
     *     expressions.
     * @throws std::runtime_error If records, identity, or dimensions
     *     are invalid.
     */
    void parse_all_files() {
        gene_index_map_.clear();
        gene_count_ = 0;
        expressions_matrix_.clear();
        // Pass 1: discover the complete gene vocabulary and validate
        // each file. Only one file's temporary duplicate-detection set
        // is resident at a time.
        for (std::size_t i = 0; i < sample_paths_.size(); ++i) {
            parse_file_records(i, [this, i](const std::string& gene_name,
                double /*value*/,
                std::size_t /*line_number*/) {
                    if (gene_index_map_.find(gene_name) ==
                    gene_index_map_.end()) {
                        if (gene_count_ >= kMaxGenes) {
                            throw std::runtime_error(std::format(
                            "Exceeded maximum distinct gene limit ({}) in {}.",
                            kMaxGenes,
                            sample_paths_[i]));
                        }
                        gene_index_map_.emplace(gene_name, gene_count_++);
                    }
                });
            std::cerr << std::format("Sample {} parsed (pass 1).\n", i + 1);
        }
        if (gene_count_ == 0) {
            throw
            std::runtime_error("No gene expression values were successfully "
                "parsed.");
        }
        if (sample_paths_.size() >
            std::numeric_limits<std::size_t>::max() / gene_count_) {
            throw
            std::length_error("Expression matrix size overflows size_t.");
        }
        const std::size_t matrix_size = sample_paths_.size() * gene_count_;
        expressions_matrix_.assign(matrix_size, 0.0);
        // Pass 2: populate the dense matrix now that every gene has a
        // stable index.
        for (std::size_t i = 0; i < sample_paths_.size(); ++i) {
            parse_file_records(i, [this, i](const std::string& gene_name,
                double value,
                std::size_t /*line_number*/) {
                    const auto it = gene_index_map_.find(gene_name);
                    if (it == gene_index_map_.end()) {
                        throw std::runtime_error(std::format(
                        "Input file changed between parsing passes: {}",
                        sample_paths_[i]));
                    }
                    expressions_matrix_[i * gene_count_ + it->second] = value;
                });
            std::cerr << std::format("Sample {} parsed (pass 2).\n", i + 1);
        }
    }
    /**
     * @brief Return the number of samples.
     * @return Sample count in ordered input paths.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return sample_paths_.size();
    }
    /**
     * @brief Return the size of the aligned gene vocabulary.
     * @return Number of distinct genes.
     */
    [[nodiscard]] std::size_t gene_count() const noexcept {
        return gene_count_;
    }
    /**
     * @brief Return sample paths in constraint-group ingestion order.
     * @return A reference to the dataset-owned ordered path list.
     */
    [[nodiscard]] const std::vector<std::string>& sample_paths()
        const noexcept {
        return sample_paths_;
    }
    // Returns the (imputed) expression vector for the given sample.
    // Genes absent from that sample's input file were filled in as 0.0
    // -- see the note on expressions_matrix_ for the statistical
    // implications of that.
    /**
     * @brief Return one aligned sample expression vector.
     * @param sample_index Zero-based sample index.
     * @return A view of gene_count values, with absent genes filled by
     *     zero.
     * @throws std::out_of_range If sample_index is outside the dataset.
     */
    [[nodiscard]] std::span<const double> expressions(std::size_t sample_index)
    const {
        if (sample_index >= sample_paths_.size()) {
            throw
            std::out_of_range("ExpressionDataset::expressions: "
                "sample_index out "
                ""
                "of range");
        }
        return std::span<const double>(
            expressions_matrix_.data() + sample_index * gene_count_,
            gene_count_);
    }
    /**
     * @brief Return the sample constraint-group identifier.
     * @param sample_index Valid zero-based sample index.
     * @return Group 1/2 for cannot links, 3 for must links, or 4
     *     otherwise.
     */
    [[nodiscard]] int constraint_group(std::size_t sample_index)
        const noexcept {
        const std::size_t cannot_group_2_start = sample_counts_by_group_[0];
        const std::size_t must_group_start = cannot_group_2_start +
        sample_counts_by_group_[1];
        const std::size_t unconstrained_group_start = must_group_start +
        sample_counts_by_group_[2];
        if (sample_index < cannot_group_2_start) return 1;
        if (sample_index < must_group_start) return 2;
        if (sample_index < unconstrained_group_start) return 3;
        return 4;
    }
private:
    std::vector<std::string> sample_paths_;
    // Dense samples x genes matrix, stored row-major in one contiguous
    // buffer (row i occupies [i * gene_count_, (i + 1) *
    // gene_count_)). A single flat allocation instead of a
    // vector-of-vectors avoids one heap allocation per sample and keeps
    // rows contiguous, which matters because this matrix is walked
    // repeatedly by the O(n^2 * genes) pairwise distance computations
    // below.
    //
    // NOTE: genes that are absent from a given sample's input file are
    // imputed as 0.0 here rather than excluded. This affects every
    // distance metric derived from a sample's mean/variance (Pearson,
    // Spearman), not just the metrics that operate value-by-value
    // (Euclidean, Manhattan): a sample with many missing genes is
    // treated as if it truly expressed those genes at zero, not as "no
    // data for those genes".
    std::vector<double> expressions_matrix_;
    std::unordered_map<std::string, std::size_t> gene_index_map_;
    std::array<std::size_t, 4> sample_counts_by_group_{};
    std::size_t gene_count_ = 0;
    /**
     * @brief Read verified literal constraint-configuration paths.
     * @param filepath Optional-empty configuration file.
     * @return Nonblank directory paths in physical order.
     * @throws std::runtime_error If provenance, encoding, or I/O fails.
     */
    static std::vector<std::string> read_constraints_file(const fs::path&
        filepath) {
        const auto provenance = fs::path(filepath.string() + ".provenance");
        const auto record = pipeline_io::verify_provenance(provenance);
        if (!record.outputs.contains(filepath.filename().string())) {
            throw
            std::runtime_error("Configuration provenance does not own input");
        }
        auto lines = pipeline_io::read_lines(filepath, true, true);
        std::erase_if(lines, [](const std::string& line) {
                return line.find_first_not_of(" \t") == std::string::npos;
            });
        return lines;
    }
    /**
     * @brief Append verified unique sample paths for one constraint
     *     group.
     * @param filepath Sample-file manifest for the group.
     * @param folder_index Group slot in the range zero through three.
     * @throws std::runtime_error If provenance, identifiers, or I/O
     *     fails.
     */
    void read_sample_manifest(const fs::path& filepath, std::size_t
        folder_index) {
        const auto record = pipeline_io::verify_provenance(
            fs::path(filepath.string() + ".provenance"));
        if (!record.outputs.contains(filepath.filename().string())) {
            throw std::runtime_error("Sample-file manifest is not committed");
        }
        for (const auto& line : pipeline_io::read_lines(filepath)) {
            sample_paths_.push_back(line);
            ++sample_counts_by_group_[folder_index];
        }
    }
    /**
     * @brief Validate one complete finite expression field.
     * @param text Decoded CSV expression field.
     * @param path_str Source path for diagnostics.
     * @param line_number One-based physical source line.
     * @return Finite binary64 expression value.
     * @throws std::runtime_error If syntax or numeric range is invalid.
     */
    double parse_finite_double(std::string_view text,
        const std::string& path_str, std::size_t line_number) const {
        return pipeline_io::parse_decimal_float(
            text, path_str + ":" + std::to_string(line_number) + ":2");
    }
    /**
     * @brief Validate a sample CSV and visit its expression records.
     * @param sample_index Index in the ordered sample-path list.
     * @param on_record Callback receiving gene, finite value, and
     *     physical line.
     * @throws std::runtime_error If header, records, identifiers, or
     *     I/O fails.
     */
    template <typename RecordHandler>
    void parse_file_records(
        std::size_t sample_index, RecordHandler&& on_record)
    const {
        const std::string& path_str = sample_paths_[sample_index];
        const auto lines =
        pipeline_io::read_lines(pipeline_io::path_from_utf8(path_str));
        const auto header = pipeline_io::csv_fields(lines.front(), path_str
            + ":1");
        if (header != std::vector<std::string>{"gene", "expression"} ||
            lines.size() < 2) {
            throw std::runtime_error(path_str
                + ": expected gene,expression and data");
        }
        std::unordered_set<std::string> seen_genes;
        for (std::size_t i = 1; i < lines.size(); ++i) {
            const auto source = path_str + ":" + std::to_string(i + 1);
            const auto fields = pipeline_io::csv_fields(lines[i], source);
            if (fields.size() != 2) {
                throw std::runtime_error(source
                    + ": expected exactly two CSV fields");
            }
            pipeline_io::validate_identifier(fields[0], source + ":1");
            if (!seen_genes.insert(fields[0]).second) {
                throw std::runtime_error(std::format("{}: duplicate gene {}",
                    source, fields[0]));
            }
            on_record(fields[0], parse_finite_double(fields[1], path_str, i +
                1), i + 1);
        }
    }
};
/**
 * @brief Constraint divisive hierarchical clustering engine for gene
 *     expression data.
 *
 * Implements a constraint divisive clustering algorithm leveraging a
 *     packed, lower-triangular 1D buffer for caching pairwise distance
 *     evaluations. Incremental statistics (e.g., partial distance sums)
 *     are updated dynamically to significantly reduce the algorithmic
 *     complexity of repeated recalculations during iterative cluster
 *     splitting.
 */
class ClusteringEngine {
public:
    /**
     * @brief Initialize immutable-tree state and packed distance
     *     storage.
     * @param dataset Validated sample dataset transferred into the
     *     engine.
     * @throws std::length_error If distance or node storage overflows.
     */
    explicit ClusteringEngine(ExpressionDataset dataset)
    : dataset_(std::move(dataset)) {
        const std::size_t n = dataset_.size();
        const std::size_t max_value = std::numeric_limits<std::size_t>::max();
        if (n > max_value / 2 + 1) {
            throw std::length_error(
                "Immutable tree node identifiers would overflow size_t.");
        }
        cur_clustering_.assign(n, 0);
        sum_dists_.assign(n, 0.0);
        dists_to_splinter_.assign(n, 0.0);
        counts_.assign(1, n);
        cluster_max_distance_.assign(1,
            -std::numeric_limits<double>::infinity());
        cluster_max_pair_.assign(1, std::nullopt);
        cluster_members_.assign(1, std::vector<std::size_t>(n));
        std::iota(cluster_members_[0].begin(), cluster_members_[0].end(), 0);
        // Working cluster slot 0 initially represents immutable tree
        // root 0. Working slots are reused internally when a source
        // cluster is split; immutable node IDs are never reused.
        active_tree_node_ids_.reserve(n);
        active_tree_node_ids_.push_back(kRootTreeNodeId);
        split_history_.reserve(n > 0 ? n - 1 : 0);
        const std::size_t pair_count = checked_triangular_size(n);
        distance_matrix_.resize(pair_count);
    }
    /**
     * @brief Compute all pairwise distances under the selected metric.
     * @param metric One supported dissimilarity definition.
     * @throws std::runtime_error If a numerical result is nonfinite.
     */
    void compute_distances(DistanceMetric metric) {
        switch (metric) {
        case DistanceMetric::Euclidean: compute_euclidean();
            break;
        case DistanceMetric::Pearson: compute_pearson();
            break;
        case DistanceMetric::Spearman: compute_spearman();
            break;
        case DistanceMetric::Manhattan: compute_manhattan();
            break;
        default:
            throw std::logic_error(std::format(
                "Unhandled DistanceMetric value: {}",
                static_cast<int>(metric)));
        }
        initialize_cluster_statistics();
    }
    /**
     * @brief Split active clusters to singleton leaves and write
     *     snapshots.
     * @param output_prefix Destination filename prefix, including
     *     directory separator.
     * @throws std::exception If an invariant or output operation fails.
     */
    void run_clustering(const std::string& output_prefix) {
        const std::size_t n = dataset_.size();
        if (n <= 1) {
            export_tree_splits(output_prefix);
            export_sample_paths(output_prefix);
            return;
        }
        while (max_cluster_ < n - 1) {
            split_next_cluster();
            const std::string out_name =
            std::format("{}iter{}.txt", output_prefix, max_cluster_);
            // cur_clustering_ contains reusable working-slot indices.
            // Persist the immutable tree-node ID represented by each
            // active slot instead, so a parent ID disappears
            // permanently after its split.
            write_atomically(out_name, [this](std::ofstream& output_stream) {
                    for (std::size_t i = 0; i < cur_clustering_.size(); ++i) {
                        const std::size_t working_slot = cur_clustering_[i];
                        if (working_slot >= active_tree_node_ids_.size()) {
                            throw std::logic_error(
                            "Internal clustering error: working slot has no "
                            "tree node ID.");
                        }
                        output_stream << active_tree_node_ids_[working_slot]
                        << (i + 1 == cur_clustering_.size() ? "" : " ");
                    }
                    output_stream << '\n';
                });
            std::cerr << std::format(
                "Clustering iteration number {} completed.\n", max_cluster_);
        }
        export_tree_splits(output_prefix);
        export_sample_paths(output_prefix);
    }
private:
    using PairKey = std::size_t;
    static constexpr std::size_t kRootTreeNodeId = 0;
    /** @brief One immutable parent-to-two-children split. */
    struct TreeSplit {
        std::size_t iteration;
        std::size_t parent;
        std::size_t retained_child;
        std::size_t extracted_child;
    };
    ExpressionDataset dataset_;
    std::vector<double> distance_matrix_;
    // Per-sample working-cluster slots. These remain dense
    // 0..max_cluster_ so the existing clustering algorithm and caches
    // keep their original layout. They are internal implementation
    // details and are not persisted as hierarchy node identifiers.
    std::vector<std::size_t> cur_clustering_;
    std::size_t max_cluster_ = 0;
    // Maps each active working slot to the immutable tree node that
    // currently represents it. Root is 0. Every split retires one
    // parent and allocates two fresh child IDs from next_tree_node_id_.
    std::vector<std::size_t> active_tree_node_ids_;
    std::size_t next_tree_node_id_ = 1;
    std::vector<TreeSplit> split_history_;
    bool first_must_group_transfer_ = true;
    bool first_cannot_group_transfer_ = true;
    // Incrementally maintained statistics.
    std::vector<double> sum_dists_;
    std::vector<double> dists_to_splinter_;
    std::vector<std::size_t> counts_;
    // Members of each cluster, always kept in ascending object-index
    // order. split_next_cluster() and recompute_cluster_max() scan this
    // instead of the whole dataset, so their cost is bounded by the
    // current cluster's size rather than by n. Ascending order is
    // preserved by construction (insert/erase both use
    // std::lower_bound), which defines deterministic tie-breaking
    // order.
    std::vector<std::vector<std::size_t>> cluster_members_;
    // Cached maximum intra-cluster distance for each cluster. A cache
    // entry is recomputed only when the previous maximizing pair was
    // split out.
    std::vector<double> cluster_max_distance_;
    std::vector<std::optional<PairKey>> cluster_max_pair_;
    /**
     * @brief Compute the packed off-diagonal matrix length safely.
     * @param n Number of samples.
     * @return N(N - 1)/2 entries.
     * @throws std::length_error If the checked product overflows
     *     size_t.
     */
    static std::size_t checked_triangular_size(std::size_t n) {
        if (n < 2) return 0;
        const std::size_t max_value = std::numeric_limits<std::size_t>::max();
        if (n > max_value / (n - 1)) {
            throw std::length_error("Distance matrix size overflows size_t.");
        }
        const std::size_t product = n * (n - 1);
        return product / 2;
    }
    // Precondition: i != j, except i == j == 0, which happens to
    // evaluate to 0 via unsigned wraparound (0 * SIZE_MAX == 0) — do
    // not call with i == j > 0.
    /**
     * @brief Map distinct sample indices into lower-triangular storage.
     * @param i First valid sample index.
     * @param j Distinct second sample index.
     * @return Packed index; callers must exclude diagonal pairs.
     */
    [[nodiscard]] static std::size_t pair_index(std::size_t i, std::size_t j)
    noexcept {
        if (i < j) std::swap(i, j);
        return i * (i - 1) / 2 + j;
    }
    /**
     * @brief Recover the two sample indices from a packed pair key.
     * @param key Valid off-diagonal packed index.
     * @return Row and column indices in the distance matrix.
     */
    [[nodiscard]] std::pair<std::size_t, std::size_t> decode_pair(PairKey key)
    const noexcept {
        // Find the smallest row r such that r(r-1)/2 > key. This path
        // is used only for cached maxima, not in numerical hot loops.
        std::size_t low = 1;
        std::size_t high = dataset_.size();
        while (low < high) {
            const std::size_t mid = low + (high - low) / 2;
            const std::size_t start = mid * (mid - 1) / 2;
            if (start <= key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        const std::size_t row = low - 1;
        const std::size_t row_start = row * (row - 1) / 2;
        return {row, key - row_start};
    }
    /**
     * @brief Return a stored distance, or zero for a diagonal pair.
     * @param i First valid sample index.
     * @param j Second valid sample index.
     * @return Pairwise dissimilarity.
     */
    [[nodiscard]] double get_dist(std::size_t i, std::size_t j) const noexcept
    {
        if (i == j) return 0.0;
        return distance_matrix_[pair_index(i, j)];
    }
    /**
     * @brief Test whether two group IDs form the primary forbidden
     *     pair.
     * @param group_a First constraint-group ID.
     * @param group_b Second constraint-group ID.
     * @return Whether the groups are 1 and 2 in either order.
     */
    static bool is_cannot_link_pair(int group_a, int group_b) noexcept {
        return group_a + group_b == 3;
    }
    /**
     * @brief Store the distance for one off-diagonal sample pair.
     * @param i First valid sample index.
     * @param j Distinct second sample index.
     * @param value Finite computed distance.
     */
    void set_distance(std::size_t i, std::size_t j, double value) noexcept {
        distance_matrix_[pair_index(i, j)] = value;
    }
    // Write to a sibling temporary file, then replace the destination
    // atomically. POSIX rename() replaces an existing destination
    // atomically; Windows needs MoveFileExW because
    // std::filesystem::rename() does not replace one there.
    /**
     * @brief Publish a fully written and checked native output.
     * @param final_path Output pathname to replace.
     * @param write_body Callback writing the complete staged content.
     * @throws std::runtime_error If writing, closure, or replacement
     *     fails.
     */
    static void write_atomically(const fs::path& final_path,
        const std::function<void(std::ofstream&)>& write_body) {
        pipeline_io::checked_write(final_path, write_body);
    }
    /**
     * @brief Fill pairwise Euclidean distances.
     * @throws std::runtime_error If a distance is nonfinite.
     */
    void compute_euclidean() {
        const std::size_t n = dataset_.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto expr_i = dataset_.expressions(i);
            for (std::size_t j = 0; j < i; ++j) {
                const auto expr_j = dataset_.expressions(j);
                double dist_sq = 0.0;
                for (std::size_t k = 0; k < dataset_.gene_count(); ++k) {
                    const double diff = expr_i[k] - expr_j[k];
                    dist_sq += diff * diff;
                }
                const double distance = std::sqrt(dist_sq);
                if (!std::isfinite(distance)) {
                    throw std::runtime_error(
                        "Euclidean distance computation produced a "
                        "non-finite value.");
                }
                set_distance(i, j, distance);
            }
            std::cerr << std::format(
                "Distances for sample {} have been computed.\n", i + 1);
        }
    }
    /**
     * @brief Fill one-minus-Pearson-correlation distances.
     * @throws std::runtime_error If a computed correlation is
     *     nonfinite.
     */
    void compute_pearson() {
        const std::size_t n = dataset_.size();
        const std::size_t genes = dataset_.gene_count();
        std::vector<double> avg(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const auto expr = dataset_.expressions(i);
            avg[i] = std::accumulate(expr.begin(), expr.end(), 0.0)
            / static_cast<double>(genes);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto expr_i = dataset_.expressions(i);
            for (std::size_t j = 0; j < i; ++j) {
                const auto expr_j = dataset_.expressions(j);
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;
                for (std::size_t k = 0; k < genes; ++k) {
                    const double dev_i = expr_i[k] - avg[i];
                    const double dev_j = expr_j[k] - avg[j];
                    sum1 += dev_i * dev_j;
                    sum2 += dev_i * dev_i;
                    sum3 += dev_j * dev_j;
                }
                if (sum2 <= 0.0 || sum3 <= 0.0) {
                    set_distance(i, j, 2.0);
                } else {
                    const double correlation =
                    sum1 / (std::sqrt(sum2) * std::sqrt(sum3));
                    if (!std::isfinite(correlation)) {
                        throw std::runtime_error(
                            "Pearson distance computation produced a "
                            "non-finite value.");
                    }
                    set_distance(i, j, 1.0 - std::clamp(correlation, -1.0,
                        1.0));
                }
            }
            std::cerr << std::format(
                "Distances for sample {} have been computed.\n", i + 1);
        }
    }
    /**
     * @brief Rank expression values, assigning average ranks to ties.
     * @param values One finite sample expression vector.
     * @return Ranks aligned with the original gene order.
     */
    static std::vector<double> compute_average_ranks(std::span<const double>
        values) {
        const std::size_t total_genes = values.size();
        std::vector<std::size_t> order(total_genes);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t
            b) {
                return values[a] < values[b];
            });
        std::vector<double> ranks(total_genes);
        std::size_t k = 0;
        while (k < total_genes) {
            std::size_t j = k;
            while (j + 1 < total_genes && values[order[j + 1]] ==
                values[order[k]]) {
                ++j;
            }
            const double avg_rank = (static_cast<double>(k + j) + 2.0) / 2.0;
            for (std::size_t t = k; t <= j; ++t) {
                ranks[order[t]] = avg_rank;
            }
            k = j + 1;
        }
        return ranks;
    }
    /**
     * @brief Fill one-minus-rank-correlation distances.
     * @throws std::runtime_error If a computed correlation is
     *     nonfinite.
     */
    void compute_spearman() {
        const std::size_t n = dataset_.size();
        const std::size_t genes = dataset_.gene_count();
        if (n != 0 && genes > std::numeric_limits<std::size_t>::max() / n) {
            throw
            std::length_error("Spearman rank matrix size overflows size_t.");
        }
        // A single contiguous allocation avoids one heap allocation per
        // sample.
        std::vector<double> rank_matrix(n * genes);
        for (std::size_t i = 0; i < n; ++i) {
            const auto ranks = compute_average_ranks(dataset_.expressions(i));
            std::copy(ranks.begin(), ranks.end(),
                rank_matrix.begin() + static_cast<std::ptrdiff_t>(i * genes));
        }
        std::vector<double> avg(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const auto row = std::span<const double>(rank_matrix.data() + i *
                genes, genes);
            avg[i] = std::accumulate(row.begin(), row.end(), 0.0)
            / static_cast<double>(genes);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto rank_i = std::span<const double>(rank_matrix.data() + i
             * genes, genes);
            for (std::size_t j = 0; j < i; ++j) {
                const auto rank_j = std::span<const double>(rank_matrix.data()
                    + j * genes, genes);
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;
                for (std::size_t k = 0; k < genes; ++k) {
                    const double dev_i = rank_i[k] - avg[i];
                    const double dev_j = rank_j[k] - avg[j];
                    sum1 += dev_i * dev_j;
                    sum2 += dev_i * dev_i;
                    sum3 += dev_j * dev_j;
                }
                if (sum2 <= 0.0 || sum3 <= 0.0) {
                    set_distance(i, j, 2.0);
                } else {
                    const double correlation =
                    sum1 / (std::sqrt(sum2) * std::sqrt(sum3));
                    if (!std::isfinite(correlation)) {
                        throw std::runtime_error(
                            "Spearman distance computation produced a "
                            "non-finite value.");
                    }
                    set_distance(i, j, 1.0 - std::clamp(correlation, -1.0,
                        1.0));
                }
            }
            std::cerr << std::format(
                "Distances for sample {} have been computed.\n", i + 1);
        }
    }
    /**
     * @brief Fill pairwise Manhattan distances.
     * @throws std::runtime_error If a distance is nonfinite.
     */
    void compute_manhattan() {
        const std::size_t n = dataset_.size();
        const std::size_t genes = dataset_.gene_count();
        for (std::size_t i = 0; i < n; ++i) {
            const auto expr_i = dataset_.expressions(i);
            for (std::size_t j = 0; j < i; ++j) {
                const auto expr_j = dataset_.expressions(j);
                double sum_abs = 0.0;
                for (std::size_t k = 0; k < genes; ++k) {
                    sum_abs += std::fabs(expr_i[k] - expr_j[k]);
                }
                if (!std::isfinite(sum_abs)) {
                    throw std::runtime_error(
                        "Manhattan distance computation produced a "
                        "non-finite value.");
                }
                set_distance(i, j, sum_abs);
            }
            std::cerr << std::format(
                "Distances for sample {} have been computed.\n", i + 1);
        }
    }
    /**
     * @brief Initialize per-sample sums and active-cluster extrema.
     */
    void initialize_cluster_statistics() {
        std::fill(sum_dists_.begin(), sum_dists_.end(), 0.0);
        if (dataset_.size() < 2) {
            cluster_max_distance_[0] =
            -std::numeric_limits<double>::infinity();
            cluster_max_pair_[0] = std::nullopt;
            return;
        }
        double max_distance = -std::numeric_limits<double>::infinity();
        std::optional<PairKey> max_pair;
        for (std::size_t i = 0; i < dataset_.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                const double d = get_dist(i, j);
                sum_dists_[i] += d;
                sum_dists_[j] += d;
                if (d > max_distance) {
                    max_distance = d;
                    max_pair = pair_index(i, j);
                }
            }
        }
        cluster_max_distance_[0] = max_distance;
        cluster_max_pair_[0] = max_pair;
    }
    /**
     * @brief Test whether both members of a pair occupy one working
     *     slot.
     * @param key Packed off-diagonal pair key.
     * @param cluster Working-cluster slot.
     * @return Whether both sample assignments equal the selected slot.
     */
    [[nodiscard]] bool pair_is_in_cluster(PairKey key, std::size_t cluster)
    const noexcept {
        const auto [u, v] = decode_pair(key);
        return cur_clustering_[u] == cluster && cur_clustering_[v] == cluster;
    }
    /**
     * @brief Recompute the diameter cache for one active cluster.
     * @param cluster Working-cluster slot.
     */
    void recompute_cluster_max(std::size_t cluster) {
        double max_distance = -std::numeric_limits<double>::infinity();
        std::optional<PairKey> max_pair;
        // cluster_members_ is kept in ascending object-index order, so
        // this reproduces the same tie-breaking behavior as scanning
        // the whole dataset in order (i increases, and j < i increases)
        // while only visiting this cluster's own members.
        const auto& members = cluster_members_[cluster];
        for (std::size_t pos = 0; pos < members.size(); ++pos) {
            const std::size_t i = members[pos];
            for (std::size_t prev = 0; prev < pos; ++prev) {
                const std::size_t j = members[prev];
                const double d = get_dist(i, j);
                if (d > max_distance) {
                    max_distance = d;
                    max_pair = pair_index(i, j);
                }
            }
        }
        cluster_max_distance_[cluster] = max_distance;
        cluster_max_pair_[cluster] = max_pair;
    }
    /**
     * @brief Invalidate a diameter cache after extracting a maximizing
     *     sample.
     * @param cluster Working-cluster slot.
     * @param extracted Sample index extracted from the source cluster.
     */
    void invalidate_cluster_max_if_needed(std::size_t cluster, std::size_t
        extracted) {
        if (!cluster_max_pair_[cluster].has_value()) return;
        const auto [u, v] = decode_pair(*cluster_max_pair_[cluster]);
        if (u == extracted || v == extracted) {
            cluster_max_pair_[cluster] = std::nullopt;
            cluster_max_distance_[cluster] =
            -std::numeric_limits<double>::infinity();
        }
    }
    // Incremental statistics reduce repeated work in typical runs, but
    // the worst-case total split work remains O(n^3) for a highly
    // unbalanced hierarchy.
    /**
     * @brief Apply one constrained divisive split and allocate fresh
     *     node IDs.
     * @throws std::exception If a split would violate a clustering
     *     invariant.
     */
    void split_next_cluster() {
        std::size_t cluster_to_take = 0;
        double max_dist = -std::numeric_limits<double>::infinity();
        for (std::size_t cluster = 0; cluster <= max_cluster_; ++cluster) {
            if (cluster_max_pair_[cluster].has_value() &&
                cluster_max_distance_[cluster] > max_dist) {
                max_dist = cluster_max_distance_[cluster];
                cluster_to_take = cluster;
            }
        }
        if (counts_[cluster_to_take] < 2) {
            throw std::runtime_error(
                "Internal clustering error: selected cluster is "
                "not splittable.");
        }
        if (cluster_to_take >= active_tree_node_ids_.size()) {
            throw std::logic_error(
                "Internal clustering error: source working slot "
                "has no tree node ID.");
        }
        const std::size_t parent_tree_node_id =
        active_tree_node_ids_[cluster_to_take];
        bool is_from_first_folder = false;
        bool is_from_second_folder = false;
        for (const std::size_t i : cluster_members_[cluster_to_take]) {
            const int f = dataset_.constraint_group(i);
            if (f == 1) is_from_first_folder = true;
            if (f == 2) is_from_second_folder = true;
        }
        const bool constraint = is_from_first_folder && is_from_second_folder;
        double max_sum_dist = -std::numeric_limits<double>::infinity();
        std::optional<std::size_t> element_to_extract;
        for (const std::size_t i : cluster_members_[cluster_to_take]) {
            if (constraint && dataset_.constraint_group(i) > 2) continue;
            if (sum_dists_[i] > max_sum_dist) {
                max_sum_dist = sum_dists_[i];
                element_to_extract = i;
            }
        }
        if (!element_to_extract.has_value()) {
            throw std::runtime_error(
                "Internal clustering error: no splittable "
                "element found in the "
                "selected cluster.");
        }
        ++max_cluster_;
        counts_.push_back(0);
        cluster_max_distance_.push_back(
            -std::numeric_limits<double>::infinity());
        cluster_max_pair_.push_back(std::nullopt);
        cluster_members_.push_back({});
        cluster_members_[max_cluster_].reserve(counts_[cluster_to_take]);
        // Elements extracted this round, in extraction order. This is
        // kept separately from cluster_members_ (which stays sorted by
        // object index) because the new cluster's initial cached
        // maximum below is computed in extraction order.
        std::vector<std::size_t> splinter_members;
        splinter_members.reserve(counts_[cluster_to_take]);
        auto extract_element = [&](std::size_t element) {
            if (cur_clustering_[element] != cluster_to_take) return;
            invalidate_cluster_max_if_needed(cluster_to_take, element);
            auto& source_members = cluster_members_[cluster_to_take];
            source_members.erase(
                std::lower_bound(source_members.begin(), source_members.end(),
                element));
            --counts_[cluster_to_take];
            ++counts_[max_cluster_];
            cur_clustering_[element] = max_cluster_;
            // Remove the extracted element's contribution from the
            // remaining source-cluster sums, and accumulate its
            // distance into dists_to_splinter_. This scans only the
            // node_means
            // still in the source cluster (source_members, just shrunk
            // above), not every element in the dataset.
            for (const std::size_t i : source_members) {
                const double d = get_dist(i, element);
                sum_dists_[i] -= d;
                dists_to_splinter_[i] += d;
            }
            // Keep sum_dists_ exact for the new cluster as well. They
            // are maintained incrementally to avoid that O(N^2)
            // rebuild.
            double new_cluster_sum = 0.0;
            for (const std::size_t member : splinter_members) {
                const double d = get_dist(member, element);
                sum_dists_[member] += d;
                new_cluster_sum += d;
            }
            sum_dists_[element] = new_cluster_sum;
            auto& dest_members = cluster_members_[max_cluster_];
            dest_members.insert(
                std::lower_bound(dest_members.begin(), dest_members.end(),
                element), element);
            splinter_members.push_back(element);
        };
        const std::size_t selected_idx = *element_to_extract;
        const int selected_group = dataset_.constraint_group(selected_idx);
        const bool cluster_has_non_must_link = std::any_of(
            cluster_members_[cluster_to_take].begin(),
            cluster_members_[cluster_to_take].end(),
            [this](std::size_t i) {
                return dataset_.constraint_group(i) != 3;
            });
        if (selected_group == 3 && first_must_group_transfer_ &&
            cluster_has_non_must_link) {
            // Snapshot before extracting: extract_element() mutates
            // cluster_members_[cluster_to_take], which would invalidate
            // an in-progress iteration over that same container.
            const auto snapshot = cluster_members_[cluster_to_take];
            for (const std::size_t i : snapshot) {
                if (i == selected_idx) continue;
                if (cur_clustering_[i] != cluster_to_take) continue;
                if (dataset_.constraint_group(i) == 3) {
                    extract_element(i);
                }
            }
            first_must_group_transfer_ = false;
        }
        if (selected_group <= 2 && first_cannot_group_transfer_) {
            const auto snapshot = cluster_members_[cluster_to_take];
            for (const std::size_t i : snapshot) {
                if (i == selected_idx) continue;
                if (cur_clustering_[i] != cluster_to_take) continue;
                if (dataset_.constraint_group(i) == selected_group) {
                    extract_element(i);
                }
            }
            first_cannot_group_transfer_ = false;
        }
        std::optional<std::size_t> current_extract = selected_idx;
        const std::size_t element_to_extract_old = selected_idx;
        while (current_extract.has_value()) {
            // Every iteration creates a non-empty cluster and leaves
            // the source non-empty, so every run ends with singleton
            // clusters, even though samples may have identical
            // transcription values.
            if (counts_[cluster_to_take] <= 1) break;
            extract_element(*current_extract);
            double max_diff = 0.0;
            current_extract = std::nullopt;
            const double source_count =
            static_cast<double>(counts_[cluster_to_take]);
            const double splinter_count =
            static_cast<double>(counts_[max_cluster_]);
            for (const std::size_t i : cluster_members_[cluster_to_take]) {
                if (is_cannot_link_pair(
                    dataset_.constraint_group(i),
                    dataset_.constraint_group(element_to_extract_old))) {
                    continue;
                }
                const double cur_diff =
                (sum_dists_[i] / source_count) -
                (dists_to_splinter_[i] / splinter_count);
                if (cur_diff > max_diff) {
                    max_diff = cur_diff;
                    current_extract = i;
                }
            }
        }
        // The extracted node_means are finalized in the new cluster.
        // Update the new cluster's cached maximum incrementally,
        // avoiding a full O(N^2) scan.
        double new_max_distance = -std::numeric_limits<double>::infinity();
        std::optional<PairKey> new_max_pair;
        for (std::size_t pos = 0; pos < splinter_members.size(); ++pos) {
            const std::size_t i = splinter_members[pos];
            for (std::size_t prev = 0; prev < pos; ++prev) {
                const std::size_t j = splinter_members[prev];
                const double d = get_dist(i, j);
                if (d > new_max_distance) {
                    new_max_distance = d;
                    new_max_pair = pair_index(i, j);
                }
            }
        }
        cluster_max_distance_[max_cluster_] = new_max_distance;
        cluster_max_pair_[max_cluster_] = new_max_pair;
        // The source cluster's cached maximum remains valid unless one
        // endpoint of the previous maximum pair was extracted.
        // Recompute only in that case.
        if (!cluster_max_pair_[cluster_to_take].has_value() &&
            counts_[cluster_to_take] >= 2) {
            recompute_cluster_max(cluster_to_take);
        } else if (counts_[cluster_to_take] < 2) {
            cluster_max_distance_[cluster_to_take] =
            -std::numeric_limits<double>::infinity();
            cluster_max_pair_[cluster_to_take] = std::nullopt;
        }
        if (counts_[cluster_to_take] == 0 || counts_[max_cluster_] == 0) {
            throw std::logic_error(
                "Internal clustering error: immutable tree "
                "split produced an empty child.");
        }
        if (next_tree_node_id_ > std::numeric_limits<std::size_t>::max() - 1) {
            throw std::length_error(
                "Immutable tree node identifiers overflow size_t.");
        }
        // Retire the immutable parent and assign fresh IDs to both
        // children. The retained child continues to use the source
        // working slot; the extracted child uses the newly appended
        // working slot. Neither child ID can ever be reused by a later
        // split.
        const std::size_t retained_child = next_tree_node_id_++;
        const std::size_t extracted_child = next_tree_node_id_++;
        active_tree_node_ids_[cluster_to_take] = retained_child;
        if (active_tree_node_ids_.size() != max_cluster_) {
            throw std::logic_error(
                "Internal clustering error: "
                "working-slot/tree-node maps are out of sync.");
        }
        active_tree_node_ids_.push_back(extracted_child);
        split_history_.push_back(TreeSplit{
                max_cluster_,
                parent_tree_node_id,
                retained_child,
                extracted_child,
            });
        // The next iteration starts with a fresh destination-cluster
        // accumulator.
        std::fill(dists_to_splinter_.begin(), dists_to_splinter_.end(), 0.0);
    }
    /**
     * @brief Write the explicit root and immutable split records.
     * @param output_prefix Output filename prefix including its
     *     separator.
     * @throws std::runtime_error If publication fails.
     */
    void export_tree_splits(const std::string& output_prefix) const {
        const std::string split_out = std::format(
            "{}{}", output_prefix, kTreeTopologyFile);
        write_atomically(split_out, [this](std::ofstream& output_stream) {
            // Self-describing topology format. Root 0 exists even for a
            // single-sample dataset, where there are no split records.
                output_stream << "root " << kRootTreeNodeId << '\n';
                for (const TreeSplit& split : split_history_) {
                    output_stream << "split "
                    << split.iteration << ' '
                    << split.parent << ' '
                    << split.retained_child << ' '
                    << split.extracted_child << '\n';
                }
            });
    }
    /**
     * @brief Write literal sample paths in assignment-vector order.
     * @param output_prefix Output filename prefix including its
     *     separator.
     * @throws std::runtime_error If publication fails.
     */
    void export_sample_paths(const std::string& output_prefix) const {
        const std::string fn_out = std::format(
            "{}{}", output_prefix, kSamplePathsFile);
        write_atomically(fn_out, [this](std::ofstream& output_stream) {
                for (const auto& fn : dataset_.sample_paths()) {
                    output_stream << fn << '\n';
                }
            });
    }
};
} // namespace
/**
 * @brief Execute the native stage with provenance and contextual
 *     errors.
 * @param argc Number of command-line arguments.
 * @param argv Program path and stage-specific arguments.
 * @return Zero on success; one for usage, validation, or execution
 *     failure.
 */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << std::format(
            "Usage: {} <output_filename_prefix> <distance_metric>\n", argv[0]);
        std::cerr <<
        "Distance metrics:\n  1 - Euclidean\n  2 - "
        "Pearson\n  3 - Spearman\n  4 - Manhattan\n";
        return 1;
    }
    try {
        const std::string metric_arg = argv[2];
        std::size_t parsed_chars = 0;
        int metric_val = 0;
        try {
            metric_val = std::stoi(metric_arg, &parsed_chars);
        } catch (const std::exception&) {
            throw std::invalid_argument(std::format(
                "Distance metric must be an integer, got "
                "\"{}\".", metric_arg));
        }
        if (parsed_chars != metric_arg.size()) {
            throw std::invalid_argument(std::format(
                "Unexpected trailing characters in distance "
                "metric argument: \"{}\".",
                metric_arg));
        }
        if (metric_val < 1 || metric_val > 4) {
            throw std::invalid_argument(std::format(
                "Unsupported distance metric identifier: {}", metric_arg));
        }
        const auto metric = static_cast<DistanceMetric>(metric_val);
        ExpressionDataset dataset;
        dataset.load_dataset_config();
        dataset.parse_all_files();
        std::vector<fs::path> inputs;
        std::vector<fs::path> parents;
        for (const auto* config :
            {kCannotLinksFile, kMustLinksFile, kNoLinksFile}) {
            inputs.emplace_back(config);
            parents.emplace_back(std::string(config) + ".provenance");
            for (const auto& folder : pipeline_io::read_lines(config, true,
                true)) {
                if (folder.empty() || folder.find_first_not_of(" \t") ==
                    std::string::npos) continue;
                inputs.emplace_back(folder + "_parsed_files.txt");
                parents.emplace_back(folder + "_parsed_files.txt.provenance");
            }
        }
        const auto sample_count = dataset.size();
        for (const auto& path : dataset.sample_paths())
        inputs.push_back(pipeline_io::path_from_utf8(path));
        const std::string prefix = argv[1];
        const fs::path commit = prefix + kClusteringProvenanceFile;
        const auto directory = fs::absolute(commit).parent_path();
        pipeline_io::invalidate_dependents({commit}, {directory,
                directory.parent_path()});
        if (fs::is_regular_file(commit)) {
            const auto previous = pipeline_io::read_provenance(commit);
            fs::remove(commit);
            for (const auto& [name, digest] : previous.outputs) {
                const auto old = directory / pipeline_io::path_from_utf8(name);
                fs::remove(old);  // removes a symbolic link, not its target
            }
        }
        ClusteringEngine engine(std::move(dataset));
        engine.compute_distances(metric);
        engine.run_clustering(argv[1]);
        std::vector<fs::path> outputs{prefix + kSamplePathsFile, prefix
            + kTreeTopologyFile};
        for (std::size_t i = 1; i < sample_count; ++i) {
            outputs.emplace_back(prefix + "iter" + std::to_string(i) + ".txt");
        }
        pipeline_io::write_provenance(commit, "clustering", outputs, inputs,
            parents, argv[0]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << std::format("Error: [clustering] {}\n", e.what());
        return 1;
    }
}
