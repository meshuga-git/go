/**
 * @file compute_variances.cpp
 * @note Requires C++20.
 * @brief Unweighted hierarchy variances for single-gene comparison
 *     terms.
 *
 * Read committed genes-by-samples integer data and sample lineages.
 *     Publish term_variances.txt and its provenance. Equality and
 *     strict
 *     inequality terms use the established operator codes. Requires
 *     C++20.
 */
#include "pipeline_files.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <version>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
using pipeline_files::kMatrixGenesFile;
using pipeline_files::kMatrixSamplesFile;
using pipeline_files::kMatrixValuesFile;
using pipeline_files::kPreparedProvenanceFile;
using pipeline_files::kSampleLineagesFile;
using pipeline_files::kTermVariancesFile;

using pipeline_files::kPrepareMarkerFile;

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
namespace {
// Term kinds: samples where a gene's value is == / < / > a threshold.
/** @brief Stable comparison codes written to variance records. */
enum class TermOperator : int {
    Equal = 0,
    LessThan = 1,
    GreaterThan = 2
};
/**
 * @brief One gene index, integer threshold, and comparison operator.
 */
struct Term {
    std::size_t gene_index;
    int value;
    TermOperator type;
};
// Owns all data associated with one loaded pipeline dataset.
//
// Keeping this state together makes dependencies explicit and allows a
// caller to construct a fresh VarianceDataset object when restarting
// from
// a pipeline step, without relying on process-global state.
//
// Each read_*_file() function parses into local containers and only
// replaces the corresponding members here once the entire input file
// has been read and validated, so a failed reload leaves a previously
// loaded dataset untouched instead of partially overwritten.
/** @brief Validated expression data and derived comparison terms. */
struct VarianceDataset {
    std::unordered_map<std::string, int> gene_indices;
    std::unordered_map<std::string, int> sample_indices;
    std::vector<std::string> gene_ids;
    std::vector<std::string> sample_ids;
    std::vector<std::vector<int>> values_matrix;
    std::vector<std::vector<int>> unique_values;
    std::vector<std::vector<int>> sample_lineages;
    std::vector<Term> terms;
};
// Owns reusable scratch storage for variance calculations.
//
// The workspace is explicitly tied to a dataset's sample/node count
// instead of being hidden in function-local static variables. It can
// therefore be discarded and reconstructed whenever a pipeline step is
// restarted.
/** @brief Reusable scratch state for one loaded dataset shape. */
struct VarianceWorkspace {
    explicit VarianceWorkspace(std::size_t sample_count);
    std::vector<std::size_t> term_sample_indices;
    std::vector<double> node_means;
    std::vector<double> sample_node_counts;
    // Tracks which entries of node_means/sample_node_counts the most
    // recent
    // call to compute_term_variance touched, so that function can reset
    // just those entries instead of the full node_count-sized vectors,
    // and can restrict its variance summation to nodes that can
    // possibly have a nonzero contribution. See compute_term_variance
    // for details.
    std::vector<bool> node_touched;
    std::vector<std::size_t> active_nodes;
};
// A full binary hierarchy with N leaves has exactly 2N - 1 immutable
// nodes. Node 0 is the root. Split k (1-based) creates sibling nodes 2k
// - 1 and 2k. Hierarchy IDs are stored as int in sample_lineages.txt,
// so
// reject dataset sizes whose largest immutable node ID could not be
// represented by that contract.
/**
 * @brief Check the integer contract and compute 2N - 1 nodes.
 * @param sample_count Positive number of sample leaves.
 * @return Full binary-tree node count.
 * @throws std::runtime_error If sample count violates the ID bounds.
 */
[[nodiscard]] std::size_t tree_node_count(std::size_t sample_count) {
    if (sample_count == 0) {
        throw std::runtime_error(
            "Cannot construct an immutable hierarchy for an empty sample set");
    }
    const std::size_t max_int =
    static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (sample_count - 1 > max_int / 2) {
        throw std::runtime_error(
            "Sample count is too large for 2N-1 immutable hierarchy node IDs "
            "stored as 32-bit int values");
    }
    return 2 * sample_count - 1;
}
// Outcome of read_strict_int_token(). Missing vs. Invalid vs.
// OutOfRange are kept distinct so callers can report a malformed field
// differently from a field that is simply absent (e.g. "fewer than N
// columns" for a genuinely short row, but a specific parse error for a
// row that has enough fields, one of which isn't a valid integer).
/** @brief Outcome of parsing a canonical integer token. */
enum class TokenParseResult {
    Success,
    Missing,
    Invalid,
    OutOfRange
};
// Reads one whitespace-delimited token from `stream` and, if present,
// strictly converts it to an int: the token must be consumed entirely
// by the conversion (no trailing characters, e.g. "12xyz" and "1 2" are
// rejected, not silently truncated) and must fit in `int`.
//
// Returns Missing without touching `out`/`bad_token` when the stream
// has no more tokens on the current line, so a caller can distinguish
// "the row is too short" from "the row has enough fields, but one of
// them is invalid". On Invalid or OutOfRange, `bad_token` (if non-null)
// receives the offending token so the caller can include it in an error
// message.
/**
 * @brief Read one complete canonical integer with checked narrowing.
 * @param stream Current physical-row token stream.
 * @param out Destination, changed only on success.
 * @param bad_token Optional destination for an invalid token.
 * @return Success, Missing, Invalid, or OutOfRange.
 */
[[nodiscard]] TokenParseResult read_strict_int_token(
    std::istringstream& stream, int& out, std::string* bad_token) {
    std::string token;
    if (!(stream >> token)) return TokenParseResult::Missing;
    if (bad_token != nullptr) *bad_token = token;
    if (!pipeline_io::canonical_integer(token)) return
    TokenParseResult::Invalid;
    int result{};
    const auto [end, error] = std::from_chars(
        token.data(), token.data() + token.size(), result);
    if (error == std::errc::result_out_of_range) return
    TokenParseResult::OutOfRange;
    if (error != std::errc{} || end != token.data() + token.size()) return
    TokenParseResult::Invalid;
    out = result;
    return TokenParseResult::Success;
}
// Reads gene names from `genes_file` into
// data.gene_indices/data.gene_ids.
//
// Gene names may not contain whitespace: see contains_whitespace() for
// why.
/**
 * @brief Load unique literal gene names in expression-row order.
 * @param filename Gene identifier file.
 * @param data Dataset updated only after successful validation.
 * @throws std::runtime_error If names, duplicates, encoding, or I/O
 *     fail.
 */
void read_genes_file(const std::string& filename, VarianceDataset& data) {
    const auto lines = pipeline_io::read_lines(filename);
    std::unordered_map<std::string, int> mapping;
    if (lines.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Too many gene identifiers");
    }
    for (std::size_t i = 0; i < lines.size(); ++i) {
        pipeline_io::validate_identifier(lines[i], filename + ":" +
            std::to_string(i + 1));
        if (!mapping.emplace(lines[i], static_cast<int>(i)).second) {
            throw std::runtime_error("Duplicate gene identifier: " + lines[i]);
        }
    }
    data.gene_indices = std::move(mapping);
    data.gene_ids = lines;
}
// Reads sample/column names from `columns_file` into
// data.sample_indices/data.sample_ids.
//
// Sample names may not contain whitespace: see contains_whitespace()
// for why -- in particular, sample_lineages.txt identifies a sample by
// matching the first whitespace-delimited token on each row against
// this list.
/**
 * @brief Load unique sample IDs in expression-column order.
 * @param filename Sample identifier file.
 * @param data Dataset updated only after successful validation.
 * @throws std::runtime_error If names, duplicates, encoding, or I/O
 *     fail.
 */
void read_samples_file(const std::string& filename, VarianceDataset& data) {
    const auto lines = pipeline_io::read_lines(filename);
    std::unordered_map<std::string, int> mapping;
    if (lines.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Too many sample identifiers");
    }
    for (std::size_t i = 0; i < lines.size(); ++i) {
        pipeline_io::validate_identifier(lines[i], filename + ":" +
            std::to_string(i + 1));
        if (!mapping.emplace(lines[i], static_cast<int>(i)).second) {
            throw std::runtime_error("Duplicate sample identifier: " +
                lines[i]);
        }
    }
    data.sample_indices = std::move(mapping);
    data.sample_ids = lines;
}
// Reads the genes x samples integer matrix from `values_file`.
//
// Strict by construction: requires exactly data.gene_ids.size() rows,
// each with exactly data.sample_ids.size() whitespace-separated
// integers and no trailing tokens. A short/misaligned file throws
// instead of silently leaving missing entries at their
// default-constructed 0. Each field must be a complete, in-range
// integer token -- a malformed field (e.g. "xyz") or an out-of-range
// one is reported distinctly from a genuinely missing one, rather than
// both being reported as "fewer than N columns".
/**
 * @brief Validate and load the rectangular integer expression matrix.
 * @param values_file One physical row per retained gene.
 * @param data Dataset containing validated gene and sample lists.
 * @throws std::runtime_error If a token, range, dimension, or read
 *     fails.
 */
void read_values_file(const std::string& values_file, VarianceDataset& data) {
    std::ifstream myfile(values_file, std::ios::binary);
    if (!myfile.is_open()) {
        throw std::runtime_error("Unable to open values file: " + values_file);
    }
    std::vector<std::vector<int>> values_matrix(
        data.gene_ids.size(), std::vector<int>(data.sample_ids.size()));
    std::string line;
    for (std::size_t i = 0; i < data.gene_ids.size(); ++i) {
        if (!std::getline(myfile, line)) {
            throw
                std::runtime_error(std::format(
                    "values file '{}' has fewer than {} rows (one "
                "expected per gene; failed at row {})", values_file,
                    data.gene_ids.size(), i + 1));
        }
        if (i == 0 && line.starts_with("\xef\xbb\xbf")) line.erase(0, 3);
        if (!myfile.eof() && !line.empty() && line.back() == '\r')
        line.pop_back();
        pipeline_io::unicode_values(line);
        if (line.find('\r') != std::string::npos) throw
        std::runtime_error("Lone CR in matrix");
        std::istringstream row_stream(line);
        for (std::size_t j = 0; j < data.sample_ids.size(); ++j) {
            std::string bad_token;
            const TokenParseResult result =
            read_strict_int_token(row_stream, values_matrix[i][j], &bad_token);
            switch (result) {
            case TokenParseResult::Missing:
                throw
                    std::runtime_error(std::format(
                        "values file '{}' row {} ('{}') has fewer than "
                    "{} columns", values_file, i + 1, data.gene_ids[i],
                        data.sample_ids.size()));
            case TokenParseResult::Invalid:
                throw
                    std::runtime_error(std::format(
                        "values file '{}' row {} ('{}') column {} is "
                    "not a valid integer: '{}'", values_file, i + 1,
                        data.gene_ids[i], j + 1, bad_token));
            case TokenParseResult::OutOfRange:
                throw
                    std::runtime_error(std::format(
                        "values file '{}' row {} ('{}') column {} is "
                    "out of range for a 32-bit int: '{}'", values_file, i + 1,
                        data.gene_ids[i], j + 1, bad_token));
            case TokenParseResult::Success:
                break;
            }
        }
        std::string extra;
        if (row_stream >> extra) {
            throw
                std::runtime_error(std::format(
                    "values file '{}' row {} ('{}') has more than "
                "{} columns", values_file, i +
                1, data.gene_ids[i], data.sample_ids.size()));
        }
    }
    while (std::getline(myfile, line)) {
        if (true) {
            throw
                std::runtime_error(std::format(
                    "values file '{}' has more than {} rows (one "
                "expected per gene)", values_file, data.gene_ids.size()));
        }
    }
    if (!myfile.eof() && myfile.fail()) {
        throw std::runtime_error("Error while reading values file: " +
            values_file);
    }
    data.values_matrix = std::move(values_matrix);
}
// Reads the sample lineages from `lineages_file`.
//
// Expected format: exactly one physical line per sample, in any order:
//
// <sample_name> <count> <id_0> <id_1> ... <id_count-1>
//
// Each sample_name must match an entry from matrix_samples.txt. For N
// samples, each id must be an immutable tree-node ID in [0, 2N-1). Each
// of <count> and every <id> must be a complete, in-range integer token;
// a malformed or out-of-range one is reported distinctly from a
// genuinely missing one.
//
// Physical line boundaries are significant. A row declaring N ids must
// contain exactly N ids on that same physical line. Missing ids cannot
// be taken from the next sample's row, and extra tokens on a row are
// rejected.
//
// Rows are matched to samples by name rather than by physical order.
/**
 * @brief Load node memberships by explicit sample identity.
 * @param lineages_file Sample ID, membership count, and node IDs per
 *     row.
 * @param data Dataset updated after complete membership validation.
 * @throws std::runtime_error If identities, counts, nodes, or I/O fail.
 */
void read_sample_lineages(const std::string& lineages_file, VarianceDataset&
    data) {
    const auto physical_lines = pipeline_io::read_lines(lineages_file);
    std::ostringstream serialized;
    for (const auto& row : physical_lines) serialized << row << '\n';
    std::istringstream myfile(serialized.str());
    const std::size_t sample_count = data.sample_ids.size();
    const std::size_t node_count = tree_node_count(sample_count);
    std::vector<std::vector<int>> sample_lineages(sample_count);
    std::vector<bool> seen_sample(sample_count, false);
    std::vector<bool> seen_node(node_count, false);
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(myfile, line)) {
        ++line_number;
        // Every physical line is supposed to represent one sample.
        if (line.find_first_not_of(" \t") == std::string::npos) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' contains an empty row at "
                "line {}", lineages_file, line_number));
        }
        std::istringstream row_stream(line);
        std::string sample_name;
        if (!(row_stream >> sample_name)) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' has a missing sample name "
                "at line {}", lineages_file, line_number));
        }
        const auto sample_it = data.sample_indices.find(sample_name);
        if (sample_it == data.sample_indices.end()) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} names unknown "
                "sample '{}' (not found in matrix_samples.txt)",
                lineages_file,
                    line_number, sample_name));
        }
        const std::size_t sample_index =
        static_cast<std::size_t>(sample_it->second);
        if (seen_sample[sample_index]) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' contains more than one row "
                "for sample '{}' (duplicate at line {})", lineages_file,
                    sample_name, line_number));
        }
        int count = 0;
        std::string bad_token;
        const TokenParseResult count_result =
        read_strict_int_token(row_stream, count, &bad_token);
        if (count_result == TokenParseResult::Missing) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} has a missing "
                "count for sample '{}'", lineages_file, line_number,
                    sample_name));
        }
        if (count_result == TokenParseResult::Invalid) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} has an invalid "
                "count for sample '{}': '{}'", lineages_file, line_number,
                    sample_name, bad_token));
        }
        if (count_result == TokenParseResult::OutOfRange) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} has a count out of "
                "range for a 32-bit int for sample '{}': '{}'",
                    lineages_file, line_number, sample_name, bad_token));
        }
        if (count < 0) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} has a negative "
                "count for sample '{}'", lineages_file, line_number,
                    sample_name));
        }
        if (count == 0) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} declares an empty "
                "lineage for sample '{}'; every immutable "
                "lineage must include root node 0", lineages_file,
                    line_number, sample_name));
        }
        // A root-to-leaf path in a full binary tree with N leaves can
        // contain at most N distinct nodes (root plus at most N-1
        // successive splits).
        if (static_cast<std::size_t>(count) > sample_count) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} declares {} ids "
                "for sample '{}', which exceeds the maximum "
                "immutable lineage length for {} samples", lineages_file,
                    line_number, count, sample_name, sample_count));
        }
        std::vector<int> row;
        row.reserve(static_cast<std::size_t>(count));
        std::set<int> seen_id;
        for (int j = 0; j < count; ++j) {
            int id = 0;
            const TokenParseResult id_result =
            read_strict_int_token(row_stream, id, &bad_token);
            if (id_result == TokenParseResult::Missing) {
                throw
                    std::runtime_error(std::format(
                        "lineage file '{}' line {} declares {} ids "
                    "for sample '{}' but fewer were found on that "
                    "physical line", lineages_file, line_number, count,
                        sample_name));
            }
            if (id_result == TokenParseResult::Invalid) {
                throw
                    std::runtime_error(std::format(
                        "lineage file '{}' line {} for sample '{}' "
                    "has an invalid id: '{}'", lineages_file, line_number,
                        sample_name, bad_token));
            }
            if (id_result == TokenParseResult::OutOfRange) {
                throw
                    std::runtime_error(std::format(
                        "lineage file '{}' line {} for sample '{}' "
                    "has an id out of range for a 32-bit int: '{}'",
                        lineages_file, line_number, sample_name, bad_token));
            }
            if (id < 0 || static_cast<std::size_t>(id) >= node_count) {
                throw
                    std::runtime_error(std::format(
                        "lineage file '{}' line {} for sample '{}' "
                    "contains out-of-range immutable node id {} "
                    "(valid range is [0, {}))", lineages_file, line_number,
                        sample_name, id, node_count));
            }
            if (!seen_id.insert(id).second) {
                throw
                    std::runtime_error(std::format(
                        "lineage file '{}' line {} for sample '{}' "
                    "contains duplicate id {}", lineages_file, line_number,
                        sample_name, id));
            }
            seen_node[static_cast<std::size_t>(id)] = true;
            row.push_back(id);
        }
        if (seen_id.find(0) == seen_id.end()) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} for sample '{}' "
                "does not contain immutable root node 0", lineages_file,
                    line_number, sample_name));
        }
        // Enforce the exact physical-row field count. Anything after
        // the declared number of ids makes this row malformed.
        std::string extra;
        if (row_stream >> extra) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' line {} for sample '{}' "
                "contains extra field '{}' after the declared "
                "{} ids", lineages_file, line_number, sample_name, extra,
                    count));
        }
        sample_lineages[sample_index] = std::move(row);
        seen_sample[sample_index] = true;
    }
    if (!myfile.eof() && myfile.fail()) {
        throw std::runtime_error("Error while reading lineage file: " +
            lineages_file);
    }
    for (std::size_t i = 0; i < sample_count; ++i) {
        if (!seen_sample[i]) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' is missing a row for "
                "sample '{}'", lineages_file, data.sample_ids[i]));
        }
    }
    // The immutable Python preparation stage requires the union of all
    // sample
    // lineages to contain every node 0..2N-2. Recheck that invariant
    // here so an old mutable hierarchy or a partially written immutable
    // hierarchy is rejected at the native boundary rather than
    // producing plausible output.
    for (std::size_t node = 0; node < node_count; ++node) {
        if (!seen_node[node]) {
            throw
                std::runtime_error(std::format(
                    "lineage file '{}' does not represent a "
                "complete immutable tree: node {} is absent "
                "(expected all node IDs 0..{})", lineages_file, node,
                    node_count - 1));
        }
    }
    data.sample_lineages = std::move(sample_lineages);
}
// Computes, per gene, the sorted list of distinct values appearing in
// that gene's row.
/**
 * @brief Collect sorted distinct expression thresholds for every gene.
 * @param data Dataset whose unique_values field is replaced.
 */
void compute_unique_values(VarianceDataset& data) {
    data.unique_values.clear();
    data.unique_values.reserve(data.values_matrix.size());
    for (const auto& row : data.values_matrix) {
        const std::set<int> distinct(row.begin(), row.end());
        data.unique_values.emplace_back(distinct.begin(), distinct.end());
    }
}
// Generates, per gene and per distinct value, an "equal" term,
// plus every nonempty strict "less than" / "greater than" term.
// The minimum threshold is excluded for "<" and the maximum threshold
// for ">" because those two strict comparisons would cover no samples.
/**
 * @brief Generate equality and nonempty strict-comparison terms.
 * @param data Dataset with per-gene distinct thresholds; terms are
 *     replaced.
 */
void generate_terms(VarianceDataset& data) {
    data.terms.clear();
    std::size_t term_count = 0;
    for (const auto& vals : data.unique_values) {
        term_count += vals.size();
        if (!vals.empty()) {
            term_count += 2 * (vals.size() - 1);
        }
    }
    data.terms.reserve(term_count);
    for (std::size_t i = 0; i < data.unique_values.size(); ++i) {
        const auto& vals = data.unique_values[i];
        for (std::size_t j = 0; j < vals.size(); ++j) {
            data.terms.push_back({i, vals[j], TermOperator::Equal});
            if (j > 0) {
                data.terms.push_back({i, vals[j], TermOperator::LessThan});
            }
            if (j + 1 < vals.size()) {
                data.terms.push_back({i, vals[j], TermOperator::GreaterThan});
            }
        }
    }
}
/**
 * @brief Allocate scratch arrays for one hierarchy shape.
 * @param sample_count Number of samples defining the node domain.
 * @throws std::exception If the node domain or allocation fails.
 */
VarianceWorkspace::VarianceWorkspace(std::size_t sample_count) {
    const std::size_t node_count = tree_node_count(sample_count);
    node_means.assign(node_count, 0.0);
    sample_node_counts.assign(node_count, 0.0);
    node_touched.assign(node_count, false);
}
// Computes the sum of unweighted population variances (ddof=0) across
// hierarchy nodes for a term's covered samples.
//
// For term s with element set E and hierarchy-node membership
// counts x_i,k:
//
// node_means[k] = mean over i in E of x_i,k
//
// result = sum_k ((1/|E|) * sum_{i in E} (x_i,k - node_means[k])^2)
//
// Performance note: a node that is never touched by any sample of this
// term necessarily has node_means[k] == 0 and, for every sample,
// sample_node_counts[k] == 0 -- it can never contribute a nonzero term
// to the
// sum below. workspace.active_nodes tracks exactly the set of touched
// nodes (in
// ascending order, matching the iteration order a full 0..node_count
// scan would visit them in, so the floating-point summation order --
// and therefore the result -- is unchanged), so both the per-call reset
// and the per-sample variance loop only need to visit that set instead
// of every node in the dataset.
/**
 * @brief Compute unweighted population variance for one covered subset.
 * @param term_index Index of the comparison term to evaluate.
 * @param data Validated expressions, memberships, and terms.
 * @param workspace Mutable scratch state sized for this dataset.
 * @return Sum of squared-deviation population variances over nodes.
 * @throws std::runtime_error If term index or workspace shape is
 *     invalid.
 */
[[nodiscard]] double compute_term_variance(
    std::size_t term_index, const VarianceDataset& data, VarianceWorkspace&
    workspace) {
    if (term_index >= data.terms.size()) {
        throw std::runtime_error("Term index is out of range");
    }
    const std::size_t node_count =
    tree_node_count(data.sample_ids.size());
    // A workspace belongs to a dataset shape. Reject accidental reuse
    // after loading a dataset with a different number of samples/nodes.
    if (workspace.node_means.size() != node_count ||
        workspace.sample_node_counts.size() != node_count ||
        workspace.node_touched.size() != node_count) {
        throw std::runtime_error("Variance workspace size does not match the "
            "loaded dataset");
    }
    workspace.term_sample_indices.clear();
    // Undo exactly what the previous call touched, rather than clearing
    // the full node_count-sized scratch vectors. Driven by
    // workspace.active_nodes rather than an assumption that the
    // previous call completed normally, this is self-healing if
    // execution restarts after a previous call was interrupted (e.g. by
    // an exception).
    for (const std::size_t node : workspace.active_nodes) {
        workspace.node_means[node] = 0.0;
        workspace.sample_node_counts[node] = 0.0;
        workspace.node_touched[node] = false;
    }
    workspace.active_nodes.clear();
    const Term& term = data.terms[term_index];
    const std::size_t gene_index = term.gene_index;
    const int value = term.value;
    const TermOperator type = term.type;
    for (std::size_t k = 0; k < data.values_matrix[gene_index].size(); ++k) {
        const int current_value = data.values_matrix[gene_index][k];
        const bool matches =
        (type == TermOperator::Equal && current_value == value) ||
        (type == TermOperator::LessThan && current_value < value) ||
        (type == TermOperator::GreaterThan && current_value > value);
        if (matches) {
            workspace.term_sample_indices.push_back(k);
        }
    }
    // A term covering zero or one sample has zero variance.
    if (workspace.term_sample_indices.size() <= 1) {
        return 0.0;
    }
    // Calculate average element counts, recording which nodes are
    // touched by any sample of this term.
    for (const std::size_t sample_index : workspace.term_sample_indices) {
        for (const int node : data.sample_lineages[sample_index]) {
            const std::size_t node_index = static_cast<std::size_t>(node);
            if (!workspace.node_touched[node_index]) {
                workspace.node_touched[node_index] = true;
                workspace.active_nodes.push_back(node_index);
            }
            workspace.node_means[node_index] += 1.0;
        }
    }
    // Visiting active_nodes in ascending order keeps the floating-point
    // operations below in the same order a full 0..node_count scan
    // would produce, so this restriction changes performance but not
    // the result.
    std::sort(workspace.active_nodes.begin(), workspace.active_nodes.end());
    const double covered_sample_count =
    static_cast<double>(workspace.term_sample_indices.size());
    for (const std::size_t node : workspace.active_nodes) {
        workspace.node_means[node] /= covered_sample_count;
    }
    // Calculate unweighted variance directly from squared deviations.
    double total_sq = 0.0;
    for (const std::size_t sample_index : workspace.term_sample_indices) {
        double cur_sq = 0.0;
        const auto& node_ids = data.sample_lineages[sample_index];
        for (const int node : node_ids) {
            workspace.sample_node_counts[
                static_cast<std::size_t>(node)] += 1.0;
        }
        for (const std::size_t k : workspace.active_nodes) {
            if (workspace.sample_node_counts[k] != workspace.node_means[k]) {
                const double diff = workspace.sample_node_counts[k] -
                workspace.node_means[k];
                cur_sq += diff * diff;
            }
        }
        for (const int node : node_ids) {
            workspace.sample_node_counts[
                static_cast<std::size_t>(node)] -= 1.0;
        }
        total_sq += cur_sq;
    }
    return total_sq / covered_sample_count;
}
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
    static_cast<void>(argc);
    const std::string output_file = kTermVariancesFile;
    const std::filesystem::path commit = kPreparedProvenanceFile;
    try {
        if (std::filesystem::exists(
            kPrepareMarkerFile)) {
            throw std::runtime_error("Interrupted prepare stage");
        }
        const auto committed = pipeline_io::verify_provenance(commit);
        for (const auto* name : {kMatrixGenesFile, kMatrixSamplesFile,
                kMatrixValuesFile, kSampleLineagesFile}) {
            if (!committed.outputs.contains(name)) {
                throw
                std::runtime_error("Prepared commit lacks required input");
            }
        }
        VarianceDataset data;
        read_genes_file(kMatrixGenesFile, data);
        read_samples_file(kMatrixSamplesFile, data);
        read_values_file(kMatrixValuesFile, data);
        read_sample_lineages(kSampleLineagesFile, data);
        compute_unique_values(data);
        generate_terms(data);
        VarianceWorkspace workspace(data.sample_ids.size());
        const auto directory = std::filesystem::current_path();
        const auto provenance = directory /
            (std::string(kTermVariancesFile) + ".provenance");
        pipeline_io::invalidate_dependents(
            {directory / output_file, provenance}, {directory,
                directory.parent_path()});
        std::filesystem::remove(provenance);
        pipeline_io::checked_write(output_file, [&](std::ofstream& output) {
                output <<
                std::setprecision(std::numeric_limits<double>::max_digits10);
                for (std::size_t i = 0; i < data.terms.size(); ++i) {
                    if ((i + 1) % 10000 == 0) std::cerr << i + 1 << '\n';
                    const double result = compute_term_variance(i, data,
                    workspace);
                    const Term& term = data.terms[i];
                    output << data.gene_ids[term.gene_index] << ' ' <<
                    term.value << ' '
                    << static_cast<int>(term.type) << ' ' << (result == 0 ?
                    0.0 : result) << '\n';
                }
            });
        pipeline_io::write_provenance(provenance, "variances", {output_file},
            {kMatrixGenesFile, kMatrixSamplesFile, kMatrixValuesFile,
            kSampleLineagesFile},
            {commit}, argv[0]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: [variances] " << error.what() << '\n';
        return 1;
    }
}
