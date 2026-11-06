/**
 * @file reconstruct_hierarchy.cpp
 * @note Requires C++20.
 * @brief Immutable-tree hierarchy-history reconstruction.
 *
 * Read a committed clustering output set. Publish total_data.csv and
 *     its provenance beside the inputs. The history is
 *     genes-independent and sample rows are lexicographically ordered.
 *     Requires C++20.
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
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
using pipeline_files::kClusteringProvenanceFile;
using pipeline_files::kTreeStatesFile;
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
    const auto [end, error] = std::from_chars(
        token.data(), token.data() + token.size(), result);
    if (error != std::errc{} || end != token.data() + token.size() ||
        !std::isfinite(result)) {
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
                if (fs::is_symlink(output)) {
                    throw std::runtime_error("Refusing symlink cleanup");
                }
                dirty.insert(fs::weakly_canonical(output));
                fs::remove(output);
            }
            it = records.erase(it);
            progress = true;
        }
    }
}
} // namespace pipeline_io
namespace {
namespace fs = std::filesystem;
// Immutable-tree input/output contract:
// * filenames.txt contains N non-empty sample names.
// * tree_splits.txt contains `root 0` followed by exactly N - 1
// records: split <iteration> <parent> <retained_child>
// <extracted_child>
// * Split k allocates immutable node IDs 2k - 1 and 2k. The parent is
// retired permanently; both child IDs are fresh and are never reused.
// * iter1.txt through iter(N-1).txt must exist. Each file contains
// exactly N immutable IDs identifying the active tree node for every
// sample after that split.
// * total_data.csv stores N hierarchy states after the filename: state
// 0 is the explicit root (0), followed by states after splits 1..N-1.
// * Samples are ordered lexicographically by their complete
// immutable-node history. Within each state column, repeated values in
// adjacent sorted rows are run-length-compressed as empty CSV fields.
//
// The history matrix intentionally uses O(N^2) storage. This is a
// design constraint of the current algorithm and should be considered
// when choosing the maximum practical sample count. See
// max_practical_sample_count() for the runtime ceiling this places on
// N.
using SampleIndex = std::size_t;
using TreeNodeId = std::uint32_t;
// Immutable tree-node history for every sample, stored in one
// contiguous row-major buffer. For N samples, each row contains N
// states: state 0 is the explicit root and states 1..N-1 are the
// post-split snapshots.
//
// Accessor bounds are checked in every build configuration.
/** @brief Owned row-major sample-by-state immutable node matrix. */
struct SampleHistories {
    std::unique_ptr<TreeNodeId[]> data;
    std::size_t sample_count = 0;
    std::size_t state_count = 0;
    // Return a non-owning view of one sample's history. std::span makes
    // the range and its extent explicit. An empty history is handled
    // explicitly because a zero-length allocation does not contain an
    // array element.
    //
    // Precondition: sample < sample_count, enforced below (not just via
    // assert()) so that an out-of-range index fails loudly in every
    // build configuration instead of performing undefined pointer
    // arithmetic in release builds.
    /**
     * @brief Return one immutable-node history without copying.
     * @param sample Zero-based sample index.
     * @return A non-owning view of exactly state_count node IDs.
     * @throws std::out_of_range If sample is outside the matrix.
     */
    [[nodiscard]] std::span<const TreeNodeId> row(SampleIndex sample) const {
        if (sample >= sample_count) {
            throw
                std::out_of_range(std::format(
                    "SampleHistories::row: sample index {} is out "
                "of range for {} samples", sample, sample_count));
        }
        if (state_count == 0) {
            return {};
        }
        return {data.get() + sample * state_count, state_count};
    }
};
// Quote a CSV field when required and double any embedded quotes. This
// follows the conventional RFC 4180 CSV escaping rules. Takes a
// string_view since this function only reads the field and never needs
// to own or extend its lifetime.
/**
 * @brief Encode one CSV field using double-quote escaping.
 * @param field Literal filename or path.
 * @return A field with doubled embedded quotes and required quoting.
 */
std::string escape_csv_field(std::string_view field) {
    // If the field contains a comma, double‑quote, newline, or carriage
    // return, it must be enclosed in double‑quotes and any embedded
    // quotes doubled.
    if (field.find_first_of(",\"\n\r") == std::string_view::npos) {
        return std::string(field);
    }
    std::string result;
    result.reserve(field.size() + 2);
    result.push_back('"');
    for (char ch : field) {
        if (ch == '"') result.push_back('"');
        result.push_back(ch);
    }
    result.push_back('"');
    return result;
}
/**
 * @brief Read ordered literal sample paths with unique sample IDs.
 * @param sample_paths_file Ordered sample-path list.
 * @return Paths in physical input order.
 * @throws std::runtime_error If transport, identity, or I/O is invalid.
 */
[[nodiscard]] std::vector<std::string> read_sample_paths(const
    std::filesystem::path& sample_paths_file) {
    auto sample_paths = pipeline_io::read_lines(sample_paths_file);
    std::unordered_set<std::string> ids;
    for (const auto& name : sample_paths) {
        const auto separator = name.find_last_of("/\\");
        const auto id = name.substr(separator == std::string::npos ? 0 :
            separator + 1);
        pipeline_io::validate_identifier(id, sample_paths_file.string());
        if (!ids.insert(id).second) throw
        std::runtime_error("Duplicate sample ID: " + id);
    }
    return sample_paths;
}
/** @brief One immutable parent-to-two-children split. */
struct TreeSplit {
    std::size_t iteration = 0;
    TreeNodeId parent = 0;
    TreeNodeId retained_child = 0;
    TreeNodeId extracted_child = 0;
};
/** @brief Validated root, ordered splits, and immutable node count. */
struct TreeTopology {
    TreeNodeId root = 0;
    std::vector<TreeSplit> splits;
    std::size_t node_count = 0;
};
/**
 * @brief Validate the complete immutable split topology.
 * @param topology_file Root and ordered split records.
 * @param sample_count Number of terminal sample leaves.
 * @return Validated topology with N - 1 splits and 2N - 1 nodes.
 * @throws std::runtime_error If a record or allocation invariant fails.
 */
[[nodiscard]] TreeTopology read_tree_topology(
    const std::filesystem::path& topology_file,
    const std::size_t sample_count) {
    if (sample_count == 0) {
        throw std::invalid_argument(
            "read_tree_topology requires a non-zero sample count");
    }
    if (sample_count >
        (static_cast<std::size_t>(std::numeric_limits<TreeNodeId>::max()) + 2)
        / 2) {
        throw std::length_error(
            "Sample count is too large for 32-bit immutable tree-node IDs");
    }
    const std::size_t expected_node_count = 2 * sample_count - 1;
    const std::size_t expected_split_count = sample_count - 1;
    const auto physical_lines = pipeline_io::read_lines(topology_file);
    std::ostringstream buffer;
    for (const auto& record : physical_lines) buffer << record << '\n';
    std::istringstream input(buffer.str());
    std::string line;
    std::size_t line_number = 0;
    if (!std::getline(input, line)) {
        throw std::runtime_error(
            "Tree topology file is empty: " + topology_file.string());
    }
    ++line_number;
    std::istringstream root_stream(line);
    std::string root_keyword;
    unsigned long long root_value{};
    std::string extra;
    if (!(root_stream >> root_keyword >> root_value) ||
        root_keyword != "root" || (root_stream >> extra)) {
        throw std::runtime_error(
            "Invalid root record at line 1 of " + topology_file.string() +
            "; expected: root 0");
    }
    if (pipeline_io::tokens(physical_lines[0]) !=
        std::vector<std::string>{"root", "0"}) {
        throw std::runtime_error("Topology root must use canonical root 0");
    }
    if (root_value != 0) {
        throw std::runtime_error(
            "Immutable tree root must have node ID 0 in " +
            topology_file.string());
    }
    TreeTopology topology;
    topology.root = 0;
    topology.node_count = expected_node_count;
    topology.splits.reserve(expected_split_count);
    std::vector<bool> seen(expected_node_count, false);
    std::vector<bool> active(expected_node_count, false);
    seen[0] = true;
    active[0] = true;
    std::size_t active_count = 1;
    for (std::size_t expected_iteration = 1;
        expected_iteration <= expected_split_count;
        ++expected_iteration) {
        if (!std::getline(input, line)) {
            throw
                std::runtime_error(std::format(
                    "Tree topology file contains too few split "
                "records: {} (expected {})", topology_file.string(),
                    expected_split_count));
        }
        ++line_number;
        std::istringstream split_stream(line);
        std::string keyword;
        unsigned long long iteration_value{};
        unsigned long long parent_value{};
        unsigned long long retained_value{};
        unsigned long long extracted_value{};
        extra.clear();
        if (!(split_stream >> keyword >> iteration_value >> parent_value >>
            retained_value >> extracted_value) ||
            keyword != "split" || (split_stream >> extra)) {
            throw
                std::runtime_error(std::format(
                    "Invalid split record at line {} of {}", line_number,
                    topology_file.string()));
        }
        const auto strict_fields = pipeline_io::tokens(line);
        for (std::size_t i = 1; i < strict_fields.size(); ++i) {
            pipeline_io::parse_integer<unsigned long long>(strict_fields[i],
                topology_file.string());
        }
        if (iteration_value != expected_iteration) {
            throw
                std::runtime_error(std::format(
                    "Tree topology split iteration out of sequence "
                "at line {}: expected {}, found {}", line_number,
                    expected_iteration, iteration_value));
        }
        const std::size_t expected_retained = 2 * expected_iteration - 1;
        const std::size_t expected_extracted = 2 * expected_iteration;
        if (retained_value != expected_retained ||
            extracted_value != expected_extracted) {
            throw
                std::runtime_error(std::format(
                    "Tree topology violates immutable node "
                "allocation at split {}: expected children {} "
                "and {}", expected_iteration, expected_retained,
                    expected_extracted));
        }
        if (parent_value >= expected_node_count ||
            retained_value >= expected_node_count ||
            extracted_value >= expected_node_count) {
            throw std::runtime_error(
                "Tree topology contains a node ID outside valid range 0.." +
                std::to_string(expected_node_count - 1));
        }
        const auto parent = static_cast<TreeNodeId>(parent_value);
        const auto retained = static_cast<TreeNodeId>(retained_value);
        const auto extracted = static_cast<TreeNodeId>(extracted_value);
        if (!active[parent]) {
            throw
                std::runtime_error(std::format(
                    "Tree topology attempts to split inactive node "
                "{} at iteration {}", parent, expected_iteration));
        }
        if (seen[retained] || seen[extracted] || retained == extracted) {
            throw std::runtime_error(
                "Tree topology reuses an immutable child node "
                "ID at iteration " +
                std::to_string(expected_iteration));
        }
        active[parent] = false;
        --active_count;
        seen[retained] = true;
        seen[extracted] = true;
        active[retained] = true;
        active[extracted] = true;
        active_count += 2;
        topology.splits.push_back(TreeSplit{
                expected_iteration, parent, retained, extracted});
    }
    while (std::getline(input, line)) {
        ++line_number;
        if (true) {
            throw
                std::runtime_error(std::format(
                    "Tree topology file contains unexpected extra "
                "content at line {}: {}", line_number,
                    topology_file.string()));
        }
    }
    if (!input.eof() && input.fail()) {
        throw std::runtime_error(
            "Error while reading tree topology file: " +
            topology_file.string());
    }
    if (active_count != sample_count) {
        throw
            std::runtime_error(std::format(
                "Tree topology ended with {} active nodes; "
            "expected one leaf per sample ({})", active_count, sample_count));
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::runtime_error(
            "Tree topology did not allocate every expected immutable node ID");
    }
    return topology;
}
/**
 * @brief Load one snapshot and validate its declared split.
 * @param iteration_file Assignment snapshot to read.
 * @param sample_count Required number of assignments.
 * @param state_index Destination history-state column.
 * @param state_count Number of columns in each sample history.
 * @param split Expected parent and fresh child nodes.
 * @param flat Mutable row-major history matrix.
 * @throws std::runtime_error If a value, count, or transition is
 *     invalid.
 */
void read_clustering_iteration(
    const std::filesystem::path& iteration_file,
    const std::size_t sample_count,
    const std::size_t state_index,
    const std::size_t state_count,
    const TreeSplit& split,
    std::span<TreeNodeId> flat) {
    const auto physical_lines = pipeline_io::read_lines(iteration_file);
    std::ostringstream buffer;
    for (const auto& record : physical_lines) buffer << record << '\n';
    std::istringstream input(buffer.str());
    std::size_t retained_count = 0;
    std::size_t extracted_count = 0;
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        std::string value_token;
        if (!(input >> value_token)) {
            if (input.eof()) {
                throw
                    std::runtime_error(std::format(
                        "Iteration file contains too few values: {} "
                    "(expected {})", iteration_file.string(), sample_count));
            }
            throw std::runtime_error(
                "Invalid tree-node value in iteration file: " +
                iteration_file.string());
        }
        const auto value = pipeline_io::parse_integer<unsigned long long>(
            value_token, iteration_file.string());
        if (!std::in_range<TreeNodeId>(value)) {
            throw std::runtime_error(
                "Tree-node value out of range in iteration file: " +
                iteration_file.string());
        }
        const TreeNodeId current = static_cast<TreeNodeId>(value);
        const TreeNodeId previous = flat[sample * state_count + state_index -
        1];
        if (previous == split.parent) {
            if (current == split.retained_child) {
                ++retained_count;
            } else if (current == split.extracted_child) {
                ++extracted_count;
            } else {
                throw
                    std::runtime_error(std::format(
                        "Iteration {} assigns a sample from parent node "
                    "{} to unexpected node {} in {}", split.iteration,
                        split.parent, current, iteration_file.string()));
            }
        } else if (current != previous) {
            throw
                std::runtime_error(std::format(
                    "Iteration {} changes a sample outside split "
                "parent node {} from node {} to node {} in {}",
                    split.iteration, split.parent, previous, current,
                    iteration_file.string()));
        }
        flat[sample * state_count + state_index] = current;
    }
    std::string extra_token;
    if (input >> extra_token) {
        throw
            std::runtime_error(std::format(
                "Iteration file contains more values than "
            "expected: {} (expected {})", iteration_file.string(),
                sample_count));
    }
    if (!input.eof() && input.fail()) {
        throw std::runtime_error(
            "Error while reading iteration file: " + iteration_file.string());
    }
    if (retained_count == 0 || extracted_count == 0) {
        throw
            std::runtime_error(std::format(
                "Iteration {} does not leave both immutable "
            "child nodes non-empty (retained={}, "
            "extracted={})", split.iteration, retained_count,
                extracted_count));
    }
}
// Practical upper bound on sample count, independent of the pure
// integer- overflow check in build_histories(). The O(N^2) history
// matrix means memory usage grows quadratically with N: at 100,000
// samples the matrix alone is already on the order of tens of
// gigabytes. This ceiling exists purely to fail fast with a clear,
// actionable message instead of letting the process exhaust system
// memory (or swap thrash) on an input that is technically valid but
// impractically large.
//
// Deployments with a larger memory budget can raise the limit via the
// CLUSTER_HISTORY_MAX_SAMPLES environment variable; a malformed
// override is ignored in favor of the default rather than failing the
// whole run.
/**
 * @brief Read the configured history-matrix resource ceiling.
 * @return Maximum samples; malformed overrides use the default.
 */
[[nodiscard]] std::size_t max_practical_sample_count() {
    constexpr std::size_t kDefaultMaxSamples = 100'000;
    if (const char* override_value =
        std::getenv("CLUSTER_HISTORY_MAX_SAMPLES")) {
        try {
            return std::stoull(override_value);
        } catch (const std::exception&) {
            // Fall through to the default below.
        }
    }
    return kDefaultMaxSamples;
}
/**
 * @brief Construct and validate the complete N-by-N node history.
 * @param directory Directory containing committed clustering outputs.
 * @param sample_paths Ordered sample paths.
 * @param topology Validated immutable tree.
 * @return Row-major history matrix, including explicit root state 0.
 * @throws std::exception If I/O, topology, or allocation limits fail.
 */
[[nodiscard]] SampleHistories build_histories(
    const std::filesystem::path& directory,
    const std::vector<std::string>& sample_paths,
    const TreeTopology& topology) {
    const std::size_t sample_count = sample_paths.size();
    if (sample_count == 0) {
        throw std::invalid_argument(
            "build_histories requires a non-empty sample_paths list");
    }
    if (topology.splits.size() != sample_count - 1 ||
        topology.node_count != 2 * sample_count - 1 ||
        topology.root != 0) {
        throw std::invalid_argument(
            "build_histories received a topology "
            "inconsistent with sample count");
    }
    if (const std::size_t max_samples = max_practical_sample_count();
        sample_count > max_samples) {
        throw
            std::length_error(std::format(
                "Sample count {} exceeds the practical limit of "
            "{} samples (history matrix memory usage is "
            "O(N^2)); set CLUSTER_HISTORY_MAX_SAMPLES to "
            "override if you have the memory budget for a "
            "larger run.", sample_count, max_samples));
    }
    // The immutable model stores one explicit root state plus N - 1
    // post-split states, so N samples yield exactly N hierarchy-state
    // columns.
    const std::size_t state_count = sample_count;
    constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();
    if (state_count > kMaxSize / sample_count) {
        throw std::length_error("Matrix size exceeds maximum addressable "
            "memory.");
    }
    const std::size_t total_elements = sample_count * state_count;
    SampleHistories histories;
    histories.sample_count = sample_count;
    histories.state_count = state_count;
    histories.data =
    std::make_unique_for_overwrite<TreeNodeId[]>(total_elements);
    std::span<TreeNodeId> data_span{
        histories.data.get(), total_elements};
    // State 0 is explicit and identical for every sample.
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        data_span[sample * state_count] = topology.root;
    }
    for (std::size_t split_index = 0;
        split_index < topology.splits.size();
        ++split_index) {
        const std::size_t iteration = split_index + 1;
        const auto iteration_file =
        directory / ("iter" + std::to_string(iteration) + ".txt");
        read_clustering_iteration(
            iteration_file,
            sample_count,
            iteration, // state index equals the one-based split number
            state_count,
            topology.splits[split_index],
            data_span);
    }
    return histories;
}
/**
 * @brief Order samples lexicographically by their complete history.
 * @param histories Validated immutable-node histories.
 * @return Original sample indices in hierarchy order.
 */
[[nodiscard]] std::vector<SampleIndex> sorted_sample_order(const
    SampleHistories& histories) {
    std::vector<SampleIndex> order(histories.sample_count);
    std::iota(order.begin(), order.end(), SampleIndex{
            0
        });
    // Fast path: for 0 or 1 samples there is nothing to reorder. This
    // is a trivial-case optimization — skipping std::sort here just
    // avoids a pointless call.
    if (histories.sample_count <= 1) {
        return order;
    }
    std::sort(
        order.begin(),
        order.end(),
        [&histories](SampleIndex lhs, SampleIndex rhs) {
            auto lhs_row = histories.row(lhs);
            auto rhs_row = histories.row(rhs);
        // Range-based mismatch
            auto mismatch = std::ranges::mismatch(lhs_row, rhs_row);
            if (mismatch.in1 != lhs_row.end()) {
                return *mismatch.in1 < *mismatch.in2;
            }
            return lhs < rhs;
        });
    return order;
}
/**
 * @brief Write compressed hierarchy history and check stream closure.
 * @param output_file Staged destination CSV.
 * @param sample_paths Original ordered sample-path list.
 * @param histories Complete immutable-node history.
 * @param order Sample indices in hierarchy order.
 * @throws std::runtime_error If opening, writing, flushing, or closing
 *     fails.
 */
void write_tree_states(
    const std::filesystem::path& output_file,
    const std::vector<std::string>& sample_paths,
    const SampleHistories& histories,
    const std::vector<SampleIndex>& order) {
    std::ofstream output(output_file, std::ios::binary);
    output.imbue(std::locale::classic());
    if (!output) {
        throw std::runtime_error(
            "Unable to open output file: " + output_file.string());
    }
    std::span<const TreeNodeId> previous_row;
    // Accumulate each complete CSV row before writing it to reduce the
    // number of stream operations without introducing a large global
    // output buffer.
    std::string line_buffer;
    line_buffer.reserve(1024); // a reasonable starting point
    for (std::size_t i = 0; i < order.size(); ++i) {
        const SampleIndex sample = order[i];
        auto current_row = histories.row(sample);
        // Start the line with the escaped filename
        line_buffer = escape_csv_field(sample_paths[sample]);
        for (std::size_t state = 0;
            state < histories.state_count;
            ++state) {
            line_buffer.push_back(',');
            if (previous_row.empty() ||
                current_row[state] != previous_row[state]) {
                line_buffer.append(std::to_string(current_row[state]));
            }
        }
        line_buffer.push_back('\n');
        output << line_buffer;
        previous_row = current_row;
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("Flush failed: " + output_file.string());
    }
    output.close();
    if (output.fail()) {
        throw std::runtime_error("Close failed: " + output_file.string());
    }
}
// Writes total_data.csv atomically: the full CSV is written to a
// temporary file in the same directory first, and only renamed into
// place once writing has fully succeeded. This guarantees that
// total_data.csv either doesn't exist yet or is always a complete,
// valid file — a mid-write failure (e.g. disk full, permissions
// revoked) can never leave a truncated CSV at the final path for a
// downstream consumer to read.
/**
 * @brief Publish complete hierarchy history using sibling replacement.
 * @param directory Destination directory for total_data.csv.
 * @param sample_paths Original ordered sample paths.
 * @param histories Complete immutable-node history.
 * @param order Sample indices in hierarchy order.
 * @throws std::runtime_error If writing or replacement fails.
 */
void write_tree_states_atomically(
    const std::filesystem::path& directory,
    const std::vector<std::string>& sample_paths,
    const SampleHistories& histories,
    const std::vector<SampleIndex>& order) {
    const auto final_path = directory / kTreeStatesFile;
    const auto temp_path = directory / (std::string(kTreeStatesFile) + ".tmp");
    try {
        write_tree_states(temp_path, sample_paths, histories, order);
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temp_path, cleanup_error);
        throw;
    }
#ifdef _WIN32
    if (!MoveFileExW(
        temp_path.wstring().c_str(),
        final_path.wstring().c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const std::error_code error(
            static_cast<int>(GetLastError()), std::system_category());
        std::error_code cleanup_error;
        std::filesystem::remove(temp_path, cleanup_error);
        throw
            std::runtime_error(std::format(
                "Unable to finalize output file: {} ({})", final_path.string(
                ), error.message()));
    }
#else
    if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
        const std::error_code error(errno, std::generic_category());
        std::error_code cleanup_error;
        std::filesystem::remove(temp_path, cleanup_error);
        throw
            std::runtime_error(std::format(
                "Unable to finalize output file: {} ({})", final_path.string(
                ), error.message()));
    }
#endif
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory>\n";
        return 1;
    }
    try {
        const std::filesystem::path directory(argv[1]);
        const auto commit = directory / kClusteringProvenanceFile;
        const auto committed = pipeline_io::verify_provenance(commit);
        if (!committed.outputs.contains(kSamplePathsFile) ||
            !committed.outputs.contains(kTreeTopologyFile)) {
            throw std::runtime_error("Clustering commit lacks required files");
        }
        const auto sample_paths = read_sample_paths(
            directory / kSamplePathsFile);
        for (std::size_t i = 1; i < sample_paths.size(); ++i) {
            if (!committed.outputs.contains("iter" + std::to_string(i)
                + ".txt")) {
                throw
                std::runtime_error("Clustering commit lacks an iteration");
            }
        }
        const auto topology = read_tree_topology(
            directory / kTreeTopologyFile, sample_paths.size());
        const auto histories = build_histories(
            directory, sample_paths, topology);
        const auto order = sorted_sample_order(histories);
        const auto output = directory / kTreeStatesFile;
        const auto provenance = directory /
            (std::string(kTreeStatesFile) + ".provenance");
        pipeline_io::invalidate_dependents(
            {output, provenance}, {directory,
                fs::absolute(directory).parent_path()});
        fs::remove(provenance);
        write_tree_states_atomically(
            directory, sample_paths, histories, order);
        pipeline_io::write_provenance(
            provenance, "reconstruct", {output}, {}, {commit}, argv[0]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: [reconstruct] " << error.what() << '\n';
        return 1;
    }
}
