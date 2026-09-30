#include "artifact/sha256.h"
#include "media/decode/decode.h"
#include "targets/qwen3_8/impl/frontend/chat_template.h"
#include "targets/qwen3_8/impl/frontend/processor.h"
#include "targets/qwen3_8/impl/frontend/tokenizer.h"
#include "targets/qwen3_8/reference/legacy_encode_rendered_chat.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::media::decode {

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::logic_error("decode_image is unavailable in the g1 differential");
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::logic_error("decode_video is unavailable in the g1 differential");
}

} // namespace ninfer::media::decode

namespace {

namespace product = ninfer::targets::qwen3_8::frontend_internal;
namespace legacy  = ninfer::targets::qwen3_8::legacy_oracle;
using ninfer::artifact::Sha256;

enum class Impl { Legacy, Candidate };
enum class Profile { Real, Synthetic };
enum class Side { Real, Synthetic, Both };

constexpr int kRealCases      = 53;
constexpr int kSyntheticCases = 74;
constexpr std::size_t kFullCutLimit = 256;
constexpr std::size_t kUint32Past =
    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL;

struct Options {
    Impl impl          = Impl::Candidate;
    Profile profile    = Profile::Real;
    std::string resources;
    std::string cases;
    std::string output;
    std::string summary;
    std::string toggle;
    std::string count_body;
    std::uint32_t seed = 20260925;
    int fuzz           = 0;
};

struct Vocab {
    std::string tokenizer_json;
    std::string config_json;
    std::string generation_json;
    std::unique_ptr<product::Tokenizer> product_tokenizer;
    std::unique_ptr<legacy::Tokenizer> legacy_tokenizer;
    std::string resources_sha;
};

struct Outcome {
    bool ok = false;
    std::vector<int> token_ids;
    product::EncodedChat encoded;
    std::string error_class;
    std::string error_message;
};

struct Templates {
    std::optional<product::CompiledChatTemplate> reasoning;
    std::optional<product::CompiledChatTemplate> toggle;
};

struct Emitter {
    std::ostream& out;
    std::set<std::string> ids;
    int rows = 0;

    void write(const std::string& case_id, const std::string& kind, const std::string& input_sha,
               const std::string& resources_sha, const nlohmann::json& row_body) {
        if (!ids.insert(case_id).second) {
            throw std::runtime_error("duplicate case_id " + case_id);
        }
        nlohmann::json row = row_body;
        row["case_id"]          = case_id;
        row["kind"]             = kind;
        row["input_sha256"]     = input_sha;
        row["resources_sha256"] = resources_sha;
        out << row.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        ++rows;
        if (rows % 2000 == 0) { out.flush(); }
    }
};

std::string sha256_bytes(std::string_view text) {
    Sha256 hasher;
    const auto* data = reinterpret_cast<const std::byte*>(text.data());
    hasher.update(std::span<const std::byte>(data, text.size()));
    const auto digest = hasher.finish();
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const auto byte = static_cast<unsigned char>(digest[i]);
        out[i * 2]      = kHex[byte >> 4];
        out[i * 2 + 1]  = kHex[byte & 0x0F];
    }
    return out;
}

std::string canonical(const nlohmann::json& value) {
    return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
}

std::string input_sha(const nlohmann::json& identity) { return sha256_bytes(canonical(identity)); }

std::string hex_encode(std::string_view text) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(text.size() * 2, '0');
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto byte = static_cast<unsigned char>(text[i]);
        out[i * 2]      = kHex[byte >> 4];
        out[i * 2 + 1]  = kHex[byte & 0x0F];
    }
    return out;
}

int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') { return ch - '0'; }
    if (ch >= 'a' && ch <= 'f') { return ch - 'a' + 10; }
    if (ch >= 'A' && ch <= 'F') { return ch - 'A' + 10; }
    throw std::invalid_argument("bad hex digit");
}

std::string hex_decode(std::string_view text) {
    if (text.size() % 2 != 0) { throw std::invalid_argument("odd hex length"); }
    std::string out(text.size() / 2, '\0');
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>((hex_value(text[i * 2]) << 4) | hex_value(text[i * 2 + 1]));
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("missing resource: " + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

// Fixture files keep a POSIX trailing newline. The product pin is those bytes
// without that newline, the same rule as read_template_fixture in test_frontend.cpp.
std::string read_pinned_template(const std::string& path) {
    std::string source = read_file(path);
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

const char* exception_class(const std::exception& ex) {
    if (dynamic_cast<const std::overflow_error*>(&ex) != nullptr) { return "overflow_error"; }
    if (dynamic_cast<const std::invalid_argument*>(&ex) != nullptr) { return "invalid_argument"; }
    if (dynamic_cast<const std::out_of_range*>(&ex) != nullptr) { return "out_of_range"; }
    if (dynamic_cast<const std::logic_error*>(&ex) != nullptr) { return "logic_error"; }
    return "exception";
}

std::string utf8_codepoint(std::uint32_t codepoint) {
    if (codepoint < 0x80) { return std::string(1, static_cast<char>(codepoint)); }
    if (codepoint < 0x800) {
        return {static_cast<char>(0xC0 | (codepoint >> 6)),
                static_cast<char>(0x80 | (codepoint & 0x3F))};
    }
    return {static_cast<char>(0xE0 | (codepoint >> 12)),
            static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)),
            static_cast<char>(0x80 | (codepoint & 0x3F))};
}

nlohmann::json byte_vocab() {
    nlohmann::json vocab = nlohmann::json::object();
    std::uint32_t next   = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool shown =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174);
        vocab[utf8_codepoint(shown ? static_cast<std::uint32_t>(byte) : next++)] = byte;
    }
    return vocab;
}

// GPT-2 byte pieces use the character itself for bytes 174..255, so U+00E9 ("é")
// is the piece for byte 0xE9. V2-SC-09 adds that same string as a token. Move the
// colliding piece to the next free codepoint; the byte id stays, and the added
// token can load. test_tokenization_equivalence avoids this with a one-entry vocab.
nlohmann::json byte_vocab_for(const nlohmann::json& added) {
    nlohmann::json vocab = byte_vocab();
    std::vector<std::pair<std::string, int>> blocked;
    for (const nlohmann::json& token : added) {
        const std::string content = token.at("content").get<std::string>();
        const auto found          = vocab.find(content);
        if (found != vocab.end()) { blocked.emplace_back(content, found->get<int>()); }
    }
    std::uint32_t spare = 324;
    for (const auto& [content, id] : blocked) {
        vocab.erase(content);
        std::string replacement;
        do {
            replacement = utf8_codepoint(spare++);
            bool used   = vocab.find(replacement) != vocab.end();
            if (!used) {
                for (const nlohmann::json& token : added) {
                    if (token.at("content").get<std::string>() == replacement) { used = true; }
                }
            }
            if (!used) { break; }
        } while (true);
        vocab[replacement] = id;
    }
    return vocab;
}

nlohmann::json added_from_spec(const nlohmann::json& spec) {
    nlohmann::json added = nlohmann::json::array();
    if (spec.is_null() || spec.empty()) { return added; }
    for (const nlohmann::json& item : spec) {
        int id = 0;
        std::string content;
        bool special = false;
        if (item.is_array()) {
            id      = item.at(0).get<int>();
            content = item.at(1).get<std::string>();
        } else {
            id      = item.at("id").get<int>();
            content = item.at("content").get<std::string>();
            special = item.value("special", false);
        }
        added.push_back(nlohmann::json{{"id", id},
                                       {"content", content},
                                       {"single_word", false},
                                       {"lstrip", false},
                                       {"rstrip", false},
                                       {"normalized", false},
                                       {"special", special}});
    }
    return added;
}

std::string directory_resources_sha(const std::string& dir) {
    static constexpr const char* kNames[] = {
        "tokenizer.json",         "tokenizer_config.json", "chat_template.jinja",
        "generation_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json"};
    nlohmann::json files = nlohmann::json::object();
    for (const char* name : kNames) {
        const std::string path = dir + "/" + name;
        std::ifstream stream(path, std::ios::binary);
        if (!stream) { throw std::runtime_error(std::string("BLOCKED missing resource: ") + path); }
        const std::string raw((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());
        files[name] = {{"bytes", raw.size()}, {"sha256", sha256_bytes(raw)}};
    }
    return sha256_bytes(canonical(files));
}

Vocab make_vocab(const nlohmann::json& added_spec, const nlohmann::json& merges, Profile profile,
                 const std::string& resources, Impl impl, bool force_real) {
    Vocab vocab;
    const bool real = force_real || (profile == Profile::Real && added_spec.empty() && merges.empty());
    if (real) {
        vocab.tokenizer_json  = read_file(resources + "/tokenizer.json");
        vocab.config_json     = read_file(resources + "/tokenizer_config.json");
        vocab.generation_json = read_file(resources + "/generation_config.json");
        vocab.resources_sha   = directory_resources_sha(resources);
    } else {
        nlohmann::json added = added_from_spec(added_spec);
        nlohmann::json merge_list = merges.is_null() ? nlohmann::json::array() : merges;
        vocab.tokenizer_json = nlohmann::json{{"model",
                                               {{"type", "BPE"},
                                                {"vocab", byte_vocab_for(added)},
                                                {"merges", std::move(merge_list)}}},
                                              {"added_tokens", added}}
                                   .dump();
        nlohmann::json decoder = nlohmann::json::object();
        for (const nlohmann::json& token : added) {
            nlohmann::json value = token;
            const std::string id = std::to_string(value.at("id").get<int>());
            value.erase("id");
            decoder[id] = std::move(value);
        }
        vocab.config_json     = nlohmann::json{{"added_tokens_decoder", std::move(decoder)}}.dump();
        vocab.generation_json = R"({"eos_token_id":0})";
        vocab.resources_sha   = sha256_bytes(vocab.tokenizer_json);
    }
    if (impl == Impl::Candidate) {
        vocab.product_tokenizer = std::make_unique<product::Tokenizer>(product::TokenizerResources{
            vocab.tokenizer_json, vocab.config_json, vocab.generation_json});
    } else {
        vocab.legacy_tokenizer = std::make_unique<legacy::Tokenizer>(legacy::TokenizerResources{
            vocab.tokenizer_json, vocab.config_json, vocab.generation_json});
    }
    return vocab;
}

Outcome encode_raw(const Vocab& vocab, Impl impl, std::string_view text, bool parse_added) {
    Outcome out;
    try {
        if (impl == Impl::Candidate) {
            product::EncodeOptions options;
            options.parse_added_tokens = parse_added;
            out.token_ids              = vocab.product_tokenizer->encode(text, options);
        } else {
            legacy::EncodeOptions options;
            options.parse_added_tokens = parse_added;
            out.token_ids              = vocab.legacy_tokenizer->encode(text, options);
        }
        out.ok = true;
    } catch (const std::exception& ex) {
        out.error_class   = exception_class(ex);
        out.error_message = ex.what();
    }
    return out;
}

Outcome encode_rendered(const Vocab& vocab, Impl impl, const product::RenderedChat& rendered) {
    Outcome out;
    try {
        if (impl == Impl::Candidate) {
            out.encoded = product::encode_rendered_chat(*vocab.product_tokenizer, rendered);
        } else {
            out.encoded = legacy::encode_rendered_chat(*vocab.legacy_tokenizer, rendered);
        }
        out.token_ids = out.encoded.input_ids;
        out.ok        = true;
    } catch (const std::exception& ex) {
        out.error_class   = exception_class(ex);
        out.error_message = ex.what();
    }
    return out;
}

const char* hint_name(product::SemanticCheckpointKind kind) {
    switch (kind) {
    case product::SemanticCheckpointKind::StablePrefix: return "StablePrefix";
    case product::SemanticCheckpointKind::StableTurn: return "StableTurn";
    case product::SemanticCheckpointKind::Rolling: return "Rolling";
    }
    throw std::logic_error("unknown checkpoint kind");
}

nlohmann::json boundary_value(const std::optional<std::uint32_t>& value) {
    if (!value) { return nullptr; }
    return *value;
}

nlohmann::json hint_array(const product::EncodedChat& encoded) {
    nlohmann::json hints = nlohmann::json::array();
    for (const auto& hint : encoded.checkpoint_hints) {
        hints.push_back({{"boundary", hint.boundary}, {"kind", hint_name(hint.kind)}});
    }
    return hints;
}

nlohmann::json rendered_semantic(const Outcome& outcome) {
    return {{"checkpoint_hints", hint_array(outcome.encoded)},
            {"stable_prefix_boundary", boundary_value(outcome.encoded.stable_prefix_boundary)},
            {"token_ids", outcome.token_ids},
            {"turn_rewrite_boundary", boundary_value(outcome.encoded.turn_rewrite_boundary)},
            {"user_turn_boundary", boundary_value(outcome.encoded.user_turn_boundary)}};
}

nlohmann::json prepared_semantic(const Outcome& outcome, const std::string& rendered,
                                 bool starts_in_reasoning) {
    // Same text-only assignment as Frontend::prepare: three identical axes, no rope shift.
    nlohmann::json semantic = rendered_semantic(outcome);
    const std::size_t count = outcome.token_ids.size();
    nlohmann::json types    = nlohmann::json::array();
    nlohmann::json positions = nlohmann::json::array();
    for (std::size_t i = 0; i < count; ++i) { types.push_back(0); }
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < count; ++i) { positions.push_back(static_cast<std::int64_t>(i)); }
    }
    semantic["token_types"]        = std::move(types);
    semantic["positions"]          = std::move(positions);
    semantic["rope_delta"]         = 0;
    semantic["starts_in_reasoning"] = starts_in_reasoning;
    semantic["rendered_sha256"]    = sha256_bytes(rendered);
    semantic["identity"]           = {{"checkpoint_hints", hint_array(outcome.encoded)},
                                      {"reusable", true},
                                      {"stable_prefix_boundary",
                                       boundary_value(outcome.encoded.stable_prefix_boundary)},
                                      {"turn_rewrite_boundary",
                                       boundary_value(outcome.encoded.turn_rewrite_boundary)},
                                      {"user_turn_boundary",
                                       boundary_value(outcome.encoded.user_turn_boundary)}};
    return semantic;
}

void emit_ok(Emitter& emitter, const std::string& case_id, const std::string& kind,
             const nlohmann::json& identity, const std::string& resources_sha,
             const nlohmann::json& semantic) {
    emitter.write(case_id, kind, input_sha(identity), resources_sha,
                  {{"semantic", semantic}, {"status", "OK"}});
}

void emit_error(Emitter& emitter, const std::string& case_id, const std::string& kind,
                const nlohmann::json& identity, const std::string& resources_sha,
                const std::string& phase, const Outcome& outcome) {
    emitter.write(case_id, kind, input_sha(identity), resources_sha,
                  {{"error",
                    {{"class", outcome.error_class},
                     {"message", outcome.error_message},
                     {"phase", phase}}},
                   {"status", "ERROR"}});
}

std::string b64_decode(std::string_view in) {
    static constexpr int kMap[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1};
    std::string out;
    int value = 0;
    int bits  = 0;
    for (unsigned char ch : in) {
        if (ch == '=') { break; }
        const int digit = kMap[ch];
        if (digit < 0) { continue; }
        value = (value << 6) | digit;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((value >> bits) & 0xFF));
        }
    }
    return out;
}

const nlohmann::json* spec_of(const nlohmann::json& row) {
    if (row.contains("spec") && row.at("spec").is_object()) { return &row.at("spec"); }
    return nullptr;
}

std::string case_text(const nlohmann::json& row) {
    const nlohmann::json* spec = spec_of(row);
    if (spec != nullptr && spec->contains("repeat_unit")) {
        const std::string unit = spec->at("repeat_unit").get<std::string>();
        const int repeats      = spec->value("repeats", 1);
        std::string text;
        text.reserve(unit.size() * static_cast<std::size_t>(repeats));
        for (int i = 0; i < repeats; ++i) { text += unit; }
        return text;
    }
    if (spec != nullptr && spec->contains("text_hex")) {
        return hex_decode(spec->at("text_hex").get<std::string>());
    }
    if (spec != nullptr && spec->contains("text") && spec->at("text").is_string()) {
        return spec->at("text").get<std::string>();
    }
    if (row.contains("text") && row.at("text").is_string()) { return row.at("text").get<std::string>(); }
    if (row.contains("raw_base64")) { return b64_decode(row.at("raw_base64").get<std::string>()); }
    return {};
}

bool has_explicit_text(const nlohmann::json& row) {
    const nlohmann::json* spec = spec_of(row);
    return row.contains("text") || row.contains("raw_base64") ||
           (spec != nullptr && (spec->contains("text") || spec->contains("text_hex") ||
                                spec->contains("repeat_unit")));
}

nlohmann::json added_spec_of(const nlohmann::json& row) {
    const nlohmann::json* spec = spec_of(row);
    nlohmann::json added       = nlohmann::json::array();
    if (row.contains("added_tokens")) {
        added = row.at("added_tokens");
    } else if (spec != nullptr && spec->contains("added_tokens")) {
        added = spec->at("added_tokens");
    } else if (spec != nullptr && spec->contains("added_tokens_hex")) {
        for (const nlohmann::json& item : spec->at("added_tokens_hex")) {
            added.push_back(nlohmann::json{{"id", item.at(0).get<int>()},
                                           {"content", hex_decode(item.at(1).get<std::string>())}});
        }
    }
    if (spec != nullptr && spec->value("special", false)) {
        nlohmann::json objects = added_from_spec(added);
        for (nlohmann::json& token : objects) { token["special"] = true; }
        return objects;
    }
    return added;
}

nlohmann::json merges_of(const nlohmann::json& row) {
    if (row.contains("merges")) { return row.at("merges"); }
    const nlohmann::json* spec = spec_of(row);
    if (spec != nullptr && spec->contains("merges")) { return spec->at("merges"); }
    return nlohmann::json::array();
}

bool parse_added_of(const nlohmann::json& row) {
    if (row.contains("parse_added_tokens")) { return row.at("parse_added_tokens").get<bool>(); }
    const nlohmann::json* spec = spec_of(row);
    if (spec != nullptr && spec->contains("parse_added_tokens")) {
        return spec->at("parse_added_tokens").get<bool>();
    }
    return true;
}

Side side_for(const nlohmann::json& row) {
    const std::string kind    = row.at("kind").get<std::string>();
    const std::string profile = row.value("resource_profile", "");
    const nlohmann::json* spec = spec_of(row);
    if (kind == "trace-lifecycle" || kind == "synthetic-added-priority" || kind == "synthetic-merge" ||
        kind == "invalid-utf8") {
        return Side::Synthetic;
    }
    if (kind == "chat-transform") { return Side::Real; }
    if (kind == "scanner-contract") {
        return spec != nullptr && spec->value("real_resources", false) ? Side::Real : Side::Synthetic;
    }
    if (kind == "boundary-contract") {
        return spec != nullptr && spec->contains("chat_transform") ? Side::Real : Side::Synthetic;
    }
    if (kind == "raw-boundaries" || kind == "rendered-contract") {
        if (profile == "synthetic-byte-vocab") { return Side::Synthetic; }
        if (profile == "real-or-synthetic-byte-vocab") { return Side::Both; }
        throw std::runtime_error("unsupported resource_profile on " + row.at("case_id").get<std::string>());
    }
    throw std::runtime_error("unsupported kind " + kind);
}

bool applies(Profile profile, Side side) {
    if (side == Side::Both) { return true; }
    return (profile == Profile::Real && side == Side::Real) ||
           (profile == Profile::Synthetic && side == Side::Synthetic);
}

product::SemanticCheckpointKind parse_kind(const std::string& kind) {
    if (kind == "StablePrefix") { return product::SemanticCheckpointKind::StablePrefix; }
    if (kind == "StableTurn") { return product::SemanticCheckpointKind::StableTurn; }
    if (kind == "Rolling") { return product::SemanticCheckpointKind::Rolling; }
    throw std::invalid_argument("unknown checkpoint kind " + kind);
}

std::optional<std::size_t> parse_offset(const nlohmann::json& value, std::size_t text_size) {
    if (value.is_null()) { return std::nullopt; }
    if (value.is_string()) {
        if (value.get<std::string>() == "EOF") { return text_size; }
        throw std::invalid_argument("unknown offset token");
    }
    return value.get<std::size_t>();
}

bool boundary_fields(const nlohmann::json& spec) {
    return spec.contains("stable_prefix_byte_offset") || spec.contains("stable_offset") ||
           spec.contains("turn_rewrite_byte_offset") || spec.contains("turn_offset") ||
           spec.contains("user_turn_byte_offset") || spec.contains("user_offset") ||
           spec.contains("checkpoint_hints") || spec.contains("hints");
}

product::RenderedChat load_rendered(const std::string& text, const nlohmann::json& spec) {
    product::RenderedChat rendered;
    rendered.text = text;
    const auto take = [&](const char* first, const char* second) -> std::optional<std::size_t> {
        if (spec.contains(first)) { return parse_offset(spec.at(first), text.size()); }
        if (second != nullptr && spec.contains(second)) {
            return parse_offset(spec.at(second), text.size());
        }
        return std::nullopt;
    };
    rendered.stable_prefix_byte_offset = take("stable_prefix_byte_offset", "stable_offset");
    rendered.turn_rewrite_byte_offset  = take("turn_rewrite_byte_offset", "turn_offset");
    rendered.user_turn_byte_offset     = take("user_turn_byte_offset", "user_offset");
    if (spec.contains("checkpoint_hints")) {
        for (const nlohmann::json& hint : spec.at("checkpoint_hints")) {
            rendered.checkpoint_hints.push_back(
                {parse_kind(hint.at("kind").get<std::string>()),
                 hint.at("byte_offset").get<std::size_t>()});
        }
    }
    if (spec.contains("hints")) {
        for (const nlohmann::json& hint : spec.at("hints")) {
            rendered.checkpoint_hints.push_back(
                {parse_kind(hint.at(0).get<std::string>()), hint.at(1).get<std::size_t>()});
        }
    }
    return rendered;
}

void emit_rendered_outcome(Emitter& emitter, const std::string& case_id, const nlohmann::json& identity,
                           const Vocab& vocab, const Outcome& outcome) {
    if (outcome.ok) {
        emit_ok(emitter, case_id, "rendered", identity, vocab.resources_sha, rendered_semantic(outcome));
    } else {
        emit_error(emitter, case_id, "rendered", identity, vocab.resources_sha, "encode", outcome);
    }
}

std::vector<std::size_t> sample_cuts(const std::string& text) {
    std::set<std::size_t> cuts;
    cuts.insert(0);
    cuts.insert(text.size());
    if (text.size() > 2) { cuts.insert(text.size() / 2); }
    static constexpr const char* kMarkers[] = {"<|im_start|>", "<|im_end|>", "<think>", "</think>",
                                               "<tool_call>",  "</tool_call>", "<a>",     "<b>",
                                               "<long>"};
    for (const char* marker : kMarkers) {
        const std::string needle(marker);
        const std::size_t first = text.find(needle);
        if (first == std::string::npos) { continue; }
        const std::size_t last = text.rfind(needle);
        for (const std::size_t pos : {first, last}) {
            cuts.insert(pos);
            if (needle.size() > 1) { cuts.insert(pos + 1); }
            cuts.insert(pos + needle.size() - 1);
            cuts.insert(std::min(text.size(), pos + needle.size()));
        }
    }
    return {cuts.begin(), cuts.end()};
}

void emit_coverage(Emitter& emitter, const Vocab& vocab, Impl impl, const std::string& case_id,
                   const std::string& text, bool parse_added, bool all_byte_cuts) {
    if (all_byte_cuts && text.size() > kFullCutLimit) {
        throw std::runtime_error(case_id + " requests all byte cuts above 256");
    }
    const bool full = text.size() <= kFullCutLimit;
    std::vector<std::size_t> cuts;
    if (full) {
        cuts.resize(text.size() + 1);
        for (std::size_t cut = 0; cut <= text.size(); ++cut) { cuts[cut] = cut; }
    } else {
        cuts = sample_cuts(text);
    }
    for (const std::size_t cut : cuts) {
        const std::string prefix = text.substr(0, cut);
        const nlohmann::json raw_id{{"cut", cut},
                                    {"parse_added_tokens", parse_added},
                                    {"text_hex", hex_encode(prefix)}};
        const Outcome raw = encode_raw(vocab, impl, prefix, parse_added);
        if (raw.ok) {
            emit_ok(emitter, case_id + "#b" + std::to_string(cut), "raw", raw_id, vocab.resources_sha,
                    {{"token_ids", raw.token_ids}});
        } else {
            emit_error(emitter, case_id + "#b" + std::to_string(cut), "raw", raw_id,
                       vocab.resources_sha, "encode", raw);
        }
        product::RenderedChat rendered;
        rendered.text                      = text;
        rendered.stable_prefix_byte_offset = cut;
        const Outcome boundary             = encode_rendered(vocab, impl, rendered);
        const nlohmann::json boundary_id{{"boundary", "stable"},
                                         {"cut", cut},
                                         {"text_hex", hex_encode(text)}};
        emit_rendered_outcome(emitter, case_id + "#s" + std::to_string(cut), boundary_id, vocab,
                              boundary);
    }
}

product::ChatMessage message_from(const nlohmann::json& item) {
    product::ChatMessage message;
    message.role = item.at("role").get<std::string>();
    if (message.role == "developer") { message.role = "system"; }
    if (item.contains("reasoning_content") && item.at("reasoning_content").is_string()) {
        message.reasoning_content = item.at("reasoning_content").get<std::string>();
    }
    if (item.contains("tool_call_id") && item.at("tool_call_id").is_string()) {
        message.tool_call_id = item.at("tool_call_id").get<std::string>();
    }
    if (item.contains("content") && !item.at("content").is_null()) {
        if (!item.at("content").is_string()) {
            throw std::runtime_error("chat content must be a string in the g1 adapter");
        }
        message.parts.push_back(product::ChatPart::text_part(item.at("content").get<std::string>()));
    }
    if (item.contains("tool_calls") && item.at("tool_calls").is_array()) {
        for (const nlohmann::json& call : item.at("tool_calls")) {
            const nlohmann::json& function = call.at("function");
            message.tool_calls.push_back(product::ToolCall{
                call.at("id").get<std::string>(), function.at("name").get<std::string>(),
                function.at("arguments").is_string() ? function.at("arguments").get<std::string>()
                                                     : function.at("arguments").dump()});
        }
    }
    return message;
}

std::vector<product::ChatMessage> messages_from(const nlohmann::json& items) {
    std::vector<product::ChatMessage> messages;
    for (const nlohmann::json& item : items) { messages.push_back(message_from(item)); }
    return messages;
}

std::vector<std::string> tool_jsons_from(const nlohmann::json& tools) {
    std::vector<std::string> out;
    if (!tools.is_array()) { return out; }
    for (const nlohmann::json& item : tools) {
        nlohmann::json normalized = item;
        nlohmann::json& function  = normalized["function"];
        if (!function.contains("parameters") || function.at("parameters").is_null()) {
            function["parameters"] = nlohmann::json{{"properties", nlohmann::json::object()},
                                                    {"type", "object"}};
        }
        function["strict"] = false;
        out.push_back(normalized.dump());
    }
    return out;
}

product::ChatRenderOptions render_options_for(const nlohmann::json& spec, const nlohmann::json& tools,
                                              bool allow_effort) {
    product::ChatRenderOptions options;
    options.add_generation_prompt    = spec.value("add_generation_prompt", true);
    options.enable_thinking          = spec.value("enable_thinking", true);
    options.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::RollingTool;
    options.tool_jsons               = tool_jsons_from(tools);
    if (spec.contains("preserve_thinking")) {
        options.preserve_thinking = spec.at("preserve_thinking").get<bool>();
    }
    if (allow_effort && options.enable_thinking && spec.contains("reasoning_effort")) {
        const std::string effort = spec.at("reasoning_effort").get<std::string>();
        if (effort == "low") {
            options.reasoning_effort = ninfer::ReasoningEffort::Low;
        } else if (effort == "medium") {
            options.reasoning_effort = ninfer::ReasoningEffort::Medium;
        } else if (effort == "xhigh") {
            options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
        } else if (effort == "none") {
            options.reasoning_effort = std::nullopt;
        } else {
            throw std::runtime_error("unsupported reasoning_effort " + effort);
        }
    } else if (allow_effort && options.enable_thinking) {
        options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    }
    return options;
}

nlohmann::json apply_messages(nlohmann::json messages, const nlohmann::json& transform) {
    if (transform.contains("keep_first_messages")) {
        const std::size_t count = transform.at("keep_first_messages").get<std::size_t>();
        if (count > messages.size()) { throw std::runtime_error("keep_first_messages past end"); }
        messages.erase(messages.begin() + static_cast<std::ptrdiff_t>(count), messages.end());
    }
    if (transform.contains("replace_message")) {
        const nlohmann::json& replace = transform.at("replace_message");
        const std::size_t index       = replace.at("index").get<std::size_t>();
        messages.at(index)            = replace.at("value");
    }
    if (transform.contains("append_messages")) {
        for (const nlohmann::json& item : transform.at("append_messages")) { messages.push_back(item); }
    }
    for (auto it = transform.begin(); it != transform.end(); ++it) {
        const std::string key = it.key();
        if (key != "keep_first_messages" && key != "replace_message" && key != "append_messages" &&
            key != "after_options" && key != "after_tools") {
            throw std::runtime_error("unsupported transform " + key);
        }
    }
    return messages;
}

void emit_prepared(Emitter& emitter, const std::string& case_id, const Vocab& vocab, Impl impl,
                   const product::CompiledChatTemplate& chat, const nlohmann::json& message_json,
                   const nlohmann::json& option_json, const nlohmann::json& tools,
                   bool allow_effort, const std::string& template_name) {
    const nlohmann::json identity{{"options", option_json},
                                  {"template", template_name},
                                  {"tools", tools}};
    try {
        const std::vector<product::ChatMessage> messages = messages_from(message_json);
        const product::ChatRenderOptions options = render_options_for(option_json, tools, allow_effort);
        const product::RenderedChat rendered     = chat.render(messages, options);
        const Outcome outcome                    = encode_rendered(vocab, impl, rendered);
        if (!outcome.ok) {
            emit_error(emitter, case_id, "prepared", identity, vocab.resources_sha, "encode", outcome);
            return;
        }
        const bool starts = options.add_generation_prompt && options.enable_thinking;
        emit_ok(emitter, case_id, "prepared", identity, vocab.resources_sha,
                prepared_semantic(outcome, rendered.text, starts));
        emit_coverage(emitter, vocab, impl, case_id, rendered.text, true, false);
    } catch (const std::exception& ex) {
        Outcome outcome;
        outcome.error_class   = exception_class(ex);
        outcome.error_message = ex.what();
        emit_error(emitter, case_id, "prepared", identity, vocab.resources_sha, "render", outcome);
    }
}

void emit_logical(Emitter& emitter, Impl impl, const std::string& resources_sha) {
    const auto run = [&](bool exact, const char* variant) {
        Outcome outcome;
        try {
            if (impl == Impl::Candidate) {
                (void)product::enforce_exact_token_prefix(exact, kUint32Past,
                                                          "semantic checkpoint prefix");
            } else {
                legacy::enforce_exact_then_uint32(exact, kUint32Past, "semantic checkpoint prefix");
            }
            outcome.ok = true;
        } catch (const std::exception& ex) {
            outcome.error_class   = exception_class(ex);
            outcome.error_message = ex.what();
        }
        const nlohmann::json identity{{"exact", exact}, {"length", kUint32Past}, {"variant", variant}};
        if (outcome.ok) {
            emit_ok(emitter, std::string("V2-BC-08#") + variant, "rendered", identity, resources_sha,
                    rendered_semantic(outcome));
        } else {
            emit_error(emitter, std::string("V2-BC-08#") + variant, "rendered", identity, resources_sha,
                       "boundary", outcome);
        }
    };
    run(true, "exact_then_overflow");
    run(false, "nonexact_before_overflow");
}

void emit_lifecycle(Emitter& emitter, const nlohmann::json& row, const Vocab& vocab, Impl impl) {
    const std::string id = row.at("case_id").get<std::string>();
    const nlohmann::json& spec = row.at("spec");
    const nlohmann::json ops = spec.value("operations", nlohmann::json::array());
    const bool copy = ops.is_array() && !ops.empty() && ops.at(0) == "construct" &&
                      std::find(ops.begin(), ops.end(), "copy") != ops.end();
    const bool move = ops.is_array() && std::find(ops.begin(), ops.end(), "move") != ops.end();
    const bool threads = spec.contains("threads");
    if (copy || move) {
        const char* label = copy ? "copy" : "move";
        std::vector<int> ids;
        if (impl == Impl::Candidate) {
            auto source = std::make_unique<product::Tokenizer>(*vocab.product_tokenizer);
            product::Tokenizer owned = copy ? *source : std::move(*source);
            source.reset();
            ids = owned.encode("<a>x");
        } else {
            auto source = std::make_unique<legacy::Tokenizer>(*vocab.legacy_tokenizer);
            legacy::Tokenizer owned = copy ? *source : std::move(*source);
            source.reset();
            ids = owned.encode("<a>x");
        }
        emit_ok(emitter, id + "#" + label, "raw",
                {{"operation", label}, {"text_hex", hex_encode("<a>x")}}, vocab.resources_sha,
                {{"token_ids", ids}});
        return;
    }
    if (threads) {
        const std::vector<std::string> inputs = {"<a>x", "café", "<a><b>", ""};
        std::vector<int> serial;
        for (const std::string& text : inputs) {
            const Outcome one = encode_raw(vocab, impl, text, true);
            if (!one.ok) {
                emit_error(emitter, id + "#threads", "raw", {{"threads", true}}, vocab.resources_sha,
                           "encode", one);
                return;
            }
            serial.insert(serial.end(), one.token_ids.begin(), one.token_ids.end());
        }
        const int thread_count = spec.value("threads", 4);
        const int iterations   = spec.value("iterations_each", 100);
        std::vector<std::string> errors(static_cast<std::size_t>(thread_count));
        std::vector<std::thread> pool;
        for (int index = 0; index < thread_count; ++index) {
            pool.emplace_back([&, index] {
                try {
                    for (int step = 0; step < iterations; ++step) {
                        for (const std::string& text : inputs) {
                            const Outcome one = encode_raw(vocab, impl, text, true);
                            if (!one.ok) { throw std::runtime_error(one.error_message); }
                        }
                    }
                } catch (const std::exception& ex) {
                    errors[static_cast<std::size_t>(index)] = ex.what();
                }
            });
        }
        for (std::thread& thread : pool) { thread.join(); }
        Outcome failed;
        for (const std::string& error : errors) {
            if (!error.empty()) {
                failed.error_class   = "exception";
                failed.error_message = error;
                emit_error(emitter, id + "#threads", "raw", {{"threads", thread_count}},
                           vocab.resources_sha, "encode", failed);
                return;
            }
        }
        emit_ok(emitter, id + "#threads", "raw",
                {{"iterations_each", iterations}, {"threads", thread_count}}, vocab.resources_sha,
                {{"token_ids", serial}});
        return;
    }
    const std::string text = spec.contains("text") ? spec.at("text").get<std::string>() : std::string{};
    const bool parse_added = spec.value("parse_added_tokens", true);
    const Outcome outcome  = encode_raw(vocab, impl, text, parse_added);
    const nlohmann::json identity{{"parse_added_tokens", parse_added}, {"text_hex", hex_encode(text)}};
    if (outcome.ok) {
        emit_ok(emitter, id + "#trace", "raw", identity, vocab.resources_sha,
                {{"token_ids", outcome.token_ids}});
    } else {
        emit_error(emitter, id + "#trace", "raw", identity, vocab.resources_sha, "encode", outcome);
    }
}

nlohmann::json lifecycle_added() {
    return nlohmann::json::array({nlohmann::json{{"id", 300}, {"content", "<a>"}},
                                  {{"id", 301}, {"content", "<b>"}}});
}

nlohmann::json boundary_user_reset_messages(bool after) {
    nlohmann::json messages = nlohmann::json::array(
        {{{"role", "system"}, {"content", "Inspect Java code. Preserve exact identifiers."}},
         {{"role", "user"}, {"content", "Fix the failing test in src/ExamService.java."}},
         {{"role", "assistant"},
          {"content", ""},
          {"reasoning_content", "Read the source first."},
          {"tool_calls",
           nlohmann::json::array({{{"id", "call_1"},
                                   {"type", "function"},
                                   {"function",
                                    {{"name", "read_file"},
                                     {"arguments", "{\"path\":\"src/ExamService.java\"}"}}}}})}},
         {{"role", "tool"},
          {"tool_call_id", "call_1"},
          {"content", "public int grade(int score) { return score; }"}}});
    if (after) {
        messages.push_back({{"role", "user"}, {"content", "What is the grade boundary?"}});
        messages.push_back(
            {{"role", "assistant"},
             {"content", ""},
             {"reasoning_content", "Check the boundary."},
             {"tool_calls",
              nlohmann::json::array({{{"id", "call_2"},
                                      {"type", "function"},
                                      {"function",
                                       {{"name", "read_file"},
                                        {"arguments", "{\"path\":\"src/ExamBoundary.java\"}"}}}}})}});
        messages.push_back({{"role", "tool"},
                            {"tool_call_id", "call_2"},
                            {"content", "expected=0 actual=-1"}});
    }
    return messages;
}

void emit_case(Emitter& emitter, const nlohmann::json& row, const Vocab& vocab, Impl impl,
               const Templates& templates) {
    const std::string kind = row.at("kind").get<std::string>();
    const std::string id   = row.at("case_id").get<std::string>();
    const nlohmann::json* spec = spec_of(row);
    if (kind == "trace-lifecycle") {
        emit_lifecycle(emitter, row, vocab, impl);
        return;
    }
    if (kind == "boundary-contract" && spec != nullptr && spec->value("logical_test_only", false)) {
        emit_logical(emitter, impl, vocab.resources_sha);
        return;
    }
    if (kind == "boundary-contract" && spec != nullptr && spec->contains("test_variants")) {
        const std::string text = "<a>x";
        for (const nlohmann::json& variant : spec->at("test_variants")) {
            const std::string name = variant.get<std::string>();
            product::RenderedChat rendered;
            rendered.text = text;
            if (name == "user_zero_only") {
                rendered.user_turn_byte_offset = 0;
            } else if (name == "stable_zero_only") {
                rendered.stable_prefix_byte_offset = 0;
            } else {
                throw std::runtime_error("unsupported boundary variant " + name);
            }
            const Outcome outcome = encode_rendered(vocab, impl, rendered);
            emit_rendered_outcome(emitter, id + "#" + name,
                                  {{"text_hex", hex_encode(text)}, {"variant", name}}, vocab, outcome);
        }
        return;
    }
    if (kind == "boundary-contract" && spec != nullptr && spec->contains("chat_transform")) {
        if (!templates.reasoning) { throw std::runtime_error("missing reasoning template"); }
        const nlohmann::json options_json{{"add_generation_prompt", true},
                                          {"enable_thinking", true},
                                          {"preserve_thinking", true},
                                          {"reasoning_effort", "medium"}};
        emit_prepared(emitter, id + "#base", vocab, impl, *templates.reasoning,
                      boundary_user_reset_messages(false), options_json, nlohmann::json::array(), true,
                      "reasoning_effort");
        emit_prepared(emitter, id + "#after", vocab, impl, *templates.reasoning,
                      boundary_user_reset_messages(true), options_json, nlohmann::json::array(), true,
                      "reasoning_effort");
        return;
    }
    if (kind == "chat-transform") {
        if (!templates.reasoning || !templates.toggle) {
            throw std::runtime_error("chat-transform requires both templates");
        }
        nlohmann::json messages = row.at("messages");
        nlohmann::json option_json = row.value("options", nlohmann::json::object());
        nlohmann::json tools       = nlohmann::json::array();
        if (row.contains("transform")) {
            const nlohmann::json& transform = row.at("transform");
            messages = apply_messages(std::move(messages), transform);
            if (transform.contains("after_options")) {
                for (auto it = transform.at("after_options").begin();
                     it != transform.at("after_options").end(); ++it) {
                    option_json[it.key()] = it.value();
                }
            }
            if (transform.contains("after_tools")) { tools = transform.at("after_tools"); }
        }
        const nlohmann::json templates_json =
            row.value("templates", nlohmann::json::array({"reasoning_effort"}));
        for (const nlohmann::json& name_json : templates_json) {
            const std::string name = name_json.get<std::string>();
            if (name == "reasoning_effort") {
                emit_prepared(emitter, id + "#chat#reasoning_effort", vocab, impl, *templates.reasoning,
                              messages, option_json, tools, true, name);
            } else if (name == "thinking_toggle_compatible_options") {
                emit_prepared(emitter, id + "#chat#thinking_toggle", vocab, impl, *templates.toggle,
                              messages, option_json, tools, false, name);
            } else {
                throw std::runtime_error("unsupported template " + name);
            }
        }
        return;
    }

    const std::string text = case_text(row);
    const bool parse_added = parse_added_of(row);
    const nlohmann::json& boundary_spec =
        row.contains("rendered") ? row.at("rendered") : spec != nullptr ? *spec : nlohmann::json::object();
    if ((kind == "rendered-contract" || kind == "boundary-contract") && boundary_fields(boundary_spec)) {
        const product::RenderedChat rendered = load_rendered(text, boundary_spec);
        const Outcome outcome                = encode_rendered(vocab, impl, rendered);
        emit_rendered_outcome(emitter, id + "#rendered",
                              {{"spec", boundary_spec}, {"text_hex", hex_encode(text)}}, vocab, outcome);
    }
    if (has_explicit_text(row)) {
        const bool all_cuts = (spec != nullptr && spec->value("all_byte_cuts", false)) ||
                              row.value("all_byte_cuts", false);
        emit_coverage(emitter, vocab, impl, id, text, parse_added, all_cuts);
    }
}

nlohmann::json fuzz_added() {
    return nlohmann::json::array({nlohmann::json{{"id", 300}, {"content", "<|im_end|>"}},
                                  {{"id", 301}, {"content", "<|im_start|>"}},
                                  {{"id", 302}, {"content", "<think>"}},
                                  {{"id", 303}, {"content", "</think>"}},
                                  {{"id", 304}, {"content", "ab"}},
                                  {{"id", 305}, {"content", "abc"}}});
}

std::string fuzz_piece(std::mt19937& rng, int index) {
    std::uniform_int_distribution<int> length(0, 80);
    std::uniform_int_distribution<int> byte(0, 255);
    const int kind = index % 8;
    if (kind == 0) {
        std::string text(static_cast<std::size_t>(length(rng)), '\0');
        for (char& ch : text) { ch = static_cast<char>(byte(rng)); }
        return text;
    }
    if (kind == 1) { return "x<|im_sta"; }
    if (kind == 2) { return "<|im_start|>assistant\n<think>\n"; }
    if (kind == 3) { return std::string("cafe") + "\xCC\x81\xCC\xA7"; }
    if (kind == 4) { return "<|im_end|"; }
    if (kind == 5) {
        return std::string("abcabc") + std::string(static_cast<std::size_t>(index % 5), '<');
    }
    if (kind == 6) {
        static constexpr char kNul[] = {'a', '\0', 'b', '<', '|', 'i', 'm', '_', 'e', 'n', 'd', '|', '>'};
        return std::string(kNul, sizeof(kNul));
    }
    std::string text = "中文";
    text.append(static_cast<std::size_t>(index % 7), 'x');
    text.push_back(static_cast<char>(0xFF));
    return text;
}

void usage() {
    std::cerr << "usage: ninfer_qwen3_8_g1_diff --impl legacy|candidate --profile real|synthetic "
                 "--resources DIR --cases JSONL --output JSONL --summary JSON "
                 "[--toggle-template FILE --seed N --fuzz N]\n"
                 "   or: ninfer_qwen3_8_g1_diff --count-body FILE --resources DIR [--impl legacy|candidate]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need             = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { throw std::runtime_error(std::string("missing ") + flag); }
            return argv[++i];
        };
        if (arg == "--impl") {
            const std::string value = need("--impl");
            if (value == "legacy") {
                options.impl = Impl::Legacy;
            } else if (value == "candidate") {
                options.impl = Impl::Candidate;
            } else {
                throw std::runtime_error("impl must be legacy or candidate");
            }
        } else if (arg == "--profile") {
            const std::string value = need("--profile");
            if (value == "real") {
                options.profile = Profile::Real;
            } else if (value == "synthetic") {
                options.profile = Profile::Synthetic;
            } else {
                throw std::runtime_error("profile must be real or synthetic");
            }
        } else if (arg == "--resources") {
            options.resources = need("--resources");
        } else if (arg == "--cases") {
            options.cases = need("--cases");
        } else if (arg == "--output") {
            options.output = need("--output");
        } else if (arg == "--summary") {
            options.summary = need("--summary");
        } else if (arg == "--toggle-template") {
            options.toggle = need("--toggle-template");
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(need("--seed")));
        } else if (arg == "--fuzz") {
            options.fuzz = std::stoi(need("--fuzz"));
        } else if (arg == "--count-body") {
            options.count_body = need("--count-body");
        } else {
            usage();
            throw std::runtime_error("unknown argument");
        }
    }
    return options;
}

int count_body(const Options& options) {
    const nlohmann::json body = nlohmann::json::parse(read_file(options.count_body));
    const bool thinking       = body.value("enable_thinking", true);
    const bool preserve       = body.value("preserve_thinking", true);
    nlohmann::json option_json{{"add_generation_prompt", true},
                               {"enable_thinking", thinking},
                               {"preserve_thinking", preserve}};
    if (thinking) { option_json["reasoning_effort"] = body.value("reasoning_effort", "medium"); }
    const Vocab vocab = make_vocab(nlohmann::json::array(), nlohmann::json::array(), Profile::Real,
                                   options.resources, options.impl, true);
    const product::CompiledChatTemplate chat = product::CompiledChatTemplate::resolve(
        read_pinned_template(options.resources + "/chat_template.jinja"));
    const product::RenderedChat rendered = chat.render(
        messages_from(body.at("messages")),
        render_options_for(option_json, body.value("tools", nlohmann::json::array()), true));
    const Outcome outcome = encode_rendered(vocab, options.impl, rendered);
    if (!outcome.ok) {
        throw std::runtime_error(outcome.error_class + ": " + outcome.error_message);
    }
    nlohmann::json row{{"hints", outcome.encoded.checkpoint_hints.size()},
                       {"impl", options.impl == Impl::Legacy ? "legacy" : "candidate"},
                       {"prompt_tokens", outcome.token_ids.size()},
                       {"rendered_chars", rendered.text.size()},
                       {"token_ids_sha256", sha256_bytes(nlohmann::json(outcome.token_ids).dump())}};
    std::cout << row.dump() << '\n';
    return 0;
}

const Vocab& vocab_for(std::map<std::string, Vocab>& vocabs, const nlohmann::json& added,
                       const nlohmann::json& merges, const Options& options, bool force_real) {
    const std::string key = added.dump() + "\n" + merges.dump() + "\n" +
                            (force_real || options.profile == Profile::Real ? "real" : "syn");
    const auto found = vocabs.find(key);
    if (found != vocabs.end()) { return found->second; }
    Vocab vocab = make_vocab(added, merges, options.profile, options.resources, options.impl, force_real);
    const auto inserted = vocabs.emplace(key, std::move(vocab));
    return inserted.first->second;
}

} // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(std::size_t) >= 8, "g1 uint32 boundary probe requires a 64-bit size_t");
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try {
        const Options options = parse_options(argc, argv);
        if (!options.count_body.empty()) { return count_body(options); }
        if (options.cases.empty() || options.output.empty() || options.resources.empty() ||
            options.summary.empty()) {
            usage();
            return 2;
        }
        std::ifstream cases(options.cases);
        if (!cases) {
            std::cerr << "BLOCKED missing cases\n";
            return 3;
        }
        std::ofstream output(options.output);
        if (!output) { throw std::runtime_error("cannot write output"); }
        Templates templates;
        if (options.profile == Profile::Real) {
            if (options.toggle.empty()) {
                throw std::runtime_error("real profile requires --toggle-template");
            }
            templates.reasoning = product::CompiledChatTemplate::resolve(
                read_pinned_template(options.resources + "/chat_template.jinja"));
            templates.toggle =
                product::CompiledChatTemplate::resolve(read_pinned_template(options.toggle));
        }
        Emitter emitter{output};
        std::map<std::string, Vocab> vocabs;
        int applied = 0;
        int skipped = 0;
        int lines   = 0;
        std::string line;
        while (std::getline(cases, line)) {
            if (line.empty()) { continue; }
            ++lines;
            const nlohmann::json row = nlohmann::json::parse(line);
            const Side side          = side_for(row);
            if (!applies(options.profile, side)) {
                ++skipped;
                continue;
            }
            const int rows_before = emitter.rows;
            const std::string kind = row.at("kind").get<std::string>();
            const bool force_real =
                options.profile == Profile::Real &&
                (kind == "chat-transform" || kind == "raw-boundaries" || kind == "rendered-contract" ||
                 (kind == "scanner-contract" &&
                  spec_of(row) != nullptr && spec_of(row)->value("real_resources", false)) ||
                 (kind == "boundary-contract" && spec_of(row) != nullptr &&
                  spec_of(row)->contains("chat_transform")));
            nlohmann::json added = added_spec_of(row);
            nlohmann::json merges = merges_of(row);
            if (kind == "trace-lifecycle") { added = lifecycle_added(); }
            if (force_real) {
                added  = nlohmann::json::array();
                merges = nlohmann::json::array();
            }
            const Vocab* vocab = nullptr;
            try {
                vocab = &vocab_for(vocabs, added, merges, options, force_real);
            } catch (const std::exception& ex) {
                Outcome outcome;
                outcome.error_class   = exception_class(ex);
                outcome.error_message = ex.what();
                const std::string id  = row.at("case_id").get<std::string>();
                emit_error(emitter, id + "#load", "raw",
                           {{"case_id", id}, {"phase", "load"}},
                           sha256_bytes(std::string("vocab-load:") + id), "load", outcome);
                ++applied;
                continue;
            }
            emit_case(emitter, row, *vocab, options.impl, templates);
            if (emitter.rows == rows_before) {
                throw std::runtime_error("case produced no rows: " + row.at("case_id").get<std::string>());
            }
            ++applied;
        }
        const int expect_applied = options.profile == Profile::Real ? kRealCases : kSyntheticCases;
        if (applied != expect_applied || applied + skipped != lines) {
            throw std::runtime_error("case routing mismatch applied=" + std::to_string(applied) +
                                     " skipped=" + std::to_string(skipped) +
                                     " lines=" + std::to_string(lines));
        }
        if (options.fuzz > 0) {
            const bool real_fuzz = options.profile == Profile::Real;
            const nlohmann::json added = real_fuzz ? nlohmann::json::array() : fuzz_added();
            const Vocab& vocab = vocab_for(vocabs, added, nlohmann::json::array(), options, real_fuzz);
            std::mt19937 rng(options.seed);
            for (int i = 0; i < options.fuzz; ++i) {
                const std::string text = fuzz_piece(rng, i);
                const bool parse_added = i % 11 != 0;
                const Outcome outcome  = encode_raw(vocab, options.impl, text, parse_added);
                char id[32];
                std::snprintf(id, sizeof(id), "fuzz-%06d", i);
                const nlohmann::json identity{{"fuzz", i},
                                              {"parse_added_tokens", parse_added},
                                              {"seed", options.seed},
                                              {"text_hex", hex_encode(text)}};
                if (outcome.ok) {
                    emit_ok(emitter, id, "raw", identity, vocab.resources_sha,
                            {{"token_ids", outcome.token_ids}});
                } else {
                    emit_error(emitter, id, "raw", identity, vocab.resources_sha, "encode", outcome);
                }
                if (i > 0 && i % 5000 == 0) { std::cerr << "fuzz " << i << '\n'; }
            }
        }
        output.flush();
        if (!output.good()) { throw std::runtime_error("output write failed"); }
        nlohmann::json summary{{"applied_cases", applied},
                               {"cut_policy", "full_when_bytes_le_256_else_marker_samples"},
                               {"fuzz", options.fuzz},
                               {"impl", options.impl == Impl::Legacy ? "legacy" : "candidate"},
                               {"lines", lines},
                               {"profile", options.profile == Profile::Real ? "real" : "synthetic"},
                               {"rows", emitter.rows},
                               {"seed", options.seed},
                               {"skipped_cases", skipped}};
        std::ofstream summary_out(options.summary);
        summary_out << summary.dump(2) << '\n';
        std::cerr << summary.dump() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "g1_diff failed: " << ex.what() << '\n';
        return 1;
    }
}
