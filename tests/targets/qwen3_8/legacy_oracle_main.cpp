#include "artifact/sha256.h"
#include "targets/qwen3_8/reference/legacy_encode_rendered_chat.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::Sha256;
using ninfer::targets::qwen3_8::frontend_internal::EncodedChat;
using ninfer::targets::qwen3_8::frontend_internal::RenderedChat;
using ninfer::targets::qwen3_8::frontend_internal::SemanticCheckpointKind;
using ninfer::targets::qwen3_8::legacy_oracle::EncodeOptions;
using ninfer::targets::qwen3_8::legacy_oracle::Tokenizer;
using ninfer::targets::qwen3_8::legacy_oracle::TokenizerResources;
using ninfer::targets::qwen3_8::legacy_oracle::encode_rendered_chat;

std::string sha256_bytes(std::string_view text) {
    Sha256 hasher;
    const auto* data = reinterpret_cast<const std::byte*>(text.data());
    hasher.update(std::span<const std::byte>(data, text.size()));
    const auto digest = hasher.finish();
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        out[i * 2]     = kHex[digest[i] >> 4];
        out[i * 2 + 1] = kHex[digest[i] & 0x0F];
    }
    return out;
}

std::string sha256_hex(std::string_view text) { return sha256_bytes(text); }

std::string directory_resources_sha(const std::string& dir) {
    static constexpr const char* kNames[] = {
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json"};
    nlohmann::json files = nlohmann::json::object();
    for (const char* name : kNames) {
        const std::string path = dir + "/" + name;
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            if (std::string(name) == "chat_template.jinja" ||
                std::string(name) == "preprocessor_config.json" ||
                std::string(name) == "video_preprocessor_config.json") {
                continue;
            }
            throw std::runtime_error(std::string("BLOCKED missing resource: ") + path);
        }
        const std::string raw((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        files[name] = {{"bytes", raw.size()}, {"sha256", sha256_bytes(raw)}};
    }
    return sha256_bytes(files.dump());
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("missing resource: " + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

nlohmann::json added_from_spec(const nlohmann::json& spec) {
    nlohmann::json added = nlohmann::json::array();
    for (const nlohmann::json& item : spec) {
        added.push_back(nlohmann::json{{"id", item.at("id").get<int>()},
                                       {"content", item.at("content").get<std::string>()},
                                       {"single_word", false},
                                       {"lstrip", false},
                                       {"rstrip", false},
                                       {"normalized", false},
                                       {"special", item.value("special", false)}});
    }
    return added;
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

Tokenizer make_inline(const nlohmann::json& added_spec) {
    const nlohmann::json added = added_from_spec(added_spec);
    const std::string tokenizer_json = nlohmann::json{{"model",
                                                       {{"type", "BPE"},
                                                        {"vocab", byte_vocab()},
                                                        {"merges", nlohmann::json::array()}}},
                                                      {"added_tokens", added}}
                                           .dump();
    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : added) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    const std::string config = nlohmann::json{{"added_tokens_decoder", std::move(decoder)}}.dump();
    const std::string generation = R"({"eos_token_id":0})";
    return Tokenizer(TokenizerResources{tokenizer_json, config, generation});
}

Tokenizer make_from_dir(const std::string& dir) {
    const std::string tokenizer_json = read_file(dir + "/tokenizer.json");
    const std::string config         = read_file(dir + "/tokenizer_config.json");
    const std::string generation     = read_file(dir + "/generation_config.json");
    return Tokenizer(TokenizerResources{tokenizer_json, config, generation});
}

SemanticCheckpointKind parse_kind(const std::string& kind) {
    if (kind == "StablePrefix") { return SemanticCheckpointKind::StablePrefix; }
    if (kind == "StableTurn") { return SemanticCheckpointKind::StableTurn; }
    if (kind == "Rolling") { return SemanticCheckpointKind::Rolling; }
    throw std::invalid_argument("unknown checkpoint kind " + kind);
}

RenderedChat rendered_from(const nlohmann::json& row) {
    RenderedChat rendered;
    rendered.text = row.at("text").get<std::string>();
    if (row.contains("stable_prefix_byte_offset") && !row.at("stable_prefix_byte_offset").is_null()) {
        rendered.stable_prefix_byte_offset = row.at("stable_prefix_byte_offset").get<std::size_t>();
    }
    if (row.contains("turn_rewrite_byte_offset") && !row.at("turn_rewrite_byte_offset").is_null()) {
        rendered.turn_rewrite_byte_offset = row.at("turn_rewrite_byte_offset").get<std::size_t>();
    }
    if (row.contains("user_turn_byte_offset") && !row.at("user_turn_byte_offset").is_null()) {
        rendered.user_turn_byte_offset = row.at("user_turn_byte_offset").get<std::size_t>();
    }
    if (row.contains("checkpoint_hints")) {
        for (const nlohmann::json& hint : row.at("checkpoint_hints")) {
            rendered.checkpoint_hints.push_back(
                {parse_kind(hint.at("kind").get<std::string>()),
                 hint.at("byte_offset").get<std::size_t>()});
        }
    }
    return rendered;
}

const char* exception_class(const std::exception& ex) {
    if (dynamic_cast<const std::overflow_error*>(&ex) != nullptr) { return "overflow_error"; }
    if (dynamic_cast<const std::invalid_argument*>(&ex) != nullptr) { return "invalid_argument"; }
    if (dynamic_cast<const std::out_of_range*>(&ex) != nullptr) { return "out_of_range"; }
    if (dynamic_cast<const std::logic_error*>(&ex) != nullptr) { return "logic_error"; }
    return "exception";
}

nlohmann::json boundary_value(const std::optional<std::uint32_t>& value) {
    if (!value) { return nullptr; }
    return *value;
}

nlohmann::json hints_json(const EncodedChat& encoded) {
    nlohmann::json hints = nlohmann::json::array();
    for (const auto& hint : encoded.checkpoint_hints) {
        const char* kind = "StablePrefix";
        if (hint.kind == SemanticCheckpointKind::StableTurn) { kind = "StableTurn"; }
        if (hint.kind == SemanticCheckpointKind::Rolling) { kind = "Rolling"; }
        hints.push_back({{"kind", kind}, {"boundary", hint.boundary}});
    }
    return hints;
}

nlohmann::json semantic_for(const std::string& kind, const EncodedChat& encoded,
                            std::string_view text) {
    nlohmann::json semantic = {{"token_ids", encoded.input_ids},
                               {"stable_prefix_boundary", boundary_value(encoded.stable_prefix_boundary)},
                               {"turn_rewrite_boundary", boundary_value(encoded.turn_rewrite_boundary)},
                               {"user_turn_boundary", boundary_value(encoded.user_turn_boundary)},
                               {"checkpoint_hints", hints_json(encoded)}};
    if (kind == "count") { return nlohmann::json{{"count", encoded.input_ids.size()}}; }
    if (kind == "raw") { return nlohmann::json{{"token_ids", encoded.input_ids}}; }
    if (kind == "prepared") {
        const std::size_t n = encoded.input_ids.size();
        nlohmann::json types = nlohmann::json::array();
        nlohmann::json positions = nlohmann::json::array();
        for (std::size_t i = 0; i < n; ++i) { types.push_back(0); }
        for (int axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < n; ++i) { positions.push_back(static_cast<int>(i)); }
        }
        semantic["token_types"]          = std::move(types);
        semantic["positions"]            = std::move(positions);
        semantic["rope_delta"]           = 0;
        semantic["identity"]             = sha256_hex(semantic["token_ids"].dump());
        semantic["starts_in_reasoning"]  = false;
        semantic["rendered_sha256"]      = sha256_hex(text);
    }
    return semantic;
}

void usage() {
    std::cerr << "usage: ninfer_qwen3_8_legacy_frontend_oracle --cases JSONL --output JSONL "
                 "[--resources DIR]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string cases_path;
    std::string output_path;
    std::string resources;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need             = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << '\n';
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--cases") {
            cases_path = need("--cases");
        } else if (arg == "--output") {
            output_path = need("--output");
        } else if (arg == "--resources") {
            resources = need("--resources");
        } else {
            usage();
            return 2;
        }
    }
    if (cases_path.empty() || output_path.empty()) {
        usage();
        return 2;
    }

    std::ifstream input(cases_path);
    if (!input) {
        std::cerr << "BLOCKED missing cases\n";
        return 3;
    }
    std::vector<nlohmann::json> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) { continue; }
        nlohmann::json row = nlohmann::json::parse(line);
        const std::string kind = row.at("kind").get<std::string>();
        if (kind != "raw" && kind != "rendered" && kind != "count" && kind != "prepared") {
            std::cerr << "unsupported kind " << kind << '\n';
            return 2;
        }
        rows.push_back(std::move(row));
    }

    std::optional<Tokenizer> from_dir;
    if (!resources.empty()) {
        try {
            from_dir = make_from_dir(resources);
        } catch (const std::exception& ex) {
            std::cerr << "BLOCKED " << ex.what() << '\n';
            return 3;
        }
    }

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "cannot write output\n";
        return 2;
    }
    for (const nlohmann::json& row : rows) {
        const std::string kind    = row.at("kind").get<std::string>();
        const std::string case_id = row.at("case_id").get<std::string>();
        const std::string text    = row.at("text").get<std::string>();
        const bool parse_added    = row.value("parse_added_tokens", true);
        if ((kind == "rendered" || kind == "prepared") && !parse_added) {
            std::cerr << "parse_added_tokens false is unsupported for " << kind << '\n';
            return 2;
        }
        const auto field_or_null = [&](const char* key) -> nlohmann::json {
            if (!row.contains(key) || row.at(key).is_null()) { return nullptr; }
            return row.at(key);
        };
        nlohmann::json identity = {{"added_tokens", field_or_null("added_tokens")},
                                   {"checkpoint_hints", field_or_null("checkpoint_hints")},
                                   {"kind", kind},
                                   {"parse_added_tokens", parse_added},
                                   {"stable_prefix_byte_offset", field_or_null("stable_prefix_byte_offset")},
                                   {"text", text},
                                   {"turn_rewrite_byte_offset", field_or_null("turn_rewrite_byte_offset")},
                                   {"user_turn_byte_offset", field_or_null("user_turn_byte_offset")}};
        const nlohmann::json added = added_from_spec(row.value("added_tokens", nlohmann::json::array()));
        const std::string inline_resources =
            nlohmann::json{{"added_tokens", added},
                           {"generation_config", R"({"eos_token_id":0})"},
                           {"model", "byte-vocab-v1"}}
                .dump();
        nlohmann::json result{{"case_id", case_id},
                              {"kind", kind},
                              {"input_sha256", sha256_hex(identity.dump())},
                              {"resources_sha256",
                               resources.empty() ? sha256_hex(inline_resources)
                                                 : directory_resources_sha(resources)}};
        try {
            if (row.value("real_resources", false) && !from_dir) {
                throw std::runtime_error("BLOCKED real resources were not provided");
            }
            Tokenizer tokenizer = row.value("real_resources", false)
                                      ? *from_dir
                                      : make_inline(row.value("added_tokens", nlohmann::json::array()));
            EncodeOptions options;
            options.parse_added_tokens = parse_added;
            EncodedChat encoded;
            if (kind == "rendered" || kind == "prepared") {
                encoded = encode_rendered_chat(tokenizer, rendered_from(row));
            } else {
                encoded.input_ids = tokenizer.encode(text, options);
            }
            result["status"]   = "OK";
            result["semantic"] = semantic_for(kind, encoded, text);
        } catch (const std::exception& ex) {
            const std::string message = ex.what();
            if (message.rfind("BLOCKED", 0) == 0) {
                std::cerr << message << '\n';
                return 3;
            }
            result["status"] = "ERROR";
            result["error"]  = {{"phase", "encode"}, {"class", exception_class(ex)}, {"message", message}};
        }
        output << result.dump() << '\n';
    }
    return 0;
}
