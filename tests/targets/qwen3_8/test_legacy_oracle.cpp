#include "targets/qwen3_8/reference/legacy_encode_rendered_chat.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::targets::qwen3_8::frontend_internal::RenderedChat;
using ninfer::targets::qwen3_8::frontend_internal::SemanticCheckpointKind;
using ninfer::targets::qwen3_8::legacy_oracle::Tokenizer;
using ninfer::targets::qwen3_8::legacy_oracle::TokenizerResources;
using ninfer::targets::qwen3_8::legacy_oracle::encode_rendered_chat;
using ninfer::targets::qwen3_8::legacy_oracle::enforce_exact_then_uint32;

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << message << '\n';
    ++g_failures;
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
        const std::uint32_t codepoint = shown ? static_cast<std::uint32_t>(byte) : next++;
        vocab[utf8_codepoint(codepoint)] = byte;
    }
    return vocab;
}

nlohmann::json added_token(int id, std::string content) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", false}};
}

Tokenizer make_tokenizer(const nlohmann::json& added) {
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
    const std::string tokenizer_config =
        nlohmann::json{{"added_tokens_decoder", std::move(decoder)}}.dump();
    const std::string generation = R"({"eos_token_id":0})";
    return Tokenizer(TokenizerResources{tokenizer_json, tokenizer_config, generation});
}

void test_added_priority() {
    const Tokenizer shorter_first =
        make_tokenizer(nlohmann::json::array({added_token(300, "ab"), added_token(301, "abc")}));
    const std::vector<int> ids = shorter_first.encode("abc");
    expect(ids == std::vector<int>({300, 99}), "lower id shorter token did not leave ordinary c");
    expect(ids == shorter_first.encode("abc"), "legacy encode was not deterministic");

    const Tokenizer longer_first =
        make_tokenizer(nlohmann::json::array({added_token(300, "abc"), added_token(301, "ab")}));
    expect(longer_first.encode("abc") == std::vector<int>({300}),
           "lower id longer token was not selected");

    // Insert the larger id first. The decoder sort, not the JSON array order, must win.
    const Tokenizer unsorted = make_tokenizer(
        nlohmann::json::array({added_token(301, "ab"), added_token(300, "abc")}));
    expect(unsorted.encode("abc") == std::vector<int>({300}),
           "id sort was skipped and insertion order won");
}

void test_overflow_order() {
    const std::size_t huge = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    bool saw_logic         = false;
    try {
        enforce_exact_then_uint32(false, huge, "stable prefix");
    } catch (const std::logic_error& ex) {
        saw_logic = std::string(ex.what()) == "stable prefix is not an exact token prefix";
        expect(dynamic_cast<const std::overflow_error*>(&ex) == nullptr,
               "exact failure was reported as overflow_error");
    }
    expect(saw_logic, "illegal exact prefix did not throw logic_error first");

    bool saw_overflow = false;
    try {
        enforce_exact_then_uint32(true, huge, "turn rewrite prefix");
    } catch (const std::overflow_error& ex) {
        saw_overflow = std::string(ex.what()) == "turn rewrite prefix token boundary exceeds uint32";
    } catch (const std::logic_error&) {
        saw_overflow = false;
    }
    expect(saw_overflow, "legal oversized prefix did not throw overflow_error");
}

void test_boundary_messages() {
    const Tokenizer tokenizer =
        make_tokenizer(nlohmann::json::array({added_token(300, "ab")}));
    RenderedChat rendered;
    rendered.text                      = "ab";
    rendered.stable_prefix_byte_offset = 99;
    bool saw                           = false;
    try {
        (void)encode_rendered_chat(tokenizer, rendered);
    } catch (const std::logic_error& ex) {
        saw = std::string(ex.what()) == "stable prefix byte offset exceeds rendered chat";
    }
    expect(saw, "overlong boundary message mismatch");

    rendered.stable_prefix_byte_offset.reset();
    rendered.user_turn_byte_offset = 0;
    const auto skipped             = encode_rendered_chat(tokenizer, rendered);
    expect(!skipped.user_turn_boundary.has_value(), "user offset 0 was validated");

    rendered.user_turn_byte_offset.reset();
    const std::string text = "\xC3\xA9";
    rendered.text          = text;
    rendered.stable_prefix_byte_offset = 1;
    bool wrapped                        = false;
    try {
        (void)encode_rendered_chat(tokenizer, rendered);
    } catch (const std::logic_error& ex) {
        wrapped = std::string(ex.what()) == "stable prefix cannot be tokenized independently";
    }
    expect(wrapped, "cut utf-8 prefix was not wrapped");

    rendered = {};
    rendered.text = "xxab";
    rendered.checkpoint_hints.push_back({SemanticCheckpointKind::Rolling, 4});
    rendered.checkpoint_hints.push_back({SemanticCheckpointKind::StableTurn, 2});
    bool unordered = false;
    try {
        (void)encode_rendered_chat(tokenizer, rendered);
    } catch (const std::logic_error& ex) {
        unordered = std::string(ex.what()) == "semantic checkpoint boundaries are not ordered";
    }
    expect(unordered, "unordered hints were accepted");
}

} // namespace

int main() {
    try {
        test_added_priority();
        test_overflow_order();
        test_boundary_messages();
    } catch (const std::exception& ex) {
        std::cerr << "unexpected exception: " << ex.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    return 0;
}
