#include "media/decode/decode.h"
#include "targets/qwen3_8/impl/frontend/processor.h"
#include "targets/qwen3_8/reference/legacy_encode_rendered_chat.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::media::decode {

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::logic_error("decode_image is unavailable in the boundary test");
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::logic_error("decode_video is unavailable in the boundary test");
}

} // namespace ninfer::media::decode

namespace {

namespace product = ninfer::targets::qwen3_8::frontend_internal;
namespace legacy  = ninfer::targets::qwen3_8::legacy_oracle;

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
        vocab[utf8_codepoint(shown ? static_cast<std::uint32_t>(byte) : next++)] = byte;
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

struct Fixture {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
};

Fixture make_fixture(const nlohmann::json& added) {
    Fixture fixture;
    fixture.tokenizer_json = nlohmann::json{{"model",
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
    fixture.tokenizer_config_json =
        nlohmann::json{{"added_tokens_decoder", std::move(decoder)}}.dump();
    fixture.generation_config_json = R"({"eos_token_id":0})";
    return fixture;
}

product::Tokenizer make_product(const Fixture& fixture) {
    return product::Tokenizer(product::TokenizerResources{
        fixture.tokenizer_json, fixture.tokenizer_config_json, fixture.generation_config_json});
}

legacy::Tokenizer make_legacy(const Fixture& fixture) {
    return legacy::Tokenizer(legacy::TokenizerResources{
        fixture.tokenizer_json, fixture.tokenizer_config_json, fixture.generation_config_json});
}

bool same_hints(const product::EncodedChat& lhs, const product::EncodedChat& rhs) {
    if (lhs.checkpoint_hints.size() != rhs.checkpoint_hints.size()) { return false; }
    for (std::size_t i = 0; i < lhs.checkpoint_hints.size(); ++i) {
        if (lhs.checkpoint_hints[i].kind != rhs.checkpoint_hints[i].kind ||
            lhs.checkpoint_hints[i].boundary != rhs.checkpoint_hints[i].boundary) {
            return false;
        }
    }
    return true;
}

void expect_same_chat(const product::Tokenizer& modern, const legacy::Tokenizer& old,
                      const product::RenderedChat& rendered, const char* label) {
    std::string modern_error;
    std::string legacy_error;
    product::EncodedChat modern_chat;
    product::EncodedChat legacy_chat;
    try {
        modern_chat = product::encode_rendered_chat(modern, rendered);
    } catch (const std::exception& ex) { modern_error = ex.what(); }
    try {
        legacy_chat = legacy::encode_rendered_chat(old, rendered);
    } catch (const std::exception& ex) { legacy_error = ex.what(); }
    if (modern_error != legacy_error) {
        std::cerr << label << " error\n modern: " << modern_error << "\n legacy: " << legacy_error
                  << '\n';
        ++g_failures;
        return;
    }
    if (!modern_error.empty()) { return; }
    expect(modern_chat.input_ids == legacy_chat.input_ids, label);
    expect(modern_chat.stable_prefix_boundary == legacy_chat.stable_prefix_boundary, label);
    expect(modern_chat.turn_rewrite_boundary == legacy_chat.turn_rewrite_boundary, label);
    expect(modern_chat.user_turn_boundary == legacy_chat.user_turn_boundary, label);
    expect(same_hints(modern_chat, legacy_chat), label);
}

void test_uint32_order() {
    const std::size_t huge = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    bool saw_logic         = false;
    try {
        (void)product::enforce_exact_token_prefix(false, huge, "stable prefix");
    } catch (const std::logic_error& ex) {
        saw_logic = std::string(ex.what()) == "stable prefix is not an exact token prefix";
        expect(dynamic_cast<const std::overflow_error*>(&ex) == nullptr,
               "exact failure was reported as overflow_error");
    }
    expect(saw_logic, "illegal exact prefix did not throw logic_error first");

    bool saw_overflow = false;
    try {
        (void)product::enforce_exact_token_prefix(true, huge, "turn rewrite prefix");
    } catch (const std::overflow_error& ex) {
        saw_overflow =
            std::string(ex.what()) == "turn rewrite prefix token boundary exceeds uint32";
    } catch (const std::logic_error&) { saw_overflow = false; }
    expect(saw_overflow, "legal oversized prefix did not throw overflow_error");
}

void test_contracts(const product::Tokenizer& modern, const legacy::Tokenizer& old) {
    product::RenderedChat full;
    full.text                      = "ab";
    full.stable_prefix_byte_offset = 2;
    const product::EncodedChat stable = product::encode_rendered_chat(modern, full);
    expect(stable.stable_prefix_boundary == 1, "stable full prefix was rejected");
    expect(stable.encoding_work.main_encode_calls == 1, "stable full encoded more than once");
    expect_same_chat(modern, old, full, "stable full");

    product::RenderedChat turn_full = full;
    turn_full.stable_prefix_byte_offset.reset();
    turn_full.turn_rewrite_byte_offset = 2;
    expect_same_chat(modern, old, turn_full, "turn full forbidden");

    product::RenderedChat skipped;
    skipped.text                   = "ab";
    skipped.user_turn_byte_offset  = 0;
    const product::EncodedChat user0 = product::encode_rendered_chat(modern, skipped);
    expect(!user0.user_turn_boundary.has_value(), "user offset 0 was validated");
    expect(user0.encoding_work.boundary_queries == 0, "user offset 0 was queried");

    product::RenderedChat invalid;
    invalid.text                      = "\xff";
    invalid.stable_prefix_byte_offset = 99;
    std::string modern_error;
    std::string legacy_error;
    try {
        (void)product::encode_rendered_chat(modern, invalid);
    } catch (const std::exception& ex) { modern_error = ex.what(); }
    try {
        (void)old.encode("\xff");
    } catch (const std::exception& ex) { legacy_error = ex.what(); }
    expect(!modern_error.empty() && modern_error == legacy_error, "main invalid was not first");
    expect(modern_error.find("byte offset exceeds") == std::string::npos,
           "main invalid was reported as a boundary overflow");

    product::RenderedChat overlong;
    overlong.text                      = "ab";
    overlong.stable_prefix_byte_offset = 99;
    expect_same_chat(modern, old, overlong, "overlong boundary");

    product::RenderedChat cut;
    cut.text                      = "\xC3\xA9";
    cut.stable_prefix_byte_offset = 1;
    expect_same_chat(modern, old, cut, "cut utf-8");

    product::RenderedChat unordered;
    unordered.text = "xxab";
    unordered.checkpoint_hints.push_back({product::SemanticCheckpointKind::Rolling, 4});
    unordered.checkpoint_hints.push_back({product::SemanticCheckpointKind::StableTurn, 2});
    expect_same_chat(modern, old, unordered, "unordered hints");

    product::RenderedChat inside;
    inside.text                      = "xxab";
    inside.stable_prefix_byte_offset = 3;
    expect_same_chat(modern, old, inside, "cut inside added token");

    product::RenderedChat same_offset;
    same_offset.text                      = "ab";
    same_offset.stable_prefix_byte_offset = 2;
    same_offset.turn_rewrite_byte_offset  = 2;
    expect_same_chat(modern, old, same_offset, "stable full then turn full");

    product::RenderedChat next_token;
    next_token.text                      = "ab<a>z";
    next_token.stable_prefix_byte_offset = 3;
    expect_same_chat(modern, old, next_token, "cut inside added token after a restart");
}

void test_local_work(const product::Tokenizer& modern, const legacy::Tokenizer& old) {
    const std::string marker = "<a>";
    const std::string body(2000, 'q');
    const std::string opener(8, 'z');
    product::RenderedChat rendered;
    rendered.user_turn_byte_offset = 0;
    for (int i = 0; i < 32; ++i) {
        rendered.text += body;
        rendered.text += marker;
        rendered.text += opener;
        rendered.checkpoint_hints.push_back(
            {product::SemanticCheckpointKind::Rolling, rendered.text.size()});
    }
    const std::size_t repeated = rendered.checkpoint_hints.back().byte_offset;
    rendered.checkpoint_hints.push_back({product::SemanticCheckpointKind::StablePrefix, repeated});
    rendered.stable_prefix_byte_offset = rendered.checkpoint_hints[1].byte_offset;
    rendered.turn_rewrite_byte_offset  = rendered.checkpoint_hints[2].byte_offset;

    const product::EncodedChat encoded = product::encode_rendered_chat(modern, rendered);
    expect_same_chat(modern, old, rendered, "prodshape-like");
    const product::BoundaryEncodingWork& work = encoded.encoding_work;
    expect(work.main_encode_calls == 1, "main encode was repeated");
    expect(work.main_input_bytes == rendered.text.size(), "main bytes were not the rendered chat");
    expect(work.checkpoint_hints == rendered.checkpoint_hints.size(), "hint count was rewritten");
    expect(work.boundary_queries == 2 + rendered.checkpoint_hints.size(), "boundary query count");
    expect(work.boundary_unique_offsets == 32, "repeated offsets were encoded as unique");
    expect(work.local_encode_calls == 32, "repeated offset encoded the local slice again");
    expect(work.memo_hits >= 1, "repeated offset missed the memo");
    expect(work.local_input_bytes == 32 * opener.size(), "local bytes were not the opener tails");
    expect(work.local_max_bytes == opener.size(), "local max was not one opener");
    expect(work.local_input_bytes * 10 < work.main_input_bytes, "local work repeated the prefix");
    expect(work.main_work.scan_positions <= work.main_input_bytes, "main scan exceeded input bytes");
    expect(work.main_work.scan_positions > 0, "main scan was not counted");

    product::RenderedChat plain;
    plain.text = std::string(1000, 'q');
    plain.stable_prefix_byte_offset = 800;
    const product::EncodedChat fallback = product::encode_rendered_chat(modern, plain);
    expect_same_chat(modern, old, plain, "no added fallback");
    expect(fallback.encoding_work.main_encode_calls == 1, "fallback main encode");
    expect(fallback.encoding_work.local_encode_calls == 1, "fallback local encode");
    expect(fallback.encoding_work.local_input_bytes == 800,
           "no-added boundary did not encode its prefix from zero");
}

} // namespace

int main() {
    try {
        test_uint32_order();
        const Fixture fixture =
            make_fixture(nlohmann::json::array({added_token(300, "ab"), added_token(301, "<a>")}));
        const product::Tokenizer modern = make_product(fixture);
        const legacy::Tokenizer old     = make_legacy(fixture);
        test_contracts(modern, old);
        test_local_work(modern, old);
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
