#include "targets/qwen3_8/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using ninfer::targets::qwen3_8::frontend_internal::EncodeOptions;
using ninfer::targets::qwen3_8::frontend_internal::Tokenizer;
using ninfer::targets::qwen3_8::frontend_internal::TokenizerResources;
using ninfer::targets::qwen3_8::frontend_internal::TokenizerWork;

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

// GPT-2 / Qwen byte-level symbols for every byte, so ordinary tails resolve
// without merges.
nlohmann::json byte_vocab() {
    nlohmann::json vocab = nlohmann::json::object();
    std::uint32_t next = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool shown =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174);
        const std::uint32_t codepoint = shown ? static_cast<std::uint32_t>(byte) : next++;
        vocab[utf8_codepoint(codepoint)] = byte;
    }
    return vocab;
}

nlohmann::json added_token(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
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
    fixture.tokenizer_config_json = nlohmann::json{{"added_tokens_decoder", std::move(decoder)}}.dump();
    fixture.generation_config_json = R"({"eos_token_id":0})";
    return fixture;
}

Tokenizer make_tokenizer(const Fixture& fixture) {
    return Tokenizer(TokenizerResources{fixture.tokenizer_json, fixture.tokenizer_config_json,
                                        fixture.generation_config_json});
}

bool same_work(const TokenizerWork& lhs, const TokenizerWork& rhs) {
    return lhs.scan_positions == rhs.scan_positions &&
           lhs.candidate_checks == rhs.candidate_checks &&
           lhs.selected_added_tokens == rhs.selected_added_tokens &&
           lhs.ordinary_segments == rhs.ordinary_segments &&
           lhs.ordinary_input_bytes == rhs.ordinary_input_bytes;
}

void test_empty_and_independence() {
    const Fixture fixture = make_fixture(nlohmann::json::array({added_token(300, "<a>")}));
    const Tokenizer tokenizer = make_tokenizer(fixture);
    TokenizerWork work;
    const std::vector<int> empty = tokenizer.encode("", {}, &work);
    expect(empty.empty(), "empty input produced tokens");
    expect(same_work(work, {}), "empty input changed counters");

    const std::vector<int> plain = tokenizer.encode("<a>tail");
    TokenizerWork counted;
    const std::vector<int> with_work = tokenizer.encode("<a>tail", {}, &counted);
    expect(plain == with_work, "work pointer changed token ids");
    expect(counted.selected_added_tokens == 1, "selected added token was not counted");
    expect(counted.ordinary_input_bytes == 4, "ordinary tail bytes were not counted");

    const TokenizerWork after_first = counted;
    tokenizer.encode("<a>", {}, &counted);
    expect(counted.selected_added_tokens == after_first.selected_added_tokens + 1,
           "second call did not accumulate on the caller-owned counter");

    TokenizerWork other;
    tokenizer.encode("<a>tail", {}, &other);
    expect(same_work(other, after_first), "a fresh counter saw another call's totals");
    tokenizer.encode("<a>tail");
    expect(same_work(other, after_first), "a null work call mutated a live counter");
}

void test_parse_added_tokens_false() {
    const Fixture fixture = make_fixture(nlohmann::json::array({added_token(300, "<a>")}));
    const Tokenizer tokenizer = make_tokenizer(fixture);
    EncodeOptions options;
    options.parse_added_tokens = false;
    TokenizerWork work;
    const std::vector<int> ids = tokenizer.encode("<a>", options, &work);
    expect(!ids.empty(), "parse-false encode dropped the ordinary text");
    expect(work.scan_positions == 0 && work.candidate_checks == 0 && work.selected_added_tokens == 0,
           "parse-false encode scanned added tokens");
    expect(work.ordinary_input_bytes == 3, "parse-false ordinary bytes mismatch");
    const std::vector<int> again = tokenizer.encode("<a>", options);
    expect(ids == again, "parse-false work path diverged");
}

void test_threads_do_not_share_counters() {
    const Fixture fixture = make_fixture(nlohmann::json::array(
        {added_token(300, "<a>"), added_token(301, "<b>")}));
    const Tokenizer tokenizer = make_tokenizer(fixture);
    constexpr int kThreads = 4;
    std::vector<TokenizerWork> works(static_cast<std::size_t>(kThreads));
    std::vector<std::vector<int>> ids(static_cast<std::size_t>(kThreads));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(kThreads));
    for (int index = 0; index < kThreads; ++index) {
        threads.emplace_back([&, index] {
            const std::string text = index % 2 == 0 ? "<a>x<b>" : "cafe<b>";
            for (int step = 0; step < 40; ++step) {
                ids[static_cast<std::size_t>(index)] =
                    tokenizer.encode(text, {}, &works[static_cast<std::size_t>(index)]);
            }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    expect(ids[0] == ids[2], "even threads disagreed");
    expect(ids[1] == ids[3], "odd threads disagreed");
    expect(works[0].selected_added_tokens == 80, "thread 0 lost or shared selections");
    expect(works[1].selected_added_tokens == 40, "thread 1 lost or shared selections");
    expect(works[0].candidate_checks == works[2].candidate_checks, "even thread counts diverged");
    expect(works[1].candidate_checks != works[0].candidate_checks, "different inputs shared a count");
}

} // namespace

int main() {
    try {
        test_empty_and_independence();
        test_parse_added_tokens_false();
        test_threads_do_not_share_counters();
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
