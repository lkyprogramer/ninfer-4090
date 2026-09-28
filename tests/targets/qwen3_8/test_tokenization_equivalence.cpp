#include "targets/qwen3_8/impl/frontend/tokenizer.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace product = ninfer::targets::qwen3_8::frontend_internal;
namespace legacy  = ninfer::targets::qwen3_8::legacy_oracle;

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << message << '\n';
    ++g_failures;
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("missing resource: " + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
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

Fixture make_fixture(const nlohmann::json& added, const nlohmann::json& vocab = {}) {
    Fixture fixture;
    const nlohmann::json used_vocab = vocab.empty() ? byte_vocab() : vocab;
    fixture.tokenizer_json = nlohmann::json{{"model",
                                             {{"type", "BPE"},
                                              {"vocab", used_vocab},
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

void expect_same(const product::Tokenizer& modern, const legacy::Tokenizer& old,
                 std::string_view text, product::EncodeOptions options, const char* label) {
    std::vector<int> got;
    std::vector<int> want;
    std::string got_error;
    std::string want_error;
    try {
        got = modern.encode(text, options);
    } catch (const std::exception& ex) {
        got_error = ex.what();
    }
    legacy::EncodeOptions legacy_options;
    legacy_options.parse_added_tokens = options.parse_added_tokens;
    try {
        want = old.encode(text, legacy_options);
    } catch (const std::exception& ex) {
        want_error = ex.what();
    }
    if (got_error != want_error || got != want) {
        std::cerr << label << " mismatch\n";
        ++g_failures;
    }
}

void test_selection_and_structure() {
    const Fixture shorter = make_fixture(
        nlohmann::json::array({added_token(300, "ab"), added_token(301, "abc")}));
    const product::Tokenizer modern_short = make_product(shorter);
    const legacy::Tokenizer old_short     = make_legacy(shorter);
    expect_same(modern_short, old_short, "abc", {}, "V2-SC-01");
    expect(modern_short.encode("abc") == std::vector<int>({300, 99}), "V2-SC-01 ids");

    const Fixture longer = make_fixture(
        nlohmann::json::array({added_token(301, "ab"), added_token(300, "abc")}));
    const product::Tokenizer modern_long = make_product(longer);
    const legacy::Tokenizer old_long     = make_legacy(longer);
    expect_same(modern_long, old_long, "abc", {}, "V2-SC-02");
    expect(modern_long.encode("abc") == std::vector<int>({300}), "V2-SC-02 ids");

    const Fixture earliest = make_fixture(
        nlohmann::json::array({added_token(307, "ab"), added_token(300, "abc")}));
    expect_same(make_product(earliest), make_legacy(earliest), "xab---abc", {}, "V2-SC-03");

    const Fixture overlap = make_fixture(
        nlohmann::json::array({added_token(300, "aba"), added_token(301, "bab")}));
    expect_same(make_product(overlap), make_legacy(overlap), "ababa", {}, "V2-SC-04");

    const Fixture adjacent = make_fixture(
        nlohmann::json::array({added_token(300, "<a>"), added_token(301, "<b>")}));
    expect_same(make_product(adjacent), make_legacy(adjacent), "<a><a><b>", {}, "V2-SC-11");

    const Fixture letters = make_fixture(
        nlohmann::json::array({added_token(300, "é"), added_token(301, "中")}),
        nlohmann::json{{"x", 0}});
    expect_same(make_product(letters), make_legacy(letters), "é中é", {}, "V2-SC-09");

    const Fixture nul = make_fixture(nlohmann::json::array({added_token(300, std::string("\0b", 2))}));
    const std::string nul_text("a\0b\0c", 5);
    expect_same(make_product(nul), make_legacy(nul), nul_text, {}, "V2-SC-10");

    const Fixture special = make_fixture(nlohmann::json::array({added_token(300, "<x>", true)}));
    const product::Tokenizer modern_special = make_product(special);
    const legacy::Tokenizer old_special     = make_legacy(special);
    expect_same(modern_special, old_special, "<x>", {}, "V2-SC-12");
    expect(modern_special.decode(modern_special.encode("<x>"), {.skip_special_tokens = true}).empty(),
           "special decode was not skipped");
    expect(modern_special.decode(modern_special.encode("<x>")) ==
               old_special.decode(old_special.encode("<x>")),
           "special decode diverged");

    product::EncodeOptions off;
    off.parse_added_tokens = false;
    const product::Tokenizer modern_adj = make_product(adjacent);
    const legacy::Tokenizer old_adj     = make_legacy(adjacent);
    product::TokenizerWork work;
    const std::vector<int> parsed_off = modern_adj.encode("<a><b>", off, &work);
    legacy::EncodeOptions legacy_off;
    legacy_off.parse_added_tokens = false;
    expect(parsed_off == old_adj.encode("<a><b>", legacy_off), "V2-SC-07 ids");
    expect(work.scan_positions == 0 && work.candidate_checks == 0, "V2-SC-07 scanned");

    expect_same(modern_adj, old_adj, "", {}, "empty");
    expect_same(modern_adj, old_adj, "你好", {}, "chinese");
    expect_same(modern_adj, old_adj, "{\"a\":1}", {}, "json");
    expect_same(modern_adj, old_adj, "\xff", {}, "invalid-utf8");

    nlohmann::json many = nlohmann::json::array();
    for (int i = 0; i < 8; ++i) { many.push_back(added_token(300 + i, "MARK" + std::to_string(i))); }
    const Fixture stress   = make_fixture(many);
    const product::Tokenizer modern_stress = make_product(stress);
    const std::string absent(2000, 'q');
    product::TokenizerWork scan;
    (void)modern_stress.encode(absent, {}, &scan);
    expect(scan.scan_positions <= absent.size(), "scan_positions exceeded input bytes");
    expect(scan.scan_positions > 0, "scan_positions was not counted");
}

void test_copy_move_threads() {
    const Fixture fixture = make_fixture(
        nlohmann::json::array({added_token(300, "<a>"), added_token(301, "<b>")}));
    const legacy::Tokenizer old = make_legacy(fixture);
    {
        auto source = std::make_unique<product::Tokenizer>(make_product(fixture));
        product::Tokenizer copy = *source;
        source.reset();
        expect(copy.encode("<a>x<b>") == old.encode("<a>x<b>"), "copy dangling index");
    }
    {
        std::optional<product::Tokenizer> source{make_product(fixture)};
        product::Tokenizer moved = std::move(*source);
        source.reset();
        expect(moved.encode("<a>x<b>") == old.encode("<a>x<b>"), "move dangling index");
    }

    const product::Tokenizer shared = make_product(fixture);
    constexpr int kThreads          = 4;
    std::vector<std::vector<int>> ids(static_cast<std::size_t>(kThreads));
    std::vector<product::TokenizerWork> works(static_cast<std::size_t>(kThreads));
    std::vector<std::thread> threads;
    for (int index = 0; index < kThreads; ++index) {
        threads.emplace_back([&, index] {
            const std::string text = index % 2 == 0 ? "<a>x" : "café<b>";
            for (int step = 0; step < 50; ++step) {
                ids[static_cast<std::size_t>(index)] =
                    shared.encode(text, {}, &works[static_cast<std::size_t>(index)]);
            }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    expect(ids[0] == ids[2] && ids[1] == ids[3], "concurrent encodes diverged");
    expect(works[0].scan_positions == works[2].scan_positions, "concurrent counters diverged");
    expect(works[0].candidate_checks != works[1].candidate_checks, "different inputs shared counts");
}

void test_real_resources(const std::string& dir) {
    const std::string tokenizer_json = read_file(dir + "/tokenizer.json");
    const std::string config         = read_file(dir + "/tokenizer_config.json");
    const std::string generation     = read_file(dir + "/generation_config.json");
    const product::Tokenizer modern(product::TokenizerResources{tokenizer_json, config, generation});
    const legacy::Tokenizer old(legacy::TokenizerResources{tokenizer_json, config, generation});
    expect_same(modern, old, "x<|im_sta", {}, "V2-SC-05");
    expect_same(modern, old, "<tts_text_bos><tts_text_bos_single>", {}, "V2-SC-08");
    expect_same(modern, old, "", {}, "real-empty");
    expect_same(modern, old, "hello", {}, "real-hello");

    std::string repeated;
    const std::string unit = "<|im_start|>assistant\n<|im_end|>";
    repeated.reserve(unit.size() * 4096);
    for (int i = 0; i < 4096; ++i) { repeated += unit; }
    product::TokenizerWork work;
    const std::vector<int> got = modern.encode(repeated, {}, &work);
    expect(got == old.encode(repeated), "V2-SC-06 ids");
    expect(work.scan_positions <= repeated.size(), "V2-SC-06 scan_positions exceeded bytes");
    std::cerr << "V2-SC-06 scan_positions=" << work.scan_positions
              << " bytes=" << repeated.size() << " tokens=" << got.size() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_selection_and_structure();
        test_copy_move_threads();
        if (g_failures != 0) { return 1; }
        if (argc < 2) {
            std::cerr << "BLOCKED real resources not provided\n";
            return 77;
        }
        test_real_resources(argv[1]);
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
