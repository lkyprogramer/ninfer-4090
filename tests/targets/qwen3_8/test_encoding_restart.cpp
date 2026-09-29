#include "targets/qwen3_8/impl/frontend/tokenizer.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

bool same_work(const product::TokenizerWork& lhs, const product::TokenizerWork& rhs) {
    return lhs.scan_positions == rhs.scan_positions &&
           lhs.candidate_checks == rhs.candidate_checks &&
           lhs.selected_added_tokens == rhs.selected_added_tokens &&
           lhs.ordinary_segments == rhs.ordinary_segments &&
           lhs.ordinary_input_bytes == rhs.ordinary_input_bytes;
}

bool same_restart(const product::EncodingRestart& lhs, const product::EncodingRestart& rhs) {
    return lhs.byte_end == rhs.byte_end && lhs.token_end == rhs.token_end;
}

product::EncodingRestart last_restart_not_after(const std::vector<product::EncodingRestart>& trace,
                                                std::size_t byte_offset) {
    if (trace.empty() || trace.front().byte_end != 0 || trace.front().token_end != 0) {
        throw std::logic_error("encoding restart trace is missing its initial point");
    }
    const auto it = std::upper_bound(
        trace.begin(), trace.end(), byte_offset,
        [](std::size_t offset, const product::EncodingRestart& restart) {
            return offset < restart.byte_end;
        });
    if (it == trace.begin()) {
        throw std::logic_error("encoding restart trace is missing its initial point");
    }
    return *std::prev(it);
}

void expect_restarts(const std::vector<product::EncodingRestart>& got,
                     const std::vector<product::EncodingRestart>& want, const char* label) {
    if (got.size() != want.size()) {
        std::cerr << label << " restart count\n";
        ++g_failures;
        return;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!same_restart(got[i], want[i])) {
            std::cerr << label << " restart " << i << " byte " << got[i].byte_end << "/"
                      << want[i].byte_end << " token " << got[i].token_end << "/"
                      << want[i].token_end << '\n';
            ++g_failures;
            return;
        }
    }
}

void expect_decomposition(const product::Tokenizer& modern, const legacy::Tokenizer& old,
                          std::string_view text, const char* label) {
    product::TokenizerWork traced_work;
    product::TokenizerWork plain_work;
    const product::TracedEncoding traced = modern.encode_traced(text, {}, &traced_work);
    const std::vector<int> plain         = modern.encode(text, {}, &plain_work);
    expect(traced.token_ids == plain, label);
    expect(same_work(traced_work, plain_work), label);
    expect(traced.restarts.size() == traced_work.selected_added_tokens + 1, label);
    expect(!traced.restarts.empty() && traced.restarts.front().byte_end == 0 &&
               traced.restarts.front().token_end == 0,
           label);
    for (std::size_t i = 1; i < traced.restarts.size(); ++i) {
        expect(traced.restarts[i].byte_end > traced.restarts[i - 1].byte_end &&
                   traced.restarts[i].token_end > traced.restarts[i - 1].token_end,
               label);
        expect(traced.restarts[i].token_end <= traced.token_ids.size(), label);
    }
    if (!traced.restarts.empty() && traced.restarts.back().byte_end == text.size()) {
        expect(traced.restarts.back().token_end == traced.token_ids.size(), label);
    }

    for (std::size_t cut = 0; cut <= text.size(); ++cut) {
        const product::EncodingRestart restart = last_restart_not_after(traced.restarts, cut);
        expect(restart.byte_end <= cut, label);

        const std::string_view prefix = text.substr(0, cut);
        const std::string_view slice  = text.substr(restart.byte_end, cut - restart.byte_end);
        std::vector<int> legacy_ids;
        std::string legacy_error;
        try {
            legacy_ids = old.encode(prefix);
        } catch (const std::exception& ex) { legacy_error = ex.what(); }
        std::vector<int> local_ids;
        std::string local_error;
        try {
            local_ids = modern.encode(slice);
        } catch (const std::exception& ex) { local_error = ex.what(); }
        if (!legacy_error.empty() || !local_error.empty()) {
            expect(legacy_error == local_error, label);
            continue;
        }
        if (restart.token_end > traced.token_ids.size()) {
            expect(false, label);
            continue;
        }
        std::vector<int> composed(traced.token_ids.begin(),
                                  traced.token_ids.begin() +
                                      static_cast<std::ptrdiff_t>(restart.token_end));
        composed.insert(composed.end(), local_ids.begin(), local_ids.end());
        expect(composed == legacy_ids, label);
    }
}

void test_restart_points() {
    const Fixture markers = make_fixture(
        nlohmann::json::array({added_token(300, "<a>"), added_token(301, "<b>")}));
    const product::Tokenizer modern = make_product(markers);
    const legacy::Tokenizer old     = make_legacy(markers);

    const product::TracedEncoding marked = modern.encode_traced("<a>x<b>y");
    expect(marked.token_ids == modern.encode("<a>x<b>y"), "marked ids changed when traced");
    expect(marked.token_ids == old.encode("<a>x<b>y"), "marked ids diverged from legacy");
    expect_restarts(marked.restarts,
                    {{0, 0}, {3, 1}, {7, 3}},
                    "ordinary tail was recorded as a restart");

    const Fixture shorter = make_fixture(
        nlohmann::json::array({added_token(300, "ab"), added_token(301, "abc")}));
    const product::Tokenizer modern_short = make_product(shorter);
    const product::TracedEncoding short_trace = modern_short.encode_traced("abc");
    expect(short_trace.token_ids == std::vector<int>({300, 99}), "shorter-id trace ids");
    expect_restarts(short_trace.restarts, {{0, 0}, {2, 1}}, "ordinary final byte became a restart");

    const Fixture longer = make_fixture(
        nlohmann::json::array({added_token(301, "ab"), added_token(300, "abc")}));
    const product::TracedEncoding long_trace = make_product(longer).encode_traced("abc");
    expect(long_trace.token_ids == std::vector<int>({300}), "longer-id trace ids");
    expect_restarts(long_trace.restarts, {{0, 0}, {3, 1}}, "selected token end at EOF was dropped");

    const product::TracedEncoding adjacent = modern.encode_traced("<a><b>");
    expect(adjacent.token_ids == std::vector<int>({300, 301}), "adjacent marker ids");
    expect_restarts(adjacent.restarts, {{0, 0}, {3, 1}, {6, 2}}, "adjacent markers lost a restart");

    product::EncodeOptions off;
    off.parse_added_tokens = false;
    const product::TracedEncoding unparsed = modern.encode_traced("<a><b>", off);
    expect_restarts(unparsed.restarts, {{0, 0}}, "parse_added_tokens false recorded a match");
    expect(unparsed.token_ids == modern.encode("<a><b>", off), "parse-false ids changed when traced");
    product::TokenizerWork unparsed_work;
    (void)modern.encode("<a><b>", off, &unparsed_work);
    expect(unparsed_work.scan_positions == 0 && unparsed_work.candidate_checks == 0,
           "parse-false trace scanned");

    const product::TracedEncoding empty = modern.encode_traced("");
    expect(empty.token_ids.empty(), "empty traced ids");
    expect_restarts(empty.restarts, {{0, 0}}, "empty input dropped the initial restart");

    const product::TracedEncoding plain = modern.encode_traced("qqq");
    expect_restarts(plain.restarts, {{0, 0}}, "ordinary EOF became a restart");

    product::TokenizerWork traced_work;
    product::TokenizerWork plain_work;
    expect(modern.encode_traced("<a>x<b>y", {}, &traced_work).token_ids ==
               modern.encode("<a>x<b>y", {}, &plain_work),
           "traced encode changed token ids");
    expect(same_work(traced_work, plain_work), "traced encode changed work counters");
}

void test_byte_cuts() {
    const Fixture markers = make_fixture(
        nlohmann::json::array({added_token(300, "<a>"), added_token(301, "<b>")}));
    const product::Tokenizer modern = make_product(markers);
    const legacy::Tokenizer old     = make_legacy(markers);
    expect_decomposition(modern, old, "", "empty cuts");
    expect_decomposition(modern, old, "<a>x<b>y", "marker cuts");
    expect_decomposition(modern, old, "<a><b>", "adjacent marker cuts");
    expect_decomposition(modern, old, "<a>\xC3\xA9", "truncated utf8 after selected token");
    expect_decomposition(modern, old, "a<b", "partial marker cuts");
    expect_decomposition(modern, old, "qqq", "ordinary cuts");
    expect_decomposition(modern, old, "\xC3\xA9", "utf8 cuts");
    std::string modern_error;
    std::string legacy_error;
    try {
        (void)modern.encode("\xff");
    } catch (const std::exception& ex) { modern_error = ex.what(); }
    try {
        (void)old.encode("\xff");
    } catch (const std::exception& ex) { legacy_error = ex.what(); }
    expect(!modern_error.empty() && modern_error == legacy_error, "invalid utf8 errors diverged");

    const Fixture priority = make_fixture(
        nlohmann::json::array({added_token(300, "ab"), added_token(301, "abc")}));
    expect_decomposition(make_product(priority), make_legacy(priority), "abc", "priority cuts");
    expect_decomposition(make_product(priority), make_legacy(priority), "zzab", "embedded cuts");
}

void test_lookup_rejects_missing_origin() {
    bool missing = false;
    try {
        (void)last_restart_not_after({}, 0);
    } catch (const std::logic_error&) { missing = true; }
    expect(missing, "empty trace was accepted");

    bool no_origin = false;
    try {
        (void)last_restart_not_after({{5, 1}}, 5);
    } catch (const std::logic_error&) { no_origin = true; }
    expect(no_origin, "trace without the zero restart was accepted");

    const product::EncodingRestart found = last_restart_not_after({{0, 0}, {5, 2}, {10, 4}}, 7);
    expect(found.byte_end == 5 && found.token_end == 2, "upper_bound picked a restart after the cut");
    const product::EncodingRestart at = last_restart_not_after({{0, 0}, {5, 2}, {10, 4}}, 5);
    expect(at.byte_end == 5 && at.token_end == 2, "cut on a restart moved backward");
}

void test_copy_and_threads() {
    const Fixture fixture = make_fixture(
        nlohmann::json::array({added_token(300, "<a>"), added_token(301, "<b>")}));
    const legacy::Tokenizer old = make_legacy(fixture);
    {
        auto source = std::make_unique<product::Tokenizer>(make_product(fixture));
        product::Tokenizer copy = *source;
        source.reset();
        const product::TracedEncoding traced = copy.encode_traced("<a>x<b>");
        expect(traced.token_ids == old.encode("<a>x<b>"), "copied trace diverged");
        expect_restarts(traced.restarts, {{0, 0}, {3, 1}, {7, 3}}, "copied restarts");
    }
    {
        std::optional<product::Tokenizer> source{make_product(fixture)};
        product::Tokenizer moved = std::move(*source);
        source.reset();
        expect(moved.encode_traced("<a>x<b>").token_ids == old.encode("<a>x<b>"),
               "moved trace diverged");
    }

    const product::Tokenizer shared = make_product(fixture);
    constexpr int kThreads          = 4;
    std::vector<product::TracedEncoding> traces(static_cast<std::size_t>(kThreads));
    std::vector<std::thread> threads;
    for (int index = 0; index < kThreads; ++index) {
        threads.emplace_back([&, index] {
            const std::string text = index % 2 == 0 ? "<a>x<b>" : "plain";
            for (int step = 0; step < 20; ++step) { traces[static_cast<std::size_t>(index)] = shared.encode_traced(text); }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    expect(traces[0].token_ids == traces[2].token_ids && traces[0].restarts.size() == traces[2].restarts.size(),
           "concurrent traces diverged");
    expect(traces[1].restarts.size() == 1 && traces[0].restarts.size() == 3,
           "concurrent traces shared restart storage");
}

} // namespace

int main() {
    try {
        test_lookup_rejects_missing_origin();
        test_restart_points();
        test_byte_cuts();
        test_copy_and_threads();
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
