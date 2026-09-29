// CPU-only comparison of three encode_rendered_chat lanes on one binary.
// legacy: frozen pre-B0 tokenizer + full-prefix boundary checks.
// b0: product Tokenizer::encode + the pre-T05 full-prefix boundary loop.
// b0-b1: product encode_rendered_chat (one traced encode, then local suffixes).
// This is not HTTP TTFT and it refuses inputs above 262144 tokens.

#include "media/decode/decode.h"
#include "targets/qwen3_8/impl/frontend/chat_template.h"
#include "targets/qwen3_8/impl/frontend/digest.h"
#include "targets/qwen3_8/impl/frontend/processor.h"
#include "targets/qwen3_8/reference/legacy_encode_rendered_chat.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace ninfer::media::decode {

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::logic_error("decode_image is unavailable in the boundary bench");
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::logic_error("decode_video is unavailable in the boundary bench");
}

} // namespace ninfer::media::decode

namespace {

namespace product = ninfer::targets::qwen3_8::frontend_internal;
namespace legacy  = ninfer::targets::qwen3_8::legacy_oracle;

constexpr std::uint64_t kHardTokenCap = 262144;
constexpr std::string_view kExpectedResourcesSha =
    "c8afe487da2dd305125cd7421a668bd55cadd632b13b0999956a7300d3917a51";
constexpr const char* kResourceNames[] = {
    "chat_template.jinja",          "generation_config.json", "preprocessor_config.json",
    "tokenizer.json",               "tokenizer_config.json",  "video_preprocessor_config.json",
};

struct Options {
    std::filesystem::path resources;
    std::filesystem::path out;
    std::filesystem::path paired;
    std::filesystem::path single;
    std::filesystem::path work;
    std::filesystem::path calibrate;
    std::uint64_t max_tokens      = kHardTokenCap;
    std::uint64_t short_target    = 6000;
    std::uint64_t short_floor     = 2000;
    std::uint64_t long_target     = 160000;
    std::uint64_t long_floor      = 120000;
    int short_history             = 4;
    int short_tail                = 2;
    int long_history              = 80;
    int long_tail                 = 10;
    int legacy_reps               = 3;
    int candidate_reps            = 7;
    int warmup_candidate          = 2;
    double legacy_budget_seconds  = 180;
    double slow_legacy_seconds    = 300;
    bool run_short                = true;
    bool run_long                 = true;
};

struct LogicalPrefix {
    std::uint64_t queries    = 0;
    std::uint64_t calls      = 0;
    std::uint64_t bytes      = 0;
    std::uint64_t max_bytes  = 0;
};

struct CohortShape {
    std::string session_id;
    int history_rounds          = 0;
    int tail_rounds             = 0;
    std::uint64_t target_tokens = 0;
    std::uint64_t floor_tokens  = 0;
    bool adapt_reps             = false;
};

struct BuiltChat {
    CohortShape shape;
    product::RenderedChat rendered;
    std::size_t body_chars      = 0;
    std::size_t message_count   = 0;
    std::size_t prompt_tokens   = 0;
    std::string input_sha256;
    std::string notes;
    LogicalPrefix logical;
};

struct LaneResult {
    bool ok = false;
    std::string error;
    product::EncodedChat chat;
    LogicalPrefix measured;
    double seconds = 0;
};

void fail(const std::string& message, int code) {
    std::cerr << "error: " << message << '\n';
    std::exit(code);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string resources_sha256(const std::filesystem::path& root) {
    std::string canonical = "{";
    for (std::size_t i = 0; i < std::size(kResourceNames); ++i) {
        const std::filesystem::path path = root / kResourceNames[i];
        const std::string bytes          = read_file(path);
        const std::string hex = product::sha256_hex(product::sha256(std::string_view(bytes)));
        if (i != 0) { canonical.push_back(','); }
        canonical += '"';
        canonical += kResourceNames[i];
        canonical += "\":{\"bytes\":";
        canonical += std::to_string(bytes.size());
        canonical += ",\"sha256\":\"";
        canonical += hex;
        canonical += "\"}";
    }
    canonical.push_back('}');
    return product::sha256_hex(product::sha256(std::string_view(canonical)));
}

product::ChatMessage text_message(std::string role, std::string text) {
    product::ChatMessage message;
    message.role = std::move(role);
    message.parts.push_back(product::ChatPart::text_part(std::move(text)));
    return message;
}

std::string fill_body(std::size_t chars, std::uint32_t salt) {
    static constexpr std::string_view kWords[] = {
        "return", "status", "offset", "buffer", "session", "request", "prefix", "token",
        "boundary", "message", "function", "result", "cache", "index", "module", "service",
        "compile", "error", "public", "class", "void", "string", "value", "field",
        "update", "check", "path", "source",
    };
    std::string out;
    out.reserve(chars + 16);
    std::size_t word = salt;
    while (out.size() + 8 < chars) {
        if (!out.empty()) { out.push_back(' '); }
        out.append(kWords[word % std::size(kWords)]);
        if ((word % 11U) == 0U) {
            out.push_back(' ');
            out.append(std::to_string(salt + static_cast<std::uint32_t>(word)));
        }
        ++word;
    }
    if (out.size() < chars) {
        out.append(chars - out.size(), 'x');
    } else if (out.size() > chars) {
        out.resize(chars);
    }
    return out;
}

product::ChatMessage assistant_with_tool(std::uint32_t salt) {
    product::ChatMessage message = text_message(
        "assistant", "Checked the requested path and recorded the comparison.");
    message.reasoning_content = "Compare the path with the previous turn before answering.";
    product::ToolCall call;
    call.id   = "call_" + std::to_string(salt);
    call.name = "read_file";
    call.arguments_json = "{\"path\":\"src/module/File" + std::to_string(salt) +
                          ".java\",\"note\":\"compare the offset\"}";
    message.tool_calls.push_back(std::move(call));
    return message;
}

product::ChatRenderOptions render_options() {
    product::ChatRenderOptions options;
    options.add_generation_prompt = true;
    options.enable_thinking       = true;
    options.reasoning_effort      = ninfer::ReasoningEffort::XHigh;
    options.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::RollingTool;
    options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"read_file","description":"Read a repository file","parameters":{"type":"object","properties":{"path":{"type":"string"},"note":{"type":"string"}},"required":["path"]}}})");
    return options;
}

std::vector<product::ChatMessage> build_messages(const CohortShape& shape, std::size_t body_chars) {
    if (shape.history_rounds < 1 || shape.tail_rounds < 1) {
        throw std::invalid_argument("history and tail rounds must be positive");
    }
    std::vector<product::ChatMessage> messages;
    messages.push_back(text_message(
        "system",
        "You are a coding assistant working in a Java repository. Prefer reading files before editing."));
    for (int round = 0; round < shape.history_rounds; ++round) {
        const auto salt = static_cast<std::uint32_t>(round + 1);
        messages.push_back(text_message("user", fill_body(body_chars, salt)));
        messages.push_back(assistant_with_tool(salt));
        messages.push_back(text_message("tool", fill_body(body_chars, salt + 1000U)));
        messages.push_back(text_message(
            "assistant", "The file contents match the surrounding request context."));
    }
    messages.push_back(text_message(
        "user", "Summarize the remaining compile error and name the file to edit."));
    for (int round = 0; round < shape.tail_rounds; ++round) {
        const auto salt = static_cast<std::uint32_t>(10000 + round + 1);
        messages.push_back(assistant_with_tool(salt));
        messages.push_back(text_message("tool", fill_body(body_chars, salt)));
        messages.push_back(text_message(
            "assistant", "Recorded the tool result against the latest compile error."));
    }
    return messages;
}

void add_offset(LogicalPrefix& prefix, std::size_t offset) {
    ++prefix.queries;
    ++prefix.calls;
    prefix.bytes += static_cast<std::uint64_t>(offset);
    if (offset > prefix.max_bytes) { prefix.max_bytes = static_cast<std::uint64_t>(offset); }
}

LogicalPrefix logical_full_prefix(const product::RenderedChat& rendered) {
    LogicalPrefix prefix;
    if (rendered.stable_prefix_byte_offset) { add_offset(prefix, *rendered.stable_prefix_byte_offset); }
    if (rendered.turn_rewrite_byte_offset) { add_offset(prefix, *rendered.turn_rewrite_byte_offset); }
    if (rendered.user_turn_byte_offset && *rendered.user_turn_byte_offset > 0) {
        add_offset(prefix, *rendered.user_turn_byte_offset);
    }
    for (const product::SemanticCheckpointByteHint& hint : rendered.checkpoint_hints) {
        add_offset(prefix, hint.byte_offset);
    }
    return prefix;
}

// Pre-T05 product boundary loop. Every queried prefix is encoded from byte 0
// with Tokenizer::encode, not encode_traced.
LaneResult encode_b0_full_prefix(const product::Tokenizer& tokenizer,
                                 const product::RenderedChat& rendered) {
    LaneResult result;
    result.chat.input_ids = tokenizer.encode(rendered.text);
    const auto encode_boundary = [&](std::size_t byte_offset, bool allow_full_prompt,
                                     std::string_view name) {
        ++result.measured.queries;
        if (byte_offset > rendered.text.size()) {
            throw std::logic_error(std::string(name) + " byte offset exceeds rendered chat");
        }
        ++result.measured.calls;
        result.measured.bytes += static_cast<std::uint64_t>(byte_offset);
        if (byte_offset > result.measured.max_bytes) {
            result.measured.max_bytes = static_cast<std::uint64_t>(byte_offset);
        }
        std::vector<int> prefix;
        try {
            prefix = tokenizer.encode(std::string_view(rendered.text).substr(0, byte_offset));
        } catch (const std::exception&) {
            throw std::logic_error(std::string(name) + " cannot be tokenized independently");
        }
        if (prefix.empty() || (!allow_full_prompt && prefix.size() >= result.chat.input_ids.size()) ||
            prefix.size() > result.chat.input_ids.size() ||
            !std::equal(prefix.begin(), prefix.end(), result.chat.input_ids.begin())) {
            throw std::logic_error(std::string(name) + " is not an exact token prefix");
        }
        if (prefix.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(std::string(name) + " token boundary exceeds uint32");
        }
        return static_cast<std::uint32_t>(prefix.size());
    };
    if (rendered.stable_prefix_byte_offset) {
        result.chat.stable_prefix_boundary =
            encode_boundary(*rendered.stable_prefix_byte_offset, true, "stable prefix");
    }
    if (rendered.turn_rewrite_byte_offset) {
        result.chat.turn_rewrite_boundary =
            encode_boundary(*rendered.turn_rewrite_byte_offset, false, "turn rewrite prefix");
    }
    if (rendered.user_turn_byte_offset && *rendered.user_turn_byte_offset > 0) {
        result.chat.user_turn_boundary =
            encode_boundary(*rendered.user_turn_byte_offset, false, "user turn prefix");
    }
    result.chat.checkpoint_hints.reserve(rendered.checkpoint_hints.size());
    std::optional<std::uint32_t> previous;
    for (const product::SemanticCheckpointByteHint& hint : rendered.checkpoint_hints) {
        const std::uint32_t boundary =
            encode_boundary(hint.byte_offset, true, "semantic checkpoint prefix");
        if (previous && boundary < *previous) {
            throw std::logic_error("semantic checkpoint boundaries are not ordered");
        }
        result.chat.checkpoint_hints.push_back({hint.kind, boundary});
        previous = boundary;
    }
    result.ok = true;
    return result;
}

std::string semantic_mismatch(const product::EncodedChat& left, const product::EncodedChat& right) {
    if (left.input_ids != right.input_ids) {
        const std::size_t n = std::min(left.input_ids.size(), right.input_ids.size());
        std::size_t index   = 0;
        while (index < n && left.input_ids[index] == right.input_ids[index]) { ++index; }
        std::ostringstream out;
        out << "input_ids differ at " << index << " sizes " << left.input_ids.size() << " and "
            << right.input_ids.size();
        return out.str();
    }
    if (left.stable_prefix_boundary != right.stable_prefix_boundary) {
        return "stable_prefix_boundary differs";
    }
    if (left.turn_rewrite_boundary != right.turn_rewrite_boundary) {
        return "turn_rewrite_boundary differs";
    }
    if (left.user_turn_boundary != right.user_turn_boundary) { return "user_turn_boundary differs"; }
    if (left.checkpoint_hints.size() != right.checkpoint_hints.size()) {
        return "checkpoint hint count differs";
    }
    for (std::size_t i = 0; i < left.checkpoint_hints.size(); ++i) {
        if (left.checkpoint_hints[i].kind != right.checkpoint_hints[i].kind ||
            left.checkpoint_hints[i].boundary != right.checkpoint_hints[i].boundary) {
            return "checkpoint hint " + std::to_string(i) + " differs";
        }
    }
    return {};
}

template <typename Fn>
LaneResult time_lane(Fn&& function) {
    LaneResult result;
    asm volatile("" ::: "memory");
    const auto start = std::chrono::steady_clock::now();
    try {
        result = function();
    } catch (const std::exception& error) {
        result.ok    = false;
        result.error = error.what();
    }
    const auto stop = std::chrono::steady_clock::now();
    asm volatile("" ::: "memory");
    result.seconds  = std::chrono::duration<double>(stop - start).count();
    if (result.ok) {
        volatile std::size_t sink = result.chat.input_ids.size();
        (void)sink;
    }
    return result;
}

LaneResult time_legacy(const legacy::Tokenizer& tokenizer, const product::RenderedChat& rendered) {
    return time_lane([&] {
        LaneResult result;
        result.chat = legacy::encode_rendered_chat(tokenizer, rendered);
        result.ok   = true;
        return result;
    });
}

LaneResult time_b0(const product::Tokenizer& tokenizer, const product::RenderedChat& rendered) {
    return time_lane([&] { return encode_b0_full_prefix(tokenizer, rendered); });
}

LaneResult time_product(const product::Tokenizer& tokenizer, const product::RenderedChat& rendered) {
    return time_lane([&] {
        LaneResult result;
        result.chat = product::encode_rendered_chat(tokenizer, rendered);
        result.ok   = true;
        return result;
    });
}

void require_ok(const LaneResult& result, const std::string& lane) {
    if (!result.ok) { fail(lane + " failed: " + result.error, 2); }
}

std::size_t count_tokens(const product::Tokenizer& tokenizer,
                         const product::CompiledChatTemplate& chat_template, const CohortShape& shape,
                         std::size_t body_chars) {
    const std::vector<product::ChatMessage> messages = build_messages(shape, body_chars);
    const product::RenderedChat rendered = chat_template.render(messages, render_options());
    const std::vector<int> ids           = tokenizer.encode(rendered.text);
    std::cerr << "probe session=" << shape.session_id << " body=" << body_chars
              << " tokens=" << ids.size() << " bytes=" << rendered.text.size()
              << " messages=" << messages.size()
              << " hints=" << rendered.checkpoint_hints.size() << '\n';
    return ids.size();
}

std::size_t choose_body(const product::Tokenizer& tokenizer,
                        const product::CompiledChatTemplate& chat_template, const CohortShape& shape,
                        std::uint64_t max_tokens) {
    const std::uint64_t ceiling = std::min(shape.target_tokens, max_tokens);
    const auto tokens_of        = [&](std::size_t body) {
        return count_tokens(tokenizer, chat_template, shape, body);
    };
    const std::size_t t1 = tokens_of(128);
    const std::size_t t2 = tokens_of(512);
    if (t1 > max_tokens) { fail(shape.session_id + " exceeds the token cap at body 128", 3); }
    if (t2 <= t1) { fail(shape.session_id + " token count did not increase from body 128 to 512", 1); }

    const double slope     = static_cast<double>(t2 - t1) / static_cast<double>(512 - 128);
    const double intercept = static_cast<double>(t1) - slope * 128.0;
    std::size_t body       = 128;
    if (slope > 0.0 && static_cast<double>(ceiling) > intercept) {
        body = static_cast<std::size_t>((static_cast<double>(ceiling) - intercept) / slope);
    }
    body = std::clamp(body, std::size_t{32}, std::size_t{64000});

    std::size_t tokens = tokens_of(body);
    for (int attempt = 0; attempt < 4 && (tokens > ceiling || tokens > max_tokens); ++attempt) {
        if (body <= 32) { break; }
        const double scale = static_cast<double>(ceiling) / static_cast<double>(tokens) * 0.97;
        const std::size_t next = std::max<std::size_t>(32, static_cast<std::size_t>(body * scale));
        if (next >= body) { break; }
        body   = next;
        tokens = tokens_of(body);
    }
    for (int attempt = 0; attempt < 3 && tokens < shape.floor_tokens && tokens <= ceiling; ++attempt) {
        const double scale = static_cast<double>(ceiling) / static_cast<double>(tokens);
        std::size_t next   = static_cast<std::size_t>(std::min(64000.0, body * scale));
        if (next <= body) { break; }
        const std::size_t grown = tokens_of(next);
        if (grown > max_tokens || grown > ceiling) { break; }
        body   = next;
        tokens = grown;
    }
    if (tokens > max_tokens) { fail(shape.session_id + " stayed above the token cap", 3); }
    if (tokens < shape.floor_tokens) {
        fail(shape.session_id + " reached " + std::to_string(tokens) +
                 " tokens, under the floor " + std::to_string(shape.floor_tokens),
             3);
    }
    return body;
}

BuiltChat build_chat(const product::Tokenizer& tokenizer,
                     const product::CompiledChatTemplate& chat_template, CohortShape shape,
                     std::size_t body_chars) {
    BuiltChat built;
    built.shape        = std::move(shape);
    built.body_chars   = body_chars;
    const std::vector<product::ChatMessage> messages = build_messages(built.shape, body_chars);
    built.message_count = messages.size();
    built.rendered      = chat_template.render(messages, render_options());
    const std::vector<int> ids = tokenizer.encode(built.rendered.text);
    built.prompt_tokens = ids.size();
    built.input_sha256  = product::sha256_hex(product::sha256(std::string_view(built.rendered.text)));
    built.logical       = logical_full_prefix(built.rendered);
    std::ostringstream notes;
    notes << "cpu encode_rendered_chat; history=" << built.shape.history_rounds
          << " tail=" << built.shape.tail_rounds << " body=" << built.body_chars
          << " messages=" << built.message_count
          << " hints=" << built.rendered.checkpoint_hints.size() << "; not http";
    built.notes = notes.str();
    return built;
}

void check_shape(const BuiltChat& built, std::uint64_t max_tokens) {
    if (built.prompt_tokens > max_tokens || built.prompt_tokens > kHardTokenCap) {
        fail(built.shape.session_id + " prompt_tokens " + std::to_string(built.prompt_tokens) +
                 " exceeds the cap",
             3);
    }
    if (built.prompt_tokens < built.shape.floor_tokens) {
        fail(built.shape.session_id + " prompt_tokens under floor", 3);
    }
    if (!built.rendered.stable_prefix_byte_offset || !built.rendered.turn_rewrite_byte_offset ||
        !built.rendered.user_turn_byte_offset || *built.rendered.user_turn_byte_offset == 0) {
        fail(built.shape.session_id + " is missing a real boundary offset", 1);
    }
    const std::size_t min_hints = built.shape.adapt_reps ? 15 : 3;
    if (built.rendered.checkpoint_hints.size() < min_hints) {
        fail(built.shape.session_id + " rendered only " +
                 std::to_string(built.rendered.checkpoint_hints.size()) + " checkpoint hints",
             1);
    }
    if (built.logical.queries == 0) { fail(built.shape.session_id + " has no boundary queries", 1); }
}

nlohmann::json work_record(const BuiltChat& built, const std::string& mode, const std::string& phase,
                           const LaneResult& result, const std::string& resources_sha) {
    nlohmann::json record = {
        {"session_id", built.shape.session_id},
        {"mode", mode},
        {"phase", phase},
        {"input_sha256", built.input_sha256},
        {"resources_sha256", resources_sha},
        {"prompt_tokens", static_cast<std::int64_t>(built.prompt_tokens)},
        {"render_bytes", static_cast<std::int64_t>(built.rendered.text.size())},
        {"tokenize_seconds", result.seconds},
        {"logical_full_prefix_queries", static_cast<std::int64_t>(built.logical.queries)},
        {"logical_full_prefix_input_bytes", static_cast<std::int64_t>(built.logical.bytes)},
        {"logical_full_prefix_max_bytes", static_cast<std::int64_t>(built.logical.max_bytes)},
    };
    if (mode == "b0") {
        record["prefix_encode_calls"] = static_cast<std::int64_t>(result.measured.calls);
        record["prefix_input_bytes"]  = static_cast<std::int64_t>(result.measured.bytes);
        record["prefix_max_bytes"]    = static_cast<std::int64_t>(result.measured.max_bytes);
        record["boundary_queries"]    = static_cast<std::int64_t>(result.measured.queries);
    }
    if (mode == "b0-b1") {
        const product::BoundaryEncodingWork& work = result.chat.encoding_work;
        record["main_encode_calls"]       = static_cast<std::int64_t>(work.main_encode_calls);
        record["main_input_bytes"]        = static_cast<std::int64_t>(work.main_input_bytes);
        record["checkpoint_hints"]        = static_cast<std::int64_t>(work.checkpoint_hints);
        record["boundary_queries"]        = static_cast<std::int64_t>(work.boundary_queries);
        record["boundary_unique_offsets"] = static_cast<std::int64_t>(work.boundary_unique_offsets);
        record["local_encode_calls"]      = static_cast<std::int64_t>(work.local_encode_calls);
        record["local_input_bytes"]       = static_cast<std::int64_t>(work.local_input_bytes);
        record["local_max_bytes"]         = static_cast<std::int64_t>(work.local_max_bytes);
        record["memo_hits"]               = static_cast<std::int64_t>(work.memo_hits);
        record["main_scan_positions"]     = static_cast<std::int64_t>(work.main_work.scan_positions);
        record["main_candidate_checks"]   = static_cast<std::int64_t>(work.main_work.candidate_checks);
        record["main_selected_added_tokens"] =
            static_cast<std::int64_t>(work.main_work.selected_added_tokens);
    }
    return record;
}

void check_product_work(const BuiltChat& built, const LaneResult& result) {
    const product::BoundaryEncodingWork& work = result.chat.encoding_work;
    if (work.main_encode_calls != 1 || work.main_input_bytes != built.rendered.text.size()) {
        fail(built.shape.session_id + " product main encode accounting is wrong", 2);
    }
    if (work.boundary_queries != built.logical.queries) {
        fail(built.shape.session_id + " product boundary_queries " +
                 std::to_string(work.boundary_queries) + " != logical " +
                 std::to_string(built.logical.queries),
             2);
    }
    if (work.main_work.scan_positions > work.main_input_bytes ||
        work.main_work.selected_added_tokens == 0) {
        fail(built.shape.session_id + " product scanner accounting is not a single pass", 2);
    }
    if (built.shape.adapt_reps && work.local_input_bytes * 5 >= work.main_input_bytes) {
        fail(built.shape.session_id + " local bytes " + std::to_string(work.local_input_bytes) +
                 " are not smaller than the main text " + std::to_string(work.main_input_bytes),
             2);
    }
}

void check_b0_accounting(const BuiltChat& built, const LaneResult& result) {
    if (result.measured.queries != built.logical.queries ||
        result.measured.bytes != built.logical.bytes ||
        result.measured.max_bytes != built.logical.max_bytes) {
        fail(built.shape.session_id + " b0 prefix accounting does not match the rendered offsets", 2);
    }
}

nlohmann::json schema_row(const BuiltChat& built, const std::string& mode, int repetition,
                          const LaneResult& result, const std::string& resources_sha, bool oracle_match) {
    std::uint64_t queries = built.logical.queries;
    if (mode == "b0") { queries = result.measured.queries; }
    if (mode == "b0-b1") { queries = result.chat.encoding_work.boundary_queries; }
    nlohmann::json row = {
        {"schema_version", 2},
        {"pair_id", built.shape.session_id + "-r" + std::to_string(repetition)},
        {"session_id", built.shape.session_id},
        {"mode", mode},
        {"input_sha256", built.input_sha256},
        {"resources_sha256", resources_sha},
        {"prompt_tokens", static_cast<std::int64_t>(built.prompt_tokens)},
        {"cache_kind", "none"},
        {"cohort", "synthetic"},
        {"tokenize_seconds", result.seconds},
        {"prepare_seconds", nullptr},
        {"ttft_seconds", nullptr},
        {"decode_tps", nullptr},
        {"status", "ok"},
        {"oracle_match", oracle_match},
        {"render_bytes", static_cast<std::int64_t>(built.rendered.text.size())},
        {"boundary_queries", static_cast<std::int64_t>(queries)},
        {"notes", built.notes},
    };
    if (mode == "b0-b1") {
        row["suffix_input_bytes"] =
            static_cast<std::int64_t>(result.chat.encoding_work.local_input_bytes);
    }
    return row;
}

void write_line(std::ofstream& out, const nlohmann::json& value) {
    out << value.dump() << '\n';
    out.flush();
}

double median_of(std::vector<double> values) {
    if (values.empty()) { return 0; }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) { return values[mid]; }
    return (values[mid - 1] + values[mid]) / 2.0;
}

std::string join_seconds(const std::vector<double>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) { out << ','; }
        out << values[i];
    }
    return out.str();
}

struct ArgMap {
    std::vector<std::pair<std::string, std::string>> items;
    bool flag(std::string_view name) const {
        return std::any_of(items.begin(), items.end(), [&](const auto& item) {
            return item.first == name && item.second == "true";
        });
    }
    std::string require(std::string_view name) const {
        for (const auto& item : items) {
            if (item.first == name) { return item.second; }
        }
        fail(std::string("missing --") + std::string(name), 1);
        return {};
    }
    std::optional<std::string> get(std::string_view name) const {
        for (const auto& item : items) {
            if (item.first == name) { return item.second; }
        }
        return std::nullopt;
    }
};

template <typename T>
T parse_number(const std::string& text, const char* name) {
    std::size_t used = 0;
    T value{};
    try {
        if constexpr (std::is_integral_v<T>) {
            value = static_cast<T>(std::stoll(text, &used));
        } else {
            value = static_cast<T>(std::stod(text, &used));
        }
    } catch (const std::exception&) { fail(std::string("bad number for --") + name, 1); }
    if (used != text.size()) { fail(std::string("bad number for --") + name, 1); }
    return value;
}

Options parse_options(int argc, char** argv) {
    ArgMap args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view token = argv[i];
        if (!token.starts_with("--") || token.size() == 2) { fail(std::string("bad argument ") + argv[i], 1); }
        const std::string name(token.substr(2));
        if (i + 1 >= argc) { fail("missing value for --" + name, 1); }
        args.items.emplace_back(name, argv[++i]);
    }
    Options options;
    options.resources = args.require("resources");
    options.out       = args.require("out");
    options.work      = args.require("work");
    options.calibrate = args.require("calibrate");
    if (const auto paired = args.get("paired")) { options.paired = *paired; }
    if (const auto single = args.get("single")) { options.single = *single; }
    if (options.paired.empty()) { options.paired = options.out.string() + ".paired.jsonl"; }
    if (options.single.empty()) { options.single = options.out.string() + ".single.jsonl"; }
    if (const auto value = args.get("max-tokens")) {
        options.max_tokens = parse_number<std::uint64_t>(*value, "max-tokens");
    }
    if (options.max_tokens == 0 || options.max_tokens > kHardTokenCap) {
        fail("max-tokens must be in 1..262144", 1);
    }
    if (const auto value = args.get("short-target")) {
        options.short_target = parse_number<std::uint64_t>(*value, "short-target");
    }
    if (const auto value = args.get("long-target")) {
        options.long_target = parse_number<std::uint64_t>(*value, "long-target");
    }
    if (const auto value = args.get("long-floor")) {
        options.long_floor = parse_number<std::uint64_t>(*value, "long-floor");
    }
    if (const auto value = args.get("legacy-reps")) {
        options.legacy_reps = parse_number<int>(*value, "legacy-reps");
    }
    if (const auto value = args.get("candidate-reps")) {
        options.candidate_reps = parse_number<int>(*value, "candidate-reps");
    }
    if (const auto value = args.get("warmup-candidate")) {
        options.warmup_candidate = parse_number<int>(*value, "warmup-candidate");
    }
    if (const auto value = args.get("legacy-budget-seconds")) {
        options.legacy_budget_seconds = parse_number<double>(*value, "legacy-budget-seconds");
    }
    if (options.legacy_reps < 1 || options.candidate_reps < 1 || options.warmup_candidate < 0) {
        fail("repetition counts must leave at least one scored run", 1);
    }
    const std::string cohorts = args.get("cohorts").value_or("short,long");
    options.run_short         = cohorts.find("short") != std::string::npos;
    options.run_long          = cohorts.find("long") != std::string::npos;
    if (!options.run_short && !options.run_long) { fail("cohorts must include short or long", 1); }
    return options;
}

std::string cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        const std::string key = "model name";
        if (line.rfind(key, 0) == 0) {
            const auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string value = line.substr(pos + 1);
                while (!value.empty() && value.front() == ' ') { value.erase(value.begin()); }
                return value;
            }
        }
    }
    return {};
}

std::string cpu_affinity() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        const std::string key = "Cpus_allowed_list:";
        if (line.rfind(key, 0) == 0) {
            std::string value = line.substr(key.size());
            while (!value.empty() && value.front() == ' ') { value.erase(value.begin()); }
            return value;
        }
    }
    return {};
}

int nice_value() {
#if defined(__linux__)
    errno = 0;
    const int value = getpriority(PRIO_PROCESS, 0);
    if (value == -1 && errno != 0) { return 0; }
    return value;
#else
    return 0;
#endif
}

struct RunPlan {
    int legacy_reps    = 0;
    int candidate_reps = 0;
    std::string note;
};

RunPlan adapt_plan(const Options& options, const BuiltChat& built, double legacy_seconds,
                   double b0_seconds) {
    RunPlan plan{options.legacy_reps, options.candidate_reps, "doc09 warmup plus interleaved scored reps"};
    if (!built.shape.adapt_reps) { return plan; }
    if (legacy_seconds > options.slow_legacy_seconds || b0_seconds > options.slow_legacy_seconds) {
        plan.legacy_reps    = 1;
        plan.candidate_reps = std::min(options.candidate_reps, 3);
        plan.note           = "scored reps reduced after a slow warmup; not a 3x7 G4 matrix";
        return plan;
    }
    if (legacy_seconds > options.legacy_budget_seconds || b0_seconds > 90.0) {
        plan.candidate_reps = std::min(options.candidate_reps, 3);
        plan.note           = "candidate reps reduced to 3 after a warmup over the budget; legacy stays 3";
    }
    return plan;
}

void run_cohort(const Options& options, const product::Tokenizer& modern, const legacy::Tokenizer& old,
                const product::CompiledChatTemplate& chat_template, const CohortShape& shape,
                const std::string& resources_sha, std::ofstream& all_rows, std::ofstream& paired_rows,
                std::ofstream& single_rows, std::ofstream& work, nlohmann::json& calibrate) {
    std::cerr << "calibrating " << shape.session_id << '\n';
    const std::size_t body = choose_body(modern, chat_template, shape, options.max_tokens);
    const BuiltChat built  = build_chat(modern, chat_template, shape, body);
    check_shape(built, options.max_tokens);
    std::cerr << "built " << built.notes << " tokens=" << built.prompt_tokens
              << " sha=" << built.input_sha256 << '\n';

    auto record_warmup = [&](const std::string& mode, int index, LaneResult result) {
        require_ok(result, mode + " warmup");
        nlohmann::json record = work_record(built, mode, "warmup", result, resources_sha);
        record["warmup_index"] = index;
        write_line(work, record);
        std::cerr << "WARMUP " << built.shape.session_id << ' ' << mode << " #" << index << ' '
                  << result.seconds << "s\n";
        return result;
    };

    std::cerr << "warmup " << built.shape.session_id << '\n';
    LaneResult legacy_warm = record_warmup("legacy", 0, time_legacy(old, built.rendered));
    LaneResult b0_warm     = record_warmup("b0", 0, time_b0(modern, built.rendered));
    LaneResult product_warm = record_warmup("b0-b1", 0, time_product(modern, built.rendered));
    const std::string legacy_vs_b0 = semantic_mismatch(legacy_warm.chat, b0_warm.chat);
    const std::string legacy_vs_product = semantic_mismatch(legacy_warm.chat, product_warm.chat);
    if (!legacy_vs_b0.empty() || !legacy_vs_product.empty()) {
        fail(built.shape.session_id + " oracle mismatch: " + legacy_vs_b0 + " | " + legacy_vs_product, 2);
    }
    check_b0_accounting(built, b0_warm);
    check_product_work(built, product_warm);
    const RunPlan plan = adapt_plan(options, built, legacy_warm.seconds, b0_warm.seconds);
    std::cerr << "PLAN " << built.shape.session_id << " legacy_reps=" << plan.legacy_reps
              << " candidate_reps=" << plan.candidate_reps << " " << plan.note << '\n';
    const bool full_warmup = plan.note.find("reduced") == std::string::npos &&
                             plan.note.find("Reduced") == std::string::npos;
    const int extra_warmups = full_warmup ? std::max(0, options.warmup_candidate - 1) : 0;
    for (int extra = 0; extra < extra_warmups; ++extra) {
        (void)record_warmup("b0", extra + 1, time_b0(modern, built.rendered));
        (void)record_warmup("b0-b1", extra + 1, time_product(modern, built.rendered));
    }

    nlohmann::json cohort = {
        {"session_id", built.shape.session_id},
        {"body_chars", static_cast<std::int64_t>(built.body_chars)},
        {"message_count", static_cast<std::int64_t>(built.message_count)},
        {"prompt_tokens", static_cast<std::int64_t>(built.prompt_tokens)},
        {"render_bytes", static_cast<std::int64_t>(built.rendered.text.size())},
        {"checkpoint_hints", static_cast<std::int64_t>(built.rendered.checkpoint_hints.size())},
        {"boundary_queries", static_cast<std::int64_t>(built.logical.queries)},
        {"input_sha256", built.input_sha256},
        {"history_rounds", built.shape.history_rounds},
        {"tail_rounds", built.shape.tail_rounds},
        {"legacy_reps", plan.legacy_reps},
        {"candidate_reps", plan.candidate_reps},
        {"protocol_note", plan.note},
        {"warmup_legacy_seconds", legacy_warm.seconds},
        {"equality_b0_seconds", b0_warm.seconds},
        {"equality_b0b1_seconds", product_warm.seconds},
        {"local_input_bytes", static_cast<std::int64_t>(product_warm.chat.encoding_work.local_input_bytes)},
        {"local_max_bytes", static_cast<std::int64_t>(product_warm.chat.encoding_work.local_max_bytes)},
        {"main_scan_positions",
         static_cast<std::int64_t>(product_warm.chat.encoding_work.main_work.scan_positions)},
        {"notes", built.notes},
    };
    calibrate["cohorts"].push_back(std::move(cohort));
    {
        std::ofstream checkpoint(options.calibrate, std::ios::trunc);
        if (checkpoint) { checkpoint << calibrate.dump(2) << '\n'; }
    }

    std::vector<double> legacy_seconds;
    std::vector<double> b0_seconds;
    std::vector<double> product_seconds;
    const int reps = std::max(plan.legacy_reps, plan.candidate_reps);
    for (int rep = 0; rep < reps; ++rep) {
        const bool score_legacy = rep < plan.legacy_reps;
        if (score_legacy) {
            LaneResult result = time_legacy(old, built.rendered);
            require_ok(result, "legacy");
            const std::string mismatch = semantic_mismatch(result.chat, legacy_warm.chat);
            if (!mismatch.empty()) { fail("legacy rep drifted: " + mismatch, 2); }
            const bool paired = rep < plan.candidate_reps;
            nlohmann::json row = schema_row(built, "legacy", rep, result, resources_sha, true);
            write_line(all_rows, row);
            write_line(paired ? paired_rows : single_rows, row);
            nlohmann::json record = work_record(built, "legacy", "time", result, resources_sha);
            record["pair_id"] = row["pair_id"];
            write_line(work, record);
            legacy_seconds.push_back(result.seconds);
            std::cerr << "TIMED " << built.shape.session_id << " r" << rep << " legacy "
                      << result.seconds << "s\n";
        }
        if (rep < plan.candidate_reps) {
            LaneResult b0_result = time_b0(modern, built.rendered);
            require_ok(b0_result, "b0");
            const std::string b0_mismatch = semantic_mismatch(b0_result.chat, b0_warm.chat);
            if (!b0_mismatch.empty()) { fail("b0 rep drifted: " + b0_mismatch, 2); }
            check_b0_accounting(built, b0_result);
            LaneResult product_result = time_product(modern, built.rendered);
            require_ok(product_result, "b0-b1");
            const std::string product_mismatch =
                semantic_mismatch(product_result.chat, product_warm.chat);
            if (!product_mismatch.empty()) { fail("b0-b1 rep drifted: " + product_mismatch, 2); }
            const bool paired = rep < plan.legacy_reps;
            const std::pair<const char*, const LaneResult*> scored[] = {
                {"b0", &b0_result},
                {"b0-b1", &product_result},
            };
            for (const auto& item : scored) {
                nlohmann::json row = schema_row(built, item.first, rep, *item.second, resources_sha, true);
                write_line(all_rows, row);
                write_line(paired ? paired_rows : single_rows, row);
                nlohmann::json record = work_record(built, item.first, "time", *item.second, resources_sha);
                record["pair_id"] = row["pair_id"];
                write_line(work, record);
            }
            b0_seconds.push_back(b0_result.seconds);
            product_seconds.push_back(product_result.seconds);
            std::cerr << "TIMED " << built.shape.session_id << " r" << rep << " b0 " << b0_result.seconds
                      << "s b0-b1 " << product_result.seconds << "s\n";
        }
    }
    std::cerr << "SUMMARY " << built.shape.session_id
              << " legacy n=" << legacy_seconds.size() << " median=" << median_of(legacy_seconds)
              << " values=" << join_seconds(legacy_seconds) << '\n';
    std::cerr << "SUMMARY " << built.shape.session_id
              << " b0 n=" << b0_seconds.size() << " median=" << median_of(b0_seconds)
              << " values=" << join_seconds(b0_seconds) << '\n';
    std::cerr << "SUMMARY " << built.shape.session_id
              << " b0-b1 n=" << product_seconds.size() << " median=" << median_of(product_seconds)
              << " values=" << join_seconds(product_seconds) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    std::cerr << "hashing resources\n";
    const std::string resources_sha = resources_sha256(options.resources);
    if (resources_sha != kExpectedResourcesSha) {
        fail("resource sha256 " + resources_sha + " != T00 " + std::string(kExpectedResourcesSha), 1);
    }
    const std::string tokenizer_json = read_file(options.resources / "tokenizer.json");
    const std::string tokenizer_config = read_file(options.resources / "tokenizer_config.json");
    const std::string generation_config = read_file(options.resources / "generation_config.json");
    const std::string template_source = read_file(options.resources / "chat_template.jinja");

    std::cerr << "loading tokenizers\n";
    const product::Tokenizer modern(product::TokenizerResources{
        tokenizer_json, tokenizer_config, generation_config});
    const legacy::Tokenizer old(legacy::TokenizerResources{
        tokenizer_json, tokenizer_config, generation_config});
    const product::CompiledChatTemplate chat_template = product::CompiledChatTemplate::resolve(template_source);
    if (chat_template.capabilities().reasoning_effort.xhigh != true) {
        fail("chat template is not the reasoning-effort template", 1);
    }

    std::ofstream all_rows(options.out, std::ios::trunc);
    std::ofstream paired_rows(options.paired, std::ios::trunc);
    std::ofstream single_rows(options.single, std::ios::trunc);
    std::ofstream work(options.work, std::ios::trunc);
    if (!all_rows || !paired_rows || !single_rows || !work) { fail("cannot open output files", 1); }

    nlohmann::json calibrate = {
        {"resources_sha256", resources_sha},
        {"max_tokens", static_cast<std::int64_t>(options.max_tokens)},
        {"cpu_model", cpu_model()},
        {"affinity", cpu_affinity()},
        {"nice", nice_value()},
        {"lanes",
         {{"legacy", "legacy tokenizer + legacy_encode_rendered_chat"},
          {"b0", "product Tokenizer::encode + pre-T05 full-prefix loop"},
          {"b0-b1", "product encode_rendered_chat"}}},
        {"cohorts", nlohmann::json::array()},
    };

    if (options.run_short) {
        CohortShape shape;
        shape.session_id    = "cpu-short";
        shape.history_rounds = options.short_history;
        shape.tail_rounds    = options.short_tail;
        shape.target_tokens  = options.short_target;
        shape.floor_tokens   = options.short_floor;
        shape.adapt_reps     = false;
        run_cohort(options, modern, old, chat_template, shape, resources_sha, all_rows, paired_rows,
                   single_rows, work, calibrate);
    }
    if (options.run_long) {
        CohortShape shape;
        shape.session_id     = "cpu-long";
        shape.history_rounds = options.long_history;
        shape.tail_rounds    = options.long_tail;
        shape.target_tokens  = options.long_target;
        shape.floor_tokens   = options.long_floor;
        shape.adapt_reps     = true;
        run_cohort(options, modern, old, chat_template, shape, resources_sha, all_rows, paired_rows,
                   single_rows, work, calibrate);
    }

    std::ofstream calibrate_out(options.calibrate, std::ios::trunc);
    if (!calibrate_out) { fail("cannot write calibrate file", 1); }
    calibrate_out << calibrate.dump(2) << '\n';
    std::cout << "RESULT ok resources=" << resources_sha << '\n';
    return 0;
}
