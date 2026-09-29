#pragma once

#include "targets/qwen3_8/impl/frontend/chat_template.h"
#include "targets/qwen3_8/impl/frontend/tokenizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8::frontend_internal {

enum class ProcessorErrorKind {
    BudgetExceeded,
};

class ProcessorError final : public std::runtime_error {
public:
    ProcessorError(ProcessorErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ProcessorErrorKind kind() const noexcept { return kind_; }

private:
    ProcessorErrorKind kind_;
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PreprocessStats {
    std::size_t media_items       = 0;
    std::uint64_t raw_patches     = 0;
    std::uint64_t vision_tokens   = 0;
    std::uint64_t attention_pairs = 0;
    std::size_t prompt_tokens     = 0;
    std::size_t patch_bytes       = 0;

    [[nodiscard]] std::string summary() const;
};

struct ProcessorOptions {
    std::uint64_t image_min_pixels         = 32ULL * 32ULL;
    std::uint64_t image_max_pixels         = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels         = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels         = 4ULL * 1024ULL * 1024ULL;
    std::size_t max_media_bytes            = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::size_t max_media_items            = 16;
    std::uint64_t max_raw_patches          = 131'072;
    std::uint64_t max_vision_tokens        = 32'768;
    std::uint64_t max_attention_pairs      = 128ULL * 1024ULL * 1024ULL;
    std::size_t max_prompt_tokens          = 32'768;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
};

// Request-local boundary accounting. It is not part of prompt identity.
struct BoundaryEncodingWork {
    std::uint64_t main_encode_calls       = 0;
    std::uint64_t main_input_bytes        = 0;
    std::uint64_t checkpoint_hints        = 0;
    std::uint64_t boundary_queries        = 0;
    std::uint64_t boundary_unique_offsets = 0;
    std::uint64_t local_encode_calls      = 0;
    std::uint64_t local_input_bytes       = 0;
    std::uint64_t local_max_bytes         = 0;
    std::uint64_t memo_hits               = 0;
    TokenizerWork main_work;
    TokenizerWork local_work;
};

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // Row-major [sum(raw_patches), 1536], in the exact merger-friendly order.
    std::vector<float> patches;
    std::vector<VisionItem> vision_items;
    std::optional<std::uint32_t> stable_prefix_boundary;
    std::optional<std::uint32_t> turn_rewrite_boundary;
    std::optional<std::uint32_t> user_turn_boundary;
    struct CheckpointHint {
        SemanticCheckpointKind kind = SemanticCheckpointKind::StablePrefix;
        std::uint32_t boundary      = 0;
    };
    std::vector<CheckpointHint> checkpoint_hints;
    PreprocessStats stats;
    BoundaryEncodingWork encoding_work;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;
};

struct EncodedChat {
    std::vector<int> input_ids;
    std::optional<std::uint32_t> stable_prefix_boundary;
    std::optional<std::uint32_t> turn_rewrite_boundary;
    std::optional<std::uint32_t> user_turn_boundary;
    std::vector<ProcessedInput::CheckpointHint> checkpoint_hints;
    BoundaryEncodingWork encoding_work;
};

// Exact-prefix rejection stays ahead of the uint32 overflow check.
[[nodiscard]] inline std::uint32_t enforce_exact_token_prefix(bool exact, std::size_t prefix_size,
                                                              std::string_view name) {
    if (!exact) { throw std::logic_error(std::string(name) + " is not an exact token prefix"); }
    if (prefix_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(name) + " token boundary exceeds uint32");
    }
    return static_cast<std::uint32_t>(prefix_size);
}

void adjust_rendered_boundaries_for_replacement(RenderedChat& rendered, std::size_t position,
                                                std::size_t needle_size,
                                                std::size_t replacement_size);

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered);

class Processor {
public:
    Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              ProcessorOptions options = {});

    ProcessedInput process(const std::vector<ChatMessage>& messages,
                           ChatRenderOptions render_options = {}) const;

private:
    const Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    ProcessorOptions options_;
};

} // namespace ninfer::targets::qwen3_8::frontend_internal
