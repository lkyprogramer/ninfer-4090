#include "targets/qwen3_8/reference/legacy_encode_rendered_chat.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8::legacy_oracle {

using frontend_internal::EncodedChat;
using frontend_internal::RenderedChat;
using frontend_internal::SemanticCheckpointByteHint;

void enforce_exact_then_uint32(bool exact, std::size_t prefix_size, std::string_view name) {
    if (!exact) { throw std::logic_error(std::string(name) + " is not an exact token prefix"); }
    if (prefix_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(name) + " token boundary exceeds uint32");
    }
}

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered) {
    EncodedChat encoded;
    encoded.input_ids = tokenizer.encode(rendered.text);
    const auto encode_boundary = [&](std::size_t byte_offset, bool allow_full_prompt,
                                     std::string_view name) {
        if (byte_offset > rendered.text.size()) {
            throw std::logic_error(std::string(name) + " byte offset exceeds rendered chat");
        }
        std::vector<int> prefix;
        try {
            prefix = tokenizer.encode(std::string_view(rendered.text).substr(0, byte_offset));
        } catch (const std::exception&) {
            throw std::logic_error(std::string(name) + " cannot be tokenized independently");
        }
        const bool exact =
            !(prefix.empty() || (!allow_full_prompt && prefix.size() >= encoded.input_ids.size()) ||
              prefix.size() > encoded.input_ids.size() ||
              !std::equal(prefix.begin(), prefix.end(), encoded.input_ids.begin()));
        enforce_exact_then_uint32(exact, prefix.size(), name);
        return static_cast<std::uint32_t>(prefix.size());
    };
    if (rendered.stable_prefix_byte_offset) {
        encoded.stable_prefix_boundary =
            encode_boundary(*rendered.stable_prefix_byte_offset, true, "stable prefix");
    }
    if (rendered.turn_rewrite_byte_offset) {
        encoded.turn_rewrite_boundary =
            encode_boundary(*rendered.turn_rewrite_byte_offset, false, "turn rewrite prefix");
    }
    if (rendered.user_turn_byte_offset && *rendered.user_turn_byte_offset > 0) {
        encoded.user_turn_boundary =
            encode_boundary(*rendered.user_turn_byte_offset, false, "user turn prefix");
    }
    encoded.checkpoint_hints.reserve(rendered.checkpoint_hints.size());
    std::optional<std::uint32_t> previous;
    for (const SemanticCheckpointByteHint& hint : rendered.checkpoint_hints) {
        const std::uint32_t boundary =
            encode_boundary(hint.byte_offset, true, "semantic checkpoint prefix");
        if (previous && boundary < *previous) {
            throw std::logic_error("semantic checkpoint boundaries are not ordered");
        }
        encoded.checkpoint_hints.push_back({hint.kind, boundary});
        previous = boundary;
    }
    return encoded;
}

} // namespace ninfer::targets::qwen3_8::legacy_oracle
