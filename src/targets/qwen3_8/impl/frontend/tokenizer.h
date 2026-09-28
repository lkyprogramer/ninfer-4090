#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_8::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

// Caller-owned, one encode accumulation. A null pointer disables counting.
struct TokenizerWork {
    std::uint64_t scan_positions         = 0;
    std::uint64_t candidate_checks       = 0;
    std::uint64_t selected_added_tokens  = 0;
    std::uint64_t ordinary_segments      = 0;
    std::uint64_t ordinary_input_bytes   = 0;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    std::vector<int> encode(std::string_view text, EncodeOptions options,
                            TokenizerWork* work) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    std::string decode_token_bytes(int id, bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
    void build_added_token_index();
    std::vector<int> encode_impl(std::string_view text, EncodeOptions options,
                                 TokenizerWork* work) const;

    std::vector<std::string> id_to_token_;
    std::vector<bool> valid_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    std::unordered_map<std::string, int> bpe_merge_ranks_;
    bool has_bpe_merges_ = true;
    std::vector<AddedToken> added_tokens_;
    // Indices into added_tokens_, grouped by first byte. Not pointers: copies stay valid.
    std::array<std::vector<std::size_t>, 256> added_token_indices_{};
    std::vector<int> default_stop_token_ids_;
};

} // namespace ninfer::targets::qwen3_8::frontend_internal
