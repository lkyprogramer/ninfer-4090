#pragma once

#include "targets/qwen3_8/impl/frontend/processor.h"
#include "targets/qwen3_8/reference/legacy_tokenizer.h"

#include <cstddef>
#include <string_view>

namespace ninfer::targets::qwen3_8::legacy_oracle {

// Exact-prefix failure is reported before the uint32 overflow check.
// prefix_size may be above uint32 max without allocating that many tokens.
void enforce_exact_then_uint32(bool exact, std::size_t prefix_size, std::string_view name);

frontend_internal::EncodedChat encode_rendered_chat(const Tokenizer& tokenizer,
                                                    const frontend_internal::RenderedChat& rendered);

} // namespace ninfer::targets::qwen3_8::legacy_oracle
