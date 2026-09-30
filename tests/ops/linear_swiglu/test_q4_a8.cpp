#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // AllowA8 shares every Q4 route below the INT8 crossover with A16 and
        // replaces the routes above it. Cases straddle the crossover at 256/257
        // and cover partial token tiles of the INT8 schedule's 256-column block.
        constexpr std::array<std::int32_t, 12> kTokenCases{
            1, 32, 128, 255, 256, 257, 300, 384, 512, 513, 640, 641,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q4_A8",
            {QType::Q4G64_F16S, 34816, 5120, 17408, 1401U, ActivationCompute::A8}, kTokenCases);
        if (failures == 77) { return 77; }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
