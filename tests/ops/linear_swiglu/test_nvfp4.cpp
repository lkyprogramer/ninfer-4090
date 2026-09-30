#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        [[maybe_unused]] constexpr std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024};
        int failures = 0;
        const int a16 = run_profile("LinearSwiGLU NVFP4_A16",
                                    {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                    kA16Cases);
        if (a16 == 77) { return 77; }
        failures += a16;
#ifndef NINFER_SM86
        const int a4 =
            run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
        if (a4 == 77) { return 77; }
        failures += a4;
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
