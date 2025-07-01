#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>
#include "core/device/context.h"
#include "core/device/cuda.h"

using namespace elaina;

#define REQUIRE_THAT_GPU(call, matcher)                   \
    do                                                    \
    {                                                     \
        Allocator &alloc = *gpContext->alloc;             \
        float *device_result = alloc.new_object<float>(); \
        CUDA_SYNC_CHECK();                                \
        GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {           \
            *device_result = call;                        \
        });                                               \
        CUDA_SYNC_CHECK();                                \
        REQUIRE_THAT(*device_result, matcher);            \
    } while (0)

