#pragma once
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <vector>
#include <fstream>
#include <optional>
#include <stddef.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <indicators.hpp>

typedef uint32_t uint;
typedef unsigned char uchar;

using nlohmann::json;
using std::string;
using std::to_string;

#define ELAINA_NAMESPACE_BEGIN \
    namespace elaina           \
    {
#define ELAINA_NAMESPACE_END }

#define ELAINA_RESTRICT __restrict__

#if defined(__CUDA_ARCH__)
#define ELAINA_DEVICE_CODE
#define ELAINA_CONSTANT __constant__
#else
#define ELAINA_CONSTANT const
#endif

#if defined(__CUDACC__)
#define ELAINA_DEVICE __device__
#define ELAINA_HOST __host__
#define ELAINA_FORCEINLINE __forceinline__
#define ELAINA_CONST __device__ const
#define ELAINA_GLOBAL __global__
// #define ELAINA_CONSTANT __constant__
#else
#define ELAINA_DEVICE      /* ignore */
#define ELAINA_HOST        /* ignore */
#define ELAINA_FORCEINLINE /* ignore */
#define ELAINA_CONST const
#define ELAINA_GLOBAL /* ignore */
// #define ELAINA_CONSTANT const
#endif

#define __both__ ELAINA_HOST ELAINA_DEVICE
#define ELAINA_CALLABLE inline ELAINA_HOST ELAINA_DEVICE
#define ELAINA_HOST_DEVICE ELAINA_HOST ELAINA_DEVICE
#define ELAINA_DEVICE_FUNCTION ELAINA_DEVICE ELAINA_FORCEINLINE
#define ELAINA_DEVICE_LAMBDA(...) [ =, *this ] ELAINA_DEVICE(__VA_ARGS__) mutable
#define ELAINA_DEVICE_LAMBDA_GLOBAL(...) [=] ELAINA_DEVICE(__VA_ARGS__) mutable

#if !defined(__CUDA_ARCH__)
extern const uint3 threadIdx, blockIdx;
extern const dim3 blockDim;
#endif

#if !defined(__CUDA_ARCH__)
#define ELAINA_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + " not implemented")
#else
#define ELAINA_NOTIMPLEMENTED                                                            \
    printf("Error: Function not implemented! File: %s, Line: %d\n", __FILE__, __LINE__); \
    __trap()
#endif
#if defined(_MSC_VER)
#define ELAINA_SHOULDNT_GO_HERE __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#define ELAINA_SHOULDNT_GO_HERE __builtin_unreachable()
#else
#define ELAINA_SHOULDNT_GO_HERE \
    do                          \
    {                           \
    } while (0)
#endif

#define ELAINA_CLASS_DEFINE NLOHMANN_DEFINE_TYPE_INTRUSIVE
#define ELAINA_ENUM_DEFINE NLOHMANN_JSON_SERIALIZE_ENUM

#define ELAINA_INIT_PROGRESS_BAR(prefix)                   \
    indicators::show_console_cursor(false);                \
    indicators::BlockProgressBar bar                       \
    {                                                      \
        indicators::option::BarWidth{60},                  \
            indicators::option::PrefixText{prefix},        \
            indicators::option::ShowElapsedTime{true},     \
            indicators::option::ShowRemainingTime { true } \
    }

#define ELAINA_UPDATE_PROGRESS_BAR(current, total)               \
    bar.set_option(indicators::option::PostfixText{              \
        std::to_string(current) + "/" + std::to_string(total)}); \
    bar.set_progress((current) * 100 / total)

#define ELAINA_DESTROY_PROGRESS_BAR() \
    bar.mark_as_completed();          \
    indicators::show_console_cursor(true)

#include "krrmath/math.h"

ELAINA_NAMESPACE_BEGIN

namespace inter
{
    template <typename T>
    class polymorphic_allocator;
}
// this allocator uses gpu memory by default.
using Allocator = inter::polymorphic_allocator<std::byte>;

const inline json &get_by_path(const json &j, const std::string &path)
{
    const json *current = &j;
    std::istringstream ss(path);
    std::string token;

    while (std::getline(ss, token, '/'))
    {
        if (token.empty())
        {
            continue;
        }
        if (current->contains(token))
        {
            current = &(*current)[token];
        }
        else
        {
            throw std::out_of_range("Path does not exist: " + path + ", at " + token);
        }
    }

    return *current;
}

template <typename T>
T json_get_or_throw(const json &j, const std::string &path)
{
    try
    {
        const json &value = get_by_path(j, path);
        if (!value.is_null())
        {
            return value.get<T>();
        }
        else
        {
            throw std::runtime_error("Path value is null: " + path);
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Failed to find json value. " + std::string(e.what()));
    }
}

template <typename T>
T json_get_optional(const json &j, const std::string &path, const T &default_value)
{
    try
    {
        const json &value = get_by_path(j, path);
        if (!value.is_null())
        {
            return value.get<T>();
        }
        else
        {
            return default_value;
        }
    }
    catch (const std::exception &)
    {
        return default_value;
    }
}

template <typename T>
std::optional<T> json_get_optional(const json &j, const std::string &path)
{
    try
    {
        const json &value = get_by_path(j, path);
        if (!value.is_null())
        {
            return value.get<T>();
        }
        else
        {
            return {};
        }
    }
    catch (const std::exception &)
    {
        return {};
    }
}

inline json load_json_file(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open JSON file: " + file_path);
    }
    json j;
    try
    {
        file >> j;
    }
    catch (const json::parse_error &e)
    {
        throw std::runtime_error("JSON parsing error (" + file_path + "): " + std::string(e.what()));
    }
    file.close();
    return j;
}

enum class ExportImageChannel {
    DIRICHLET_SDF,
    NEUMANN_SDF,
    SOURCE,
    SOLUTION,
    CHANNEL_COUNT
};

ELAINA_NAMESPACE_END
