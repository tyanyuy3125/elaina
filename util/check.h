#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "core/logger.h"

#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t rc = call;                                                    \
        if (rc != cudaSuccess)                                                    \
        {                                                                         \
            cudaError_t err = rc; /*cudaGetLastError();*/                         \
            ELAINA_LOG(Error, "CUDA Error (%s: line %d): %s (%s)\n", __FILE__, __LINE__, \
                cudaGetErrorName(err), cudaGetErrorString(err));                  \
        }                                                                         \
    } while (0)

#define CUDA_SYNC(call)          \
    do                           \
    {                            \
        cudaDeviceSynchronize(); \
        call;                    \
        cudaDeviceSynchronize(); \
    } while (0)

#define CUDA_SYNC_CHECK()                                               \
    do                                                                  \
    {                                                                   \
        cudaDeviceSynchronize();                                        \
        cudaError_t error = cudaGetLastError();                         \
        if (error != cudaSuccess)                                       \
        {                                                               \
            ELAINA_LOG(Error, "Error (%s: line %d): %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error));                             \
        }                                                               \
    } while (0)

#define CHECK_LOG(EXPR, LOG, ...)                                                       \
    do                                                                                  \
    {                                                                                   \
        if (!(EXPR))                                                                    \
        {                                                                               \
            ELAINA_LOG(Fatal, "Error (%s: line %d): " LOG, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                               \
    } while (0)

#define ELAINA_CHECK(x) assert(x)
#define CHECK_IMPL(a, b, op) assert((a)op(b))

#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

#ifdef ELAINA_DEBUG_BUILD

#define DCHECK(x) (ELAINA_CHECK(x))
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)

#else

#define EMPTY_CHECK \
    do              \
    {               \
    } while (false) /* swallow semicolon */

#define DCHECK(x) EMPTY_CHECK

#define DCHECK_EQ(a, b) EMPTY_CHECK
#define DCHECK_NE(a, b) EMPTY_CHECK
#define DCHECK_GT(a, b) EMPTY_CHECK
#define DCHECK_GE(a, b) EMPTY_CHECK
#define DCHECK_LT(a, b) EMPTY_CHECK
#define DCHECK_LE(a, b) EMPTY_CHECK

#endif
