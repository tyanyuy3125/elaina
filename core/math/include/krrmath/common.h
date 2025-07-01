#pragma once
#define NOMINMAX
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20014
#pragma nv_diag_suppress 20015
#pragma nv_diag_suppress 20208
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#define _USE_MATH_DEFINES // Make MSVC happy
#include <cmath>
#include <algorithm>
#include <sstream>
#include <vector>

#include <Eigen/Core>

#ifndef ELAINA_NAMESPACE_BEGIN

typedef uint32_t uint;
typedef unsigned char uchar;

#define ELAINA_NAMESPACE_BEGIN namespace elaina {
#define ELAINA_NAMESPACE_END }

#if defined(__CUDA_ARCH__)
#define ELAINA_DEVICE_CODE
#endif

#if defined(__CUDACC__)
# define ELAINA_DEVICE   __device__
# define ELAINA_HOST     __host__
# define ELAINA_FORCEINLINE __forceinline__
# define ELAINA_CONST	__device__ const 
# define ELAINA_GLOBAL	__global__
#else
# define ELAINA_DEVICE			/* ignore */
# define ELAINA_HOST			/* ignore */
# define ELAINA_FORCEINLINE	/* ignore */
# define ELAINA_CONST	const
# define ELAINA_GLOBAL			/* ignore */
#endif

# define __both__   ELAINA_HOST ELAINA_DEVICE
# define ELAINA_CALLABLE inline ELAINA_HOST ELAINA_DEVICE
# define ELAINA_HOST_DEVICE ELAINA_HOST ELAINA_DEVICE
# define ELAINA_DEVICE_FUNCTION ELAINA_DEVICE ELAINA_FORCEINLINE
# define ELAINA_DEVICE_LAMBDA(...) [ =, *this ] ELAINA_DEVICE(__VA_ARGS__) mutable 

#endif

#if defined(INCLUDE_NLOHMANN_JSON_HPP_)
#define ELAINA_MATH_JSON
#endif

ELAINA_NAMESPACE_BEGIN

namespace math = Eigen;

ELAINA_NAMESPACE_END
