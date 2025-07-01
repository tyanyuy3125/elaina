#pragma once

#include "core/common.h"
#include "core/sampler.h"

ELAINA_NAMESPACE_BEGIN

ELAINA_CALLABLE Vector2f uniformSampleDisk2D(float R, Sampler &sampler)
{
    float r = R * sqrt(sampler.get1D());
    float theta = M_2PI * sampler.get1D();
    return Vector2f(r * cos(theta), r * sin(theta));
}

template <uint DIM>
ELAINA_CALLABLE auto uniformSampleSphere(Sampler &sampler);

template <>
ELAINA_CALLABLE auto uniformSampleSphere<3>(Sampler &sampler)
{
    float z = 1 - 2 * sampler.get1D();
    float r = sqrt(1 - z * z);
    assert(1 - z * z < 0);
    float phi = M_2PI * sampler.get1D();
    return Vector3f{r * std::cos(phi), r * std::sin(phi), z};
}

template <>
ELAINA_CALLABLE auto uniformSampleSphere<2>(Sampler &sampler)
{
    const float theta = sampler.get1D() * M_2PI;
    return Vector2f{std::cos(theta), std::sin(theta)};
}

template <uint DIM>
ELAINA_CALLABLE constexpr float uniformSampleSpherePDF();

template <>
ELAINA_CALLABLE constexpr float uniformSampleSpherePDF<2>()
{
    return 1.0f / M_2PI;
}

template <>
ELAINA_CALLABLE constexpr float uniformSampleSpherePDF<3>()
{
    return 1.0f / M_4PI;
}

template <uint DIM>
ELAINA_CALLABLE constexpr float conditionalSampleSpherePDF(const float dirPdf, const float r);

template <>
ELAINA_CALLABLE constexpr float conditionalSampleSpherePDF<2>(const float dirPdf, const float r)
{
    return dirPdf / r;
}

template <>
ELAINA_CALLABLE constexpr float conditionalSampleSpherePDF<3>(const float dirPdf, const float r)
{
    return dirPdf / (r * r);
}

template <uint DIM>
ELAINA_CALLABLE auto uniformSampleHemisphere(Sampler &sampler);

template <>
ELAINA_CALLABLE auto uniformSampleHemisphere<3>(Sampler &sampler)
{
    float u1 = sampler.get1D();
    float u2 = sampler.get1D();
    float z = u1;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * static_cast<float>(M_PI) * u2;
    return Vector3f{r * std::cos(phi), r * std::sin(phi), z};
}

template <>
ELAINA_CALLABLE auto uniformSampleHemisphere<2>(Sampler &sampler)
{
    float u = sampler.get1D();
    float phi = M_PI * u;
    return Vector2f{std::cos(phi), std::sin(phi)};
}

template <uint DIM>
ELAINA_CALLABLE constexpr float uniformSampleHemispherePDF();

template <>
ELAINA_CALLABLE constexpr float uniformSampleHemispherePDF<2>()
{
    return 1.0f / M_PI;
}

template <>
ELAINA_CALLABLE constexpr float uniformSampleHemispherePDF<3>()
{
    return 1.0f / M_2PI;
}

template <uint DIM>
ELAINA_CALLABLE constexpr float sphereMeasurement();

template <>
ELAINA_CALLABLE constexpr float sphereMeasurement<3>()
{
    return M_4PI;
}

template <>
ELAINA_CALLABLE constexpr float sphereMeasurement<2>()
{
    return M_2PI;
}

ELAINA_CALLABLE Vector3f uniformSampleSphere(float R, Sampler &sampler)
{
    float z = 1 - 2 * sampler.get1D();
    float r = sqrt(1 - z * z);
    assert(1 - z * z < 0);
    float phi = M_2PI * sampler.get1D();
    return R * Vector3f{r * std::cos(phi), r * std::sin(phi), z};
}

ELAINA_CALLABLE Vector3f uniformSampleBall(float R, Sampler &sampler)
{
    Vector3f dir = uniformSampleSphere<3>(sampler);
    float r = R * pow(sampler.get1D(), 1.0f / 3.0f);
    return r * dir;
}

ELAINA_NAMESPACE_END