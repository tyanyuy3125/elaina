#pragma once
#include "core/common.h"
#include "core/sampler.h"

#include <cmath>

ELAINA_NAMESPACE_BEGIN

#define ELAINA_GREEN_FUNC_R_CLAMP 1e-4f

template <uint DIM>
class HarmonicGreenBall;

template <>
class HarmonicGreenBall<2>
{
public:
    ELAINA_CALLABLE HarmonicGreenBall(const float R_) : R(R_) {
        bound = 1.5f / R_;
    }

    ELAINA_CALLABLE float eval(const float r) const
    {
        return std::log(R / r) / M_2PI;
    }

    ELAINA_CALLABLE float norm() const
    {
        return R * R / 4.0f;
    }

    ELAINA_CALLABLE float pdfRadius(float r) const
    {
        return 4.0 * r * std::log(R / r) / (R * R);
    }

    ELAINA_CALLABLE thrust::tuple<float /* r */, float /* pdf */> sample(Sampler &sampler) const
    {
        return rejectionSampleHarmonicGreenBallsFn(sampler);
    }
private:
    float R;
    float bound;

    ELAINA_CALLABLE thrust::tuple<float /* r */, float /* pdf */> rejectionSampleHarmonicGreenBallsFn(Sampler &sampler) const
    {
        int iter = 0;
        float r;
        float pdf;
        do
        {
            float u = sampler.get1D();
            r = sampler.get1D() * R;
            pdf = eval(r) / norm();
            float pdfRadius = pdf / uniformSampleSpherePDF<2>();
            ++iter;

            if (u < pdfRadius / bound)
            {
                break;
            }
        } while (iter < 1000);
        
        r = std::max(ELAINA_GREEN_FUNC_R_CLAMP, r);
        if (r > R)
        {
            r = R / 2.0f;
        }

        pdf = pdfRadius(r);

        return thrust::make_tuple(r, pdf);
    }
};

template <>
class HarmonicGreenBall<3>
{
public:
    ELAINA_CALLABLE HarmonicGreenBall(const float R_) : R(R_) {}

    ELAINA_CALLABLE float eval(const float r) const
    {
        float ret = (1.0f / r - 1.0f / R) / M_4PI;
        if (std::isnan(ret))
        {
            printf("r: %f, R: %f\n", r, R);
        }
        return ret;
    }

    ELAINA_CALLABLE float norm() const
    {
        return R * R / 6.0f;
    }

    ELAINA_CALLABLE float pdfRadius(float r) const
    {
        return 6.0f * r * (R - r) / (R * R * R);
    }

    ELAINA_CALLABLE thrust::tuple<float /* r */, float /* pdf */> sample(Sampler &sampler) const
    {
        float u1 = sampler.get1D();
        float u2 = sampler.get1D();
        float phi = M_2PI * u2;

        float r = (1.0f + std::sqrt(1.0f - std::cbrt(u1 * u1)) * std::cos(phi)) * R / 2.0f;
        r = std::max(ELAINA_GREEN_FUNC_R_CLAMP, r);
        if (r > R)
        {
            r = R / 2.0f;
        }
        float pdf = pdfRadius(r);
        return thrust::make_tuple(r, pdf);
    }
private:
    float R;
};

ELAINA_NAMESPACE_END