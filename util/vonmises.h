#pragma once

#include "core/common.h"
#include "core/sampler.h"
#include "transformation.h"
#include "check.h"

#ifdef small // Make MSVC happy.
#undef small
#endif

ELAINA_NAMESPACE_BEGIN

// ELAINA_CALLABLE float evalPoly(float y, )
#define COEF_SMALL_ORDER 7
#define COEF_LARGE_ORDER 9

ELAINA_CONSTANT float COEF_SMALL[2][COEF_SMALL_ORDER] =
    {
        {
            1.0,
            3.5156229,
            3.0899424,
            1.2067492,
            0.2659732,
            0.360768e-1,
            0.45813e-2,
        },
        {
            0.5,
            0.87890594,
            0.51498869,
            0.15084934,
            0.2658733e-1,
            0.301532e-2,
            0.32411e-3,
        }};

ELAINA_CONSTANT float COEF_LARGE[2][COEF_LARGE_ORDER] =
    {
        {
            0.39894228,
            0.1328592e-1,
            0.225319e-2,
            -0.157565e-2,
            0.916281e-2,
            -0.2057706e-1,
            0.2635537e-1,
            -0.1647633e-1,
            0.392377e-2,
        },
        {
            0.39894228,
            -0.3988024e-1,
            -0.362018e-2,
            0.163801e-2,
            -0.1031555e-1,
            0.2282967e-1,
            -0.2895312e-1,
            0.1787654e-1,
            -0.420059e-2,
        }};

ELAINA_CALLABLE float evalPoly(const float y, const float *const coeff, const int coeff_size)
{
    DCHECK_GT(coeff_size, 3);
    float ret = coeff[coeff_size - 1];
    for (int i = coeff_size - 2; i >= 0; --i)
    {
        ret = coeff[i] + y * ret;
    }
    return ret;
}

ELAINA_CALLABLE float logModifiedBesselFn(float x, int order = 0)
{
    ELAINA_CHECK(order == 0 || order == 1);

    float y = x / 3.75f;
    y *= y;
    float small = evalPoly(y, COEF_SMALL[order], COEF_SMALL_ORDER);
    if (order == 1)
    {
        small = abs(x) * small;
    }
    small = log(small);

    y = 3.75f / x;
    float large = x - 0.5f * log(x) + log(evalPoly(y, COEF_LARGE[order], COEF_LARGE_ORDER));

    float ret = (x < 3.75) ? small : large;
    return ret;
}

ELAINA_CALLABLE float rejectionSample(float kappa, double proposal_r, Sampler &sampler)
{
    if(kappa < 1e-3f)
    {
        return M_2PI * sampler.get1D();
    }

    while (true)
    // for(int i = 0; i < 2048; ++i)
    {
        double u1 = sampler.get1D64();
        double u2 = sampler.get1D64();
        double u3 = sampler.get1D64();
        double z = cos(M_PI * u1);
        double f = (1.0 + static_cast<double>(proposal_r) * z) / (static_cast<double>(proposal_r) + z);
        double c = static_cast<double>(kappa) * (static_cast<double>(proposal_r) - f);

        bool accept = ((c * (2.0 - c) - u2) > 0.0) || (log(c / u2) + 1.0 - c >= 0.0);
        if (accept)
        {
            return static_cast<float>(fmod((copysign(1.0, u3 - 0.5) * acos(f)) + M_PI, 2 * M_PI) - M_PI);
        }
    }
}

class VonMises
{
public:
    ELAINA_CALLABLE VonMises(float kappa) : kappa(kappa)
    {
        proposalR = proposalR_compute();
    }

    ELAINA_CALLABLE float log_eval(float cosTheta)
    {
        float ret = kappa * cosTheta;
        ret = (ret - log(M_2PI) - logModifiedBesselFn(kappa, 0));
        return ret;
    }

    ELAINA_CALLABLE float d_log_eval_d_kappa(float cosTheta)
    {
        if (kappa < 3.75f)
        {
            const float *const coeff = COEF_SMALL[0];

            const float coef = 0.0711111111111111f;
            const float c142 = 0.142222222222222f;
            const float c010 = 0.0101135802469136f;

            float kappa2 = kappa * kappa;
            float term7 = coeff[6] * kappa2;
            float term6 = coeff[5] + coef * term7;
            float term5 = coeff[4] + coef * kappa2 * term6;
            float term4 = coeff[3] + coef * kappa2 * term5;
            float term3 = coeff[2] + coef * kappa2 * term4;
            float term2 = coeff[1] + coef * kappa2 * term3;

            float numerator = coef * kappa2 * (coef * kappa2 * (coef * kappa2 * (coef * kappa2 * (c010 * coeff[6] * kappa * kappa2 + c142 * kappa * term6) + c142 * kappa * term5) + c142 * kappa * term4) + c142 * kappa * term3) + c142 * kappa * term2;
            float denominator = coeff[0] + coef * kappa2 * term2;

            return cosTheta - (numerator / denominator);
        }
        else
        {
            const float *const coeff = COEF_LARGE[0];
            float K_1 = coeff[0], K_2 = coeff[1], K_3 = coeff[2], K_4 = coeff[3];
            float K_5 = coeff[4], K_6 = coeff[5], K_7 = coeff[6], K_8 = coeff[7], K_9 = coeff[8];
            float x = kappa;
            float x2 = x * x;
            float x3 = x * x * x;

            return cosTheta - 1 - (3.75 * (3.75 * (3.75 * (3.75 * (3.75 * (3.75 * (-14.0625 * K_9 / x3 - 3.75 * (K_8 + 3.75 * K_9 / x) / x2) / x - 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x2) / x - 3.75 * (K_6 + 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x) / x2) / x - 3.75 * (K_5 + 3.75 * (K_6 + 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x) / x) / x2) / x - 3.75 * (K_4 + 3.75 * (K_5 + 3.75 * (K_6 + 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x) / x) / x) / x2) / x - 3.75 * (K_3 + 3.75 * (K_4 + 3.75 * (K_5 + 3.75 * (K_6 + 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x) / x) / x) / x) / x2) / x - 3.75 * (K_2 + 3.75 * (K_3 + 3.75 * (K_4 + 3.75 * (K_5 + 3.75 * (K_6 + 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x) / x) / x) / x) / x) / x2) / (K_1 + 3.75 * (K_2 + 3.75 * (K_3 + 3.75 * (K_4 + 3.75 * (K_5 + 3.75 * (K_6 + 3.75 * (K_7 + 3.75 * (K_8 + 3.75 * K_9 / x) / x) / x) / x) / x) / x) / x) / x) + 0.5 / x;
        }
    }

    ELAINA_CALLABLE float d_eval_d_kappa(float cosTheta)
    {
        return eval(cosTheta) * d_log_eval_d_kappa(cosTheta);
    }

    ELAINA_CALLABLE float eval(float cosTheta)
    {
        if(kappa < 1e-3f)
        {
            return 1.0f / M_2PI;
        }
        return exp(log_eval(cosTheta));
    }

    ELAINA_CALLABLE Vector2f sample(Sampler &sampler)
    {
        const float theta = rejectionSample(kappa, proposalR, sampler);
        return Vector2f(cos(theta), sin(theta));
    }

    ELAINA_CALLABLE Vector2f sample(Sampler &sampler, const Vector2f &mu)
    {
        return frameFromTangent(mu).toWorld(sample(sampler));
    }

public:
    ELAINA_CALLABLE double proposalR_compute()
    {
        double tau = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa);
        double rho = (tau - sqrt(2.0 * tau)) / (2.0 * kappa);
        double proposalR = (1.0 + rho * rho) / (2.0 * rho);
        double proposalRTaylor = 1.0 / kappa + kappa;
        return (kappa < 1e-5) ? proposalRTaylor : proposalR;
    }

private:
    float kappa{};
    double proposalR{};
};

ELAINA_NAMESPACE_END