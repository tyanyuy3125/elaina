#pragma once

#include "core/common.h"
#include "util/vonmises.h"
#include "util/vmf.h"
#include "util/beta.h"
#include "core/sampler.h"
#include "integrator/guided/parameters.h"
#include "train.h"

ELAINA_NAMESPACE_BEGIN

template <uint DIM>
class VMFKernel;

template <uint DIM>
class VMFBetaKernel;

template <>
class VMFKernel<2>
{
public:
    VMFKernel() = default;

    ELAINA_CALLABLE float pdf(const Vector2f &wi) const
    {
        return VonMises(kappa).eval(dot(wi, mu));
    }

    ELAINA_CALLABLE Vector2f sample(Sampler &sampler) const
    {
        return VonMises(kappa).sample(sampler, mu);
    }

    ELAINA_CALLABLE float d_pdf_d_kappa(const Vector2f &wi) const
    {
        return VonMises(kappa).d_eval_d_kappa(dot(wi, mu));
    }

public:
    float lambda;
    float kappa;
    Vector2f mu;
    Vector2f mu_original;
};

template <>
class VMFKernel<3>
{
public:
    VMFKernel() = default;

    ELAINA_CALLABLE Vector3f sample(Sampler &sampler) const
    { // TODO: do not pass sampler
        return VMF(kappa).sample(sampler, mu);
    }

    ELAINA_CALLABLE float pdf(const Vector3f &wi) const
    {
        return VMF(kappa).eval(wi, mu);
    }

    float lambda;         // [0, +inf), clamped to exp(-+10)
    float kappa;          // [0, +inf), clamped to exp(-+10)
    Vector3f mu;          // Normalized
    Vector3f mu_original; // Not normalized
};

template <>
class VMFBetaKernel<2>
{
public:
    VMFBetaKernel() = default;

    ELAINA_CALLABLE Vector3f sample(Sampler &sampler) const
    {
        auto direction = VonMises(kappa).sample(sampler, mu);
        auto radius = BetaDistribution(alpha, beta).sample(sampler);
        return Vector3f(direction.x(), direction.y(), radius);
    }

    ELAINA_CALLABLE float pdf(const Vector3f &wi_r) const
    {
        auto wi = Vector2f(wi_r.x(), wi_r.y());
        auto r = wi_r.z();
        return VonMises(kappa).eval(dot(wi, mu)) * BetaDistribution(alpha, beta).eval(r);
    }

public:
    // vMF
    float lambda;
    float kappa;
    Vector2f mu;
    Vector2f mu_original;

    // Beta
    float alpha;
    float beta;
};

template <>
class VMFBetaKernel<3>
{
public:
    VMFBetaKernel() = default;

    ELAINA_CALLABLE Vector4f sample(Sampler &sampler) const
    {
        auto direction = VMF(kappa).sample(sampler, mu);
        auto radius = BetaDistribution(alpha, beta).sample(sampler);
        return Vector4f(direction.x(), direction.y(), direction.z(), radius);
    }

    ELAINA_CALLABLE float pdf(const Vector4f &wi_r) const
    {
        auto wi = Vector3f(wi_r.x(), wi_r.y(), wi_r.z());
        auto r = wi_r.w();
        return VMF(kappa).eval(wi, mu) * BetaDistribution(alpha, beta).eval(r);
    }

public:
    // vMF
    float lambda;
    float kappa;
    Vector3f mu;
    Vector3f mu_original;

    // Beta
    float alpha;
    float beta;
};

template <uint DIM, uint N>
class VMM;

template <uint N>
class VMM<2, N>
{
public:
    using VMFKernel = VMFKernel<2>;

    static constexpr auto N_COMP = N;

    VMM() = default;

    template <typename T>
    ELAINA_CALLABLE VMM(T *data)
    {
        using namespace common2d;

        totalWeight = 0.0f;
        for (int i = 0; i < N; ++i)
        {
            int idx = i * N_DIM_VMF;
            mSG[i].lambda = network_to_params((float)data[idx], ACTIVATION_LAMBDA);
            mSG[i].kappa = network_to_params((float)data[idx + 1], ACTIVATION_KAPPA);
            const float x = network_to_params((float)data[idx + 2], ACTIVATION_COORDINATES);
            const float y = network_to_params((float)data[idx + 3], ACTIVATION_COORDINATES);
            mSG[i].mu_original = Vector2f(x, y);
            mSG[i].mu = mSG[i].mu_original.normalized();
            totalWeight += mSG[i].lambda;
        }
        DCHECK_NE(totalWeight, 0.0f);
        for (int i = 0; i < N; ++i)
        {
            weight[i] = mSG[i].lambda / totalWeight;
        }
    }

    ELAINA_CALLABLE float pdf(const Vector2f &wi) const
    {
        float val = 0.0f;
        for (int i = 0; i < N; ++i)
        {
            val += weight[i] * mSG[i].pdf(wi);
        }
        return val;
    }

    ELAINA_CALLABLE float pdf(uint i, const Vector2f &wi) const
    {
        DCHECK_LT(i, N);
        return mSG[i].pdf(wi);
    }

    ELAINA_CALLABLE Vector2f sample(Sampler &sampler) const
    {
        float u = sampler.get1D();
        for (int i = 0; i < N; ++i)
        {
            if (u < weight[i])
            {
                return mSG[i].sample(sampler);
            }
            u -= weight[i];
        }
        return mSG[0].sample(sampler);
    }

    template <typename T>
    ELAINA_CALLABLE float gradients_probability(const Vector2f &wi, const bool isOnNeumannBoundary, const Vector2f &neumannBoundaryNormal, T *output) const
    {
        using namespace common2d;

        float probability = 0.0f;
        Vector2f wiReflected;
        if (isOnNeumannBoundary)
        {
            wiReflected = reflect(wi, neumannBoundaryNormal);
        }
        for (int sg = 0; sg < N_COMP; ++sg)
        {
            precision_t *cur_gradient = output + sg * N_DIM_VMF;

            float lambda = mSG[sg].lambda, kappa = mSG[sg].kappa;
            float mu_original_x = mSG[sg].mu_original[0], mu_original_y = mSG[sg].mu_original[1];

            float vm = pdf(sg, wi);
            probability += weight[sg] * vm;
            float vmReflected = 0.0f;
            if (isOnNeumannBoundary)
            {
                vmReflected = pdf(sg, wiReflected);
                probability += weight[sg] * vmReflected;
            }

            float dF_dlambda = (vm + vmReflected) * (totalWeight - lambda) / pow2(totalWeight);
            for (int k = 0; k < N; ++k)
            {
                if (k != sg)
                {
                    dF_dlambda -= weight[k] / totalWeight * mSG[k].pdf(wi); // Not divided by pow2(totalWeight) for weight[k] has already been preprocessed.
                    if (isOnNeumannBoundary)
                    {
                        dF_dlambda -= weight[k] / totalWeight * mSG[k].pdf(wiReflected);
                    }
                }
            }
            float dF_dkappa = weight[sg] * mSG[sg].d_pdf_d_kappa(wi);
            if (isOnNeumannBoundary)
            {
                dF_dkappa += weight[sg] * mSG[sg].d_pdf_d_kappa(wiReflected);
            }
            float denom = pow(pow2(mu_original_x) + pow2(mu_original_y), 1.5f);
            if (denom < M_EPSILON)
            {
                denom = M_EPSILON;
            }
            float x = wi[0], y = wi[1];
            float xReflected = wiReflected[0], yReflected = wiReflected[1];
            float dF_dx = weight[sg] * vm * kappa * mu_original_y * (-mu_original_x * y + mu_original_y * x) / denom;
            if (isOnNeumannBoundary)
            {
                dF_dx += weight[sg] * vmReflected * kappa * mu_original_y * (-mu_original_x * yReflected + mu_original_y * xReflected) / denom;
            }
            float dF_dy = weight[sg] * vm * kappa * mu_original_x * (mu_original_x * y - mu_original_y * x) / denom;
            if (isOnNeumannBoundary)
            {
                dF_dy += weight[sg] * vmReflected * kappa * mu_original_x * (mu_original_x * yReflected - mu_original_y * xReflected) / denom;
            }
            cur_gradient[0] = dF_dlambda, cur_gradient[1] = dF_dkappa, cur_gradient[2] = dF_dx, cur_gradient[3] = dF_dy;
        }
        return probability;
    }

    ELAINA_HOST void print() const
    {
        for (int i = 0; i < N; ++i)
        {
            ELAINA_LOG(Info, "Component %d: lambda = %f, kappa = %f, mu = (%f, %f)", i, mSG[i].lambda, mSG[i].kappa, mSG[i].mu[0], mSG[i].mu[1]);
        }
    }

protected:
    VMFKernel mSG[N];
    float weight[N], totalWeight{};
};

template <uint N>
class VMM<3, N>
{
public:
    using VMFKernel = VMFKernel<3>;

    static constexpr auto N_COMP = N;

    VMM() = default;

    template <typename T>
    ELAINA_CALLABLE VMM(T *data)
    {
        using namespace common3d;

        totalWeight = 0;
        for (int i = 0; i < N; i++)
        {
            int idx = i * N_DIM_VMF;
            mSG[i].lambda = network_to_params((float)data[idx], ACTIVATION_LAMBDA);
            mSG[i].kappa = network_to_params((float)data[idx + 1], ACTIVATION_KAPPA);
            const float x = network_to_params((float)data[idx + 2], ACTIVATION_COORDINATES);
            const float y = network_to_params((float)data[idx + 3], ACTIVATION_COORDINATES);
            const float z = network_to_params((float)data[idx + 4], ACTIVATION_COORDINATES);
            mSG[i].mu_original = Vector3f(x, y, z);
            mSG[i].mu = mSG[i].mu_original.normalized();
            totalWeight += mSG[i].lambda;
        }
        DCHECK_NE(totalWeight, 0);
        for (int i = 0; i < N; i++)
        {
            weight[i] = mSG[i].lambda / totalWeight;
        }
    }

    ELAINA_CALLABLE float pdf(Vector3f wi) const
    {
        DCHECK(wi.any());
        float pdf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            pdf += weight[i] * mSG[i].pdf(wi);
        }
        return pdf;
    }

    // Evaluate pdf of the i-th vMF component.
    ELAINA_CALLABLE float pdf(uint i, Vector3f wi) const
    {
        DCHECK_LT(i, N);
        return mSG[i].pdf(wi);
    }

    ELAINA_CALLABLE Vector3f sample(Sampler &sampler) const
    {
        float u = sampler.get1D();
        for (int i = 0; i < N; i++)
        {
            if (u < weight[i])
            {
                return mSG[i].sample(sampler);
            }
            u -= weight[i];
        }
        return mSG[0].sample(sampler);
    }

    /* Computes d_VMM(omega_i) / d_params (params applied activation) */
    template <typename T>
    ELAINA_CALLABLE float gradients_probability(const Vector3f wi, const bool isOnNeumannBoundary, const Vector3f &neumannBoundaryNormal, T *output) const
    {
        using namespace common3d;

        float probability = 0.0f; // calculate the probability by the way...
        Vector3f wiReflected;
        if (isOnNeumannBoundary)
        {
            wiReflected = reflect(wi, neumannBoundaryNormal);
        }

        for (int sg = 0; sg < N_COMP; sg++)
        {
            precision_t *cur_gradient = output + sg * 5;

            // [note] there's a divergence around kappa, theta, phi=0, when calculating dL_dparam.
            // So check the implementation of exponential activation, which uses a clamp to force x>0!
            float lambda = mSG[sg].lambda, kappa = mSG[sg].kappa,
                  mu_x = mSG[sg].mu[0], mu_y = mSG[sg].mu[1], mu_z = mSG[sg].mu[2],
                  mu_original_x = mSG[sg].mu_original[0], mu_original_y = mSG[sg].mu_original[1], mu_original_z = mSG[sg].mu_original[2];
            // Check the numerical stable version of the derivatives!
            float vmf = pdf(sg, wi);
            probability += weight[sg] * vmf;
            float vmfReflected = 0.0f;
            if (isOnNeumannBoundary)
            {
                vmfReflected = pdf(sg, wiReflected);
                probability += weight[sg] * vmfReflected;
            }

            float dF_dlambda = (vmf + vmfReflected) * (totalWeight - lambda) / pow2(totalWeight);
            for (int k = 0; k < N; k++)
            {
                if (k != sg)
                {
                    dF_dlambda -= weight[k] / totalWeight * mSG[k].pdf(wi);
                    if (isOnNeumannBoundary)
                    {
                        dF_dlambda -= weight[k] / totalWeight * mSG[k].pdf(wiReflected);
                    }
                }
            }
            float inv_kappa_minus_inv_tanh_kappa;
            if (kappa < 1)
            {
                inv_kappa_minus_inv_tanh_kappa = 0.000962f + -0.344883f * kappa + 0.030147f * pow2(kappa);
            }
            else
            {
                inv_kappa_minus_inv_tanh_kappa = 1 / kappa - (1 + expf(-2 * kappa)) / (1 - expf(-2 * kappa));
            }
            float x = wi[0], y = wi[1], z = wi[2];
            float xReflected = wiReflected[0], yReflected = wiReflected[1], zReflected = wiReflected[2];
            float dF_dkappa = weight[sg] * vmf * (x * mu_x + y * mu_y + z * mu_z + inv_kappa_minus_inv_tanh_kappa);
            if (isOnNeumannBoundary)
            {
                dF_dkappa += weight[sg] * vmfReflected * (xReflected * mu_x + yReflected * mu_y + zReflected * mu_z + inv_kappa_minus_inv_tanh_kappa);
            }
            float denom = pow(pow2(mu_original_x) + pow2(mu_original_y) + pow2(mu_original_z), 1.5f);
            if (denom < M_EPSILON)
            {
                denom = M_EPSILON;
            }
            float dF_dx = weight[sg] * vmf * kappa * (-mu_original_x * mu_original_y * y - mu_original_x * mu_original_z * z + pow2(mu_original_y) * x + pow2(mu_original_z) * x) / denom;
            if (isOnNeumannBoundary)
            {
                dF_dx += weight[sg] * vmfReflected * kappa * (-mu_original_x * mu_original_y * yReflected - mu_original_x * mu_original_z * zReflected + pow2(mu_original_y) * xReflected + pow2(mu_original_z) * xReflected) / denom;
            }
            float dF_dy = weight[sg] * vmf * kappa * (-mu_original_x * mu_original_y * x - mu_original_y * mu_original_z * z + pow2(mu_original_x) * y + pow2(mu_original_z) * y) / denom;
            if (isOnNeumannBoundary)
            {
                dF_dy += weight[sg] * vmfReflected * kappa * (-mu_original_x * mu_original_y * xReflected - mu_original_y * mu_original_z * zReflected + pow2(mu_original_x) * yReflected + pow2(mu_original_z) * yReflected) / denom;
            }
            float dF_dz = weight[sg] * vmf * kappa * (-mu_original_x * mu_original_z * x - mu_original_y * mu_original_z * y + pow2(mu_original_x) * z + pow2(mu_original_y) * z) / denom;
            if (isOnNeumannBoundary)
            {
                dF_dz += weight[sg] * vmfReflected * kappa * (-mu_original_x * mu_original_z * xReflected - mu_original_y * mu_original_z * yReflected + pow2(mu_original_x) * zReflected + pow2(mu_original_y) * zReflected) / denom;
            }

            cur_gradient[0] = dF_dlambda, cur_gradient[1] = dF_dkappa;
            cur_gradient[2] = dF_dx, cur_gradient[3] = dF_dy, cur_gradient[4] = dF_dz;
        }
        return probability;
    }

    ELAINA_HOST void print() const
    {
        for (int i = 0; i < N; i++)
        {
            ELAINA_LOG(Info, "Component %d: lambda = %f, kappa = %f, mu = (%f, %f, %f)", i, mSG[i].lambda, mSG[i].kappa, mSG[i].mu[0], mSG[i].mu[1], mSG[i].mu[2]);
        }
    }

protected:
    VMFKernel mSG[N];
    float weight[N], totalWeight{};
};

ELAINA_NAMESPACE_END