#pragma once

#include "core/common.h"
#include "core/sampler.h"
#include "util/sampling.h"
#include "transformation.h"
#include "check.h"

ELAINA_NAMESPACE_BEGIN

class BetaDistribution {
public:
    BetaDistribution() = default;

    ELAINA_CALLABLE BetaDistribution(float alpha, float beta) : m_alpha(alpha), m_beta(beta) {}

    ELAINA_CALLABLE float eval(float x) const {
        return powf(x, m_alpha - 1) * powf(1 - x, m_beta - 1) / std::beta(m_alpha, m_beta);
    }

    ELAINA_CALLABLE float sample(Sampler &sampler) const {
        // Generate two Gamma distributed random variables
        float gamma_alpha = sampleGamma(m_alpha, sampler);
        float gamma_beta = sampleGamma(m_beta, sampler);

        // Generate Beta distributed random variable
        return gamma_alpha / (gamma_alpha + gamma_beta);
    }

private:
    float m_alpha{};
    float m_beta{};

    // Helper function to sample from Gamma distribution
    ELAINA_CALLABLE float sampleGamma(float shape, Sampler &sampler) const {
        if (shape < 1.0f) {
            // Use the method for shape < 1
            return sampleGammaLessThanOne(shape, sampler);
        } else {
            // Use the method for shape >= 1
            return sampleGammaGreaterEqualOne(shape, sampler);
        }
    }

    // Helper function to sample from Gamma distribution when shape < 1
    ELAINA_CALLABLE float sampleGammaLessThanOne(float shape, Sampler &sampler) const {
        float b = (shape + 1.0f) / M_E;
        while (true) {
            float u = sampler.get1D();
            float v = sampler.get1D();
            float x = b * u;
            if (x <= 0.0f) continue;
            float y = -logf(sampler.get1D());
            if (v <= powf(x, shape - 1.0f)) {
                return x;
            }
        }
    }

    // Helper function to sample from Gamma distribution when shape >= 1
    ELAINA_CALLABLE float sampleGammaGreaterEqualOne(float shape, Sampler &sampler) const {
        float d = shape - 1.0f / 3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        while (true) {
            float z = 0.0f;
            do {
                z = sampler.get1D() * 2.0f - 1.0f;
            } while (z <= -1.0f || z >= 1.0f);
            float v = 1.0f + c * z;
            if (v <= 0.0f) continue;
            float v_cubed = v * v * v;
            float u = sampler.get1D();
            if (u < 1.0f - 0.0331f * z * z * z * z) {
                return d * v_cubed;
            }
            if (logf(u) < 0.5f * z * z + d * (1.0f - v_cubed + logf(v_cubed))) {
                return d * v_cubed;
            }
        }
    }
};

ELAINA_NAMESPACE_END
