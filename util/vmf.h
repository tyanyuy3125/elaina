#pragma once

#include "core/common.h"
#include "core/sampler.h"
#include "util/sampling.h"
#include "transformation.h"
#include "check.h"

#include <cmath>

// The method present in 
// A Numerically Stable Implementation of the von Mises–Fisher Distribution on S^2
// seems to perform slightly worse(?) when kappa falls in the ordinary range.
//
// So we temporary comment it out and use the implementation of Jakob [2012].
//
// TODO: double check.

ELAINA_NAMESPACE_BEGIN

class VMF {
public:
	VMF() = default;
	
	ELAINA_CALLABLE VMF(float kappa) : m_kappa(kappa) {}

	ELAINA_CALLABLE float eval(float cosTheta) const { 
		if (m_kappa < M_EPSILON)
			return M_INV_4PI;
		return expf(m_kappa * min(0.f, cosTheta - 1.f)) * m_kappa /
			   (M_2PI * (1 - expf(-2 * m_kappa)));
	}

	ELAINA_CALLABLE float eval(const Vector3f& wi, const Vector3f& mu) const { 
		return eval(dot(wi, mu)); 
	}

	// ELAINA_CALLABLE float eval(const Vector3f &wi, const Vector3f &mu) const {
	// 	if (m_kappa < M_EPSILON)
	// 		return M_INV_4PI;
	// 	auto d = wi - mu;
	// 	return expf(-0.5f * m_kappa * d.squaredNorm()) * x_over_expm1(-2.0f * m_kappa) / (M_4PI);
	// }
	
	ELAINA_CALLABLE Vector3f sample(Sampler &sampler) const {
		if (m_kappa < M_EPSILON)
			return uniformSampleSphere<3>(sampler);
        Vector2f u = sampler.get2D();
		float cosTheta = 1 + log1p(-u[0] + expf(-2 * m_kappa) * u[0]) / m_kappa;
		float sinTheta = safe_sqrt(1 - cosTheta * cosTheta), sinPhi, cosPhi;
		float phi	   = M_2PI * u[1];
		sinPhi		   = sin(phi);
		cosPhi		   = cos(phi);
		return Vector3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}

	// ELAINA_CALLABLE Vector3f sample(Sampler &sampler) const {
	// 	if (m_kappa < M_EPSILON)
	// 		return uniformSampleSphere<3>(sampler);
	// 	Vector2f u = sampler.get2D();
	// 	float phi = M_2PI * u[0];
	// 	float r = log1p(u[1] * expm1(-2.0f * m_kappa)) / m_kappa;
	// 	float cos_theta = 1.0f + r;
	// 	float sin_theta = sqrt(-fma(r, r, 2.0f * r));
	// 	return Vector3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
	// }

	ELAINA_CALLABLE Vector3f sample(Sampler &sampler, Vector3f mu) const { 
		return frameFromNormal(mu).toWorld(sample(sampler));
	}
	
private:
	float m_kappa{};

	ELAINA_CALLABLE static float x_over_expm1(const float x) {
		float u = expf(x);
		if (u == 1.0f) { return 1.0f; }
		float y = u - 1.0f;
		if (abs(x) < 1.0f) {
			return logf(u) / y;
		}
		return x / y;
	}
};

ELAINA_NAMESPACE_END