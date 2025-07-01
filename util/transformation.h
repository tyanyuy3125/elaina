#pragma once

#include "core/common.h"

ELAINA_NAMESPACE_BEGIN

template <uint DIM>
struct Frame;

template <>
struct Frame<3> {
    ELAINA_CALLABLE Frame(Vector3f n, Vector3f t, Vector3f b)
        : N(n), T(t), B(b)
        {}

	ELAINA_CALLABLE Vector3f toWorld(const Vector3f& v) const {
		return T * v[0] + B * v[1] + N * v[2];
	}

	ELAINA_CALLABLE Vector3f toLocal(const Vector3f& v) const {
		return { dot(T, v), dot(B, v), dot(N, v) };
	}

	Vector3f N;
	Vector3f T;
	Vector3f B;
};

template <>
struct Frame<2> {
    ELAINA_CALLABLE Frame(const Vector2f &n, const Vector2f &t)
        : N(n), T(t)
        {}

    ELAINA_CALLABLE Vector2f toWorld(const Vector2f& v) const {
        return T * v[0] + N * v[1];
    }

    ELAINA_CALLABLE Vector2f getNormal() const { return N; }

    ELAINA_CALLABLE Vector2f getTangent() const { return T; }

    Vector2f N;
    Vector2f T;
};

ELAINA_CALLABLE Frame<2> frameFromTangent(const Vector2f &t)
{
    return Frame<2>(utils::getPerpendicular(t), t);
}

ELAINA_CALLABLE Frame<2> frameFromNormal(const Vector2f &n)
{
    return Frame<2>(n, -utils::getPerpendicular(n));
}

ELAINA_CALLABLE Frame<3> frameFromTangent(const Vector3f &t)
{
    ELAINA_NOTIMPLEMENTED;
}

ELAINA_CALLABLE Frame<3> frameFromNormal(const Vector3f &n)
{
    const auto t = utils::getPerpendicular(n);
    const auto b = normalize(cross(n, t));
    return Frame<3>(n, t, b);
}

template <typename VectorType>
ELAINA_CALLABLE VectorType reflect(const VectorType& dirVec, const VectorType& normal) {
    return dirVec - 2 * dirVec.dot(normal) * normal;
}

ELAINA_NAMESPACE_END