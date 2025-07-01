#pragma once
#include "common.h"
#include <Eigen/Dense>

#include "vector.h"
#include "matrix.h"
#include "constants.h"

ELAINA_NAMESPACE_BEGIN

#ifdef ELAINA_DEVICE_CODE
using ::abs;
using ::copysign;
using ::fmod;
using ::isnan;
using ::max;
using ::min;
#else
using std::abs;
using std::copysign;
using std::fmod;
using std::isnan;
using std::max;
using std::min;
#endif

using ::cos;
using ::cosh;
using ::pow;
using ::sin;
using ::sinh;
using ::tan;
using ::tanh;

template <typename T>
ELAINA_CALLABLE auto clamp(T v, T lo, T hi)
{
	return std::max(std::min(v, hi), lo);
}

template <typename DerivedV, typename DerivedB>
ELAINA_CALLABLE auto clamp(const Eigen::MatrixBase<DerivedV> &v, DerivedB lo, DerivedB hi)
{
	return v.cwiseMin(hi).cwiseMax(lo);
}

template <typename DerivedV, typename DerivedB>
ELAINA_CALLABLE auto clamp(const Eigen::ArrayBase<DerivedV> &v, DerivedB lo, DerivedB hi)
{
	return v.min(hi).max(lo);
}

template <typename DerivedV, typename DerivedB>
ELAINA_CALLABLE auto clamp(const Eigen::EigenBase<DerivedV> &v, DerivedB lo, DerivedB hi)
{
	return clamp(v.eval(), lo, hi);
}

template <typename DerivedA, typename DerivedB, typename DerivedT>
ELAINA_CALLABLE auto lerp(const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
						  DerivedT t)
{
	return (a.eval() * (1 - t) + b.eval() * t).eval();
}

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedT>
ELAINA_CALLABLE auto barycentric_interpolate(
	const Eigen::DenseBase<DerivedA> &a,
	const Eigen::DenseBase<DerivedB> &b,
	const Eigen::DenseBase<DerivedC> &c,
	DerivedT u,
	DerivedT v)
{
	DerivedT w = 1 - u - v;
	return (a.eval() * w + b.eval() * u + c.eval() * v).eval();
}

template <uint DIM, typename T, typename UVType>
ELAINA_CALLABLE auto geometric_interpolate(
	const T a[DIM],
	const UVType &uv)
{
	if constexpr (DIM == 2)
	{
		return lerp(a[0], a[1], uv);
	}
	else if constexpr (DIM == 3)
	{
		return barycentric_interpolate(a[0], a[1], a[2], uv[0], uv[1]);
	}
	else
	{
		static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
	}
}

// overload unary opeartors

template <typename DerivedV>
ELAINA_CALLABLE auto normalize(const Eigen::MatrixBase<DerivedV> &v)
{
	return v.normalized();
}

template <typename DerivedV>
ELAINA_CALLABLE auto abs(const Eigen::MatrixBase<DerivedV> &v)
{
	return v.cwiseAbs();
}

template <typename DerivedV>
ELAINA_CALLABLE auto length(const Eigen::MatrixBase<DerivedV> &v)
{
	return v.norm();
}

template <typename DerivedV>
ELAINA_CALLABLE auto squaredLength(const Eigen::MatrixBase<DerivedV> &v)
{
	return v.SquaredNorm();
}

template <typename DerivedV>
ELAINA_CALLABLE auto any(const Eigen::DenseBase<DerivedV> &v)
{
	return v.any();
}

// overload binary operators

template <typename DerivedA, typename DerivedB>
ELAINA_CALLABLE auto cross(const Eigen::MatrixBase<DerivedA> &a,
						   const Eigen::MatrixBase<DerivedB> &b)
{
	return a.cross(b);
}

template <typename DerivedA, typename DerivedB>
ELAINA_CALLABLE auto dot(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b)
{
	return a.dot(b);
}

template <typename DerivedA, typename DerivedB>
ELAINA_CALLABLE auto operator/(const Eigen::MatrixBase<DerivedA> &a,
							   const Eigen::MatrixBase<DerivedB> &b)
{
	return a.cwiseQuotient(b);
}

// power shortcuts
template <typename T>
ELAINA_CALLABLE constexpr T pow1(T x) { return x; }
template <typename T>
ELAINA_CALLABLE constexpr T pow2(T x) { return x * x; }
template <typename T>
ELAINA_CALLABLE constexpr T pow3(T x) { return x * x * x; }
template <typename T>
ELAINA_CALLABLE constexpr T pow4(T x) { return x * x * x * x; }
template <typename T>
ELAINA_CALLABLE constexpr T pow5(T x) { return x * x * x * x * x; }

ELAINA_CALLABLE float sqrt(const float v) { return sqrtf(v); }

template <typename T>
ELAINA_CALLABLE T mod(T a, T b)
{
	T result = a - (a / b) * b;
	return (T)((result < 0) ? result + b : result);
}

template <typename T>
ELAINA_CALLABLE T safe_sqrt(T value)
{
	return sqrt(max((T)0, value));
}

ELAINA_CALLABLE float saturate(const float &f) { return min(1.f, max(0.f, f)); }

ELAINA_CALLABLE float rcp(float f) { return 1.f / f; }

ELAINA_CALLABLE float logistic(const float x) { return 1 / (1.f + expf(-x)); }

ELAINA_CALLABLE float csch(const float x) { return 1 / sinh(x); }

ELAINA_CALLABLE float coth(const float x) { return 1 / tanh(x); }

ELAINA_CALLABLE float sech(const float x) { return 1 / cosh(x); }

ELAINA_CALLABLE float radians(const float degree) { return degree * M_PI / 180.f; }

/* space transformations (all in left-handed coordinate) */

template <typename T, int Options = math::ColMajor>
ELAINA_CALLABLE Matrix<T, 4, 4, Options> perspective(T fovy, T aspect, T zNear, T zFar)
{
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

	T const tanHalfFovy = tan(fovy / static_cast<T>(2));
	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Zero()};

	result(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovy);
	result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
	result(2, 2) = -(zFar + zNear) / (zFar - zNear);
	result(2, 3) = -static_cast<T>(1);
	result(3, 2) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
	return result;
}

template <typename T, int Options = math::ColMajor>
ELAINA_CALLABLE Matrix<T, 4, 4, Options> orthogonal(T left, T right, T bottom, T top)
{
	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};

	result(0, 0) = static_cast<T>(2) / (right - left);
	result(1, 1) = static_cast<T>(2) / (top - bottom);
	result(2, 2) = -static_cast<T>(1);
	result(3, 0) = -(right + left) / (right - left);
	result(3, 1) = -(top + bottom) / (top - bottom);
	return result;
}

template <typename T, int Options = math::ColMajor>
Matrix<T, 4, 4, Options> look_at(Vector3<T> const &eye, Vector3<T> const &center,
								 Vector3<T> const &up)
{
	Vector3<T> const f(normalize(center - eye));
	Vector3<T> const s(normalize(cross(up, f)));
	Vector3<T> const u(cross(f, s));

	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};
	result(0, 0) = s.x;
	result(1, 0) = s.y;
	result(2, 0) = s.z;
	result(0, 1) = u.x;
	result(1, 1) = u.y;
	result(2, 1) = u.z;
	result(0, 2) = f.x;
	result(1, 2) = f.y;
	result(2, 2) = f.z;
	result(3, 0) = -dot(s, eye);
	result(3, 1) = -dot(u, eye);
	result(3, 2) = -dot(f, eye);
	return result;
}

ELAINA_NAMESPACE_END