#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "vector.h"

ELAINA_NAMESPACE_BEGIN

template <typename T>
class Quaternion : public Eigen::Quaternion<T> {
public:
	using Eigen::Quaternion<T>::Quaternion;

	ELAINA_CALLABLE Quaternion(void) : Eigen::Quaternion<T>() {}

	template <typename OtherDerived>
	ELAINA_CALLABLE Quaternion(const Eigen::QuaternionBase<OtherDerived> &other) : Eigen::Quaternion<T>(other) {}

	template <typename OtherDerived>
	ELAINA_CALLABLE Quaternion &operator=(const Eigen::QuaternionBase<OtherDerived> &other) {
		this->Eigen::Quaternion<T>::operator=(other);
		return *this;
	}

	ELAINA_CALLABLE static Quaternion fromAxisAngle(const Vector3f &axis, T angle) {
		return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(angle, axis));
	}

	ELAINA_CALLABLE static Quaternion fromEuler(T yaw, T pitch, T roll) {
		return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(yaw, elaina::Vector3<T>::UnitY()) *
									Eigen::AngleAxis<T>(roll, elaina::Vector3<T>::UnitZ()) *
									Eigen::AngleAxis<T>(pitch, elaina::Vector3<T>::UnitX()));
	}
};

using Quaternionf = Quaternion<float>;

ELAINA_NAMESPACE_END