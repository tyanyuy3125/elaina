#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "vector.h"

ELAINA_NAMESPACE_BEGIN
	
template <typename T, int Size>
class AxisAligned : public Eigen::AlignedBox<T, Size> {
public:
	using Eigen::AlignedBox<T, Size>::AlignedBox;
	using VectorType = typename Eigen::AlignedBox<T, Size>::VectorType;
	
	ELAINA_CALLABLE AxisAligned() : Eigen::AlignedBox<T, Size>() {}

	ELAINA_CALLABLE AxisAligned(const Eigen::AlignedBox<T, Size> &other) : Eigen::AlignedBox<T, Size>(other) {}

	ELAINA_CALLABLE AxisAligned &operator=(const Eigen::AlignedBox<T, Size> &other) {
		this->Eigen::AlignedBox<T, Size>::operator=(other);
		return *this;
	}

	ELAINA_CALLABLE void inflate(T inflation) {
		this->m_min -= VectorType::Constant(inflation);
		this->m_max += VectorType::Constant(inflation);
	}

	ELAINA_CALLABLE VectorType clip(const VectorType& p) const {
		return p.cwiseMin(this->m_max).cwiseMax(this->m_min);
	}
};
	
using AABB3f = AxisAligned<float, 3>;
using AABB2f = AxisAligned<float, 2>;

ELAINA_NAMESPACE_END