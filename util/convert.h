#pragma once
#include "core/common.h"

ELAINA_NAMESPACE_BEGIN

template <typename T, typename U>
ELAINA_CALLABLE T convert(const U &u)
{
    return static_cast<T>(u);
}

template <>
ELAINA_CALLABLE float2 convert(const Vector2f &v)
{
    return make_float2(v.x(), v.y());
}

template <>
ELAINA_CALLABLE float3 convert(const Vector3f &v)
{
    return make_float3(v.x(), v.y(), v.z());
}

template <>
ELAINA_CALLABLE Vector2f convert(const float2 &v)
{
    return Vector2f(v.x, v.y);
}

template <>
ELAINA_CALLABLE Vector3f convert(const float3 &v)
{
    return Vector3f(v.x, v.y, v.z);
}

template <>
ELAINA_CALLABLE Vector2i convert(const int2 &v)
{
    return Vector2i(v.x, v.y);
}

template <>
ELAINA_CALLABLE Vector3i convert(const int3 &v)
{
    return Vector3i(v.x, v.y, v.z);
}

template <>
ELAINA_CALLABLE Color convert(const nanovdb::Vec3f &v)
{
    return Color(v[0], v[1], v[2]);
}

ELAINA_NAMESPACE_END