#pragma once

#include "common.h"

ELAINA_NAMESPACE_BEGIN

template <unsigned int DIM>
class EvaluationGrid;

template <>
class EvaluationGrid<2>
{
public:
    using SharedPtr = std::shared_ptr<EvaluationGrid<2>>;

    struct ProbeData
    {
        float scale{1.0f};
        Vector2f pos{0.0f, 0.0f}; // center
        Vector2f up{0.0f, 1.0f};  // up vector

        ELAINA_CLASS_DEFINE(ProbeData, scale, pos, up);
    };

    EvaluationGrid() = default;

    ELAINA_CALLABLE Vector2f getEvaluationPoint(Vector2i pixel, Vector2i frameSize)
    {
        Vector2f ndc = 2.0f * Vector2f(pixel) / Vector2f(frameSize) + Vector2f(-1.0f);
        Vector2f u{mData.up.y(), -mData.up.x()};
        Vector2f v = mData.up;
        return mData.scale * (ndc.x() * u + ndc.y() * v) + mData.pos;
    }

public:
    ProbeData mData;

public:
    ELAINA_CLASS_DEFINE(EvaluationGrid<2>, mData);
};

template <>
class EvaluationGrid<3>
{
public:
    using SharedPtr = std::shared_ptr<EvaluationGrid<3>>;

    struct ProbeData
    {
        float scale{1.0f};
        Vector3f pos{0.0f, 0.0f, 0.0f};
        Vector3f up{0.0f, 0.0f, 1.0f}; // +z as up vector.
        Vector3f right{1.0f, 0.0f, 0.0f}; // +x as right vector.

        ELAINA_CLASS_DEFINE(ProbeData, scale, pos, up, right);
    };

    EvaluationGrid() = default;

    ELAINA_CALLABLE Vector3f getEvaluationPoint(Vector2i pixel, Vector2i frameSize)
    {
        Vector2f ndc = 2.0f * Vector2f(pixel) / Vector2f(frameSize) + Vector2f(-1.0f);
        return mData.scale * (ndc.x() * mData.right + ndc.y() * mData.up) + mData.pos;
    }

private:
    ProbeData mData;

public:
    ELAINA_CLASS_DEFINE(EvaluationGrid<3>, mData);
};

ELAINA_NAMESPACE_END