#pragma once
#include "core/common.h"
#include "core/sampler.h"

ELAINA_NAMESPACE_BEGIN

/* Remember to copy these definitions to workitem.soa whenever changing them. */

namespace common2d
{

    struct PixelState
    {
        Color solution;
        PCGSampler sampler;
    };

    struct EvaluationPointWorkItem
    {
        Vector2f point;

        uint depth;
        uint pixelId;

        Color thp;

        bool isOnNeumannBoundary;
        Vector2f neumannNormal;
    };

    struct InShellPointWorkItem
    {
        Vector2f point;

        uint depth;
        uint pixelId;

        Color thp;

        float R_D;

        // Interaction
        int side;
        Vector2i indices;
        float uv;
    };

    struct OutShellPointWorkItem
    {
        Vector2f point;

        uint depth;
        uint pixelId;

        Color thp;

        float R_D;
        float R_B;

        bool isOnNeumannBoundary;
        Vector2f neumannNormal;
    };

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "integrator/uniform/uniform2d_workitem_soa.h"
#pragma warning(pop)

}

namespace common3d
{

    struct PixelState
    {
        Color solution;
        PCGSampler sampler;
    };

    struct EvaluationPointWorkItem
    {
        Vector3f point;

        uint depth;
        uint pixelId;

        Color thp;

        bool isOnNeumannBoundary;
        Vector3f neumannNormal;
    };

    struct InShellPointWorkItem
    {
        Vector3f point;

        uint depth;
        uint pixelId;

        Color thp;

        float R_D;

        // Interaction
        int side;
        Vector3i indices;
        Vector2f uv;
    };

    struct OutShellPointWorkItem
    {
        Vector3f point;

        uint depth;
        uint pixelId;

        Color thp;

        float R_D;
        float R_B;

        bool isOnNeumannBoundary;
        Vector3f neumannNormal;
    };

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "integrator/uniform/uniform3d_workitem_soa.h"
#pragma warning(pop)

}

ELAINA_NAMESPACE_END