#pragma once
#include "core/common.h"
#include "core/sampler.h"
#include "core/problem.h"
#include "integrator/uniform/workqueue.h"
#include "integrator/guided/guideditem.h"

ELAINA_NAMESPACE_BEGIN

namespace common2d {

struct GuidedPixelStateBuffer : public SOA<GuidedPixelState>
{
public:
    GuidedPixelStateBuffer() = default;
    GuidedPixelStateBuffer(int n, Allocator alloc) : SOA<GuidedPixelState>(n, alloc) {}

    ELAINA_CALLABLE void reset(int pixelId)
    {
        // reset a guided state (call when begining a new frame...
        curDepth[pixelId] = 0;
    }

    /* Records raw (unnormalized) vertex data along the path of the current pixel */
    ELAINA_CALLABLE void incrementDepth(int pixelId,
                                        const Vector2f pos,
                                        const Vector2f dir,
                                        const float dirPdf, // effective pdf of the scatter direction
                                        const Color &thp,
                                        bool isOnNeumannBoundary,
                                        const Vector2f neumannBoundaryNormal,
                                        const Color &solution = {} // current solution
    )
    {
        int depth = curDepth[pixelId];
        if (depth >= MAX_TRAIN_DEPTH)
            return;
        records[depth].solution[pixelId] = solution;
        records[depth].pos[pixelId] = pos;
        records[depth].dir[pixelId] = dir;
        records[depth].dirPdf[pixelId] = dirPdf;
        records[depth].thp[pixelId] = thp;
        records[depth].isOnNeumannBoundary[pixelId] = isOnNeumannBoundary;
        records[depth].neumannBoundaryNormal[pixelId] = neumannBoundaryNormal;
        curDepth[pixelId] = depth + 1;
    }

    ELAINA_CALLABLE void recordSolution(int pixelId,
                                        const Color &solution)
    {
        int depth = min(curDepth[pixelId], (uint)MAX_TRAIN_DEPTH);
        for (int i = 0; i < depth; i++)
        {
            const Color &prev = records[i].solution[pixelId];
            records[i].solution[pixelId] = prev + solution;
        }
    }

    ELAINA_CALLABLE void recordSourceContribution(int pixelId,
                                              const Color &solution)
    {
        int depth = min(curDepth[pixelId], (uint)MAX_TRAIN_DEPTH);
        for (int i = 0; i <= depth; i++)
        {
            const Color &prev = records[i].solution[pixelId];
            records[i].solution[pixelId] = prev + solution;
        }
    }
};

class UniformWalkQueue : public WorkQueue<UniformWalkWorkItem>
{
public:
    using WorkQueue::push;
    using WorkQueue::WorkQueue;

    ELAINA_CALLABLE int push(uint index)
    {
        return push(UniformWalkWorkItem{index});
    }
};

class GuidedInferenceQueue : public WorkQueue<GuidedInferenceWorkItem>
{
public:
    using WorkQueue::push;
    using WorkQueue::WorkQueue;

    ELAINA_CALLABLE int push(uint index) { return push(GuidedInferenceWorkItem{index}); }
};

class PixelStateBuffer;

class TrainState
{
public:
    ELAINA_CALLABLE bool isEnableGuiding() const { return enableGuiding; }

    ELAINA_CALLABLE bool isEnableTraining() const { return enableTraining; }

    ELAINA_CALLABLE bool isTrainingPixel(uint pixelId) const
    {
        return enableTraining && (pixelId - trainPixelOffset) % trainPixelStride == 0;
    }

    bool enableTraining{false};
    bool enableGuiding{false};
    uint trainPixelOffset{0};
    uint trainPixelStride{1};
};

}

namespace common3d {

struct GuidedPixelStateBuffer : public SOA<GuidedPixelState>
{
public:
    GuidedPixelStateBuffer() = default;
    GuidedPixelStateBuffer(int n, Allocator alloc) : SOA<GuidedPixelState>(n, alloc) {}

    ELAINA_CALLABLE void reset(int pixelId)
    {
        // reset a guided state (call when begining a new frame...
        curDepth[pixelId] = 0;
    }

    /* Records raw (unnormalized) vertex data along the path of the current pixel */
    ELAINA_CALLABLE void incrementDepth(int pixelId,
                                        const Vector3f pos,
                                        const Vector3f wi,
                                        const float dirPdf, // effective pdf of the scatter direction
                                        const Color &thp,
                                        bool isOnNeumannBoundary,
                                        const Vector3f neumannBoundaryNormal,
                                        const Color &solution = {} // current solution
    )
    {
        int depth = curDepth[pixelId];
        if (depth >= MAX_TRAIN_DEPTH)
            return;
        records[depth].solution[pixelId] = solution;
        records[depth].pos[pixelId] = pos;
        records[depth].dir[pixelId] = wi;
        records[depth].dirPdf[pixelId] = dirPdf;
        records[depth].thp[pixelId] = thp;
        records[depth].isOnNeumannBoundary[pixelId] = isOnNeumannBoundary;
        records[depth].neumannBoundaryNormal[pixelId] = neumannBoundaryNormal;
        curDepth[pixelId] = depth + 1;
    }

    ELAINA_CALLABLE void recordSolution(int pixelId,
                                        const Color &solution)
    {
        int depth = min(curDepth[pixelId], (uint)MAX_TRAIN_DEPTH);
        for (int i = 0; i < depth; i++)
        {
            const Color &prev = records[i].solution[pixelId];
            records[i].solution[pixelId] = prev + solution;
        }
    }

    ELAINA_CALLABLE void recordSourceContribution(int pixelId,
                                              const Color &solution)
    {
        int depth = min(curDepth[pixelId], (uint)MAX_TRAIN_DEPTH);
        for (int i = 0; i <= depth; i++)
        {
            const Color &prev = records[i].solution[pixelId];
            records[i].solution[pixelId] = prev + solution;
        }
    }
};

class UniformWalkQueue : public WorkQueue<UniformWalkWorkItem>
{
public:
    using WorkQueue::push;
    using WorkQueue::WorkQueue;

    ELAINA_CALLABLE int push(uint index)
    {
        return push(UniformWalkWorkItem{index});
    }
};

class GuidedInferenceQueue : public WorkQueue<GuidedInferenceWorkItem>
{
public:
    using WorkQueue::push;
    using WorkQueue::WorkQueue;

    ELAINA_CALLABLE int push(uint index) { return push(GuidedInferenceWorkItem{index}); }
};

class PixelStateBuffer;

class TrainState
{
public:
    ELAINA_CALLABLE bool isEnableGuiding() const { return enableGuiding; }

    ELAINA_CALLABLE bool isEnableTraining() const { return enableTraining; }

    ELAINA_CALLABLE bool isTrainingPixel(uint pixelId) const
    {
        return enableTraining && (pixelId - trainPixelOffset) % trainPixelStride == 0;
    }

    bool enableTraining{false};
    bool enableGuiding{false};
    uint trainPixelOffset{0};
    uint trainPixelStride{1};
};

}

ELAINA_NAMESPACE_END