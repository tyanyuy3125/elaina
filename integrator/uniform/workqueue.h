#pragma once
#include "core/common.h"
#include <atomic>

#include "core/device/cuda.h"
#include "core/device/atomic.h"
#include "core/logger.h"
#include "workitem.h"

ELAINA_NAMESPACE_BEGIN

namespace common2d
{

    class PixelStateBuffer : public SOA<PixelState>
    {
    public:
        PixelStateBuffer() = default;
        PixelStateBuffer(int n, Allocator alloc) : SOA<PixelState>(n, alloc) {}

        ELAINA_CALLABLE void setColor(int pixelId, Color solution_val)
        {
            solution[pixelId] = solution_val;
        }
        ELAINA_CALLABLE void addColor(int pixelId, Color solution_val)
        {
            solution_val = solution_val + Color(solution[pixelId]);
            solution[pixelId] = solution_val;
        }
    };

    template <typename WorkItem>
    class WorkQueue : public SOA<WorkItem>
    {
    public:
        WorkQueue() = default;
        ELAINA_HOST WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
        ELAINA_HOST WorkQueue &operator=(const WorkQueue &w)
        {
            SOA<WorkItem>::operator=(w);
            m_size.store(w.m_size);
            return *this;
        }

        ELAINA_CALLABLE int size() const
        {
            return m_size.load();
        }
        ELAINA_CALLABLE void reset()
        {
            m_size.store(0);
        }

        ELAINA_CALLABLE int push(WorkItem w)
        {
            int index = allocateEntry();
            (*this)[index] = w;
            return index;
        }

    protected:
        ELAINA_CALLABLE int allocateEntry()
        {
            return m_size.fetch_add(1);
        }

    private:
        atomic<int> m_size{0};
    };

    template <typename F, typename WorkItem>
    void ForAllQueued(const WorkQueue<WorkItem> *q, int nElements,
                      F &&func, CUstream stream = 0)
    {
        GPUParallelFor(nElements, [=] ELAINA_DEVICE(int index) mutable
                       {
        if (index >= q->size())
            return;
        func((*q)[index]); }, stream);
    }

    template <typename F, typename WorkItem>
    void ForAllQueuedWithIndex(const WorkQueue<WorkItem> *q, int nElements,
                               F &&func, CUstream stream = 0)
    {
        GPUParallelFor(nElements, [=] ELAINA_DEVICE(int index) mutable
                       {
        if (index >= q->size())
            return;
        func((*q)[index], index); }, stream);
    }

    class EvaluationPointQueue : public WorkQueue<EvaluationPointWorkItem>
    {
    public:
        using WorkQueue::push;
        using WorkQueue::WorkQueue;

        ELAINA_CALLABLE void pushEvaluationPoint(const Vector2f &point, uint pixelId)
        {
            EvaluationPointWorkItem item;
            item.point = point;
            item.depth = 0;
            item.pixelId = pixelId;
            item.thp = Color::Ones();
            item.isOnNeumannBoundary = false;
            item.neumannNormal = Vector2f::Zero();

            push(item);
        }
    };

    class InShellPointQueue : public WorkQueue<InShellPointWorkItem>
    {
    public:
        using WorkQueue::push;
        using WorkQueue::WorkQueue;
    };

    class OutShellPointQueue : public WorkQueue<OutShellPointWorkItem>
    {
    public:
        using WorkQueue::push;
        using WorkQueue::WorkQueue;
    };

}

namespace common3d
{

    class PixelStateBuffer : public SOA<PixelState>
    {
    public:
        PixelStateBuffer() = default;
        PixelStateBuffer(int n, Allocator alloc) : SOA<PixelState>(n, alloc) {}

        ELAINA_CALLABLE void setColor(int pixelId, Color solution_val)
        {
            solution[pixelId] = solution_val;
        }
        ELAINA_CALLABLE void addColor(int pixelId, Color solution_val)
        {
            // if (solution_val.hasNaN())
            // {
            //     printf("solution_val has NaN\n");
            // }
            solution_val = solution_val + Color(solution[pixelId]);
            solution[pixelId] = solution_val;
        }
    };

    template <typename WorkItem>
    class WorkQueue : public SOA<WorkItem>
    {
    public:
        WorkQueue() = default;
        ELAINA_HOST WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
        ELAINA_HOST WorkQueue &operator=(const WorkQueue &w)
        {
            SOA<WorkItem>::operator=(w);
            m_size.store(w.m_size);
            return *this;
        }

        ELAINA_CALLABLE int size() const
        {
            return m_size.load();
        }
        ELAINA_CALLABLE void reset()
        {
            m_size.store(0);
        }

        ELAINA_CALLABLE int push(WorkItem w)
        {
            int index = allocateEntry();
            (*this)[index] = w;
            return index;
        }

    protected:
        ELAINA_CALLABLE int allocateEntry()
        {
            return m_size.fetch_add(1);
        }

    private:
        atomic<int> m_size{0};
    };

    template <typename F, typename WorkItem>
    void ForAllQueued(const WorkQueue<WorkItem> *q, int nElements,
                      F &&func, CUstream stream = 0)
    {
        GPUParallelFor(nElements, [=] ELAINA_DEVICE(int index) mutable
                       {
        if (index >= q->size())
            return;
        func((*q)[index]); }, stream);
    }

    template <typename F, typename WorkItem>
    void ForAllQueuedWithIndex(const WorkQueue<WorkItem> *q, int nElements,
                               F &&func, CUstream stream = 0)
    {
        GPUParallelFor(nElements, [=] ELAINA_DEVICE(int index) mutable
                       {
        if (index >= q->size())
            return;
        func((*q)[index], index); }, stream);
    }

    class EvaluationPointQueue : public WorkQueue<EvaluationPointWorkItem>
    {
    public:
        using WorkQueue::push;
        using WorkQueue::WorkQueue;

        ELAINA_CALLABLE void pushEvaluationPoint(const Vector3f &point, uint pixelId)
        {
            EvaluationPointWorkItem item;
            item.point = point;
            item.depth = 0;
            item.pixelId = pixelId;
            item.thp = Color::Ones();
            item.isOnNeumannBoundary = false;
            item.neumannNormal = Vector3f::Zero();

            push(item);
        }
    };

    class InShellPointQueue : public WorkQueue<InShellPointWorkItem>
    {
    public:
        using WorkQueue::push;
        using WorkQueue::WorkQueue;
    };

    class OutShellPointQueue : public WorkQueue<OutShellPointWorkItem>
    {
    public:
        using WorkQueue::push;
        using WorkQueue::WorkQueue;
    };

}

ELAINA_NAMESPACE_END