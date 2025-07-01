#pragma once
#include "core/common.h"
#include "core/problem.h"
#include "util/check.h"
#include "cuda.h"
#include "core/device/buffer.h"
#include "core/device/cuda.h"
#include "core/device/context.h"
#include "core/file.h"
#include "util/tonemapping.cuh"

#include "integrator/uniform/workqueue.h"

#include <snch_lbvh/scene.cuh>
#include <snch_lbvh/lbvh.cuh>

ELAINA_NAMESPACE_BEGIN

class Film;
class EvaluationGrid<2>;
class EvaluationGrid<3>;

namespace detail {
    template <typename UniformIntegratorContext>
    ELAINA_HOST void initializeImpl(UniformIntegratorContext &ctx);
}

struct UniformIntegratorSettings
{
    Vector2i frameSize{800, 800};
    uint debugPixel{0};
    int samplesPerPixel{512};
    uint maxWalkingDepth{32};
    int saveSppMetricsDuration{-1};
    int saveSppMetricsUntil{1024};
    int saveTimeMetricsDuration{-1};
    float epsilonShell{M_EPSILON};

    ELAINA_CLASS_DEFINE(
        UniformIntegratorSettings,
        frameSize,
        debugPixel,
        samplesPerPixel,
        maxWalkingDepth,
        saveSppMetricsDuration,
        saveSppMetricsUntil,
        saveTimeMetricsDuration,
        epsilonShell);
};

template <unsigned int DIM>
class UniformIntegrator;

template <>
class UniformIntegrator<2>
{
public:
    using IntegratorSettings = UniformIntegratorSettings;

    using InShellPointWorkItem = common2d::InShellPointWorkItem;
    using OutShellPointWorkItem = common2d::OutShellPointWorkItem;
    using EvaluationPointWorkItem = common2d::EvaluationPointWorkItem;

    using EvaluationPointQueue = common2d::EvaluationPointQueue;
    using InShellPointQueue = common2d::InShellPointQueue;
    using OutShellPointQueue = common2d::OutShellPointQueue;
    using PixelStateBuffer = common2d::PixelStateBuffer;

    using AABBType = AABB2f;
    using VectorType = Vector2f;
    using ProbeType = EvaluationGrid<2>;
    using ProblemType = Problem<2>;

public:
    UniformIntegrator(Problem<2> &problem, const IntegratorSettings &settings, const fs::path &basePath_);
    ~UniformIntegrator() = default;

private:
    EvaluationPointQueue *evaluationPointQueue[2]{};
    InShellPointQueue *inShellPointQueue{};
    OutShellPointQueue *outShellPointQueue{};
    PixelStateBuffer *pixelStateBuffer{};

public:
    ELAINA_HOST void initialize();

public:
    ELAINA_CALLABLE auto *currentEvaluationPointQueue(int depth) { return evaluationPointQueue[depth & 1]; }
    ELAINA_CALLABLE auto *nextEvaluationPointQueue(int depth) { return evaluationPointQueue[(depth & 1) ^ 1]; }
    ELAINA_CALLABLE auto *get_inShellPointQueue() { return inShellPointQueue; }
    ELAINA_CALLABLE auto *get_outShellPointQueue() { return outShellPointQueue; }
    ELAINA_CALLABLE auto *get_pixelStateBuffer() { return pixelStateBuffer; }

public:
    template <typename... Args>
    ELAINA_DEVICE_FUNCTION void debugPrint(Args &&...args);

public:
    ELAINA_HOST uint64_t solve();

public:
    ELAINA_HOST void exportImage(ExportImageChannel imageType, const string &file_name);
    ELAINA_HOST void exportEnergy(ExportImageChannel imageType, ToneMapping tone, const string &file_name);
    ELAINA_HOST void renderDirichletSDF();
    ELAINA_HOST void renderSilhouetteSDF();
    ELAINA_HOST void renderSource();
    ELAINA_HOST void queryNetwork(const VectorType &p);

private:
    Film *renderedImage[static_cast<size_t>(ExportImageChannel::CHANNEL_COUNT)] = {nullptr};
    int maxQueueSize{};
    EvaluationGrid<2> *evaluation_grid{nullptr};
    IntegratorSettings integratorSettings{};
    fs::path basePath{};
    Problem<2> &problem;

public:
    ELAINA_CALLABLE const auto &get_integratorSettings() const { return integratorSettings; }
    ELAINA_CALLABLE auto *get_dirichletSDFChannel() { return renderedImage[0]; }
    ELAINA_CALLABLE auto *get_neumannSDFChannel() { return renderedImage[1]; }
    ELAINA_CALLABLE auto *get_sourceChannel() { return renderedImage[2]; }
    ELAINA_CALLABLE auto *get_solutionChannel() { return renderedImage[3]; }
    ELAINA_CALLABLE const auto get_maxQueueSize() { return maxQueueSize; }
    ELAINA_CALLABLE const auto &get_basePath() { return basePath; }
    ELAINA_CALLABLE const auto get_probe() { return evaluation_grid; }
    ELAINA_CALLABLE const auto &get_problem() { return problem; }

public:
    template <typename UniformIntegratorContext>
    friend void detail::initializeImpl(UniformIntegratorContext &ctx);
};

template <>
class UniformIntegrator<3>
{
public:
    using IntegratorSettings = UniformIntegratorSettings;

    using InShellPointWorkItem = common3d::InShellPointWorkItem;
    using OutShellPointWorkItem = common3d::OutShellPointWorkItem;
    using EvaluationPointWorkItem = common3d::EvaluationPointWorkItem;

    using EvaluationPointQueue = common3d::EvaluationPointQueue;
    using InShellPointQueue = common3d::InShellPointQueue;
    using OutShellPointQueue = common3d::OutShellPointQueue;
    using PixelStateBuffer = common3d::PixelStateBuffer;

    using AABBType = AABB3f;
    using VectorType = Vector3f;
    using ProbeType = EvaluationGrid<3>;
    using ProblemType = Problem<3>;

public:
    UniformIntegrator(Problem<3> &problem, const IntegratorSettings &settings, const fs::path &basePath_);
    ~UniformIntegrator() = default;

private:
    EvaluationPointQueue *evaluationPointQueue[2]{};
    InShellPointQueue *inShellPointQueue{};
    OutShellPointQueue *outShellPointQueue{};
    PixelStateBuffer *pixelStateBuffer{};

public:
    ELAINA_HOST void initialize();

public:
    ELAINA_CALLABLE auto *currentEvaluationPointQueue(int depth) { return evaluationPointQueue[depth & 1]; }
    ELAINA_CALLABLE auto *nextEvaluationPointQueue(int depth) { return evaluationPointQueue[(depth & 1) ^ 1]; }
    ELAINA_CALLABLE auto *get_inShellPointQueue() { return inShellPointQueue; }
    ELAINA_CALLABLE auto *get_outShellPointQueue() { return outShellPointQueue; }
    ELAINA_CALLABLE auto *get_pixelStateBuffer() { return pixelStateBuffer; }

public:
    template <typename... Args>
    ELAINA_DEVICE_FUNCTION void debugPrint(Args &&...args);

public:
    ELAINA_HOST uint64_t solve();

public:
    ELAINA_HOST void exportImage(ExportImageChannel imageType, const string &file_name);
    ELAINA_HOST void exportEnergy(ExportImageChannel imageType, ToneMapping tone, const string &file_name);
    ELAINA_HOST void renderDirichletSDF();
    ELAINA_HOST void renderSilhouetteSDF();
    ELAINA_HOST void renderSource();
    ELAINA_HOST void queryNetwork(const VectorType &p);

private:
    Film *renderedImage[static_cast<size_t>(ExportImageChannel::CHANNEL_COUNT)] = {nullptr};
    int maxQueueSize{};
    EvaluationGrid<3> *evaluation_grid{nullptr};
    IntegratorSettings integratorSettings{};
    fs::path basePath{};
    Problem<3> &problem;

public:
    ELAINA_CALLABLE const auto &get_integratorSettings() const { return integratorSettings; }
    ELAINA_CALLABLE auto *get_dirichletSDFChannel() { return renderedImage[0]; }
    ELAINA_CALLABLE auto *get_neumannSDFChannel() { return renderedImage[1]; }
    ELAINA_CALLABLE auto *get_sourceChannel() { return renderedImage[2]; }
    ELAINA_CALLABLE auto *get_solutionChannel() { return renderedImage[3]; }
    ELAINA_CALLABLE const auto get_maxQueueSize() { return maxQueueSize; }
    ELAINA_CALLABLE const auto &get_basePath() { return basePath; }
    ELAINA_CALLABLE const auto get_probe() { return evaluation_grid; }
    ELAINA_CALLABLE const auto &get_problem() { return problem; }

public:
    template <typename UniformIntegratorContext>
    friend void detail::initializeImpl(UniformIntegratorContext &ctx);
};

ELAINA_NAMESPACE_END