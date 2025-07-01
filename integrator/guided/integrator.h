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

#include "integrator/guided/train.h"
#include "util/network.h"

#include <nlohmann/json.hpp>
#include <tiny-cuda-nn/common.h>

#include <snch_lbvh/scene.cuh>
#include <snch_lbvh/lbvh.cuh>

namespace tcnn
{
    template <typename T>
    class Loss;
    template <typename T>
    class Optimizer;
    template <typename T>
    class Encoding;
    template <typename T>
    class GPUMemory;
    template <typename T>
    class GPUMatrixDynamic;
    template <typename T, typename PARAMS_T>
    class Network;
    template <typename T, typename PARAMS_T, typename COMPUTE_T>
    class Trainer;
    template <uint32_t N_DIMS, uint32_t RANK, typename T>
    class TrainableBuffer;
}

ELAINA_NAMESPACE_BEGIN

namespace detail
{
    template <typename GuidedIntegratorContext>
    ELAINA_HOST void initializeImpl(GuidedIntegratorContext &ctx);
}

class Film;
class EvaluationGrid<2>;
class EvaluationGrid<3>;

using nlohmann::json;
using precision_t = tcnn::network_precision_t;

struct GuidedIntegratorSettings
{
    Vector2i frameSize{800, 800};
    uint debugPixel{0};
    int samplesPerPixel{512};

    uint trainSppCount{150};
    float uniformFractionInTrainingPhase{0.5f};
    float uniformFractionInGuidingPhase{0.5f};
    uint maxGuidedDepthInTrainingPhase{10};
    uint maxGuidedDepthInGuidingPhase{10};

    uint maxWalkingDepth{32};
    int saveSppMetricsDuration{-1};
    int saveSppMetricsUntil{1024};
    int saveTimeMetricsDuration{-1};

    float epsilonShell{M_EPSILON};

    ELAINA_CLASS_DEFINE(
        GuidedIntegratorSettings,
        frameSize,
        debugPixel,
        samplesPerPixel,
        trainSppCount,
        uniformFractionInTrainingPhase,
        uniformFractionInGuidingPhase,
        maxGuidedDepthInTrainingPhase,
        maxGuidedDepthInGuidingPhase,
        maxWalkingDepth,
        saveSppMetricsDuration,
        saveSppMetricsUntil,
        saveTimeMetricsDuration,
        epsilonShell);
};

template <unsigned int DIM>
class GuidedIntegrator;

template <>
class GuidedIntegrator<2>
{
public:
    using IntegratorSettings = GuidedIntegratorSettings;

    using EvaluationPointQueue = common2d::EvaluationPointQueue;
    using InShellPointQueue = common2d::InShellPointQueue;
    using OutShellPointQueue = common2d::OutShellPointQueue;

    using UniformWalkQueue = common2d::UniformWalkQueue;
    using GuidedInferenceQueue = common2d::GuidedInferenceQueue;

    using PixelStateBuffer = common2d::PixelStateBuffer;
    using GuidedPixelStateBuffer = common2d::GuidedPixelStateBuffer;
    using TrainBuffer = common2d::TrainBuffer;

    using OutShellPointWorkItem = common2d::OutShellPointWorkItem;
    using InShellPointWorkItem = common2d::InShellPointWorkItem;
    using EvaluationPointWorkItem = common2d::EvaluationPointWorkItem;
    using UniformWalkWorkItem = common2d::UniformWalkWorkItem;
    using GuidedInferenceWorkItem = common2d::GuidedInferenceWorkItem;

    using GuidedInput = common2d::GuidedInput;
    using GuidedOutput = common2d::GuidedOutput;

    using AABBType = AABB2f;
    using VectorType = Vector2f;
    using ProbeType = EvaluationGrid<2>;
    using ProblemType = Problem<2>;

public:
    GuidedIntegrator(ProblemType &problem, const IntegratorSettings &settings, const fs::path &basePath_);
    ~GuidedIntegrator() = default;

private:
    EvaluationPointQueue *evaluationPointQueue[2]{};
    InShellPointQueue *inShellPointQueue{};
    OutShellPointQueue *outShellPointQueue{};

    UniformWalkQueue *uniformWalkQueue{};
    GuidedInferenceQueue *guidedInferenceQueue{};

    PixelStateBuffer *pixelStateBuffer{};
    GuidedPixelStateBuffer *guidedPixelStateBuffer{};
    TrainBuffer *trainBuffer{};

    AABBType *sceneAABB{};

public:
    ELAINA_HOST void initialize();

public:
    template <typename F>
    void ParallelFor(int nElements, F &&func, CUstream stream = 0)
    {
        DCHECK_GT(nElements, 0);
        GPUParallelFor(nElements, func, stream);
    }

public:
    ELAINA_CALLABLE auto *currentEvaluationPointQueue(int depth) { return evaluationPointQueue[depth & 1]; }
    ELAINA_CALLABLE auto *nextEvaluationPointQueue(int depth) { return evaluationPointQueue[(depth & 1) ^ 1]; }
    ELAINA_CALLABLE auto *get_inShellPointQueue() { return inShellPointQueue; }
    ELAINA_CALLABLE auto *get_outShellPointQueue() { return outShellPointQueue; }
    ELAINA_CALLABLE auto *get_uniformWalkQueue() { return uniformWalkQueue; }
    ELAINA_CALLABLE auto *get_guidedInferenceQueue() { return guidedInferenceQueue; }
    ELAINA_CALLABLE auto *get_pixelStateBuffer() { return pixelStateBuffer; }
    ELAINA_CALLABLE auto *get_guidedPixelStateBuffer() { return guidedPixelStateBuffer; }
    ELAINA_CALLABLE auto *get_trainBuffer() { return trainBuffer; }
    ELAINA_CALLABLE auto *get_problemAABB() { return sceneAABB; }

public:
    template <typename... Args>
    ELAINA_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

public:
    ELAINA_HOST uint64_t solve();

public:
    ELAINA_HOST void resetNetwork(json config);
    ELAINA_HOST void resetTraining();

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
    ProbeType *evaluation_grid{nullptr};
    IntegratorSettings integratorSettings{};
    ProblemType &problem;
    fs::path basePath{};

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
    template <typename GuidedIntegratorContext>
    friend void detail::initializeImpl(GuidedIntegratorContext &ctx);

public:
    class Guidance
    {
    public:
        ELAINA_CALLABLE bool isEnableGuiding() const { return trainState.isEnableGuiding(); }

        ELAINA_CALLABLE bool isEnableTraining() const { return trainState.isEnableTraining(); }

        ELAINA_CALLABLE bool isEnableGuiding(uint depth) const
        {
            return trainState.isEnableGuiding() && depth < maxGuidedDepth;
        }

        ELAINA_CALLABLE bool isEnableTraining(uint depth) const
        {
            return trainState.isEnableTraining() && depth < maxTrainDepth;
        }

        ELAINA_CALLABLE bool isTrainingPixel(uint pixelId) const
        {
            return trainState.isTrainingPixel(pixelId);
        }

        common2d::TrainState trainState;
        PCGSampler sampler;
        cudaStream_t stream{};
        float uniformSamplingFraction{0.5f};
        uint maxGuidedDepth{10};
        uint maxTrainDepth{3};
        uint batchPerFrame{5};
        uint batchSize{TRAIN_BATCH_SIZE};
        uint trainSppCount{150};

        json config;
        EDivergence divergence_type{EDivergence::KL};
        std::shared_ptr<tcnn::Network<float, precision_t>> network;
        std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer;
        std::shared_ptr<tcnn::Encoding<precision_t>> encoding;
        std::shared_ptr<tcnn::Loss<precision_t>> loss;
        std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer;
    } mGuiding;

public:
    ELAINA_CALLABLE auto &get_mGuiding() { return mGuiding; }
};

template <>
class GuidedIntegrator<3>
{
public:
    using IntegratorSettings = GuidedIntegratorSettings;

    using EvaluationPointQueue = common3d::EvaluationPointQueue;
    using InShellPointQueue = common3d::InShellPointQueue;
    using OutShellPointQueue = common3d::OutShellPointQueue;

    using UniformWalkQueue = common3d::UniformWalkQueue;
    using GuidedInferenceQueue = common3d::GuidedInferenceQueue;

    using PixelStateBuffer = common3d::PixelStateBuffer;
    using GuidedPixelStateBuffer = common3d::GuidedPixelStateBuffer;
    using TrainBuffer = common3d::TrainBuffer;

    using OutShellPointWorkItem = common3d::OutShellPointWorkItem;
    using InShellPointWorkItem = common3d::InShellPointWorkItem;
    using EvaluationPointWorkItem = common3d::EvaluationPointWorkItem;
    using UniformWalkWorkItem = common3d::UniformWalkWorkItem;
    using GuidedInferenceWorkItem = common3d::GuidedInferenceWorkItem;

    using GuidedInput = common3d::GuidedInput;
    using GuidedOutput = common3d::GuidedOutput;

    using AABBType = AABB3f;
    using VectorType = Vector3f;
    using ProbeType = EvaluationGrid<3>;
    using ProblemType = Problem<3>;

public:
    GuidedIntegrator(ProblemType &problem, const IntegratorSettings &settings, const fs::path &basePath_);
    ~GuidedIntegrator() = default;

private:
    EvaluationPointQueue *evaluationPointQueue[2]{};
    InShellPointQueue *inShellPointQueue{};
    OutShellPointQueue *outShellPointQueue{};

    UniformWalkQueue *uniformWalkQueue{};
    GuidedInferenceQueue *guidedInferenceQueue{};

    PixelStateBuffer *pixelStateBuffer{};
    GuidedPixelStateBuffer *guidedPixelStateBuffer{};
    TrainBuffer *trainBuffer{};

    AABBType *sceneAABB{};

public:
    ELAINA_HOST void initialize();

public:
    template <typename F>
    void ParallelFor(int nElements, F &&func, CUstream stream = 0)
    {
        DCHECK_GT(nElements, 0);
        GPUParallelFor(nElements, func, stream);
    }

public:
    ELAINA_CALLABLE auto *currentEvaluationPointQueue(int depth) { return evaluationPointQueue[depth & 1]; }
    ELAINA_CALLABLE auto *nextEvaluationPointQueue(int depth) { return evaluationPointQueue[(depth & 1) ^ 1]; }
    ELAINA_CALLABLE auto *get_inShellPointQueue() { return inShellPointQueue; }
    ELAINA_CALLABLE auto *get_outShellPointQueue() { return outShellPointQueue; }
    ELAINA_CALLABLE auto *get_uniformWalkQueue() { return uniformWalkQueue; }
    ELAINA_CALLABLE auto *get_guidedInferenceQueue() { return guidedInferenceQueue; }
    ELAINA_CALLABLE auto *get_pixelStateBuffer() { return pixelStateBuffer; }
    ELAINA_CALLABLE auto *get_guidedPixelStateBuffer() { return guidedPixelStateBuffer; }
    ELAINA_CALLABLE auto *get_trainBuffer() { return trainBuffer; }
    ELAINA_CALLABLE auto *get_problemAABB() { return sceneAABB; }

public:
    template <typename... Args>
    ELAINA_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

public:
    ELAINA_HOST uint64_t solve();

public:
    ELAINA_HOST void resetNetwork(json config);
    ELAINA_HOST void resetTraining();

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
    ProbeType *evaluation_grid{nullptr};
    IntegratorSettings integratorSettings{};
    ProblemType &problem;
    fs::path basePath{};

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
    template <typename GuidedIntegratorContext>
    friend void detail::initializeImpl(GuidedIntegratorContext &ctx);

public:
    class Guidance
    {
    public:
        ELAINA_CALLABLE bool isEnableGuiding() const { return trainState.isEnableGuiding(); }

        ELAINA_CALLABLE bool isEnableTraining() const { return trainState.isEnableTraining(); }

        ELAINA_CALLABLE bool isEnableGuiding(uint depth) const
        {
            return trainState.isEnableGuiding() && depth < maxGuidedDepth;
        }

        ELAINA_CALLABLE bool isEnableTraining(uint depth) const
        {
            return trainState.isEnableTraining() && depth < maxTrainDepth;
        }

        ELAINA_CALLABLE bool isTrainingPixel(uint pixelId) const
        {
            return trainState.isTrainingPixel(pixelId);
        }

        common3d::TrainState trainState;
        PCGSampler sampler;
        cudaStream_t stream{};
        float uniformSamplingFraction{0.5f};
        uint maxGuidedDepth{10};
        uint maxTrainDepth{3};
        uint batchPerFrame{5};
        uint batchSize{TRAIN_BATCH_SIZE};
        uint trainSppCount{150};

        json config;
        // ELoss loss_type{ ELoss::L2 };	// legacy, optimize as radiance field
        EDivergence divergence_type{EDivergence::KL};
        std::shared_ptr<tcnn::Network<float, precision_t>> network;
        std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer;
        std::shared_ptr<tcnn::Encoding<precision_t>> encoding;
        std::shared_ptr<tcnn::Loss<precision_t>> loss;
        std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer;
    } mGuiding;

public:
    ELAINA_CALLABLE auto &get_mGuiding() { return mGuiding; }
};

ELAINA_NAMESPACE_END