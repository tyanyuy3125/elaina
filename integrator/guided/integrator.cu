#include "integrator.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include "util/film.h"
#include "util/ema.h"
#include "util/sampling.h"
#include "util/green.h"
#include "core/evaluation_grid.h"
#include "core/logger.h"
#include "core/config.h"
#include "util/convert.h"
#include "integrator/guided/distribution.h"
#include "integrator/common.h"

ELAINA_NAMESPACE_BEGIN
using namespace tcnn;

template <typename T>
using GPUMatrix = tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor>;

namespace
{
    // training buffers / memories
    tcnn::GPUMemory<precision_t> trainOutputBuffer;
    tcnn::GPUMemory<float> inferenceOutputBuffer;
    tcnn::GPUMemory<precision_t> gradientBuffer;
    tcnn::GPUMemory<float> lossBuffer;
    tcnn::GPUMemory<float> inferenceInputBuffer;

    // lossgraph and training info logging / plotting
    std::vector<float> lossGraph(LOSS_GRAPH_SIZE, 0);
    size_t numLossSamples{0};
    size_t numTrainingSamples{0};
    Ema curLossScalar{Ema::Type::Time, 50};
}

namespace detail
{
    template <typename GuidedIntegratorContext>
    ELAINA_HOST void initializeImpl(GuidedIntegratorContext &ctx)
    {
        Allocator &alloc = *gpContext->alloc;
        ctx.maxQueueSize = ctx.integratorSettings.frameSize[0] * ctx.integratorSettings.frameSize[1];
        CUDA_SYNC_CHECK();
        // Initialize queues
        for (int i = 0; i < 2; ++i)
        {
            if (ctx.evaluationPointQueue[i])
                ctx.evaluationPointQueue[i]->resize(ctx.maxQueueSize, alloc);
            else
                ctx.evaluationPointQueue[i] = alloc.new_object<typename GuidedIntegratorContext::EvaluationPointQueue>(ctx.maxQueueSize, alloc);
        }
        if (ctx.inShellPointQueue)
            ctx.inShellPointQueue->resize(ctx.maxQueueSize, alloc);
        else
            ctx.inShellPointQueue = alloc.new_object<typename GuidedIntegratorContext::InShellPointQueue>(ctx.maxQueueSize, alloc);
        if (ctx.outShellPointQueue)
            ctx.outShellPointQueue->resize(ctx.maxQueueSize, alloc);
        else
            ctx.outShellPointQueue = alloc.new_object<typename GuidedIntegratorContext::OutShellPointQueue>(ctx.maxQueueSize, alloc);
        if (ctx.uniformWalkQueue)
            ctx.uniformWalkQueue->resize(ctx.maxQueueSize, alloc);
        else
            ctx.uniformWalkQueue = alloc.new_object<typename GuidedIntegratorContext::UniformWalkQueue>(ctx.maxQueueSize, alloc);
        if (ctx.guidedInferenceQueue)
            ctx.guidedInferenceQueue->resize(ctx.maxQueueSize, alloc);
        else
            ctx.guidedInferenceQueue = alloc.new_object<typename GuidedIntegratorContext::GuidedInferenceQueue>(ctx.maxQueueSize, alloc);

        // Initialize buffers
        if (ctx.pixelStateBuffer)
            ctx.pixelStateBuffer->resize(ctx.maxQueueSize, alloc);
        else
            ctx.pixelStateBuffer = alloc.new_object<typename GuidedIntegratorContext::PixelStateBuffer>(ctx.maxQueueSize, alloc);
        if (ctx.guidedPixelStateBuffer)
            ctx.guidedPixelStateBuffer->resize(ctx.maxQueueSize, alloc);
        else
            ctx.guidedPixelStateBuffer = alloc.new_object<typename GuidedIntegratorContext::GuidedPixelStateBuffer>(ctx.maxQueueSize, alloc);
        if (!ctx.trainBuffer)
            ctx.trainBuffer = alloc.new_object<typename GuidedIntegratorContext::TrainBuffer>(TRAIN_BUFFER_SIZE);

        // Initialize rendered image
        for (int i = 0; i < static_cast<size_t>(ExportImageChannel::CHANNEL_COUNT); ++i)
        {
            auto &renderedImage = ctx.renderedImage[i];
            if (renderedImage)
                renderedImage->resize(ctx.integratorSettings.frameSize);
            else
                renderedImage = alloc.new_object<Film>(ctx.integratorSettings.frameSize);
            renderedImage->reset();
        }

        // Initialize evaluation_grid
        if (!ctx.evaluation_grid)
            ctx.evaluation_grid = alloc.new_object<typename GuidedIntegratorContext::ProbeType>();
        if (!ctx.sceneAABB)
            ctx.sceneAABB = alloc.new_object<typename GuidedIntegratorContext::AABBType>();

        CUDA_SYNC_CHECK();
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void prepareSolveImpl(GuidedIntegratorContext &ctx)
    {
        const auto &integratorSettings = ctx.get_integratorSettings();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto &mGuiding = ctx.get_mGuiding();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();

        ParallelFor(
            maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) { // reset per-pixel sample state
                Vector2i pixelCoord = {pixelId % integratorSettings.frameSize[0], pixelId / integratorSettings.frameSize[0]};
                pixelStateBuffer->solution[pixelId] = 0;
                pixelStateBuffer->sampler[pixelId].setPixelSample(pixelCoord, 0);
                pixelStateBuffer->sampler[pixelId].advance(256 * pixelId);
            });

        mGuiding.trainState.trainPixelOffset = mGuiding.trainState.trainPixelStride <= 1 ? 0 : mGuiding.sampler.get1D() * mGuiding.trainState.trainPixelStride;
        mGuiding.trainState.enableTraining = true;
        mGuiding.trainState.enableGuiding = true;
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void generateEvaluationPointsImpl(GuidedIntegratorContext &ctx)
    {
        const auto &integratorSettings = ctx.get_integratorSettings();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *generatedEvaluationPointQueue = ctx.currentEvaluationPointQueue(0);
        const auto evaluation_grid = ctx.get_probe();
        const auto &problem = ctx.get_problem();
        const auto mask_ptr = problem.get_mask_device();

        ParallelFor(
            maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
                if (mask_ptr[pixelId] == false)
                {
                    return;
                }
                Vector2i pixelCoord = {pixelId % integratorSettings.frameSize[0], pixelId / integratorSettings.frameSize[0]};
                auto evaluationPoint = evaluation_grid->getEvaluationPoint(pixelCoord, integratorSettings.frameSize);
                generatedEvaluationPointQueue->pushEvaluationPoint(evaluationPoint, pixelId);
            });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void separateEvaluationPointImpl(GuidedIntegratorContext &ctx, uint depth)
    {
        auto *currentQueue = ctx.currentEvaluationPointQueue(depth);
        auto *inShellPointQueue = ctx.get_inShellPointQueue();
        auto *outShellPointQueue = ctx.get_outShellPointQueue();
        const auto &problem = ctx.get_problem();
        const auto &integratorSettings = ctx.get_integratorSettings();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto isDirichletEnabled = problem.isDirichletEnabled();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        auto vertices_ptr = problem.get_problem_dirichlet_ptr()->vertices_d.data().get();

        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_dirichlet_bvh_device;
        if (isDirichletEnabled)
        {
            problem_dirichlet_bvh_device = problem.get_problem_dirichlet_bvh_device();
        }
        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;

        ForAllQueued(currentQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::EvaluationPointWorkItem &w) {
            constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
            using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
            query_point_type query_point = convert<query_point_type>(w.point);

            float R_D = INFINITY;
            if(isDirichletEnabled)
            {
                auto nearest_query_result = lbvh::query_device(problem_dirichlet_bvh_device, lbvh::nearest(query_point), lbvh::scene<DIM>::distance_calculator());
                int nearest_object_index = nearest_query_result.first;
                const auto nearest_object = problem_dirichlet_bvh_device.objects[nearest_object_index];
                const auto p0 = vertices_ptr[nearest_object.vertex_indices.x];
                const auto p1 = vertices_ptr[nearest_object.vertex_indices.y];
                using uv_type = std::conditional_t<(DIM == 2), float, float2>;
                int side; 
                uv_type uv;
                if constexpr (DIM == 2)
                {
                    side = lbvh::checkPointSide(p0, p1, query_point);
                    uv = lbvh::computeProjectionRatio(p0, p1, query_point);
                }
                else
                {
                    const auto p2 = vertices_ptr[nearest_object.vertex_indices.z];
                    side = lbvh::checkPointSide(p0, p1, p2, query_point);
                    uv = lbvh::computeProjectionRatio(p0, p1, p2, query_point);
                }
                R_D = nearest_query_result.second;
                bool inEpsilonShell = R_D < integratorSettings.epsilonShell;
                if constexpr (DIM == 2)
                {
                    inEpsilonShell &= (uv > 0.0f && uv < 1.0f);
                }
                else
                {
                    inEpsilonShell &= (uv.x > 0.0f);
                    inEpsilonShell &= (uv.y > 0.0f);
                    inEpsilonShell &= (uv.x + uv.y < 1.0f);
                }
                if(inEpsilonShell)
                {
                    typename GuidedIntegratorContext::InShellPointWorkItem inShellPointWorkItem;
                    inShellPointWorkItem.point = w.point;
                    inShellPointWorkItem.depth = w.depth;
                    inShellPointWorkItem.pixelId = w.pixelId;
                    inShellPointWorkItem.thp = w.thp;
                    inShellPointWorkItem.R_D = R_D;
                    inShellPointWorkItem.side = side;
                    inShellPointWorkItem.indices = convert<decltype(inShellPointWorkItem.indices)>(nearest_object.vertex_indices);
                    inShellPointWorkItem.uv = uv;
                    inShellPointQueue->push(inShellPointWorkItem);
                    return;
                }
            }
            float R_N = INFINITY;
            if (isNeumannEnabled)
            {
                R_N = lbvh::query_device(problem_neumann_bvh_device, lbvh::nearest_silhouette(query_point, false), lbvh::scene<DIM>::silhouette_distance_calculator());
            }
            float R_B = max(1e-4f, min(R_D, R_N));
            // R_B *= 0.99f;

            typename GuidedIntegratorContext::OutShellPointWorkItem outShellPointWorkItem;
            outShellPointWorkItem.point = w.point;
            outShellPointWorkItem.depth = w.depth;
            outShellPointWorkItem.pixelId = w.pixelId;
            outShellPointWorkItem.thp = w.thp;
            outShellPointWorkItem.R_D = R_D;
            outShellPointWorkItem.R_B = R_B;
            outShellPointWorkItem.isOnNeumannBoundary = w.isOnNeumannBoundary;
            outShellPointWorkItem.neumannNormal = w.neumannNormal;
            outShellPointQueue->push(outShellPointWorkItem); });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void handleBoundaryImpl(GuidedIntegratorContext &ctx)
    {
        const auto &problem = ctx.get_problem();
        auto vertex_color_dirichlet = problem.get_vertex_color_dirichlet_device();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *inShellPointQueue = ctx.get_inShellPointQueue();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();
        auto &mGuiding = ctx.get_mGuiding();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto dirichlet_intensity = problem.get_dirichlet_intensity();

        ForAllQueued(inShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::InShellPointWorkItem &w) {
            constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
            Color interpolated_color = computeSurfaceColor<DIM>(vertex_color_dirichlet, w.indices, w.side, w.uv);
            interpolated_color *= dirichlet_intensity;
            interpolated_color *= w.thp;
            pixelStateBuffer->addColor(w.pixelId, interpolated_color);
            if (mGuiding.isTrainingPixel(w.pixelId))
            {
                guidedPixelStateBuffer->recordSolution(w.pixelId, interpolated_color);
            } });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void sampleSourceImpl(GuidedIntegratorContext &ctx)
    {
        auto *outShellPointQueue = ctx.get_outShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto source_vdb_ptr = problem.get_source_vdb_ptr();
        const auto source_intensity = problem.get_source_intensity();
        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        auto &mGuiding = ctx.get_mGuiding();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();

        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::OutShellPointWorkItem &w) {
            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;

            if (isinf(w.R_B))
            {
                return;
            }

            VectorType dirVec;
            float dirPdf;
            float alpha = 1.0f;
            if (w.isOnNeumannBoundary)
            {
                dirVec = uniformSampleHemisphere<DIM>(sampler);
                dirVec = frameFromNormal(w.neumannNormal).toWorld(dirVec);
                dirPdf = uniformSampleHemispherePDF<DIM>();
                alpha = 0.5f;
            }
            else
            {
                dirVec = uniformSampleSphere<DIM>(sampler);
                dirPdf = uniformSampleSpherePDF<DIM>();
            }

            float dist = w.R_B;
            VectorType offset_point = w.point + integratorSettings.epsilonShell * dirVec;
            if (isNeumannEnabled)
            {
                using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
                query_point_type query_origin = convert<query_point_type>(offset_point);
                query_point_type query_dir = convert<query_point_type>(dirVec);
                auto neumann_intersect_result = lbvh::query_device(problem_neumann_bvh_device, lbvh::ray_intersect(lbvh::ray<float, DIM>(query_origin, query_dir), dist), lbvh::scene<DIM>::intersect_test());
                bool intersected = thrust::get<0>(neumann_intersect_result);
                if (intersected)
                {
                    dist = min(thrust::get<1>(neumann_intersect_result), dist);
                }
            }

            auto harmonic_green = HarmonicGreenBall<DIM>(w.R_B);
            auto [r, radiusPdf] = harmonic_green.sample(sampler);
            if (r <= dist)
            {
                nanovdb::Vec3f sourcePt(w.point.x() + r * dirVec.x(),
                                        w.point.y() + r * dirVec.y(),
                                        0.0f);
                if constexpr (DIM == 3)
                {
                    sourcePt[2] = w.point.z() + r * dirVec.z();
                }
                nanovdb::Vec3f gridIndex = source_vdb_ptr->worldToIndex(sourcePt);
                nanovdb::math::SampleFromVoxels<nanovdb::Vec3fTree, 1, false> sampler(source_vdb_ptr->tree());
                Color gridValueVec = convert<Color>(sampler(gridIndex));
                gridValueVec *= source_intensity;
                auto contrib = w.thp * gridValueVec * harmonic_green.norm() * conditionalSampleSpherePDF<DIM>(uniformSampleSpherePDF<DIM>(), r) / conditionalSampleSpherePDF<DIM>(dirPdf, r) / alpha;
                pixelStateBuffer->addColor(w.pixelId, contrib);
                if (mGuiding.isTrainingPixel(w.pixelId))
                {
                    guidedPixelStateBuffer->recordSourceContribution(w.pixelId, contrib);
                }
            } });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void sampleNeumannImpl(GuidedIntegratorContext &ctx)
    {
        // This function assumes that the scene has Neumann boundary.
        const auto &problem = ctx.get_problem();
        auto *outShellPointQueue = ctx.get_outShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *pixelStateBuffer = ctx.get_pixelStateBuffer();
        auto vertices_ptr = problem.get_problem_neumann_ptr()->vertices_d.data().get();
        const auto &integratorSettings = ctx.get_integratorSettings();
        const auto neumann_intensity = problem.get_neumann_intensity();

        auto vertex_color_neumann = problem.get_vertex_color_neumann_device();
        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();

        constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        auto &mGuiding = ctx.get_mGuiding();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();

        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::OutShellPointWorkItem &w) {
            constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
            using color_pair_type = thrust::pair<Vector3f, Vector3f>;
            using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
            query_point_type query_point = convert<query_point_type>(w.point);

            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            float u[DIM];
            for (int i = 0; i < DIM; ++i)
            {
                u[i] = sampler.get1D();
            }
            auto sample_object_result = lbvh::sample_object_in_sphere(problem_neumann_bvh_device,
                                                                      lbvh::sphere_intersect(lbvh::sphere<float, DIM>(query_point, w.R_B)),
                                                                      lbvh::scene<DIM>::intersect_sphere(),
                                                                      lbvh::scene<DIM>::measurement_getter(),
                                                                      lbvh::scene<DIM>::green_weight(),
                                                                      u[0]);
            int object_index = sample_object_result.first;
            float pdf = sample_object_result.second;

            if (object_index != -1 && pdf > 0)
            {
                const auto sampled_object = problem_neumann_bvh_device.objects[object_index];
                auto sample_point_result = lbvh::sample_on_object(problem_neumann_bvh_device,
                                                                  object_index,
                                                                  lbvh::scene<DIM>::sample_on_object(),
                                                                  u[1], u[2]);

                VectorType sample_point = convert<VectorType>(sample_point_result);
                float r = (sample_point - w.point).norm();
                
                if(r < w.R_B && r > 0)
                {
                    // Check if the intersection is the first intersection.
                    {
                        auto origin_point = w.point;
                        if(w.isOnNeumannBoundary)
                        {
                            origin_point += integratorSettings.epsilonShell * w.neumannNormal;
                        }
                        VectorType ray_dir = sample_point - origin_point;
                        float clamp_dist = ray_dir.norm();
                        if(clamp_dist > 0)
                        {
                            ray_dir /= clamp_dist;
                        }
                        if(lbvh::query_device(problem_neumann_bvh_device, 
                                            lbvh::ray_intersect<true>(
                                                    lbvh::ray<float, DIM>(convert<query_point_type>(origin_point), 
                                                    convert<query_point_type>(ray_dir)), 
                                                    clamp_dist - integratorSettings.epsilonShell),
                                            lbvh::scene<DIM>::intersect_test()))
                        {
                            return;
                        }
                    }

                    // Perform color query and interpolation.
                    using uv_type = std::conditional_t<(DIM == 2), float, float2>;
                    int side; 
                    uv_type uv;

                    const auto p0 = vertices_ptr[sampled_object.vertex_indices.x];
                    const auto p1 = vertices_ptr[sampled_object.vertex_indices.y];
                    if constexpr (DIM == 2)
                    {
                        side = lbvh::checkPointSide(p0, p1, query_point);
                        uv = lbvh::computeProjectionRatio(p0, p1, sample_point_result);
                    }
                    else
                    {
                        const auto p2 = vertices_ptr[sampled_object.vertex_indices.z];
                        side = lbvh::checkPointSide(p0, p1, p2, query_point);
                        uv = lbvh::computeProjectionRatio(p0, p1, p2, sample_point_result);
                    }
                    if (w.isOnNeumannBoundary)
                    {
                        VectorType current_normal = convert<VectorType>(problem_neumann_bvh_device.objects[object_index].normal());
                        side = utils::sign(current_normal.dot(w.neumannNormal));
                    }
                    if(side == 0)
                    {
                        return;
                    }

                    Color interpolated_color = computeSurfaceColor<DIM>(
                        vertex_color_neumann, 
                        convert<std::conditional_t<(DIM == 2), Vector2i, Vector3i>>(sampled_object.vertex_indices), 
                        side, 
                        convert<std::conditional_t<(DIM == 2), float, Vector2f>>(uv));

                    interpolated_color *= neumann_intensity;

                    float alpha = 1.0f;
                    if (w.isOnNeumannBoundary)
                    {
                        alpha = 0.5f;
                    }

                    auto harmonic_green = HarmonicGreenBall<DIM>(w.R_B);
                    interpolated_color *= w.thp * harmonic_green.eval(r) / alpha / pdf;
                    pixelStateBuffer->addColor(w.pixelId, -interpolated_color);
                    if (mGuiding.isTrainingPixel(w.pixelId))
                    {
                        guidedPixelStateBuffer->recordSourceContribution(w.pixelId, -interpolated_color);
                    }
                }
            } });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void handleOutShellPointImpl(GuidedIntegratorContext &ctx)
    {
        auto outShellPointQueue = ctx.get_outShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto &mGuiding = ctx.get_mGuiding();
        auto guidedInferenceQueue = ctx.get_guidedInferenceQueue();
        auto uniformWalkQueue = ctx.get_uniformWalkQueue();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto sceneAABB = ctx.get_problemAABB();

        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto NUM_VMF_COMPONENTS = (DIM == 3) ? common3d::NUM_VMF_COMPONENTS : common2d::NUM_VMF_COMPONENTS;
        static constexpr auto N_DIM_OUTPUT = (DIM == 3) ? common3d::N_DIM_OUTPUT : common2d::N_DIM_OUTPUT;
        static constexpr auto N_DIM_VMF = (DIM == 3) ? common3d::N_DIM_VMF : common2d::N_DIM_VMF;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        float *output = inferenceOutputBuffer.data();
        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::OutShellPointWorkItem &w) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const float* dist_data = output + tid * N_DIM_OUTPUT;
            float selection_probability = network_to_params(dist_data[NUM_VMF_COMPONENTS * N_DIM_VMF], ACTIVATION_SELECTION_PROBABILITY);
            if(mGuiding.isEnableGuiding(w.depth) && (mGuiding.uniformSamplingFraction == 0 || pixelStateBuffer->sampler[w.pixelId].get1D() < selection_probability) && sceneAABB->contains(w.point))
            {
                guidedInferenceQueue->push(tid);
            }
            else
            {
                uniformWalkQueue->push(tid);
            } });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void inferenceStepImpl(GuidedIntegratorContext &ctx)
    {
        auto &mGuiding = ctx.get_mGuiding();
        auto outShellPointQueue = ctx.get_outShellPointQueue();
        auto *sceneAABB = ctx.get_problemAABB();

        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto N_DIM_INPUT = (DIM == 3) ? common3d::N_DIM_INPUT : common2d::N_DIM_INPUT;
        static constexpr auto N_DIM_OUTPUT = (DIM == 3) ? common3d::N_DIM_OUTPUT : common2d::N_DIM_OUTPUT;

        // CUDA_SYNC_CHECK();
        cudaStreamSynchronize(0);
        const cudaStream_t &stream = mGuiding.stream;
        std::shared_ptr<Network<float, precision_t>> network = mGuiding.network;
        if (!network)
            logFatal("Network not initialized!");
        int numInferenceSamples = outShellPointQueue->size();
        if (numInferenceSamples == 0)
            return;

        {
            // Data preparation
            LinearKernel(generate_inference_data<typename GuidedIntegratorContext::OutShellPointQueue, typename GuidedIntegratorContext::AABBType>, stream, MAX_INFERENCE_NUM,
                         outShellPointQueue, inferenceInputBuffer.data(), sceneAABB);
        }
        int paddedBatchSize = next_multiple(numInferenceSamples, 128);
        GPUMatrix<float> networkInputs(inferenceInputBuffer.data(), N_DIM_INPUT, paddedBatchSize);
        GPUMatrix<float> networkOutputs(inferenceOutputBuffer.data(), N_DIM_OUTPUT, paddedBatchSize);

        {
            // Network inference
            network->inference(stream, networkInputs, networkOutputs);
            // CUDA_SYNC_CHECK();
        }
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void queryNetworkImpl(GuidedIntegratorContext &ctx, const std::conditional_t<ExtractDim<GuidedIntegratorContext>::value == 2, Vector2f, Vector3f> &queryPoint)
    {
        auto &mGuiding = ctx.get_mGuiding();
        auto *sceneAABB = ctx.get_problemAABB();

        auto normalizedQueryPoint = normalizeSpatialCoord(queryPoint, *sceneAABB);

        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto NUM_VMF_COMPONENTS = (DIM == 3) ? common3d::NUM_VMF_COMPONENTS : common2d::NUM_VMF_COMPONENTS;
        static constexpr auto N_DIM_INPUT = (DIM == 3) ? common3d::N_DIM_INPUT : common2d::N_DIM_INPUT;
        static constexpr auto N_DIM_OUTPUT = (DIM == 3) ? common3d::N_DIM_OUTPUT : common2d::N_DIM_OUTPUT;

        CUDA_SYNC_CHECK();
        const cudaStream_t &stream = mGuiding.stream;
        std::shared_ptr<Network<float, precision_t>> network = mGuiding.network;

        if (!network)
            logFatal("Network not initialized!");

        tcnn::GPUMemory<float> queryInputBuffer(1 * N_DIM_INPUT);
        tcnn::GPUMemory<float> queryOutputBuffer(1 * N_DIM_OUTPUT);
        float *queryInputData = queryInputBuffer.data();
        GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
            *(reinterpret_cast<std::decay_t<decltype(queryPoint)> *>(queryInputData)) = normalizedQueryPoint;
        });
        int paddedBatchSize = next_multiple(1, 128);
        GPUMatrix<float> networkInputs(queryInputData, N_DIM_INPUT, paddedBatchSize);
        GPUMatrix<float> networkOutputs(queryOutputBuffer.data(), N_DIM_OUTPUT, paddedBatchSize);

        {
            // Network inference
            network->inference(stream, networkInputs, networkOutputs);
            CUDA_SYNC_CHECK();
        }
        thrust::device_vector<float> outputData(1 * N_DIM_OUTPUT);
        outputData.assign(queryOutputBuffer.data(), queryOutputBuffer.data() + 1 * N_DIM_OUTPUT);
        thrust::host_vector<float> outputDataHost = outputData;

        VMM<DIM, NUM_VMF_COMPONENTS> vmm(outputDataHost.data());
        if constexpr (DIM == 2)
        {
            ELAINA_LOG(Info, "VMM @ (%f, %f): ", queryPoint.x(), queryPoint.y());
            vmm.print();
        }
        else
        {
            ELAINA_LOG(Info, "VMM @ (%f, %f, %f): ", queryPoint.x(), queryPoint.y(), queryPoint.z());
            vmm.print();
        }
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void trainStepImpl(GuidedIntegratorContext &ctx)
    {
        auto &mGuiding = ctx.get_mGuiding();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *trainBuffer = ctx.get_trainBuffer();
        auto *guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();
        auto *sceneAABB = ctx.get_problemAABB();

        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto N_DIM_INPUT = (DIM == 3) ? common3d::N_DIM_INPUT : common2d::N_DIM_INPUT;
        static constexpr auto N_DIM_PADDED_OUTPUT = (DIM == 3) ? common3d::N_DIM_PADDED_OUTPUT : common2d::N_DIM_PADDED_OUTPUT;

        const cudaStream_t &stream = mGuiding.stream;
        std::shared_ptr<Network<float, precision_t>> network = mGuiding.network;
        if (!network)
            logFatal("Network not initialized!");
        uint numTrainPixels = maxQueueSize / mGuiding.trainState.trainPixelStride;
        LinearKernel(generate_training_data<typename GuidedIntegratorContext::TrainBuffer, typename GuidedIntegratorContext::GuidedPixelStateBuffer, typename GuidedIntegratorContext::AABBType>, stream, numTrainPixels,
                     mGuiding.trainState.trainPixelOffset, mGuiding.trainState.trainPixelStride,
                     trainBuffer, guidedPixelStateBuffer, sceneAABB);
        CUDA_SYNC_CHECK();
        numTrainingSamples = trainBuffer->size();

        uint numTrainBatches = min((uint)numTrainingSamples / mGuiding.batchSize + 1, mGuiding.batchPerFrame);
        for (int iter = 0; iter < numTrainBatches; iter++)
        {
            size_t localBatchSize = min(numTrainingSamples - iter * mGuiding.batchSize, (size_t)mGuiding.batchSize);
            localBatchSize -= localBatchSize % 128;
            if (localBatchSize < MIN_TRAIN_BATCH_SIZE)
                break;
            float *inputData = (float *)(trainBuffer->inputs() + iter * mGuiding.batchSize);
            typename GuidedIntegratorContext::GuidedOutput *outputData = trainBuffer->outputs() + iter * mGuiding.batchSize;

            GPUMatrix<float> networkInputs(inputData, N_DIM_INPUT, localBatchSize);
            GPUMatrix<precision_t> networkOutputs(trainOutputBuffer.data(), N_DIM_PADDED_OUTPUT, localBatchSize);
            GPUMatrix<precision_t> dL_doutput(gradientBuffer.data(), N_DIM_PADDED_OUTPUT, localBatchSize);
            {
                std::unique_ptr<tcnn::Context> ctx = network->forward(stream, networkInputs, &networkOutputs, false, false);
                CUDA_SYNC_CHECK();

                LinearKernel(compute_dL_doutput_divergence<typename GuidedIntegratorContext::GuidedOutput>, stream, localBatchSize,
                             networkOutputs.data(), outputData, dL_doutput.data(), lossBuffer.data(), TRAIN_LOSS_SCALE, mGuiding.divergence_type);

                network->backward(stream, *ctx, networkInputs, networkOutputs, dL_doutput, nullptr, false, EGradientMode::Overwrite);
                mGuiding.trainer->optimizer_step(stream, TRAIN_LOSS_SCALE);
                float loss = thrust::reduce(thrust::device, lossBuffer.data(), lossBuffer.data() + localBatchSize, 0.f, thrust::plus<float>());
                curLossScalar.update(loss / localBatchSize);
                lossGraph[numLossSamples++ % LOSS_GRAPH_SIZE] = curLossScalar.emaVal();
            }
        }
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void handleUniformSamplingImpl(GuidedIntegratorContext &ctx)
    {
        auto &mGuiding = ctx.get_mGuiding();
        auto *uniformWalkQueue = ctx.get_uniformWalkQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto outShellPointQueue = ctx.get_outShellPointQueue();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();
        const auto &integratorSettings = ctx.get_integratorSettings();
        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }
        const auto sceneAABB = ctx.get_problemAABB();

        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto NUM_VMF_COMPONENTS = (DIM == 3) ? common3d::NUM_VMF_COMPONENTS : common2d::NUM_VMF_COMPONENTS;
        static constexpr auto N_DIM_OUTPUT = (DIM == 3) ? common3d::N_DIM_OUTPUT : common2d::N_DIM_OUTPUT;
        static constexpr auto N_DIM_VMF = (DIM == 3) ? common3d::N_DIM_VMF : common2d::N_DIM_VMF;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        float *guidingOutput = inferenceOutputBuffer.data();

        ForAllQueued(uniformWalkQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::UniformWalkWorkItem &id) {
            const float *distData = guidingOutput + id.itemId * N_DIM_OUTPUT;
            float selection_probability = network_to_params(distData[NUM_VMF_COMPONENTS * N_DIM_VMF], ACTIVATION_SELECTION_PROBABILITY);
            
            const typename GuidedIntegratorContext::OutShellPointWorkItem &w = outShellPointQueue->operator[](id.itemId);
            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            VectorType current_point = w.point;
            float alpha = 1.0f;

            VectorType dirVec;
            float pdf;
            if(w.isOnNeumannBoundary)
            {
                dirVec = uniformSampleHemisphere<DIM>(sampler);
                dirVec = frameFromNormal(w.neumannNormal).toWorld(dirVec);
                pdf = uniformSampleHemispherePDF<DIM>();
                alpha = 0.5f;

                current_point += integratorSettings.epsilonShell * w.neumannNormal;

                if(sceneAABB->contains(w.point) && mGuiding.isEnableGuiding(w.depth))// && uniformSamplingFraction < 1.0f) // TODO: duplicated call of aabb query.
                {
                    VMM<DIM, NUM_VMF_COMPONENTS> distribution(distData);
                    auto dirVecReflected = reflect(dirVec, w.neumannNormal);
                    float guidedPdf = distribution.pdf(dirVec) + distribution.pdf(dirVecReflected);
                    pdf = selection_probability * guidedPdf + (1.0f - selection_probability) * pdf;
                }
            }
            else
            {
                dirVec = uniformSampleSphere<DIM>(sampler);
                pdf = uniformSampleSpherePDF<DIM>();

                if(sceneAABB->contains(w.point) && mGuiding.isEnableGuiding(w.depth))// && uniformSamplingFraction < 1.0f) // TODO: duplicated call of aabb query.
                {
                    VMM<DIM, NUM_VMF_COMPONENTS> distribution(distData);
                    float guidedPdf = distribution.pdf(dirVec);
                    pdf = selection_probability * guidedPdf + (1.0f - selection_probability) * pdf;
                }
            }

            VectorType next_point = w.point + w.R_B * dirVec;
            VectorType normal;
            bool intersected = false;
            float dist = INFINITY;
            unsigned int obj_idx = 0xFFFFFFFF;
            if (isNeumannEnabled)
            {
                using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
                query_point_type query_origin = convert<query_point_type>(current_point);
                query_point_type query_dir = convert<query_point_type>(dirVec);
                auto neumann_intersect_result = lbvh::query_device(problem_neumann_bvh_device, lbvh::ray_intersect(lbvh::ray<float, DIM>(query_origin, query_dir), w.R_B), lbvh::scene<DIM>::intersect_test());
                intersected = thrust::get<0>(neumann_intersect_result);
                dist = thrust::get<1>(neumann_intersect_result);
                obj_idx = thrust::get<3>(neumann_intersect_result);
                if (intersected)
                {
                    auto normal_raw = problem_neumann_bvh_device.objects[obj_idx].normal();
                    normal = convert<VectorType>(normal_raw);
                    // Get shading normal.
                    if (normal.dot(dirVec) > 0)
                    {
                        normal = -normal;
                    }
                    next_point = current_point + dist * dirVec;
                }
            }

            typename GuidedIntegratorContext::EvaluationPointWorkItem next_item;
            next_item.point = next_point;
            next_item.depth = w.depth + 1;
            next_item.pixelId = w.pixelId;
            next_item.thp = w.thp / pdf / alpha / sphereMeasurement<DIM>();
            next_item.isOnNeumannBoundary = intersected;
            next_item.neumannNormal = normal;
            ctx.nextEvaluationPointQueue(w.depth)->push(next_item);
            
            if (mGuiding.isEnableTraining(w.depth) && mGuiding.isTrainingPixel(w.pixelId))
            {
                guidedPixelStateBuffer->incrementDepth(w.pixelId, w.point, dirVec, pdf, w.thp, w.isOnNeumannBoundary, w.neumannNormal, 0.0f);
            }
        });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void handleGuidedSamplingImpl(GuidedIntegratorContext &ctx)
    {
        auto &mGuiding = ctx.get_mGuiding();
        auto *guidedInferenceQueue = ctx.get_guidedInferenceQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto outShellPointQueue = ctx.get_outShellPointQueue();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();
        const auto &integratorSettings = ctx.get_integratorSettings();
        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto NUM_VMF_COMPONENTS = (DIM == 3) ? common3d::NUM_VMF_COMPONENTS : common2d::NUM_VMF_COMPONENTS;
        static constexpr auto N_DIM_OUTPUT = (DIM == 3) ? common3d::N_DIM_OUTPUT : common2d::N_DIM_OUTPUT;
        static constexpr auto N_DIM_VMF = (DIM == 3) ? common3d::N_DIM_VMF : common2d::N_DIM_VMF;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        float *output = inferenceOutputBuffer.data();
        ForAllQueued(guidedInferenceQueue, MAX_INFERENCE_NUM, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::GuidedInferenceWorkItem &id) {
            const float* dist_data = output + id.itemId * N_DIM_OUTPUT;
            float selection_probability = network_to_params(dist_data[NUM_VMF_COMPONENTS * N_DIM_VMF], ACTIVATION_SELECTION_PROBABILITY);

            const typename GuidedIntegratorContext::OutShellPointWorkItem& w = outShellPointQueue->operator[](id.itemId);
            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            VectorType current_point = w.point;
            float alpha = 1.0f;
            
            VMM<DIM, NUM_VMF_COMPONENTS> distribution(dist_data);
            VectorType dirVec = distribution.sample(sampler);
            // if (dirVec.hasNaN()) {
            //     printf("Found NaN in sampled guiding distribution.\n");
            //     return;
            // } 
            float guidedPdf	 = distribution.pdf(dirVec);
            float uniformPdf = uniformSampleSpherePDF<DIM>();

            if (w.isOnNeumannBoundary)
            {
                uniformPdf = uniformSampleHemispherePDF<DIM>();
                alpha = 0.5f;
                auto dirVecReflect = reflect(dirVec, w.neumannNormal);
                if (w.neumannNormal.dot(dirVec) <= 0)
                {
                    dirVec = dirVecReflect;
                }
                guidedPdf += distribution.pdf(dirVecReflect);

                current_point += integratorSettings.epsilonShell * w.neumannNormal;
            }

            float pdf = selection_probability * guidedPdf + (1.0f - selection_probability) * uniformPdf;

            VectorType next_point = w.point + w.R_B * dirVec;
            VectorType normal;
            bool intersected = false;
            float dist = INFINITY;
            unsigned int obj_idx = 0xFFFFFFFF;
            if (isNeumannEnabled)
            {
                using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
                query_point_type query_origin = convert<query_point_type>(current_point);
                query_point_type query_dir = convert<query_point_type>(dirVec);
                auto neumann_intersect_result = lbvh::query_device(problem_neumann_bvh_device, lbvh::ray_intersect(lbvh::ray<float, DIM>(query_origin, query_dir), w.R_B), lbvh::scene<DIM>::intersect_test());
                intersected = thrust::get<0>(neumann_intersect_result);
                dist = thrust::get<1>(neumann_intersect_result);
                obj_idx = thrust::get<3>(neumann_intersect_result);
                if (intersected)
                {
                    auto normal_raw = problem_neumann_bvh_device.objects[obj_idx].normal();
                    normal = convert<VectorType>(normal_raw);
                    if (normal.dot(dirVec) > 0)
                    {
                        normal = -normal;
                    }
                    next_point = current_point + dist * dirVec;
                }
            }

            typename GuidedIntegratorContext::EvaluationPointWorkItem next_item;
            next_item.point = next_point;
            next_item.depth = w.depth + 1;
            next_item.pixelId = w.pixelId;
            next_item.thp = w.thp / pdf / alpha / sphereMeasurement<DIM>();
            next_item.isOnNeumannBoundary = intersected;
            next_item.neumannNormal = normal;
            ctx.nextEvaluationPointQueue(w.depth)->push(next_item);

            if(mGuiding.isEnableTraining(w.depth) && mGuiding.isTrainingPixel(w.pixelId))
            {
                guidedPixelStateBuffer->incrementDepth(w.pixelId, w.point, dirVec, pdf, w.thp, w.isOnNeumannBoundary, w.neumannNormal, 0.0f);
            }
        });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void oneStepWalkImpl(GuidedIntegratorContext &ctx)
    {
        auto &mGuiding = ctx.get_mGuiding();
        auto *outShellPointQueue = ctx.get_outShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();
        const auto &integratorSettings = ctx.get_integratorSettings();
        typename GuidedIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename GuidedIntegratorContext::OutShellPointWorkItem &w) {
            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            VectorType current_point = w.point;
            float alpha = 1.0f;
            
            VectorType dirVec;
            float pdf;
            if (w.isOnNeumannBoundary)
            {
                dirVec = uniformSampleHemisphere<DIM>(sampler);
                dirVec = frameFromNormal(w.neumannNormal).toWorld(dirVec);
                pdf = uniformSampleHemispherePDF<DIM>();
                alpha = 0.5f;

                current_point += integratorSettings.epsilonShell * w.neumannNormal;
            }
            else
            {
                dirVec = uniformSampleSphere<DIM>(sampler);
                pdf = uniformSampleSpherePDF<DIM>();
            }

            VectorType next_point = w.point + w.R_B * dirVec;
            VectorType normal;
            bool intersected = false;
            float dist = INFINITY;
            unsigned int obj_idx = 0xFFFFFFFF;
            if (isNeumannEnabled)
            {
                using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
                query_point_type query_origin = convert<query_point_type>(current_point);
                query_point_type query_dir = convert<query_point_type>(dirVec);
                auto neumann_intersect_result = lbvh::query_device(problem_neumann_bvh_device, lbvh::ray_intersect(lbvh::ray<float, DIM>(query_origin, query_dir), w.R_B), lbvh::scene<DIM>::intersect_test());
                intersected = thrust::get<0>(neumann_intersect_result);
                dist = thrust::get<1>(neumann_intersect_result);
                obj_idx = thrust::get<3>(neumann_intersect_result);
                if (intersected)
                {
                    auto normal_raw = problem_neumann_bvh_device.objects[obj_idx].normal();
                    normal = convert<VectorType>(normal_raw);
                    // Get shading normal.
                    if (normal.dot(dirVec) > 0)
                    {
                        normal = -normal;
                    }
                    next_point = current_point + dist * dirVec;
                }
            }

            typename GuidedIntegratorContext::EvaluationPointWorkItem next_item;
            next_item.point = next_point;
            next_item.depth = w.depth + 1;
            next_item.pixelId = w.pixelId;
            next_item.thp = w.thp / pdf / alpha / sphereMeasurement<DIM>();
            next_item.isOnNeumannBoundary = intersected;
            next_item.neumannNormal = normal;
            ctx.nextEvaluationPointQueue(w.depth)->push(next_item);

            if(mGuiding.isEnableTraining(w.depth) && mGuiding.isTrainingPixel(w.pixelId))
            {
                guidedPixelStateBuffer->incrementDepth(w.pixelId, w.point, dirVec, pdf, w.thp, w.isOnNeumannBoundary, w.neumannNormal, 0.0f);
            }
        });
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void solveImpl(GuidedIntegratorContext &ctx)
    {
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto inShellPointQueue = ctx.get_inShellPointQueue();
        auto outShellPointQueue = ctx.get_outShellPointQueue();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();
        auto renderedImage = ctx.get_solutionChannel();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        const auto &basePath = ctx.get_basePath();
        const auto &problem = ctx.get_problem();
        auto &mGuiding = ctx.get_mGuiding();
        auto guidedPixelStateBuffer = ctx.get_guidedPixelStateBuffer();
        auto uniformWalkQueue = ctx.get_uniformWalkQueue();
        auto trainBuffer = ctx.get_trainBuffer();
        auto guidedInferenceQueue = ctx.get_guidedInferenceQueue();

        ELAINA_INIT_PROGRESS_BAR("Solving... ");

        auto start = std::chrono::high_resolution_clock::now();

        prepareSolveImpl(ctx);
        for (int sampleId = 0; sampleId < integratorSettings.samplesPerPixel; ++sampleId) // spp
        {
            if (sampleId == mGuiding.trainSppCount)
            {
                mGuiding.trainState.enableTraining = false;
                mGuiding.uniformSamplingFraction = integratorSettings.uniformFractionInGuidingPhase;
                mGuiding.maxGuidedDepth = integratorSettings.maxGuidedDepthInGuidingPhase;
            }
            ParallelFor(
                maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
                    guidedPixelStateBuffer->reset(pixelId);
                });
            GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() { trainBuffer->clear(); });
            GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() { ctx.currentEvaluationPointQueue(0)->reset(); });
            generateEvaluationPointsImpl(ctx);
            for (int depth = 0; depth < integratorSettings.maxWalkingDepth; ++depth)
            {
                GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
                    ctx.nextEvaluationPointQueue(depth)->reset();
                    inShellPointQueue->reset();
                    outShellPointQueue->reset();
                    uniformWalkQueue->reset();
                    guidedInferenceQueue->reset();
                });
                separateEvaluationPointImpl(ctx, depth);
                if (mGuiding.isEnableGuiding(depth))
                {
                    inferenceStepImpl(ctx);
                }
                handleBoundaryImpl(ctx);
                if (problem.isSourceEnabled())
                {
                    sampleSourceImpl(ctx);
                }
                if (problem.isNeumannEnabled())
                {
                    sampleNeumannImpl(ctx);
                }
                if (mGuiding.isEnableGuiding(depth))
                {
                    cudaStreamSynchronize(mGuiding.stream);
                    handleOutShellPointImpl(ctx);
                    // Sample according to uniformSamplingFraction.
                    if (mGuiding.uniformSamplingFraction < 1.0f)
                    {
                        handleGuidedSamplingImpl(ctx);
                    }
                    handleUniformSamplingImpl(ctx);
                }
                else
                {
                    oneStepWalkImpl(ctx);
                }
            }

            if (mGuiding.isEnableTraining())
            {
                trainStepImpl(ctx);
            }

            if (integratorSettings.saveSppMetricsDuration > 0 &&
                sampleId % integratorSettings.saveSppMetricsDuration == 0 &&
                sampleId < integratorSettings.saveSppMetricsUntil)
            {
                renderedImage->reset();
                ParallelFor(
                    maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
                        Color solution = pixelStateBuffer->solution[pixelId] / float(sampleId + 1);
                        renderedImage->put(Color4f(solution, 1.0f), pixelId);
                    });
                CUDA_SYNC_CHECK();
                renderedImage->save(basePath / "frames" / (std::to_string(sampleId) + ".exr"));
                renderedImage->save(basePath / "frames" / (std::to_string(sampleId) + ".png"));
            }

            if (integratorSettings.saveTimeMetricsDuration > 0 &&
                sampleId % integratorSettings.saveTimeMetricsDuration == 0)
            {
                renderedImage->reset();
                ParallelFor(
                    maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
                        Color solution = pixelStateBuffer->solution[pixelId] / float(sampleId + 1);
                        renderedImage->put(Color4f(solution, 1.0f), pixelId);
                    });
                CUDA_SYNC_CHECK();
                auto current = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
                renderedImage->save(basePath / "frames_time" / (std::to_string(elapsed.count()) + ".exr"));
                renderedImage->save(basePath / "frames_time" / (std::to_string(elapsed.count()) + ".png"));
            }

            ELAINA_UPDATE_PROGRESS_BAR(sampleId + 1, integratorSettings.samplesPerPixel);
        }

        renderedImage->reset();
        ParallelFor(
            maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
                Color solution = pixelStateBuffer->solution[pixelId] / float(integratorSettings.samplesPerPixel);
                renderedImage->put(Color4f(solution, 1.0f), pixelId);
            });
        ELAINA_DESTROY_PROGRESS_BAR();
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void resetNetworkImpl(GuidedIntegratorContext &ctx, json &config)
    {
        auto &mGuiding = ctx.get_mGuiding();
        static constexpr auto DIM = ExtractDim<GuidedIntegratorContext>::value;
        static constexpr auto N_DIM_INPUT = (DIM == 3) ? common3d::N_DIM_INPUT : common2d::N_DIM_INPUT;
        static constexpr auto N_DIM_OUTPUT = (DIM == 3) ? common3d::N_DIM_OUTPUT : common2d::N_DIM_OUTPUT;
        static constexpr auto N_DIM_PADDED_OUTPUT = (DIM == 3) ? common3d::N_DIM_PADDED_OUTPUT : common2d::N_DIM_PADDED_OUTPUT;

        mGuiding.config = config;
        json &encoding_config = config["encoding"];
        json &optimizer_config = config["optimizer"];
        json &network_config = config["network"];
        json &loss_config = config["loss"];
        if (!mGuiding.stream)
            cudaStreamCreate(&mGuiding.stream);

        mGuiding.optimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
        mGuiding.encoding.reset(tcnn::create_encoding<precision_t>(N_DIM_INPUT, encoding_config));
        mGuiding.loss.reset(tcnn::create_loss<precision_t>(loss_config));
        mGuiding.network = std::make_shared<GuidingNetwork<precision_t>>(mGuiding.encoding, N_DIM_OUTPUT, network_config);

        mGuiding.trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
            mGuiding.network, mGuiding.optimizer, mGuiding.loss, ELAINA_DEFAULT_RNG_SEED);

        ELAINA_LOG(Info, "Network has a padded output width of %d", mGuiding.network->padded_output_width());
        CHECK_LOG(next_multiple(N_DIM_OUTPUT, 16u) == N_DIM_PADDED_OUTPUT,
                  "Padded network output width seems wrong!");
        CHECK_LOG(mGuiding.network->padded_output_width() == N_DIM_PADDED_OUTPUT,
                  "Padded network output width seems wrong!");

        trainOutputBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT * TRAIN_BATCH_SIZE);
        gradientBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT * TRAIN_BATCH_SIZE);
        lossBuffer = GPUMemory<float>(TRAIN_BATCH_SIZE);
        inferenceInputBuffer = GPUMemory<float>(N_DIM_INPUT * MAX_INFERENCE_NUM);
        inferenceOutputBuffer = GPUMemory<float>(N_DIM_OUTPUT * MAX_INFERENCE_NUM);

        mGuiding.trainer->initialize_params();
        mGuiding.sampler.setSeed(ELAINA_DEFAULT_RNG_SEED);
        mGuiding.sampler.initialize();
        CUDA_SYNC_CHECK();
    }

    template <typename GuidedIntegratorContext>
    ELAINA_HOST void resetTrainingImpl(GuidedIntegratorContext &ctx)
    {
        auto &mGuiding = ctx.get_mGuiding();

        mGuiding.trainer->initialize_params();
        numLossSamples = 0;
    }
}

GuidedIntegrator<2>::GuidedIntegrator(Problem<2> &problem, const IntegratorSettings &settings, const fs::path &basePath_)
    : problem(problem),
      integratorSettings(settings),
      basePath(basePath_)
{
    detail::initializeImpl(*this);
    CUDA_CHECK(cudaMemcpy(evaluation_grid, &problem.getProbe(), sizeof(EvaluationGrid<2>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sceneAABB, &problem.getAABB(), sizeof(AABB2f), cudaMemcpyHostToDevice));

    mGuiding.trainSppCount = settings.trainSppCount;
    mGuiding.uniformSamplingFraction = settings.uniformFractionInTrainingPhase;
    mGuiding.maxGuidedDepth = settings.maxGuidedDepthInTrainingPhase;
}

template <typename... Args>
ELAINA_DEVICE_FUNCTION void GuidedIntegrator<2>::debugPrint(uint pixelId, const char *fmt, Args &&...args)
{
    detail::debugPrintImpl(*this, std::forward<Args>(args)...);
}

ELAINA_HOST void GuidedIntegrator<2>::renderDirichletSDF()
{
    detail::renderDirichletSDFImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<2>::renderSilhouetteSDF()
{
    detail::renderNeumannSDFImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<2>::renderSource()
{
    detail::renderSourceImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<2>::queryNetwork(const Vector2f &p)
{
    detail::queryNetworkImpl(*this, p);
}

ELAINA_HOST uint64_t GuidedIntegrator<2>::solve()
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    detail::solveImpl(*this);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

ELAINA_HOST void GuidedIntegrator<2>::resetNetwork(json config)
{
    detail::resetNetworkImpl(*this, config);
}

ELAINA_HOST void GuidedIntegrator<2>::resetTraining()
{
    detail::resetTrainingImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<2>::exportImage(ExportImageChannel imageType, const string &file_name)
{
    detail::exportImageImpl(*this, imageType, file_name);
}

ELAINA_HOST void GuidedIntegrator<2>::exportEnergy(ExportImageChannel imageType, ToneMapping tone, const string &file_name)
{
    detail::exportEnergyImpl(*this, imageType, tone, file_name);
}

GuidedIntegrator<3>::GuidedIntegrator(Problem<3> &problem, const IntegratorSettings &settings, const fs::path &basePath_)
    : problem(problem),
      integratorSettings(settings),
      basePath(basePath_)
{
    detail::initializeImpl(*this);
    CUDA_CHECK(cudaMemcpy(evaluation_grid, &problem.getProbe(), sizeof(EvaluationGrid<3>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sceneAABB, &problem.getAABB(), sizeof(AABB3f), cudaMemcpyHostToDevice));

    mGuiding.trainSppCount = settings.trainSppCount;
    mGuiding.uniformSamplingFraction = settings.uniformFractionInTrainingPhase;
    mGuiding.maxGuidedDepth = settings.maxGuidedDepthInTrainingPhase;
}

ELAINA_HOST void GuidedIntegrator<3>::initialize()
{
    detail::initializeImpl(*this);
}

template <typename... Args>
ELAINA_DEVICE_FUNCTION void GuidedIntegrator<3>::debugPrint(uint pixelId, const char *fmt, Args &&...args)
{
    detail::debugPrintImpl(*this, std::forward<Args>(args)...);
}

ELAINA_HOST void GuidedIntegrator<3>::renderDirichletSDF()
{
    detail::renderDirichletSDFImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<3>::renderSilhouetteSDF()
{
    detail::renderNeumannSDFImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<3>::renderSource()
{
    detail::renderSourceImpl(*this);
}

ELAINA_HOST void GuidedIntegrator<3>::queryNetwork(const Vector3f &p)
{
    detail::queryNetworkImpl(*this, p);
}

ELAINA_HOST uint64_t GuidedIntegrator<3>::solve()
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    detail::solveImpl(*this);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

ELAINA_HOST void GuidedIntegrator<3>::exportImage(ExportImageChannel imageType, const string &file_name)
{
    detail::exportImageImpl(*this, imageType, file_name);
}

ELAINA_HOST void GuidedIntegrator<3>::exportEnergy(ExportImageChannel imageType, ToneMapping tone, const string &file_name)
{
    detail::exportEnergyImpl(*this, imageType, tone, file_name);
}

ELAINA_HOST void GuidedIntegrator<3>::resetNetwork(json config)
{
    detail::resetNetworkImpl(*this, config);
}

ELAINA_HOST void GuidedIntegrator<3>::resetTraining()
{
    detail::resetTrainingImpl(*this);
}

ELAINA_NAMESPACE_END
