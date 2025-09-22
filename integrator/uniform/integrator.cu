#include "integrator.h"

#include "util/film.h"
#include "util/sampling.h"
#include "util/transformation.h"
#include "core/evaluation_grid.h"
#include "core/logger.h"
#include "integrator/common.h"
#include "util/green.h"
#include "util/convert.h"
#include "util/math_utils.h"

ELAINA_NAMESPACE_BEGIN

namespace detail
{
    template <typename UniformIntegratorContext>
    ELAINA_HOST void initializeImpl(UniformIntegratorContext &ctx)
    {
        constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;

        Allocator &alloc = *gpContext->alloc;
        ctx.maxQueueSize = ctx.integratorSettings.frameSize[0] * ctx.integratorSettings.frameSize[1];
        CUDA_SYNC_CHECK();
        for (int i = 0; i < 2; ++i)
        {
            if (ctx.evaluationPointQueue[i])
                ctx.evaluationPointQueue[i]->resize(ctx.maxQueueSize, alloc);
            else
                ctx.evaluationPointQueue[i] = alloc.new_object<typename UniformIntegratorContext::EvaluationPointQueue>(ctx.maxQueueSize, alloc);
        }
        if (ctx.inShellPointQueue)
            ctx.inShellPointQueue->resize(ctx.maxQueueSize, alloc);
        else
            ctx.inShellPointQueue = alloc.new_object<typename UniformIntegratorContext::InShellPointQueue>(ctx.maxQueueSize, alloc);
        if (ctx.outShellPointQueue)
            ctx.outShellPointQueue->resize(ctx.maxQueueSize, alloc);
        else
            ctx.outShellPointQueue = alloc.new_object<typename UniformIntegratorContext::OutShellPointQueue>(ctx.maxQueueSize, alloc);

        // Initialize buffers
        if (ctx.pixelStateBuffer)
            ctx.pixelStateBuffer->resize(ctx.maxQueueSize, alloc);
        else
            ctx.pixelStateBuffer = alloc.new_object<typename UniformIntegratorContext::PixelStateBuffer>(ctx.maxQueueSize, alloc);

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
            ctx.evaluation_grid = alloc.new_object<EvaluationGrid<DIM>>();
        CUDA_SYNC_CHECK();
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void prepareSolveImpl(UniformIntegratorContext &ctx)
    {
        const auto &integratorSettings = ctx.get_integratorSettings();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();

        ParallelFor(
            maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) { // reset per-pixel sample state
                Vector2i pixelCoord = {pixelId % integratorSettings.frameSize[0], pixelId / integratorSettings.frameSize[0]};
                pixelStateBuffer->solution[pixelId] = 0;
                pixelStateBuffer->sampler[pixelId].setPixelSample(pixelCoord, 0);
                pixelStateBuffer->sampler[pixelId].advance(256 * pixelId);
            });
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void generateEvaluationPointsImpl(UniformIntegratorContext &ctx)
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

    template <typename UniformIntegratorContext>
    ELAINA_HOST void separateEvaluationPointImpl(UniformIntegratorContext &ctx, uint depth)
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

        typename UniformIntegratorContext::ProblemType::DeviceBVHType problem_dirichlet_bvh_device;
        if (isDirichletEnabled)
        {
            problem_dirichlet_bvh_device = problem.get_problem_dirichlet_bvh_device();
        }
        typename UniformIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;

        ForAllQueued(currentQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename UniformIntegratorContext::EvaluationPointWorkItem &w) {
            constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;
            using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
            query_point_type query_point = convert<query_point_type>(w.point);

            // WoSt paper Alg. 1: Line 1-2
            // d_\text{Dirichlet}
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
                    typename UniformIntegratorContext::InShellPointWorkItem inShellPointWorkItem;
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
            // WoSt paper Alg. 1: Line 5-6
            // d_\text{silhouette}
            float R_N = INFINITY;
            if (isNeumannEnabled)
            {
                R_N = lbvh::query_device(problem_neumann_bvh_device, lbvh::nearest_silhouette(query_point, false), lbvh::scene<DIM>::silhouette_distance_calculator());
            }
            // WoSt paper Alg. 1: Line 7-8
            // r
            float R_B = max(1e-4f, min(R_D, R_N));
            // For numerical stability, the code below is also introduced in Zombie.
            R_B *= 0.99f;

            if (isinf(R_B))
            {
                return;
            }

            typename UniformIntegratorContext::OutShellPointWorkItem outShellPointWorkItem;
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

    template <typename UniformIntegratorContext>
    ELAINA_HOST void handleBoundaryImpl(UniformIntegratorContext &ctx)
    {
        const auto &problem = ctx.get_problem();
        auto *inShellPointQueue = ctx.get_inShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *pixelStateBuffer = ctx.get_pixelStateBuffer();
        auto vertex_color_dirichlet = problem.get_vertex_color_dirichlet_device();
        auto dirichlet_intensity = problem.get_dirichlet_intensity();

        ForAllQueued(inShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename UniformIntegratorContext::InShellPointWorkItem &w) { 
            // WoSt paper Alg. 1: Line 3-4
            constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;
            // g(\bar{x})
            Color interpolated_color = computeSurfaceColor<DIM>(vertex_color_dirichlet, w.indices, w.side, w.uv);
            interpolated_color *= dirichlet_intensity;
            interpolated_color *= w.thp;
            pixelStateBuffer->addColor(w.pixelId, interpolated_color); });
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void sampleSourceImpl(UniformIntegratorContext &ctx)
    {
        auto *outShellPointQueue = ctx.get_outShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto source_vdb_ptr = problem.get_source_vdb_ptr();
        const auto source_intensity = problem.get_source_intensity();
        typename UniformIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename UniformIntegratorContext::OutShellPointWorkItem &w) {
            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;

            if (isinf(w.R_B))
            {
                return;
            }

            // We do not adopt the sample reuse strategy in Alg. 1: Line 25 of WoSt paper, for fairness.
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

            // WoSt paper Alg. 1: Line 23-24
            auto harmonic_green = HarmonicGreenBall<DIM>(w.R_B);
            // t_\text{source}
            auto [r, radiusPdf] = harmonic_green.sample(sampler);
            // WoSt paper Alg. 1: Line 26
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
            } });
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void sampleNeumannImpl(UniformIntegratorContext &ctx)
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
        typename UniformIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();

        constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename UniformIntegratorContext::OutShellPointWorkItem &w) {
            constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;
            using color_pair_type = thrust::pair<Vector3f, Vector3f>;
            using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
            query_point_type query_point = convert<query_point_type>(w.point);

            Sampler sampler = &pixelStateBuffer->sampler[w.pixelId];
            float u[DIM];
            for (int i = 0; i < DIM; ++i)
            {
                u[i] = sampler.get1D();
            }
            // WoSt paper Alg. 1: Line 17-18
            auto sample_object_result = lbvh::sample_object_in_sphere(problem_neumann_bvh_device,
                                                                      lbvh::sphere_intersect(lbvh::sphere<float, DIM>(query_point, w.R_B)),
                                                                      lbvh::scene<DIM>::intersect_sphere(),
                                                                      lbvh::scene<DIM>::measurement_getter(),
                                                                      lbvh::scene<DIM>::green_weight(),
                                                                      u[0]);
            int object_index = sample_object_result.first;
            float pdf = sample_object_result.second;

            // WoSt paper Alg. 1: Line 19-20
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
                    
                    // WoSt paper Alg. 1: Line 21
                    float alpha = 1.0f;
                    if (w.isOnNeumannBoundary)
                    {
                        alpha = 0.5f;
                    }

                    // WoSt paper Alg. 1: Line 22
                    auto harmonic_green = HarmonicGreenBall<DIM>(w.R_B);
                    interpolated_color *= w.thp * harmonic_green.eval(r) / alpha / pdf;
                    pixelStateBuffer->addColor(w.pixelId, -interpolated_color);
                }
            } });
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void oneStepWalkImpl(UniformIntegratorContext &ctx)
    {
        auto *outShellPointQueue = ctx.get_outShellPointQueue();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        auto *pixelStateBuffer = ctx.get_pixelStateBuffer();
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        const auto &integratorSettings = ctx.get_integratorSettings();
        typename UniformIntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }

        constexpr auto DIM = ExtractDim<UniformIntegratorContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;

        ForAllQueued(outShellPointQueue, maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(const typename UniformIntegratorContext::OutShellPointWorkItem &w) {
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
            
            // Determine fallback x_{k+1}.
            // x_{k+1} := x_k + R_B * v, where R_B = min(R_D, R_N)
            VectorType next_point = w.point + w.R_B * dirVec;
            VectorType normal;
            bool intersected = false;
            float dist = INFINITY;
            unsigned int obj_idx = 0xFFFFFFFF;
            // Project x_{k+1} to the Neumann boundary (if exists at direction v).
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

            typename UniformIntegratorContext::EvaluationPointWorkItem next_item;
            next_item.point = next_point;
            next_item.depth = w.depth + 1;
            next_item.pixelId = w.pixelId;
            next_item.thp = w.thp / pdf / alpha / sphereMeasurement<DIM>();
            next_item.isOnNeumannBoundary = intersected;
            next_item.neumannNormal = normal;
            ctx.nextEvaluationPointQueue(w.depth)->push(next_item);
        });
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void solveImpl(UniformIntegratorContext &ctx)
    {
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto inShellPointQueue = ctx.get_inShellPointQueue();
        auto outShellPointQueue = ctx.get_outShellPointQueue();
        auto pixelStateBuffer = ctx.get_pixelStateBuffer();
        auto renderedImage = ctx.get_solutionChannel();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        const auto &basePath = ctx.get_basePath();
        const auto &problem = ctx.get_problem();

        ELAINA_INIT_PROGRESS_BAR("Solving... ");

        auto start = std::chrono::high_resolution_clock::now();

        // Elaina paper Fig. 5

        // Preparation
        prepareSolveImpl(ctx);
        for (int sampleId = 0; sampleId < integratorSettings.samplesPerPixel; ++sampleId)
        {
            GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() { ctx.currentEvaluationPointQueue(0)->reset(); });
            generateEvaluationPointsImpl(ctx);
            for (int depth = 0; depth < integratorSettings.maxWalkingDepth; ++depth)
            {
                // Logic Stage
                GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
                    ctx.nextEvaluationPointQueue(depth)->reset();
                    inShellPointQueue->reset();
                    outShellPointQueue->reset();
                });
                separateEvaluationPointImpl(ctx, depth);
                
                // Evaluation Stage
                handleBoundaryImpl(ctx);
                if (problem.isSourceEnabled())
                {
                    sampleSourceImpl(ctx);
                }
                if (problem.isNeumannEnabled())
                {
                    sampleNeumannImpl(ctx);
                }
                
                // Walk Stage
                oneStepWalkImpl(ctx);
            }

            // Metrics
            if (integratorSettings.saveSppMetricsDuration > 0 &&
                sampleId % integratorSettings.saveSppMetricsDuration == 0 &&
                sampleId < integratorSettings.saveSppMetricsUntil)
            {
                CUDA_SYNC_CHECK();
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
                CUDA_SYNC_CHECK();
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

        CUDA_SYNC_CHECK();
        renderedImage->reset();
        ParallelFor(
            maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
                Color solution = pixelStateBuffer->solution[pixelId] / float(integratorSettings.samplesPerPixel);
                renderedImage->put(Color4f(solution, 1.0f), pixelId);
            });
        CUDA_SYNC_CHECK();
        ELAINA_DESTROY_PROGRESS_BAR();
    }
}

UniformIntegrator<2>::UniformIntegrator(Problem<2> &problem, const IntegratorSettings &settings, const fs::path &basePath_)
    : problem(problem),
      integratorSettings(settings),
      basePath(basePath_)
{
    initialize();
    CUDA_CHECK(cudaMemcpy(evaluation_grid, &problem.getProbe(), sizeof(EvaluationGrid<2>), cudaMemcpyHostToDevice));
}

ELAINA_HOST void UniformIntegrator<2>::initialize()
{
    detail::initializeImpl(*this);
}

template <typename... Args>
ELAINA_DEVICE_FUNCTION void UniformIntegrator<2>::debugPrint(Args &&...args)
{
    detail::debugPrintImpl(*this, std::forward<Args>(args)...);
}

ELAINA_HOST void UniformIntegrator<2>::renderDirichletSDF()
{
    detail::renderDirichletSDFImpl(*this);
}

ELAINA_HOST void UniformIntegrator<2>::renderSilhouetteSDF()
{
    detail::renderNeumannSDFImpl(*this);
}

ELAINA_HOST void UniformIntegrator<2>::renderSource()
{
    detail::renderSourceImpl(*this);
}

ELAINA_HOST void UniformIntegrator<2>::queryNetwork(const Vector2f &p)
{
    ELAINA_NOTIMPLEMENTED;
}

ELAINA_HOST uint64_t UniformIntegrator<2>::solve()
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    detail::solveImpl(*this);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

ELAINA_HOST void UniformIntegrator<2>::exportImage(ExportImageChannel imageType, const string &file_name)
{
    detail::exportImageImpl(*this, imageType, file_name);
}

ELAINA_HOST void UniformIntegrator<2>::exportEnergy(ExportImageChannel imageType, ToneMapping tone, const string &file_name)
{
    detail::exportEnergyImpl(*this, imageType, tone, file_name);
}

UniformIntegrator<3>::UniformIntegrator(Problem<3> &problem, const IntegratorSettings &settings, const fs::path &basePath_)
    : problem(problem),
      integratorSettings(settings),
      basePath(basePath_)
{
    initialize();
    CUDA_CHECK(cudaMemcpy(evaluation_grid, &problem.getProbe(), sizeof(EvaluationGrid<3>), cudaMemcpyHostToDevice));
}

ELAINA_HOST void UniformIntegrator<3>::initialize()
{
    detail::initializeImpl(*this);
}

template <typename... Args>
ELAINA_DEVICE_FUNCTION void UniformIntegrator<3>::debugPrint(Args &&...args)
{
    detail::debugPrintImpl(*this, std::forward<Args>(args)...);
}

ELAINA_HOST void UniformIntegrator<3>::renderDirichletSDF()
{
    detail::renderDirichletSDFImpl(*this);
}

ELAINA_HOST void UniformIntegrator<3>::renderSilhouetteSDF()
{
    detail::renderNeumannSDFImpl(*this);
}

ELAINA_HOST void UniformIntegrator<3>::renderSource()
{
    detail::renderSourceImpl(*this);
}

ELAINA_HOST void UniformIntegrator<3>::queryNetwork(const Vector3f &p)
{
    ELAINA_NOTIMPLEMENTED;
}

ELAINA_HOST uint64_t UniformIntegrator<3>::solve()
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    detail::solveImpl(*this);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

ELAINA_HOST void UniformIntegrator<3>::exportImage(ExportImageChannel imageType, const string &file_name)
{
    detail::exportImageImpl(*this, imageType, file_name);
}

ELAINA_HOST void UniformIntegrator<3>::exportEnergy(ExportImageChannel imageType, ToneMapping tone, const string &file_name)
{
    detail::exportEnergyImpl(*this, imageType, tone, file_name);
}

ELAINA_NAMESPACE_END
