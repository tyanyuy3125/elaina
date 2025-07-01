#pragma once
#include "core/common.h"
#include "util/film.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>

ELAINA_NAMESPACE_BEGIN

template <unsigned int DIM>
class UniformIntegrator;

template <unsigned int DIM>
class GuidedIntegrator;

namespace detail
{
    template <typename F>
    void ParallelFor(int nElements, F &&func, CUstream stream = 0)
    {
        DCHECK_GT(nElements, 0);
        GPUParallelFor(nElements, func, stream);
    }

    template <typename T>
    struct ExtractDim;

    template <unsigned int DIM>
    struct ExtractDim<GuidedIntegrator<DIM>>
    {
        static constexpr unsigned int value = DIM;
    };

    template <unsigned int DIM>
    struct ExtractDim<UniformIntegrator<DIM>>
    {
        static constexpr unsigned int value = DIM;
    };

    template <typename IntegratorContext, typename... Args>
    ELAINA_DEVICE_FUNCTION void debugPrintImpl(IntegratorContext &ctx, uint pixelId, const char *fmt, Args &&...args)
    {
        const auto &integratorSettings = ctx.get_integratorSettings();

        if (pixelId == integratorSettings.debugPixel)
        {
            printf(fmt, std::forward<Args>(args)...);
        }
    }

    template <typename IntegratorContext>
    ELAINA_HOST void renderDirichletSDFImpl(IntegratorContext &ctx)
    {
        const auto &problem = ctx.get_problem();
        auto isDirichletEnabled = problem.isDirichletEnabled();
        typename IntegratorContext::ProblemType::DeviceBVHType problem_dirichlet_bvh_device;
        if (isDirichletEnabled)
        {
            problem_dirichlet_bvh_device = problem.get_problem_dirichlet_bvh_device();
        }
        const auto maxQueueSize = ctx.get_maxQueueSize();
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto renderedImage = ctx.get_dirichletSDFChannel();
        const auto evaluation_grid = ctx.get_probe();

        ParallelFor(maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
            constexpr auto DIM = ExtractDim<IntegratorContext>::value;
            Vector2i pixelCoord = { pixelId % integratorSettings.frameSize[0], pixelId / integratorSettings.frameSize[0] };
            auto p = evaluation_grid->getEvaluationPoint(pixelCoord, integratorSettings.frameSize);
            using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
            query_point_type query_point;
            if constexpr (DIM == 2)
            {
                query_point = make_float2(p.x(), p.y());
            }
            else
            {
                query_point = make_float3(p.x(), p.y(), p.z());
            }
            float dist_val = INFINITY;
            if(isDirichletEnabled)
            {
                dist_val = lbvh::query_device(problem_dirichlet_bvh_device, lbvh::nearest(query_point), lbvh::scene<DIM>::distance_calculator()).second;
            }
            renderedImage->put(Color4f(Color(dist_val), 1.0f), pixelId); });
    }

    template <typename IntegratorContext>
    ELAINA_HOST void renderNeumannSDFImpl(IntegratorContext &ctx)
    {
        const auto &problem = ctx.get_problem();
        auto isNeumannEnabled = problem.isNeumannEnabled();
        typename IntegratorContext::ProblemType::DeviceBVHType problem_neumann_bvh_device;
        if (isNeumannEnabled)
        {
            problem_neumann_bvh_device = problem.get_problem_neumann_bvh_device();
        }
        const auto maxQueueSize = ctx.get_maxQueueSize();
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto renderedImage = ctx.get_neumannSDFChannel();
        const auto evaluation_grid = ctx.get_probe();

        ParallelFor(maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
            constexpr auto DIM = ExtractDim<IntegratorContext>::value;
            Vector2i pixelCoord = { pixelId % integratorSettings.frameSize[0], pixelId / integratorSettings.frameSize[0] };
            auto p = evaluation_grid->getEvaluationPoint(pixelCoord, integratorSettings.frameSize);
            using query_point_type = std::conditional_t<(DIM == 2), float2, float3>;
            query_point_type query_point;
            if constexpr (DIM == 2)
            {
                query_point = make_float2(p.x(), p.y());
            }
            else
            {
                query_point = make_float3(p.x(), p.y(), p.z());
            }
            float dist_val = INFINITY;
            if(isNeumannEnabled)
            {
                dist_val = lbvh::query_device(problem_neumann_bvh_device, lbvh::nearest_silhouette(query_point, false), lbvh::scene<DIM>::silhouette_distance_calculator());
            }
            renderedImage->put(Color4f(Color(dist_val), 1.0f), pixelId); });
    }

    template <typename IntegratorContext>
    ELAINA_HOST void renderSourceImpl(IntegratorContext &ctx)
    {
        const auto &problem = ctx.get_problem();
        const auto maxQueueSize = ctx.get_maxQueueSize();
        const auto &integratorSettings = ctx.get_integratorSettings();
        auto renderedImage = ctx.get_sourceChannel();
        const auto evaluation_grid = ctx.get_probe();
        nanovdb::Vec3fGrid *source_vdb_ptr;
        auto isSourceEnabled = problem.isSourceEnabled();
        if (isSourceEnabled)
        {
            source_vdb_ptr = problem.get_source_vdb_ptr();
        }
        const auto source_intensity = problem.get_source_intensity();

        ParallelFor(maxQueueSize, ELAINA_DEVICE_LAMBDA_GLOBAL(int pixelId) {
            constexpr auto DIM = ExtractDim<IntegratorContext>::value;
            Vector2i pixelCoord = { pixelId % integratorSettings.frameSize[0], pixelId / integratorSettings.frameSize[0] };
            auto p = evaluation_grid->getEvaluationPoint(pixelCoord, integratorSettings.frameSize);
            Vector3f gridValueVec = Vector3f::Zero();
            if (isSourceEnabled)
            {
                auto source_vdb_ptr_captured = source_vdb_ptr;
                nanovdb::Vec3f sourcePt(p.x(), p.y(), 0.0f);
                if constexpr (DIM == 3)
                {
                    sourcePt[2] = p.z();
                }
                nanovdb::Vec3f gridIndex = source_vdb_ptr_captured->worldToIndex(sourcePt);
                nanovdb::math::SampleFromVoxels<nanovdb::Vec3fTree, 1, false> sampler(source_vdb_ptr_captured->tree());
                nanovdb::Vec3f gridValue = sampler(gridIndex);
                gridValueVec = Vector3f(gridValue[0], gridValue[1], gridValue[2]);
                gridValueVec *= source_intensity;
            }
            renderedImage->put(Color4f(gridValueVec, 1.0f), pixelId);
            // renderedImage->put(Color4f(source_intensity * gridValueVec, 1.0f), pixelId);
        });
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void exportImageImpl(UniformIntegratorContext &ctx, ExportImageChannel imageType, const string &file_name)
    {
        Film *renderedImage;
        const auto &basePath = ctx.get_basePath();
        switch (imageType)
        {
        case ExportImageChannel::SOLUTION:
        {
            renderedImage = ctx.get_solutionChannel();
            break;
        }
        case ExportImageChannel::SOURCE:
        {
            renderedImage = ctx.get_sourceChannel();
            break;
        }
        case ExportImageChannel::DIRICHLET_SDF:
        {
            renderedImage = ctx.get_dirichletSDFChannel();
            break;
        }
        case ExportImageChannel::NEUMANN_SDF:
        {
            renderedImage = ctx.get_neumannSDFChannel();
            break;
        }
        default:
            ELAINA_SHOULDNT_GO_HERE;
            break;
        }

        std::function<void(Film *, const fs::path &)> saveFunction = &Film::save;
        logInfo("Exporting image to " + (basePath / file_name).string() + ".exr");
        saveFunction(renderedImage, basePath / (file_name + ".exr"));
        logInfo("Exporting image to " + (basePath / file_name).string() + ".png");
        saveFunction(renderedImage, basePath / (file_name + ".png"));
    }

    template <typename UniformIntegratorContext>
    ELAINA_HOST void exportEnergyImpl(UniformIntegratorContext &ctx, ExportImageChannel imageType, ToneMapping tone, const string &file_name)
    {
        Film *renderedImage;
        const auto &basePath = ctx.get_basePath();
        switch (imageType)
        {
        case ExportImageChannel::SOLUTION:
        {
            renderedImage = ctx.get_solutionChannel();
            break;
        }
        case ExportImageChannel::SOURCE:
        {
            renderedImage = ctx.get_sourceChannel();
            break;
        }
        case ExportImageChannel::DIRICHLET_SDF:
        {
            renderedImage = ctx.get_dirichletSDFChannel();
            break;
        }
        case ExportImageChannel::NEUMANN_SDF:
        {
            renderedImage = ctx.get_neumannSDFChannel();
            break;
        }
        default:
            ELAINA_SHOULDNT_GO_HERE;
            break;
        }

        logInfo("Exporting image to " + (basePath / file_name).string() + ".exr");
        renderedImage->saveEnergy(basePath / (file_name + ".exr"), tone);
        logInfo("Exporting image to " + (basePath / file_name).string() + ".png");
        renderedImage->saveEnergy(basePath / (file_name + ".png"), tone);
    }

    template <uint DIM, typename ColorPairArrayType, typename IndexVectorType, typename UVType>
    ELAINA_DEVICE_FUNCTION Color computeSurfaceColor(const ColorPairArrayType &color_pairs, const IndexVectorType &indices, const int side, const UVType &uv)
    {
        Color colors[DIM];

        #pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
            if (side >= 0)
            {
                colors[i] = color_pairs[indices[i]].first;
            }
            else
            {
                colors[i] = color_pairs[indices[i]].second;
            }
        }
        return geometric_interpolate<DIM>(colors, uv);
    }
}

ELAINA_NAMESPACE_END