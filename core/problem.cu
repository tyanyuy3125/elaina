#include "problem.h"
#include "logger.h"
#include "texture.h"

#include <filesystem>
#include <magic_enum/magic_enum.hpp>

ELAINA_NAMESPACE_BEGIN

namespace detail
{
    template <typename T>
    struct ExtractDim;

    template <unsigned int DIM>
    struct ExtractDim<Problem<DIM>>
    {
        static constexpr unsigned int value = DIM;
    };

    template <typename SceneContext>
    void loadModelFileImpl(SceneContext &ctx,
                           const std::optional<string> &dirichlet_path,
                           const std::optional<string> &neumann_path)
    {
        static constexpr auto DIM = ExtractDim<SceneContext>::value;
        if (dirichlet_path.has_value())
        {
            ctx.scene_dirichlet_loader = std::make_unique<lbvh::scene_loader<DIM>>(dirichlet_path.value());

            ctx.scene_dirichlet_ptr = std::make_shared<lbvh::scene<DIM>>(
                ctx.scene_dirichlet_loader->get_vertices().begin(),
                ctx.scene_dirichlet_loader->get_vertices().end(),
                ctx.scene_dirichlet_loader->get_indices().begin(),
                ctx.scene_dirichlet_loader->get_indices().end());
            ctx.scene_dirichlet_ptr->compute_silhouettes();
            ctx.scene_dirichlet_ptr->build_bvh();

            ctx.enable_dirichlet = true;

            ctx.scene_stat.dirichlet_vertices_size = ctx.scene_dirichlet_loader->get_vertices().size();
            ctx.scene_stat.dirichlet_primitives_size = ctx.scene_dirichlet_loader->get_indices().size();
        }
        if (neumann_path.has_value())
        {
            ctx.scene_neumann_loader = std::make_unique<lbvh::scene_loader<DIM>>(neumann_path.value());

            ctx.scene_neumann_ptr = std::make_shared<lbvh::scene<DIM>>(
                ctx.scene_neumann_loader->get_vertices().begin(),
                ctx.scene_neumann_loader->get_vertices().end(),
                ctx.scene_neumann_loader->get_indices().begin(),
                ctx.scene_neumann_loader->get_indices().end());
            ctx.scene_neumann_ptr->compute_silhouettes();
            ctx.scene_neumann_ptr->build_bvh();

            ctx.enable_neumann = true;

            ctx.scene_stat.neumann_vertices_size = ctx.scene_neumann_loader->get_vertices().size();
            ctx.scene_stat.neumann_primitives_size = ctx.scene_neumann_loader->get_indices().size();
        }
    }

    thrust::host_vector<thrust::pair<Vector3f, Vector3f>> parseVertexColorFile(const string &vertex_color_path)
    {
        json vertex_color_conf = load_json_file(vertex_color_path);
        auto color_configurations = json_get_or_throw<json>(vertex_color_conf, "ColorConfigurations");

        if (color_configurations.is_array())
        {
            const int color_configurations_size = color_configurations.size();
            thrust::host_vector<thrust::pair<Vector3f, Vector3f>> vertex_color_host(color_configurations_size);
            for (int i = 0; i < color_configurations_size; ++i)
            {
                const json color_config = color_configurations[i];
                const int vertexID = json_get_or_throw<int>(color_config, "vertexID");
                const float leftColorR = json_get_or_throw<float>(color_config, "leftColor/R");
                const float leftColorG = json_get_or_throw<float>(color_config, "leftColor/G");
                const float leftColorB = json_get_or_throw<float>(color_config, "leftColor/B");
                const float rightColorR = json_get_or_throw<float>(color_config, "rightColor/R");
                const float rightColorG = json_get_or_throw<float>(color_config, "rightColor/G");
                const float rightColorB = json_get_or_throw<float>(color_config, "rightColor/B");
                if (vertexID != i + 1)
                {
                    throw std::runtime_error("The configurations should be sorted.");
                }
                vertex_color_host[i] = thrust::make_pair(
                    Vector3f(leftColorR, leftColorG, leftColorB),
                    Vector3f(rightColorR, rightColorG, rightColorB));
            }
            return vertex_color_host;
        }
        else
        {
            throw std::runtime_error("The ColorConfigurations item is not an array.");
        }
    }

    template <typename SceneContext>
    void loadVertexColorFileImpl(SceneContext &ctx,
                                 const std::optional<string> &vertex_color_dirichlet_path,
                                 const std::optional<string> &vertex_color_neumann_path)
    {
        if (ctx.enable_dirichlet)
        {
            if (vertex_color_dirichlet_path.has_value())
            {
                ctx.vertex_color_dirichlet.clear();
                const auto vertex_color_host = parseVertexColorFile(vertex_color_dirichlet_path.value());
                ctx.vertex_color_dirichlet.resize(vertex_color_host.size());
                ctx.vertex_color_dirichlet = vertex_color_host;
            }
            else
            {
                ctx.vertex_color_dirichlet.clear();
                ctx.vertex_color_dirichlet.resize(ctx.scene_stat.dirichlet_vertices_size, thrust::pair<Vector3f, Vector3f>(Vector3f::Zero(), Vector3f::Zero()));
            }
        }
        if (ctx.enable_neumann)
        {
            if (vertex_color_neumann_path.has_value())
            {
                ctx.vertex_color_neumann.clear();
                const auto vertex_color_host = parseVertexColorFile(vertex_color_neumann_path.value());
                ctx.vertex_color_neumann.resize(vertex_color_host.size());
                ctx.vertex_color_neumann = vertex_color_host;
            }
            else
            {
                ctx.vertex_color_neumann.clear();
                ctx.vertex_color_neumann.resize(ctx.scene_stat.neumann_vertices_size, thrust::pair<Vector3f, Vector3f>(Vector3f::Zero(), Vector3f::Zero()));
            }
        }
    }

    template <typename SceneContext>
    void loadSourceImpl(SceneContext &ctx,
                        const std::optional<string> &source_path)
    {
        if (source_path.has_value())
        {
            ctx.source_vdb_handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(source_path.value());
            ctx.source_vdb_handle.deviceUpload(0, false);
            ctx.enable_source = true;
            ctx.scene_stat.source_grid_count = ctx.source_vdb_handle.gridCount();
            ctx.scene_stat.source_grid_size = ctx.source_vdb_handle.gridSize(0);
            ctx.scene_stat.source_grid_type = ctx.source_vdb_handle.gridType(0);
            ctx.source_vdb_ptr = ctx.source_vdb_handle.template deviceGrid<nanovdb::Vec3f>();
        }
    }

    template <typename SceneContext>
    void loadConfigImpl(SceneContext &ctx, const json &config)
    {
        static constexpr auto DIM = ExtractDim<SceneContext>::value;
        using VectorType = std::conditional_t<DIM == 3, Vector3f, Vector2f>;
        using AABBType = std::conditional_t<DIM == 3, AABB3f, AABB2f>;

        json probeConfig = json_get_or_throw<json>(config, "evaluation_grid");
        auto aabbMin = json_get_or_throw<VectorType>(config, "aabb/min");
        auto aabbMax = json_get_or_throw<VectorType>(config, "aabb/max");
        ctx.mAABB = AABBType{aabbMin, aabbMax};
        ctx.mpProbe = std::make_shared<EvaluationGrid<DIM>>(probeConfig);

        json meshConfig = json_get_or_throw<json>(config, "mesh");
        loadModelFileImpl(ctx,
                          json_get_optional<string>(meshConfig, "dirichlet_path"),
                          json_get_optional<string>(meshConfig, "neumann_path"));

        loadVertexColorFileImpl(ctx,
                                json_get_optional<string>(meshConfig, "vertex_color_dirichlet_path"),
                                json_get_optional<string>(meshConfig, "vertex_color_neumann_path"));

        loadSourceImpl(ctx,
                       json_get_optional<string>(config, "source_path"));

        loadMaskImpl(ctx,
                     json_get_optional<string>(config, "mask_path"));

        ctx.source_intensity = json_get_optional<float>(config, "source_intensity", 1.0f);
        ctx.dirichlet_intensity = json_get_optional<float>(config, "dirichlet_intensity", 1.0f);
        ctx.neumann_intensity = json_get_optional<float>(config, "neumann_intensity", 1.0f);

        if (ctx.verbose)
        {
            ELAINA_LOG(Success, "Problem: loadConfig is completed.");
            ELAINA_LOG(Info, DOUBLE_SPLIT_LINE);
            ELAINA_LOG(Info, "Problem statistics");
            ELAINA_LOG(Info, SINGLE_SPLIT_LINE);
            if (ctx.enable_dirichlet)
            {
                ELAINA_LOG(Info, "Is Dirichlet enabled = True");
                ELAINA_LOG(Info, "Dirichlet vertices = %d", ctx.scene_stat.dirichlet_vertices_size);
                ELAINA_LOG(Info, "Dirichlet primitives = %d", ctx.scene_stat.dirichlet_primitives_size);
                ELAINA_LOG(Info, "Dirichlet intensity = %f", ctx.dirichlet_intensity);
            }
            if (ctx.enable_neumann)
            {
                ELAINA_LOG(Info, "Is Neumann enabled = True");
                ELAINA_LOG(Info, "Neumann vertices = %d", ctx.scene_stat.neumann_vertices_size);
                ELAINA_LOG(Info, "Neumann primitives = %d", ctx.scene_stat.neumann_primitives_size);
                ELAINA_LOG(Info, "Neumann intensity = %f", ctx.neumann_intensity);
            }
            if (ctx.enable_source)
            {
                ELAINA_LOG(Info, "Is source enabled = True");
                ELAINA_LOG(Info, "Source grid count (Note that Elaina only supports grid 0) = %d", ctx.scene_stat.source_grid_count);
                ELAINA_LOG(Info, "Source grid size = %d", ctx.scene_stat.source_grid_size);
                ELAINA_LOG(Info, "Source grid type (Note that Elaina only supports Vec3f) = %s", std::string(magic_enum::enum_name(ctx.scene_stat.source_grid_type)).c_str());
                ELAINA_LOG(Info, "Source intensity = %f", ctx.source_intensity);
            }
            ELAINA_LOG(Info, DOUBLE_SPLIT_LINE);
        }
    }

    template <typename SceneContext>
    void loadMaskImpl(SceneContext &ctx,
                      const std::optional<string> &mask_path)
    {
        if (mask_path.has_value())
        {
            Image image;
            image.loadImage(mask_path.value(), true, false);
            uchar* maskData = image.data();
            int channels = image.getChannels();
            Vector2i size = image.getSize();
            thrust::host_vector<bool> mask_host(size[0] * size[1]);
            for (int i = 0; i < size[0] * size[1]; ++i)
            {
                uchar* pixel = maskData + i * channels;
                mask_host[i] = false;
                for (int c = 0; c < 3; ++c)
                {
                    if ((int)pixel[c] != 0)
                    {
                        mask_host[i] = true;
                        break;
                    }
                }
            }
            ctx.mask.resize(size[0] * size[1]);
            ctx.mask = mask_host;
        }
        else
        {
            thrust::host_vector<bool> mask_host(1024 * 1024, true);
            ctx.mask.resize(1024 * 1024);
            ctx.mask = mask_host;
        }
    }
}

Problem<2>::Problem(const bool verbose)
    : verbose(verbose)
{
}

void Problem<2>::loadConfig(const json &config)
{
    detail::loadConfigImpl(*this, config);
}

Problem<3>::Problem(const bool verbose)
    : verbose(verbose)
{
}

void Problem<3>::loadConfig(const json &config)
{
    detail::loadConfigImpl(*this, config);
}

ELAINA_NAMESPACE_END