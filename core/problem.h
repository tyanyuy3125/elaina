#pragma once

#include "common.h"
#include "evaluation_grid.h"
#include <snch_lbvh/scene_loader.cuh>
#include <snch_lbvh/scene.cuh>
#include <memory>
#include <optional>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/io/IO.h>

ELAINA_NAMESPACE_BEGIN

namespace detail
{
    template <typename SceneContext>
    void loadConfigImpl(SceneContext &ctx, const json &config);

    template <typename SceneContext>
    void loadModelFileImpl(SceneContext &ctx,
                           const std::optional<string> &dirichlet_path,
                           const std::optional<string> &neumann_path);

    template <typename SceneContext>
    void loadVertexColorFileImpl(SceneContext &ctx,
                                 const std::optional<string> &vertex_color_dirichlet_path,
                                 const std::optional<string> &vertex_color_neumann_path);

    template <typename SceneContext>
    void loadSourceImpl(SceneContext &ctx,
                        const std::optional<string> &source_path);

    template <typename SceneContext>
    void loadMaskImpl(SceneContext &ctx,
                      const std::optional<string> &mask_path);
}

struct ProblemStatistics
{
    std::size_t dirichlet_vertices_size{0};
    std::size_t dirichlet_primitives_size{0};
    std::size_t neumann_vertices_size{0};
    std::size_t neumann_primitives_size{0};
    std::size_t source_grid_count{0};
    std::size_t source_grid_size{0};
    nanovdb::GridType source_grid_type{nanovdb::GridType::Unknown};
};

template <unsigned int DIM>
class Problem;

template <>
class Problem<2>
{
public:
    Problem(const bool verbose = true);

public:
    using SharedPtr = std::shared_ptr<Problem<2>>;
    using ColorPair = thrust::pair<Vector3f, Vector3f>;
    template <typename T>
    using DeviceVec = thrust::device_vector<T>;
    using SceneLoader = lbvh::scene_loader<2>;
    using DeviceScene = lbvh::scene<2>;
    using SceneProbe = EvaluationGrid<2>;

    using DeviceBVHType = lbvh::bvh_device<float, 2U, lbvh::scene<2U>::line_segment>;

public:
    void loadConfig(const json &config);

private:
    SceneProbe::SharedPtr mpProbe;
    AABB2f mAABB;

    std::unique_ptr<SceneLoader> scene_dirichlet_loader{};
    std::unique_ptr<SceneLoader> scene_neumann_loader{};

    DeviceVec<ColorPair> vertex_color_dirichlet{};
    DeviceVec<ColorPair> vertex_color_neumann{};

    std::shared_ptr<DeviceScene> scene_dirichlet_ptr{nullptr};
    std::shared_ptr<DeviceScene> scene_neumann_ptr{nullptr};

    bool enable_dirichlet{false};
    bool enable_neumann{false};

    bool verbose{false};

    ProblemStatistics scene_stat;

    bool enable_source{false};
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> source_vdb_handle;
    nanovdb::Vec3fGrid *source_vdb_ptr{nullptr};
    float source_intensity{1.0f};
    float dirichlet_intensity{1.0f};
    float neumann_intensity{1.0f};

    DeviceVec<bool> mask{};

public:
    const auto &isDirichletEnabled() const
    {
        return enable_dirichlet;
    }
    const auto &isNeumannEnabled() const
    {
        return enable_neumann;
    }
    const auto &isSourceEnabled() const
    {
        return enable_source;
    }
    const auto &getProbe() const
    {
        return *mpProbe;
    }
    const auto &getAABB() const
    {
        return mAABB;
    }
    const auto &get_problem_dirichlet_bvh_device() const
    {
        return scene_dirichlet_ptr->get_bvh_device_ptr();
    }
    const auto &get_problem_neumann_bvh_device() const
    {
        return scene_neumann_ptr->get_bvh_device_ptr();
    }
    const auto get_vertex_color_dirichlet_device() const
    {
        return vertex_color_dirichlet.data().get();
    }
    const auto get_vertex_color_neumann_device() const
    {
        return vertex_color_neumann.data().get();
    }
    const auto &get_problem_dirichlet_ptr() const
    {
        return scene_dirichlet_ptr;
    }
    const auto &get_problem_neumann_ptr() const
    {
        return scene_neumann_ptr;
    }
    const auto &get_problem_stat() const
    {
        return scene_stat;
    }
    const auto get_source_vdb_ptr() const
    {
        return source_vdb_ptr;
    }
    const auto get_source_intensity() const
    {
        return source_intensity;
    }
    const auto get_dirichlet_intensity() const
    {
        return dirichlet_intensity;
    }
    const auto get_neumann_intensity() const
    {
        return neumann_intensity;
    }
    const auto get_mask_device() const
    {
        return mask.data().get();
    }

public:
    template <typename SceneContext>
    friend void detail::loadConfigImpl(SceneContext &ctx, const json &config);

    template <typename SceneContext>
    friend void detail::loadModelFileImpl(SceneContext &ctx,
                                          const std::optional<string> &dirichlet_path,
                                          const std::optional<string> &neumann_path);

    template <typename SceneContext>
    friend void detail::loadVertexColorFileImpl(SceneContext &ctx,
                                                const std::optional<string> &vertex_color_dirichlet_path,
                                                const std::optional<string> &vertex_color_neumann_path);

    template <typename SceneContext>
    friend void detail::loadSourceImpl(SceneContext &ctx,
                                       const std::optional<string> &source_path);

    template <typename SceneContext>
    friend void detail::loadMaskImpl(SceneContext &ctx,
                                     const std::optional<string> &mask_path);
};

template <>
class Problem<3>
{
public:
    Problem(const bool verbose = true);

public:
    using SharedPtr = std::shared_ptr<Problem<3>>;
    using ColorPair = thrust::pair<Vector3f, Vector3f>;
    template <typename T>
    using DeviceVec = thrust::device_vector<T>;
    using SceneLoader = lbvh::scene_loader<3>;
    using DeviceScene = lbvh::scene<3>;
    using SceneProbe = EvaluationGrid<3>;

    using DeviceBVHType = lbvh::bvh_device<float, 3U, lbvh::scene<3U>::triangle>;

public:
    void loadConfig(const json &config);

private:
    SceneProbe::SharedPtr mpProbe;
    AABB3f mAABB;

    std::unique_ptr<SceneLoader> scene_dirichlet_loader{};
    std::unique_ptr<SceneLoader> scene_neumann_loader{};

    DeviceVec<ColorPair> vertex_color_dirichlet{};
    DeviceVec<ColorPair> vertex_color_neumann{};

    std::shared_ptr<DeviceScene> scene_dirichlet_ptr{nullptr};
    std::shared_ptr<DeviceScene> scene_neumann_ptr{nullptr};

    bool enable_dirichlet{false};
    bool enable_neumann{false};

    bool verbose{false};

    ProblemStatistics scene_stat;

    bool enable_source{false};
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> source_vdb_handle;
    nanovdb::Vec3fGrid *source_vdb_ptr{nullptr};
    float source_intensity{1.0f};
    float dirichlet_intensity{1.0f};
    float neumann_intensity{1.0f};

    DeviceVec<bool> mask{};

public:
    const auto &isDirichletEnabled() const
    {
        return enable_dirichlet;
    }
    const auto &isNeumannEnabled() const
    {
        return enable_neumann;
    }
    const auto &isSourceEnabled() const
    {
        return enable_source;
    }
    const auto &getProbe() const
    {
        return *mpProbe;
    }
    const auto &getAABB() const
    {
        return mAABB;
    }
    const auto &get_problem_dirichlet_bvh_device() const
    {
        return scene_dirichlet_ptr->get_bvh_device_ptr();
    }
    const auto &get_problem_neumann_bvh_device() const
    {
        return scene_neumann_ptr->get_bvh_device_ptr();
    }
    const auto get_vertex_color_dirichlet_device() const
    {
        return vertex_color_dirichlet.data().get();
    }
    const auto get_vertex_color_neumann_device() const
    {
        return vertex_color_neumann.data().get();
    }
    const auto &get_problem_dirichlet_ptr() const
    {
        return scene_dirichlet_ptr;
    }
    const auto &get_problem_neumann_ptr() const
    {
        return scene_neumann_ptr;
    }
    const auto &get_problem_stat() const
    {
        return scene_stat;
    }
    const auto get_source_vdb_ptr() const
    {
        return source_vdb_ptr;
    }
    const auto get_source_intensity() const
    {
        return source_intensity;
    }
    const auto get_dirichlet_intensity() const
    {
        return dirichlet_intensity;
    }
    const auto get_neumann_intensity() const
    {
        return neumann_intensity;
    }
    const auto get_mask_device() const
    {
        return mask.data().get();
    }

public:
    template <typename SceneContext>
    friend void detail::loadConfigImpl(SceneContext &ctx, const json &config);

    template <typename SceneContext>
    friend void detail::loadModelFileImpl(SceneContext &ctx,
                                          const std::optional<string> &dirichlet_path,
                                          const std::optional<string> &neumann_path);

    template <typename SceneContext>
    friend void detail::loadVertexColorFileImpl(SceneContext &ctx,
                                                const std::optional<string> &vertex_color_dirichlet_path,
                                                const std::optional<string> &vertex_color_neumann_path);

    template <typename SceneContext>
    friend void detail::loadSourceImpl(SceneContext &ctx,
                                       const std::optional<string> &source_path);

    template <typename SceneContext>
    friend void detail::loadMaskImpl(SceneContext &ctx,
                                     const std::optional<string> &mask_path);
};

ELAINA_NAMESPACE_END