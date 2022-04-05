#pragma once

#include <unordered_map>
#include <vector>

#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/Image.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <Eigen/Dense>

namespace misc3d {

namespace pose_estimation {

/**
 * @brief Ray cast renderer for depth map, instance map, point clouds and
 * normals map generation.
 * Currently only support CPU.
 * TODO: Add CUDA suppot.
 */
class RayCastRenderer {
public:
    /**
     * @brief Construct a new Ray Cast Renderer defined by a camera intrinsic.
     * The rays start from the origin [0, 0, 0] will be created based on the
     * intrinsic inside constructor.
     *
     * @param intrinsic
     */
    explicit RayCastRenderer(
        const open3d::camera::PinholeCameraIntrinsic &intrinsic);

    virtual ~RayCastRenderer() {}

    /**
     * @brief Compute the first intersection of the rays with the scene.
     * The result is stored in a map with the following keys:
     * \b t_hit: the distance of each hit
     * \b geometry_ids: the instance id of the geometries
     * \b primitive_normals: the normals of the geometries
     *
     * @param mesh_list
     * @param pose_list
     * @return true
     * @return false
     */
    bool CastRays(const std::vector<open3d::geometry::TriangleMesh> &mesh_list,
                  const std::vector<Eigen::Matrix4d> &pose_list);

    /**
     * @brief Get Depth Map
     *
     * @return open3d::core::Tensor
     */
    open3d::core::Tensor GetDepthMap() const;

    /**
     * @brief Get Instance Map
     * The id is from 0 to num_instances - 1.
     *
     * @return open3d::core::Tensor
     */
    open3d::core::Tensor GetInstanceMap() const;

    /**
     * @brief Get PointCloud from depth map with valid value.
     *
     * @return open3d::geometry::PointCloud
     */
    open3d::geometry::PointCloud GetPointCloud() const;

    /**
     * @brief Get Instance PointCloud individully.
     *
     * @return std::vector<open3d::geometry::PointCloud>
     */
    std::vector<open3d::geometry::PointCloud> GetInstancePointCloud() const;

private:
    std::unordered_map<std::string, open3d::core::Tensor> ray_cast_results_;
    int width_;
    int height_;
    open3d::core::Tensor rays_;
    size_t num_instance_;
};

}  // namespace pose_estimation

}  // namespace misc3d