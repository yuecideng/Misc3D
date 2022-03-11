#pragma once

#include <array>

#include <open3d/geometry/Geometry3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/visualizer/Visualizer.h>
#include <Eigen/Core>

namespace misc3d {

namespace vis {

/**
 * @brief Draw a pose in 3D space
 *
 * @param vis
 * @param pose
 * @param size
 */
void DrawPose(const std::shared_ptr<open3d::visualization::Visualizer> &vis,
              const Eigen::Matrix4d &pose, double size);

/**
 * @brief Draw geometry in Open3D visualizer (PointCloud, TriangleMesh,
 * BoundingBox) with apllied transformation and color
 *
 * @param vis
 * @param geometry
 * @param color
 * @param pose
 * @param size
 */
void DrawGeometry3D(
    const std::shared_ptr<open3d::visualization::Visualizer> &vis,
    const std::shared_ptr<open3d::geometry::Geometry3D> &geometry,
    const std::array<float, 3> &color = {0, 0, 0},
    const Eigen::Matrix4d &pose = Eigen::Matrix4d::Identity(),
    float size = 3.0);

}  // namespace vis

}  // namespace misc3d
