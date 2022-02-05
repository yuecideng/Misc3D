#pragma once

#include <array>

#include <open3d/geometry/PointCloud.h>
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
 * @brief Draw point clouds with colors and transformation
 * If no given color, use color in PointCloud.
 * If color is given, paint the color to PointCloud.
 *
 * @param vis
 * @param pc
 * @param color
 * @param pose
 * @param size
 */
void DrawPointCloud(const std::shared_ptr<open3d::visualization::Visualizer> &vis,
                    const open3d::geometry::PointCloud &pc,
                    const std::array<float, 3> &color = {0, 0, 0},
                    const Eigen::Matrix4d &pose = Eigen::Matrix4d::Identity(),
                    float size = 3.0);

}  // namespace vis

}  // namespace misc3d
