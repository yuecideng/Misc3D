#pragma once

#include <memory>
#include <vector>

#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/geometry/PointCloud.h>
#include <Eigen/Core>

namespace misc3d {

namespace features {

/**
 * @brief Detect boundary from point clouds
 *
 * @param pc
 * @param param nearest neighbor search parameter
 * @param angle_threshold angle threshold to decide if a point is a boundary point
 * @return std::vector<size_t>
 */
std::vector<size_t> DetectBoundaryPoints(
    const open3d::geometry::PointCloud& pc,
    const open3d::geometry::KDTreeSearchParam& param, double angle_threshold = 90.0);

}  // namespace features

}  // namespace misc3d
