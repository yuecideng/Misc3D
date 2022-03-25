#pragma once

#include <limits>
#include <vector>

#include <misc3d/logging.h>
#include <misc3d/utils.h>
#include <open3d/geometry/PointCloud.h>
#include <Eigen/Core>

namespace misc3d {

namespace segmentation {

/**
 * @brief Segment plane from point cloud using iterative ransac plane fitting
 *
 * @param pcd
 * @param threshold Ransac threshold for plane fitting
 * @param max_iteration maximum number of iteration for ransac loop.
 * @param min_ratio The minimum cluster ratio to end the segmentation.
 * Defaults to 0.05.
 * @return std::vector<std::pair<Eigen::Vector4d, open3d::geometry::PointCloud>>
 */
std::vector<std::pair<Eigen::Vector4d, open3d::geometry::PointCloud>>
SegmentPlaneIterative(const open3d::geometry::PointCloud &pcd,
                      const double threshold, const int max_iteration = 100,
                      const double min_ratio = 0.05);

}  // namespace segmentation
}  // namespace misc3d