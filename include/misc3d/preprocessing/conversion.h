#pragma once

#include <tuple>

#include <misc3d/utils.h>

namespace misc3d {

namespace preprocessing {

/**
 * @brief Crop point clouds within ROI, This method requires point cloud size
 * has the same number with height * width
 *
 * @param pc
 * @param roi
 * @param shape
 * @return PointCloudPtr
 */
PointCloudPtr CropROIPointCloud(const open3d::geometry::PointCloud &pc,
                                const std::tuple<int, int, int, int> &roi,
                                const std::tuple<int, int> &shape);

}  // namespace preprocessing
}  // namespace misc3d