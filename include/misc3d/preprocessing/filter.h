#pragma once

#include <tuple>

#include <misc3d/utils.h>

namespace misc3d {

namespace preprocessing {

/**
 * @brief Smapling point clouds using farthest point smapling algorithm.
 * 
 * @param pc 
 * @param num_points 
 * @return std::vector<size_t>
 */
std::vector<size_t> FarthestPointSampling(const open3d::geometry::PointCloud &pc,
                                    int num_points);

/**
 * @brief Crop point clouds within ROI, This method requires point cloud size
 * has the same number with height * width
 *
 * @param pc
 * @param roi (xmin, ymin, xmax, ymax)
 * @param shape
 * @return PointCloudPtr
 */
PointCloudPtr CropROIPointCloud(const open3d::geometry::PointCloud &pc,
                                const std::tuple<int, int, int, int> &roi,
                                const std::tuple<int, int> &shape);

/**
 * @brief Project point clouds into a plane where z is linear.
 *
 * @param pc
 * @return PointCloudPtr
 */
PointCloudPtr ProjectIntoPlane(const open3d::geometry::PointCloud &pc);

}  // namespace preprocessing
}  // namespace misc3d