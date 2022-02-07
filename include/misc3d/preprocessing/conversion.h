#pragma once

#include <tuple>

#include <misc3d/utils.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/Image.h>
#include <open3d/geometry/RGBDImage.h>
#include <Eigen/Dense>

namespace misc3d {

namespace preprocessing {

/**
 * @brief Convert Open3D depth data which are within roi region to point clouds.
 * NOTES: This function will convert depth inclding nan values to points to make
 * sure the return point cloud has pointmap structure.
 *
 * @param depth
 * @param roi (xmin, ymin, xmax, ymax)
 * @param intrinsic
 * @param depth_scale
 * @param depth_trunc
 * @param stride
 * @return PointCloudPtr
 */
PointCloudPtr DepthROIToPointCloud(
    const open3d::geometry::Image &depth,
    const std::tuple<int, int, int, int> &roi,
    const open3d::camera::PinholeCameraIntrinsic &intrinsic,
    float depth_scale = 1000.0, float depth_trunc = 3.0, int stride = 1);

/**
 * @brief Convert Open3D RGBD data which are within roi region to point clouds.
 * NOTES: This function will convert depth inclding nan values to points to make
 * sure the return point cloud has pointmap structure.
 *
 * @param rgbd
 * @param roi (xmin, ymin, xmax, ymax)
 * @param intrinsic
 * @param depth_scale
 * @param depth_trunc
 * @param stride
 * @return PointCloudPtr
 */
PointCloudPtr RGBDROIToPointCloud(
    const open3d::geometry::RGBDImage &rgbd,
    const std::tuple<int, int, int, int> &roi,
    const open3d::camera::PinholeCameraIntrinsic &intrinsic,
    float depth_scale = 1000.0, float depth_trunc = 3.0, int stride = 1);

}  // namespace preprocessing
}  // namespace misc3d