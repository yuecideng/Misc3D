#include <memory>
#include <numeric>

#include <misc3d/preprocessing/conversion.h>

namespace misc3d {

namespace preprocessing {

PointCloudPtr DepthROIToPointCloud(
    const open3d::geometry::Image &depth,
    const std::tuple<int, int, int, int> &roi,
    const open3d::camera::PinholeCameraIntrinsic &intrinsic, float depth_scale,
    float depth_trunc, int stride) {
    auto pointcloud = std::make_shared<open3d::geometry::PointCloud>();
    if (depth.num_of_channels_ != 1) {
        return pointcloud;
    }

    const int tl_x = std::get<0>(roi);
    const int tl_y = std::get<1>(roi);
    const int br_x = std::get<2>(roi);
    const int br_y = std::get<3>(roi);
    const int weight = br_x - tl_x;
    const int height = br_y - tl_y;
    const auto focal_length = intrinsic.GetFocalLength();
    const auto principal_point = intrinsic.GetPrincipalPoint();

    const int size = int(weight / stride) * int(height / stride);
    pointcloud->points_.resize(size);
    int cnt = 0;
    for (int v = 0; v < depth.height_; v += stride) {
        for (int u = 0; u < depth.width_; u += stride) {
            if (u < tl_x || u > br_x || v < tl_y || v > br_y) {
                continue;
            }

            float *p = depth.PointerAt<float>(u, v);
            if (depth.bytes_per_channel_ != 4) {
                *p /= depth_scale;
                if (*p >= depth_trunc) {
                    *p = 0.0f;
                }
            }

            if (*p > 0) {
                double z = (double)(*p);
                double x = (u - principal_point.first) * z / focal_length.first;
                double y =
                    (v - principal_point.second) * z / focal_length.second;
                pointcloud->points_[cnt++] = Eigen::Vector3d(x, y, z);
            } else {
                double z = std::numeric_limits<float>::quiet_NaN();
                double x = std::numeric_limits<float>::quiet_NaN();
                double y = std::numeric_limits<float>::quiet_NaN();
                pointcloud->points_[cnt++] = Eigen::Vector3d(x, y, z);
            }
        }
    }
    return pointcloud;
}

template <typename TC, int NC>
PointCloudPtr CreatePointCloudFromRGBDImageT(
    const open3d::geometry::RGBDImage &image,
    const std::tuple<int, int, int, int> &roi,
    const open3d::camera::PinholeCameraIntrinsic &intrinsic, float depth_scale,
    float depth_trunc, int stride) {
    auto pointcloud = std::make_shared<open3d::geometry::PointCloud>();

    const int tl_x = std::get<0>(roi);
    const int tl_y = std::get<1>(roi);
    const int br_x = std::get<2>(roi);
    const int br_y = std::get<3>(roi);
    const int weight = br_x - tl_x;
    const int height = br_y - tl_y;
    const auto focal_length = intrinsic.GetFocalLength();
    const auto principal_point = intrinsic.GetPrincipalPoint();
    const double scale = (sizeof(TC) == 1) ? 255.0 : 1.0;
    const int size = int(weight / stride) * int(height / stride);

    pointcloud->points_.resize(size);
    pointcloud->colors_.resize(size);
    int cnt = 0;
    for (int v = 0; v < image.depth_.height_; v += stride) {
        float *p = (float *)(image.depth_.data_.data() +
                             v * image.depth_.BytesPerLine());
        TC *pc =
            (TC *)(image.color_.data_.data() + v * image.color_.BytesPerLine());
        for (int u = 0; u < image.depth_.width_;
             u += stride, p += stride, pc += stride * NC) {
            if (u < tl_x || u > br_x || v < tl_y || v > br_y) {
                continue;
            }

            if (image.depth_.bytes_per_channel_ != 4) {
                *p /= depth_scale;
                if (*p >= depth_trunc) {
                    *p = 0.0f;
                }
            }

            if (*p > 0) {
                double z = (double)(*p);
                double x = (u - principal_point.first) * z / focal_length.first;
                double y =
                    (v - principal_point.second) * z / focal_length.second;

                pointcloud->points_[cnt] = Eigen::Vector3d(x, y, z);
                pointcloud->colors_[cnt++] =
                    Eigen::Vector3d(pc[0], pc[(NC - 1) / 2], pc[NC - 1]) /
                    scale;
            } else {
                double z = std::numeric_limits<float>::quiet_NaN();
                double x = std::numeric_limits<float>::quiet_NaN();
                double y = std::numeric_limits<float>::quiet_NaN();
                pointcloud->points_[cnt] = Eigen::Vector3d(x, y, z);
                pointcloud->colors_[cnt++] =
                    Eigen::Vector3d(std::numeric_limits<TC>::quiet_NaN(),
                                    std::numeric_limits<TC>::quiet_NaN(),
                                    std::numeric_limits<TC>::quiet_NaN());
            }
        }
    }
    return pointcloud;
}

PointCloudPtr RGBDROIToPointCloud(
    const open3d::geometry::RGBDImage &rgbd,
    const std::tuple<int, int, int, int> &roi,
    const open3d::camera::PinholeCameraIntrinsic &intrinsic, float depth_scale,
    float depth_trunc, int stride) {
    if (rgbd.depth_.num_of_channels_ == 1 &&
        rgbd.depth_.bytes_per_channel_ == 4) {
        if (rgbd.color_.bytes_per_channel_ == 1 &&
            rgbd.color_.num_of_channels_ == 3) {
            return CreatePointCloudFromRGBDImageT<uint8_t, 3>(
                rgbd, roi, intrinsic, depth_scale, depth_trunc, stride);
        } else if (rgbd.color_.bytes_per_channel_ == 1 &&
                   rgbd.color_.num_of_channels_ == 4) {
            return CreatePointCloudFromRGBDImageT<uint8_t, 4>(
                rgbd, roi, intrinsic, depth_scale, depth_trunc, stride);
        } else if (rgbd.color_.bytes_per_channel_ == 4 &&
                   rgbd.color_.num_of_channels_ == 1) {
            return CreatePointCloudFromRGBDImageT<float, 1>(
                rgbd, roi, intrinsic, depth_scale, depth_trunc, stride);
        }
    }
    return std::make_shared<open3d::geometry::PointCloud>();
}

}  // namespace preprocessing
}  // namespace misc3d