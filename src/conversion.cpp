#include <memory>
#include <numeric>
#include <vector>

#include <misc3d/logger.h>
#include <misc3d/preprocessing/conversion.h>

namespace misc3d {

namespace preprocessing {

PointCloudPtr CropROIPointCloud(const open3d::geometry::PointCloud &pc,
                                const std::tuple<int, int, int, int> &roi,
                                const std::tuple<int, int> &shape) {
    const size_t num = pc.points_.size();
    const auto width = std::get<0>(shape);
    const auto height = std::get<1>(shape);
    if (num != width * height) {
        MISC3D_ERROR("The size of point cloud is wrong.");
        return std::make_shared<open3d::geometry::PointCloud>();
    }

    const auto tl_x = std::get<0>(roi);
    const auto tl_y = std::get<1>(roi);
    const auto br_x = std::get<2>(roi);
    const auto br_y = std::get<3>(roi);
    const int roi_w = br_x - tl_x;
    const int roi_h = br_y - tl_y;

    const bool has_normals = pc.HasNormals();
    const bool has_colors = pc.HasColors();

    const int size = (roi_w + 1) * (roi_h + 1);
    std::vector<size_t> indices(size);
    
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_.resize(size);
    if (has_normals) {
        pcd->normals_.resize(size);
    }
    if (has_colors) {
        pcd->colors_.resize(size);
    }

#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        const size_t ind = (i / roi_w + tl_y) * width + (i % roi_w) + tl_x;
        // indices[i] = ind;
        pcd->points_[i] = pc.points_[ind];
        if (has_normals) {
            pcd->normals_[i] = pc.normals_[ind];
        }
        if (has_colors) {
            pcd->colors_[i] = pc.colors_[ind];
        }
    }

    return pcd;
}

}  // namespace preprocessing
}  // namespace misc3d