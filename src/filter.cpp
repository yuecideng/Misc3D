#include <memory>
#include <numeric>
#include <vector>

#include <misc3d/logging.h>
#include <misc3d/preprocessing/filter.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace preprocessing {

std::vector<size_t> FarthestPointSampling(
    const open3d::geometry::PointCloud &pc, int num_samples) {
    std::vector<size_t> indices;
    if (num_samples == 0) {
        return indices;
    } else if (num_samples == pc.points_.size()) {
        indices.resize(pc.points_.size());
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    } else if (num_samples > pc.points_.size()) {
        misc3d::LogError(
            "Illegal number of samples: {}, must <= point size: {}",
            num_samples, pc.points_.size());
    } else {
        indices.resize(num_samples);

        const size_t num_points = pc.points_.size();
        std::vector<double> distances(num_points,
                                      std::numeric_limits<double>::infinity());
        size_t farthest_index = 0;
        for (size_t i = 0; i < num_samples; i++) {
            indices[i] = farthest_index;
            const Eigen::Vector3d &selected = pc.points_[farthest_index];
            double max_dist = 0;
            for (size_t j = 0; j < num_points; j++) {
                double dist = (pc.points_[j] - selected).squaredNorm();
                distances[j] = std::min(distances[j], dist);
                if (distances[j] > max_dist) {
                    max_dist = distances[j];
                    farthest_index = j;
                }
            }
        }

        return indices;
    }
}

PointCloudPtr CropROIPointCloud(const open3d::geometry::PointCloud &pc,
                                const std::tuple<int, int, int, int> &roi,
                                const std::tuple<int, int> &shape) {
    const size_t num = pc.points_.size();
    const auto width = std::get<0>(shape);
    const auto height = std::get<1>(shape);
    if (num != width * height) {
        misc3d::LogError("The size of point cloud is wrong.");
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

PointCloudPtr ProjectIntoPlane(const open3d::geometry::PointCloud &pc) {
    auto pcd = std::make_shared<open3d::geometry::PointCloud>(pc);
    pcd->RemoveNonFinitePoints();
    const size_t size = pcd->points_.size();
    if (size < 3) {
        misc3d::LogError("You should provide more than 3 points");
        return pcd;
    }

    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    Y.resize(size);
    X.setOnes(size, 3);
    for (size_t i = 0; i < size; i++) {
        const auto &p = pcd->points_[i];
        X(i, 0) = p(0);
        X(i, 1) = p(1);
        Y(i) = p(2);
    }

    const Eigen::MatrixXd W = (X.transpose() * X).inverse() * X.transpose() * Y;
    const auto Y_ = X * W;
    X.col(2) = Y_;

    // compute normal
    const Eigen::Vector3d p0(X.row(0));
    const Eigen::Vector3d p1(X.row(1));
    const Eigen::Vector3d p2(X.row(2));

    const auto pose = CalcCoordinateTransform<double>(p0, p1, p2);
    Eigen::Vector3d normal = pose.block<3, 1>(0, 2);
    if (normal.dot(p0) > 0) {
        normal *= -1;
    }

    pcd->normals_.clear();
    pcd->normals_.resize(size);
    for (size_t i = 0; i < size; i++) {
        pcd->points_[i] = X.row(i).transpose();
        pcd->normals_[i] = normal;
    }
    return pcd;
}

}  // namespace preprocessing
}  // namespace misc3d