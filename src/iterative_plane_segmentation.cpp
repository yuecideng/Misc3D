#include <misc3d/common/ransac.h>
#include <misc3d/segmentation/iterative_plane_segmentation.h>

namespace misc3d {

namespace segmentation {
std::vector<std::pair<Eigen::Vector4d, open3d::geometry::PointCloud>>
SegmentPlaneIterative(const open3d::geometry::PointCloud &pcd,
                      const double threshold, const int max_iteration,
                      const double min_ratio) {
    std::vector<std::pair<Eigen::Vector4d, open3d::geometry::PointCloud>>
        result;
    const size_t num_points = pcd.points_.size();
    if (num_points < 3) {
        misc3d::LogWarning("Point cloud size has less than 3.");
        return result;
    }

    // Initialize RANSAC plane estimator
    misc3d::common::RANSACPlane estimator;
    estimator.SetMaxIteration(max_iteration);
    misc3d::common::Plane plane;
    std::vector<size_t> inliers;

    auto pcd_copy = std::make_shared<open3d::geometry::PointCloud>(pcd.points_);

    size_t count = 0;
    const size_t target_size = (1 - min_ratio) * num_points;
    while (count < target_size) {
        estimator.SetPointCloud(*pcd_copy);
        estimator.FitModel(threshold, plane, inliers);
        const auto cluster = pcd_copy->SelectByIndex(inliers);
        pcd_copy = pcd_copy->SelectByIndex(inliers, true);
        result.emplace_back(std::make_pair(plane.parameters_, *cluster));
        count += inliers.size();
    }

    return result;
}

}  // namespace segmentation
}  // namespace misc3d