#include <iostream>
#include <numeric>

#include <misc3d/logging.h>
#include <misc3d/registration/transform_estimation.h>
#include <misc3d/utils.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/pipelines/registration/TransformationEstimation.h>
#include <teaser/registration.h>

namespace misc3d {

namespace registration {

bool CheckValid(const open3d::geometry::PointCloud& src,
                const open3d::geometry::PointCloud& dst) {
    if (src.points_.size() < 3 || dst.points_.size() < 3) {
        misc3d::LogError("The number of points pair is less than 3.");
        return false;
    } else if (src.points_.size() != dst.points_.size()) {
        misc3d::LogError("The number of points pair is not equal.");
        return false;
    }
    return true;
}

bool CheckValid(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& dst) {
    if (src.cols() < 3 || dst.cols() < 3) {
        misc3d::LogError("The number of points pair is less than 3.");
        return false;
    } else if (src.cols() != dst.cols()) {
        misc3d::LogError("The number of points pair is not equal.");
        return false;
    }
    return true;
}

Eigen::Matrix4d TransformationSolver::Solve(
    const open3d::geometry::PointCloud& src,
    const open3d::geometry::PointCloud& dst) const {
    return Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d TransformationSolver::Solve(const Eigen::Matrix3Xd& src,
                                            const Eigen::Matrix3Xd& dst) const {
    return Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d SVDSolver::Solve(
    const open3d::geometry::PointCloud& src,
    const open3d::geometry::PointCloud& dst) const {
    Eigen::Matrix3Xd src_mat, dst_mat;
    VectorToEigenMatrix<double>(src.points_, src_mat);
    VectorToEigenMatrix<double>(dst.points_, dst_mat);

    return Solve(src_mat, dst_mat);
}

Eigen::Matrix4d SVDSolver::Solve(const Eigen::Matrix3Xd& src,
                                 const Eigen::Matrix3Xd& dst) const {
    if (!CheckValid(src, dst)) {
        return Eigen::Matrix4d::Identity();
    }

    return Eigen::umeyama(src, dst, false);
}

Eigen::Matrix4d TeaserSolver::Solve(
    const open3d::geometry::PointCloud& src,
    const open3d::geometry::PointCloud& dst) const {
    Eigen::Matrix3Xd src_mat, dst_mat;
    VectorToEigenMatrix<double>(src.points_, src_mat);
    VectorToEigenMatrix<double>(dst.points_, dst_mat);

    return Solve(src_mat, dst_mat);
}

Eigen::Matrix4d TeaserSolver::Solve(const Eigen::Matrix3Xd& src,
                                    const Eigen::Matrix3Xd& dst) const {
    if (!CheckValid(src, dst)) {
        return Eigen::Matrix4d::Identity();
    }

    // larger number of correspondences would cause memory error
    constexpr int max_num = 5000;
    Eigen::Matrix4d res = Eigen::Matrix4d::Identity();

    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = noise_bound_;
    params.cbar2 = 1.0;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 10000;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::
        ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 1e-16;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);
    if (src.cols() > max_num) {
        misc3d::LogWarning(
            "The number of correspondences is too large, random selecting "
            "{} correspondences instead.",
            max_num);
        const size_t num = src.cols();
        RandomSampler<size_t> sampler(num);
        const std::vector<size_t> indices =
            sampler.SampleWithoutDuplicate(max_num);
        solver.solve(SelectByIndexEigenMat(src, indices),
                     SelectByIndexEigenMat(dst, indices));
    } else {
        solver.solve(src, dst);
    }

    auto solution = solver.getSolution();

    res.block<3, 3>(0, 0) = solution.rotation;
    res.block<3, 1>(0, 3) = solution.translation;

    return res;
}

Eigen::Matrix4d RANSACSolver::Solve(
    const open3d::geometry::PointCloud& src,
    const open3d::geometry::PointCloud& dst,
    const std::pair<std::vector<size_t>, std::vector<size_t>>& corres) const {
    Eigen::Matrix4d res = Eigen::Matrix4d::Identity();

    if (src.points_.size() < 3 || dst.points_.size() < 3) {
        misc3d::LogError("The number of points pair is less than 3.");
        return res;
    }

    const size_t num = corres.first.size();
    open3d::pipelines::registration::CorrespondenceSet corres_set(num);
#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        corres_set[i] = Eigen::Vector2i(corres.first[i], corres.second[i]);
    }

    std::vector<std::reference_wrapper<
        const open3d::pipelines::registration::CorrespondenceChecker>>
        checkers;
    auto edge_length_checker =
        open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
            edge_length_threshold_);
    checkers.push_back(edge_length_checker);
    auto distance_checer =
        open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(
            threshold_);
    checkers.push_back(distance_checer);

    auto result = open3d::pipelines::registration::
        RegistrationRANSACBasedOnCorrespondence(
            src, dst, corres_set, threshold_,
            open3d::pipelines::registration::
                TransformationEstimationPointToPoint(false),
            3, checkers,
            open3d::pipelines::registration::RANSACConvergenceCriteria(
                max_iter_));

    return result.transformation_;
}

}  // namespace registration

}  // namespace misc3d
