#pragma once

#include <vector>

#include <open3d/geometry/PointCloud.h>
#include <Eigen/Dense>

namespace misc3d {

namespace registration {

/**
 * @brief Abstract class for transformation matrix estimation.
 *  Transformation matrix is represented as a 4 X 4 matrix.
 *
 */
class TransformationSolver {
public:
    virtual ~TransformationSolver() {}

    /**
     * @brief Solve the transformation matrix between two corresponding point clouds.
     * The correspondence relationship must be found before this function using flann
     * matcher other matching mehtods.
     *
     * @param src
     * @param dst
     * @return Eigen::Matrix4d
     */
    virtual Eigen::Matrix4d Solve(const open3d::geometry::PointCloud &src,
                                  const open3d::geometry::PointCloud &dst) const;

    enum class SolverType {
        SVD = 0,
        TEASER = 1,
        RANSAC = 2,
    };

    /**
     * @brief Get the Solver Type
     *
     * @return SolverType
     */
    SolverType GetSolverType() const { return solver_type_; }

protected:
    TransformationSolver(SolverType type) : solver_type_(type) {}

private:
    SolverType solver_type_;
};

/**
 * @brief 3D-3D outlier-free correspondences based least-squares estimation problem
 * using SVD.
 *
 */
class SVDSolver : public TransformationSolver {
public:
    SVDSolver() : TransformationSolver(SolverType::SVD) {}

    Eigen::Matrix4d Solve(const open3d::geometry::PointCloud &src,
                          const open3d::geometry::PointCloud &dst) const;
};

/**
 * @brief 3D-3D correspondences with outlier transformation matrix estimation using
 * teaser robust solver. reference: https://github.com/MIT-SPARK/TEASER-plusplus
 *
 */
class TeaserSolver : public TransformationSolver {
public:
    /**
     * @brief Construct a TeaserSolver object
     *
     * @param noise_bound
     */
    TeaserSolver(double noise_bound = 0.01)
        : TransformationSolver(SolverType::TEASER), noise_bound_(noise_bound) {}

    Eigen::Matrix4d Solve(const open3d::geometry::PointCloud &src,
                          const open3d::geometry::PointCloud &dst) const;

private:
    double noise_bound_;
};

class RANSACSolver : public TransformationSolver {
public:
    /**
     * @brief Construct a TeaserSolver object
     *
     * @param noise_bound
     */
    RANSACSolver(double threshold, int max_iter = 100000,
                 double edge_length_threshold = 0.9)
        : TransformationSolver(SolverType::RANSAC)
        , threshold_(threshold)
        , max_iter_(max_iter)
        , edge_length_threshold_(edge_length_threshold_) {}

    /**
     * @brief This function is not override from base class due to it need original
     * point clouds for calculation.
     *
     * @param src
     * @param dst
     * @param corres
     * @return Eigen::Matrix4d
     */
    Eigen::Matrix4d Solve(
        const open3d::geometry::PointCloud &src,
        const open3d::geometry::PointCloud &dst,
        const std::pair<std::vector<int>, std::vector<int>> &corres) const;

private:
    double threshold_;
    int max_iter_;
    double edge_length_threshold_;
};

}  // namespace registration

}  // namespace misc3d
