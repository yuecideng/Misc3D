#pragma once

#include <iostream>
#include <limits>
#include <vector>

#include <misc3d/logging.h>
#include <misc3d/utils.h>
#include <open3d/geometry/KDTreeFlann.h>
#include <Eigen/Core>

namespace misc3d {

namespace segmentation {

/**
 * @brief Base Proximity Evaluation class.
 *
 */
class BaseProximityEvaluator {
public:
    /**
     * @brief
     *
     * @param i the index of src point
     * @param j the index of dst point
     * @param dist the euclidean distance between src and dst point
     * @return true
     * @return false
     */
    virtual bool operator()(size_t i, size_t j, double dist) const = 0;
};

class DistanceProximityEvaluator : public BaseProximityEvaluator {
public:
    /**
     * @brief Construct a new Distance Proximity Evaluator with a given distance
     * threshold The distance threshold is used to determine whether two points
     * are close enough to be considered as proximal.
     *
     * @param dist_thresh
     */
    DistanceProximityEvaluator(double dist_thresh)
        : max_distance_(dist_thresh) {}

    inline bool operator()(size_t i, size_t j, double dist) const {
        return dist < max_distance_;
    }

private:
    double max_distance_;
};

class NormalsProximityEvaluator : public BaseProximityEvaluator {
public:
    /**
     * @brief Construct a new Normals Proximity Evaluator with computed normals
     * and angle threshold
     *
     * @param normals the normal of each point.
     * @param angle_thresh angle in degree
     */
    NormalsProximityEvaluator(const std::vector<Eigen::Vector3d> &normals,
                              double angle_thresh)
        : normals_(normals), max_angle_(Deg2Rad(angle_thresh)) {}

    inline bool operator()(size_t i, size_t j, double dist) const {
        if (i > normals_.size() || j > normals_.size()) {
            misc3d::LogError("Index exceed size of data!");
            return false;
        }
        const double angle = std::acos(normals_[i].dot(normals_[j]));
        if (max_angle_ >= 0.0) {
            return angle <= max_angle_;
        } else {
            return std::min(angle, M_PI - angle) <= -max_angle_;
        }
    }

private:
    std::vector<Eigen::Vector3d> normals_;
    double max_angle_;
};

class DistanceNormalsProximityEvaluator : public BaseProximityEvaluator {
public:
    /**
     * @brief Construct a new Points Normals Proximity Evaluator
     *
     * @param normals normal data
     * @param dist_thresh  distance in meter
     * @param angle_thresh angle in degree
     */
    DistanceNormalsProximityEvaluator(
        const std::vector<Eigen::Vector3d> &normals, double dist_thresh,
        double angle_thresh)
        : normals_(normals)
        , max_distance_(dist_thresh)
        , max_angle_(Deg2Rad(angle_thresh)) {}

    inline bool operator()(size_t i, size_t j, double dist) const {
        if (i > normals_.size() || j > normals_.size()) {
            misc3d::LogError("Index exceed size of data!");
            return false;
        }
        if (dist >= max_distance_)
            return false;
        const double angle = std::acos(normals_[i].dot(normals_[j]));
        if (max_angle_ >= 0.0) {
            return angle <= max_angle_;
        } else {
            return std::min(angle, M_PI - angle) <= -max_angle_;
        }
    }

private:
    std::vector<Eigen::Vector3d> normals_;
    double max_distance_;
    double max_angle_;
};

class ProximityExtractor {
private:
    /**
     * @brief Find nearest point map of each input points
     *
     * @param pc
     * @param search_radius
     * @param nn_map
     */
    void SearchNeighborhoodSet(
        const open3d::geometry::PointCloud &pc, double search_radius,
        std::vector<std::vector<std::pair<size_t, double>>> &nn_map);

    /**
     * @brief Build nearest point map from nearest indices
     *
     * @param pc
     * @param nn_indices
     * @param nn_map
     */
    void BuildNeighborhoodSet(
        const open3d::geometry::PointCloud &pc,
        const std::vector<std::vector<size_t>> &nn_indices,
        std::vector<std::vector<std::pair<size_t, double>>> &nn_map);

    /**
     * @brief Segment point clouds into cluster given nearest neighbor map
     *
     * @param pc
     * @param nn_map
     * @param evaluator
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> Segment(
        const open3d::geometry::PointCloud &pc,
        const std::vector<std::vector<std::pair<size_t, double>>> &nn_map,
        const BaseProximityEvaluator &evaluator);

public:
    ProximityExtractor()
        : min_cluster_size_(1)
        , max_cluster_size_(std::numeric_limits<size_t>::max())
        , cluster_num_(0)
        , points_num_(0) {}
    ProximityExtractor(size_t min_cluster_size)
        : min_cluster_size_(min_cluster_size)
        , max_cluster_size_(std::numeric_limits<size_t>::max())
        , cluster_num_(0)
        , points_num_(0) {}
    ProximityExtractor(size_t min_cluster_size, size_t max_cluster_size)
        : min_cluster_size_(min_cluster_size)
        , max_cluster_size_(max_cluster_size)
        , cluster_num_(0)
        , points_num_(0) {}

    /**
     * @brief Segment input point clouds given radius for nearest searching
     *
     * @param points
     * @param search_radius
     * @param evaluator
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> Segment(
        const open3d::geometry::PointCloud &pc, const double search_radius,
        const BaseProximityEvaluator &evaluator);

    /**
     * @brief Segment input point clouds given nearest neighbor map of points
     *
     * @param points
     * @param nn_indices
     * @param evaluator
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> Segment(
        const open3d::geometry::PointCloud &pc,
        const std::vector<std::vector<size_t>> &nn_indices,
        const BaseProximityEvaluator &evaluator);

    /**
     * @brief Get cluster id of each point clouds, the point with largest id is
     * noise.
     *
     * @return std::vector<size_t>
     */
    std::vector<size_t> GetClusterIndexMap();

    /**
     * @brief Return the number of cluster
     *
     * @return size_t
     */
    size_t GetClusterNum();

private:
    size_t min_cluster_size_;
    size_t max_cluster_size_;
    size_t cluster_num_;
    size_t points_num_;
    std::vector<size_t> indices_map_;
    std::vector<std::vector<size_t>> clustered_indices_map_;
};

}  // namespace segmentation
}  // namespace misc3d