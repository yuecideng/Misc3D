#pragma once

#include <math.h>
#include <iostream>
#include <limits>
#include <vector>

#include <open3d/geometry/KDTreeFlann.h>
#include <primitives_fitting/Utils.h>
#include <Eigen/Core>

#define NORMAL_SEARCH_RADIUS 30

namespace primitives_fitting {

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
     * @brief Construct a new Distance Proximity Evaluator with a given distance threshold
     * The distance threshold is used to determine whether two points are close enough to be
     * considered as proximal.
     *
     * @param dist_thresh
     */
    DistanceProximityEvaluator(double dist_thresh) : m_max_distance(dist_thresh) {}

    inline bool operator()(size_t i, size_t j, double dist) const { return dist < m_max_distance; }

private:
    double m_max_distance;
};

class NormalsProximityEvaluator : public BaseProximityEvaluator {
public:
    /**
     * @brief Construct a new Normals Proximity Evaluator with computed normals and angle threshold
     *
     * @param normals the normal of each point.
     * @param angle_thresh angle in degree
     */
    NormalsProximityEvaluator(const std::vector<Eigen::Vector3d> &normals, double angle_thresh)
        : m_normals(normals), m_max_angle(utils::Deg2Rad(angle_thresh)) {}

    inline bool operator()(size_t i, size_t j, double dist) const {
        if (i > m_normals.size() || j > m_normals.size()) {
            std::cout << "(ERROR) index exceed size of data!" << std::endl;
            return false;
        }
        const double angle = std::acos(m_normals[i].dot(m_normals[j]));
        if (m_max_angle >= 0.0) {
            return angle <= m_max_angle;
        } else {
            return std::min(angle, M_PI - angle) <= -m_max_angle;
        }
    }

private:
    std::vector<Eigen::Vector3d> m_normals;
    double m_max_angle;
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
    DistanceNormalsProximityEvaluator(const std::vector<Eigen::Vector3d> &normals,
                                      double dist_thresh, double angle_thresh)
        : m_normals(normals)
        , m_max_distance(dist_thresh)
        , m_max_angle(utils::Deg2Rad(angle_thresh)) {}

    inline bool operator()(size_t i, size_t j, double dist) const {
        if (i > m_normals.size() || j > m_normals.size()) {
            std::cout << "(ERROR) index exceed size of data!" << std::endl;
            return false;
        }
        if (dist >= m_max_distance)
            return false;
        const double angle = std::acos(m_normals[i].dot(m_normals[j]));
        if (m_max_angle >= 0.0) {
            return angle <= m_max_angle;
        } else {
            return std::min(angle, M_PI - angle) <= -m_max_angle;
        }
    }

private:
    std::vector<Eigen::Vector3d> m_normals;
    double m_max_distance;
    double m_max_angle;
};

class ProximityExtractor {
private:
    /**
     * @brief Find nearest point map of each input points
     *
     * @param points
     * @param search_radius
     * @param nn_map
     */
    void SearchNeighborhoodSet(const std::vector<Eigen::Vector3d> &points,
                               const double search_radius,
                               std::vector<std::vector<std::pair<size_t, double>>> &nn_map);

    /**
     * @brief Build nearest point map from nearest indices
     *
     * @param points
     * @param nn_indices
     * @param nn_map
     */
    void BuildNeighborhoodSet(const std::vector<Eigen::Vector3d> &points,
                              const std::vector<std::vector<size_t>> &nn_indices,
                              std::vector<std::vector<std::pair<size_t, double>>> &nn_map);
    /**
     * @brief Segment point clouds into cluster given nearest neighbor map
     *
     * @param points
     * @param nn_map
     * @param evaluator
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> Segment(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<std::vector<std::pair<size_t, double>>> &nn_map,
        const BaseProximityEvaluator &evaluator);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    ProximityExtractor()
        : m_min_cluster_size(1), m_max_cluster_size(std::numeric_limits<size_t>::max()) {}
    ProximityExtractor(size_t min_cluster_size)
        : m_min_cluster_size(min_cluster_size)
        , m_max_cluster_size(std::numeric_limits<size_t>::max()) {}
    ProximityExtractor(size_t min_cluster_size, size_t max_cluster_size)
        : m_min_cluster_size(min_cluster_size), m_max_cluster_size(max_cluster_size) {}

    /**
     * @brief Segment input point clouds given radius for nearest searching
     *
     * @param points
     * @param search_radius
     * @param evaluator
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> Segment(const std::vector<Eigen::Vector3d> &points,
                                             const double search_radius,
                                             const BaseProximityEvaluator &evaluator);

    /**
     * @brief Segment input point clouds given nearest neighbor map of points
     *
     * @param points
     * @param nn_indices
     * @param evaluator
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> Segment(const std::vector<Eigen::Vector3d> &points,
                                             const std::vector<std::vector<size_t>> &nn_indices,
                                             const BaseProximityEvaluator &evaluator);

    /**
     * @brief Get cluster id of each point clouds, the point with largest id is noise.
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
    size_t m_min_cluster_size;
    size_t m_max_cluster_size;
    size_t m_cluster_num;
    size_t m_points_num;
    std::vector<size_t> m_indices_map;
    std::vector<std::vector<size_t>> m_clustered_indices_map;
};

}  // namespace segmentation
}  // namespace primitives_fitting