#pragma once

#include <open3d/geometry/Geometry.h>
#include <open3d/geometry/KDTreeSearchParam.h>
#include <open3d/pipelines/registration/Feature.h>
#include <Eigen/Dense>
#include "annoylib.h"
#include "kissrandom.h"

#define ANNOY_BUILD_THREADS 4

namespace misc3d {

namespace common {

/**
 * @brief This is a K nearest neighbor search class based on annoy.
 * It is recommanded to use this class instead of the KDTreeFlann class when the
 * feature dimension is large, eg. for descriptor-based feature matching usage
 * As for point cloud based nearest neighbor search, the KDTreeFlann class is
 * more suitable.
 *
 */
class KNearestSearch {
public:
    KNearestSearch();

    KNearestSearch(int n_trees);

    KNearestSearch(const Eigen::MatrixXd &data, int n_trees = 4);

    KNearestSearch(const open3d::geometry::Geometry &geometry, int n_trees = 4);

    KNearestSearch(const open3d::pipelines::registration::Feature &feature,
                   int n_trees = 4);

    ~KNearestSearch();
    KNearestSearch(const KNearestSearch &) = delete;
    KNearestSearch &operator=(const KNearestSearch &) = delete;

public:
    bool SetMatrixData(const Eigen::MatrixXd &data);

    bool SetGeometry(const open3d::geometry::Geometry &geometry);

    bool SetFeature(const open3d::pipelines::registration::Feature &feature);

    int Search(const Eigen::VectorXd &query,
               const open3d::geometry::KDTreeSearchParam &param,
               std::vector<size_t> &indices,
               std::vector<double> &distance2) const;

    int SearchKNN(const Eigen::VectorXd &query, int knn,
                  std::vector<size_t> &indices,
                  std::vector<double> &distance2) const;

    int SearchHybrid(const Eigen::VectorXd &query, double radius, int knn,
                     std::vector<size_t> &indices,
                     std::vector<double> &distance2) const;

private:
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

private:
    int n_trees_;
    using AnnoyIndex_t =
        Annoy::AnnoyIndex<size_t, double, Annoy::Euclidean, Annoy::Kiss64Random,
                          Annoy::AnnoyIndexMultiThreadedBuildPolicy>;

    std::unique_ptr<AnnoyIndex_t> annoy_index_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace common
}  // namespace misc3d