#include <iostream>
#include <vector>

#include <misc3d/common/knn.h>
#include <misc3d/utils.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>

namespace misc3d {

namespace common {

KNearestSearch::KNearestSearch() : n_trees_(4) {}

KNearestSearch::KNearestSearch(int n_trees) : n_trees_(n_trees) {}

KNearestSearch::KNearestSearch(const Eigen::MatrixXd &data, int n_trees)
    : n_trees_(n_trees) {
    SetMatrixData(data);
}

KNearestSearch::KNearestSearch(const open3d::geometry::Geometry &geometry,
                               int n_trees)
    : n_trees_(n_trees) {
    SetGeometry(geometry);
}

KNearestSearch::KNearestSearch(
    const open3d::pipelines::registration::Feature &feature, int n_trees)
    : n_trees_(n_trees) {
    SetFeature(feature);
}

KNearestSearch::~KNearestSearch() {}

bool KNearestSearch::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();
    if (dimension_ == 0 || dataset_size_ == 0) {
        return false;
    }

    annoy_index_ = std::make_unique<AnnoyIndex_t>(dimension_);
    for (size_t i = 0; i < dataset_size_; i++) {
        annoy_index_->add_item(i, data.col(i).data());
    }
    annoy_index_->build(n_trees_, ANNOY_BUILD_THREADS);
    return true;
}

bool KNearestSearch::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
        data.data(), data.rows(), data.cols()));
}

bool KNearestSearch::SetGeometry(const open3d::geometry::Geometry &geometry) {
    switch (geometry.GetGeometryType()) {
    case open3d::geometry::Geometry::GeometryType::PointCloud:
        return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            (const double *)((const open3d::geometry::PointCloud &)geometry)
                .points_.data(),
            3,
            ((const open3d::geometry::PointCloud &)geometry).points_.size()));
    case open3d::geometry::Geometry::GeometryType::TriangleMesh:
        return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            (const double *)((const open3d::geometry::TriangleMesh &)geometry)
                .vertices_.data(),
            3,
            ((const open3d::geometry::TriangleMesh &)geometry)
                .vertices_.size()));
    default:
        return false;
    }
}

bool KNearestSearch::SetFeature(
    const open3d::pipelines::registration::Feature &feature) {
    return SetMatrixData(feature.data_);
}

int KNearestSearch::Search(const Eigen::VectorXd &query,
                           const open3d::geometry::KDTreeSearchParam &param,
                           std::vector<size_t> &indices,
                           std::vector<double> &distance) const {
    switch (param.GetSearchType()) {
    case open3d::geometry::KDTreeSearchParam::SearchType::Knn:
        return SearchKNN(
            query, ((const open3d::geometry::KDTreeSearchParamKNN &)param).knn_,
            indices, distance);
    case open3d::geometry::KDTreeSearchParam::SearchType::Hybrid:
        return SearchHybrid(
            query,
            ((const open3d::geometry::KDTreeSearchParamHybrid &)param).radius_,
            ((const open3d::geometry::KDTreeSearchParamHybrid &)param).max_nn_,
            indices, distance);
    default:
        // Not support radius search for Annoy algorithm
        return -1;
    }
    return -1;
}

int KNearestSearch::SearchKNN(const Eigen::VectorXd &query, int knn,
                              std::vector<size_t> &indices,
                              std::vector<double> &distance) const {
    if (dataset_size_ <= 0 || size_t(query.rows()) != dimension_ || knn < 0) {
        return -1;
    }

    annoy_index_->get_nns_by_vector(query.data(), knn, -1, &indices, &distance);

    return indices.size();
}

int KNearestSearch::SearchHybrid(const Eigen::VectorXd &query, double radius,
                                 int knn, std::vector<size_t> &indices,
                                 std::vector<double> &distance) const {
    if (dataset_size_ <= 0 || size_t(query.rows()) != dimension_ || knn < 0) {
        return -1;
    }

    annoy_index_->get_nns_by_vector(query.data(), knn, -1, &indices, &distance);
    size_t i;
    for (i = 0; i < indices.size(); i++) {
        if (distance[i] > radius) {
            break;
        }
    }
    const size_t num = i - 1;
    if (num > 0) {
        indices.resize(num);
        distance.resize(num);
        return num;
    } else {
        indices.clear();
        distance.clear();
        return 0;
    }
}

}  // namespace common

}  // namespace misc3d