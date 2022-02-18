#include <numeric>

#include <misc3d/logger.h>
#include <misc3d/registration/correspondence_matching.h>
#include <misc3d/utils.h>
#include <misc3d/common/knn.h>
#include <open3d/geometry/KDTreeFlann.h>

namespace misc3d {

namespace registration {

/**
 * @brief Search nearest data of src query from dst data using  KdTree
 *
 * @param src
 * @param dst
 * @return std::vector<int>
 */
std::vector<size_t> NearestSearch(
    const open3d::pipelines::registration::Feature& src,
    const open3d::pipelines::registration::Feature& dst) {
    // init kdtree from dst
    common::KNearestSearch kdtree(dst);

    const int num = src.Num();
    std::vector<size_t> nn_inds(num);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num; i++) {
        std::vector<size_t> ret_indices;
        std::vector<double> out_dists_sqr;
        const Eigen::VectorXd& temp_pt = src.data_.col(i);
        const int knn =
            kdtree.SearchKNN(temp_pt, 1, ret_indices, out_dists_sqr);
        nn_inds[i] = ret_indices[0];
    }

    return nn_inds;
}

std::pair<std::vector<size_t>, std::vector<size_t>> FLANNMatcher::Match(
    const open3d::pipelines::registration::Feature& src,
    const open3d::pipelines::registration::Feature& dst) const {
    std::pair<std::vector<size_t>, std::vector<size_t>> res;

    const auto corres01_idx1 = NearestSearch(src, dst);

    std::vector<size_t> corres01_idx0(corres01_idx1.size());
    std::iota(corres01_idx0.begin(), corres01_idx0.end(), 0);

    if (!cross_check_) {
        res.first = corres01_idx0;
        res.second = corres01_idx1;
        return res;
    }

    const auto corres10_idx0 = NearestSearch(dst, src);

    std::vector<size_t> corres_idx0;
    std::vector<size_t> corres_idx1;
    for (int i = 0; i < corres01_idx0.size(); i++) {
        if (corres10_idx0[corres01_idx1[i]] == i) {
            corres_idx0.emplace_back(corres01_idx0[i]);
            corres_idx1.emplace_back(corres01_idx1[i]);
        }
    }

    res.first = corres_idx0;
    res.second = corres_idx1;
    return res;
}

}  // namespace registration

}  // namespace misc3d
