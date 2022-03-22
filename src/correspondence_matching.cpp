#include <numeric>
#include <thread>

#include <misc3d/common/knn.h>
#include <misc3d/registration/correspondence_matching.h>
#include <misc3d/utils.h>
#include <open3d/geometry/KDTreeFlann.h>

namespace misc3d {

namespace registration {

void NearestSearch(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst,
                   std::vector<size_t>& nn_inds,
                   const MatchMethod& match_method, int n_tress) {
    if (match_method == MatchMethod::ANNOY) {
        common::KNearestSearch kdtree(dst, n_tress);
        const int num = src.cols();
        nn_inds.resize(num);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < num; i++) {
            std::vector<size_t> ret_indices;
            std::vector<double> out_dists_sqr;
            const Eigen::VectorXd& temp_pt = src.col(i);
            const int knn =
                kdtree.SearchKNN(temp_pt, 1, ret_indices, out_dists_sqr);
            nn_inds[i] = ret_indices[0];
        }

    } else if (match_method == MatchMethod::FLANN) {
        open3d::geometry::KDTreeFlann kdtree(dst);
        const int num = src.cols();
        nn_inds.resize(num);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < num; i++) {
            std::vector<int> ret_indices;
            std::vector<double> out_dists_sqr;
            const Eigen::VectorXd& temp_pt = src.col(i);
            const int knn =
                kdtree.SearchKNN(temp_pt, 1, ret_indices, out_dists_sqr);
            nn_inds[i] = ret_indices[0];
        }
    }
}

std::pair<std::vector<size_t>, std::vector<size_t>> ANNMatcher::Match(
    const open3d::pipelines::registration::Feature& src,
    const open3d::pipelines::registration::Feature& dst) const {
    return Match(src.data_, dst.data_);
}

std::pair<std::vector<size_t>, std::vector<size_t>> ANNMatcher::Match(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) const {
    std::pair<std::vector<size_t>, std::vector<size_t>> res;

    std::vector<size_t> corres01_idx1, corres10_idx0;

    // start two threads to enable cross nn search
    std::thread src_dst(NearestSearch, src, dst, std::ref(corres01_idx1),
                        match_method_, n_tress_);
    std::thread dst_src(NearestSearch, dst, src, std::ref(corres10_idx0),
                        match_method_, n_tress_);

    src_dst.join();
    dst_src.join();

    std::vector<size_t> corres01_idx0(corres01_idx1.size());
    std::iota(corres01_idx0.begin(), corres01_idx0.end(), 0);

    // cross check
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
