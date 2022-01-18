#include <iostream>
#include <numeric>
#include <set>

#include <open3d/geometry/PointCloud.h>
#include <primitives_fitting/Segmentation.h>

namespace primitives_fitting {

namespace segmentation {

void ProximityExtractor::SearchNeighborhoodSet(
    const std::vector<Eigen::Vector3d> &points, const double search_radius,
    std::vector<std::vector<std::pair<size_t, double>>> &nn_map) {
    const open3d::geometry::PointCloud o3d_pts(points);
    open3d::geometry::KDTreeFlann kdtree(o3d_pts);

    const size_t size = points.size();
    nn_map.resize(size);

#pragma omp parallel for shared(nn_map)
    for (size_t i = 0; i < size; i++) {
        std::vector<int> ret_indices;
        std::vector<double> out_dists_sqr;
        const int n = kdtree.SearchRadius(points[i], search_radius, ret_indices, out_dists_sqr);

        nn_map[i].resize(n);
        for (size_t j = 0; j < n; j++) {
            nn_map[i][j] = std::make_pair(ret_indices[j], sqrt(out_dists_sqr[j]));
        }
    }
}

void ProximityExtractor::BuildNeighborhoodSet(
    const std::vector<Eigen::Vector3d> &points, const std::vector<std::vector<size_t>> &nn_indices,
    std::vector<std::vector<std::pair<size_t, double>>> &nn_map) {
    const size_t size = points.size();
    nn_map.resize(size);

#pragma omp parallel for shared(nn_map)
    for (size_t i = 0; i < size; i++) {
        const size_t sub_size = nn_indices[i].size();
        nn_map[i].resize(sub_size);
        for (size_t j = 0; j < sub_size; j++) {
            const double dis = (points[i] - points[nn_indices[i][j]]).norm();
            nn_map[i][j] = std::make_pair(nn_indices[i][j], dis);
        }
    }
}

std::vector<std::vector<size_t>> ProximityExtractor::Segment(
    const std::vector<Eigen::Vector3d> &points, const double search_radius,
    const BaseProximityEvaluator &evaluator) {
    std::vector<std::vector<std::pair<size_t, double>>> nn_map;
    SearchNeighborhoodSet(points, search_radius, nn_map);
    return Segment(points, nn_map, evaluator);
}

std::vector<std::vector<size_t>> ProximityExtractor::Segment(
    const std::vector<Eigen::Vector3d> &points, const std::vector<std::vector<size_t>> &nn_indices,
    const BaseProximityEvaluator &evaluator) {
    if (points.size() != nn_indices.size()) {
        std::cout << "the number of input data size are not equal!" << std::endl;
        std::vector<std::vector<size_t>> result;
        return result;
    }

    std::vector<std::vector<std::pair<size_t, double>>> nn_map;
    BuildNeighborhoodSet(points, nn_indices, nn_map);
    return Segment(points, nn_map, evaluator);
}

std::vector<std::vector<size_t>> ProximityExtractor::Segment(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<std::vector<std::pair<size_t, double>>> &nn_map,
    const BaseProximityEvaluator &evaluator) {
    const size_t unassigned = std::numeric_limits<size_t>::max();
    const size_t num = points.size();
    m_points_num = num;

    std::vector<size_t> current_label(num, unassigned);

    std::vector<size_t> frontier_set;
    frontier_set.reserve(num);

    // set all input points as seeds.
    std::vector<size_t> seeds_indices(num);
    std::iota(std::begin(seeds_indices), std::end(seeds_indices), 0);
    std::vector<std::set<size_t>> seeds_to_merge_with(num);
    std::vector<bool> seed_active(num, false);

#pragma omp parallel for shared(seeds_indices, current_label, seed_active, \
                                seeds_to_merge_with) private(frontier_set)
    for (size_t i = 0; i < num; i++) {
        if (current_label[seeds_indices[i]] != unassigned)
            continue;

        seeds_to_merge_with[i].insert(i);

        frontier_set.clear();
        frontier_set.emplace_back(seeds_indices[i]);

        current_label[seeds_indices[i]] = i;
        seed_active[i] = true;

        while (!frontier_set.empty()) {
            const size_t curr_seed = frontier_set.back();
            frontier_set.pop_back();

            const std::vector<std::pair<size_t, double>> &nn(nn_map[curr_seed]);
            for (size_t j = 1; j < nn.size(); j++) {
                const size_t curr_lbl = current_label[nn[j].first];
                if (curr_lbl == i || evaluator(curr_seed, nn[j].first, nn[j].second)) {
                    if (curr_lbl == unassigned) {
                        frontier_set.emplace_back(nn[j].first);
                        current_label[nn[j].first] = i;
                    } else {
                        if (curr_lbl != i)
                            seeds_to_merge_with[i].insert(curr_lbl);
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
        for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
            seeds_to_merge_with[*it].insert(i);
        }
    }

    std::vector<size_t> seed_repr(seeds_indices.size(), unassigned);
    size_t seed_cluster_num = 0;
    for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
        if (seed_active[i] == false || seed_repr[i] != unassigned)
            continue;

        frontier_set.clear();
        frontier_set.emplace_back(i);
        seed_repr[i] = seed_cluster_num;

        while (!frontier_set.empty()) {
            const size_t curr_seed = frontier_set.back();
            frontier_set.pop_back();
            for (auto it = seeds_to_merge_with[curr_seed].begin();
                 it != seeds_to_merge_with[curr_seed].end(); ++it) {
                if (seed_active[i] == true && seed_repr[*it] == unassigned) {
                    frontier_set.emplace_back(*it);
                    seed_repr[*it] = seed_cluster_num;
                }
            }
        }

        seed_cluster_num++;
    }

    std::vector<std::vector<size_t>> segment_to_point_map_tmp(seed_cluster_num);
    for (size_t i = 0; i < current_label.size(); i++) {
        if (current_label[i] == unassigned)
            continue;
        const auto ind = seed_repr[current_label[i]];
        if (segment_to_point_map_tmp[ind].size() <= m_max_cluster_size) {
            segment_to_point_map_tmp[ind].emplace_back(i);
        }
    }

    std::vector<std::vector<size_t>> clustered_indices_map;
    for (size_t i = 0; i < segment_to_point_map_tmp.size(); i++) {
        if (segment_to_point_map_tmp[i].size() >= m_min_cluster_size &&
            segment_to_point_map_tmp[i].size() <= m_max_cluster_size) {
            clustered_indices_map.emplace_back(std::move(segment_to_point_map_tmp[i]));
        }
    }

    std::sort(clustered_indices_map.begin(), clustered_indices_map.end(),
              [](std::vector<size_t> &indices1, std::vector<size_t> &indices2) {
                  return indices1.size() > indices2.size();
              });

    const size_t cluster_num = clustered_indices_map.size();
    m_clustered_indices_map.assign(clustered_indices_map.begin(), clustered_indices_map.end());
    m_cluster_num = cluster_num;

    std::cout << "find " << cluster_num << " clusters after segmentation." << std::endl;
    return clustered_indices_map;
}

size_t ProximityExtractor::GetClusterNum() {
    return m_cluster_num;
}

std::vector<size_t> ProximityExtractor::GetClusterIndexMap() {
    // noise points will be assign largest label
    m_indices_map.resize(m_points_num, m_cluster_num);

#pragma omp parallel for shared(m_clustered_indices_map, m_indices_map)
    for (size_t i = 0; i < m_clustered_indices_map.size(); i++) {
        for (size_t j = 0; j < m_clustered_indices_map[i].size(); j++) {
            m_indices_map[m_clustered_indices_map[i][j]] = i;
        }
    }
    return m_indices_map;
}

}  // namespace segmentation
}  // namespace primitives_fitting
