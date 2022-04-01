#include <numeric>
#include <set>

#include <misc3d/segmentation/proximity_extraction.h>

namespace misc3d {

namespace segmentation {

void ProximityExtractor::SearchNeighborhoodSet(
    const open3d::geometry::PointCloud &pc, double search_radius,
    std::vector<std::vector<std::pair<size_t, double>>> &nn_map) {
    open3d::geometry::KDTreeFlann kdtree(pc);
    const size_t size = pc.points_.size();
    nn_map.resize(size);

#pragma omp parallel for shared(nn_map)
    for (int i = 0; i < size; i++) {
        std::vector<int> ret_indices;
        std::vector<double> out_dists_sqr;
        const int n = kdtree.SearchRadius(pc.points_[i], search_radius,
                                          ret_indices, out_dists_sqr);

        nn_map[i].resize(n);
        for (size_t j = 0; j < n; j++) {
            nn_map[i][j] =
                std::make_pair(ret_indices[j], sqrt(out_dists_sqr[j]));
        }
    }
}

void ProximityExtractor::BuildNeighborhoodSet(
    const open3d::geometry::PointCloud &pc,
    const std::vector<std::vector<size_t>> &nn_indices,
    std::vector<std::vector<std::pair<size_t, double>>> &nn_map) {
    const size_t size = pc.points_.size();
    nn_map.resize(size);

#pragma omp parallel for shared(nn_map)
    for (int i = 0; i < size; i++) {
        const size_t sub_size = nn_indices[i].size();
        nn_map[i].resize(sub_size);
        for (size_t j = 0; j < sub_size; j++) {
            const double dis =
                (pc.points_[i] - pc.points_[nn_indices[i][j]]).norm();
            nn_map[i][j] = std::make_pair(nn_indices[i][j], dis);
        }
    }
}

std::vector<std::vector<size_t>> ProximityExtractor::Segment(
    const open3d::geometry::PointCloud &pc, const double search_radius,
    const BaseProximityEvaluator &evaluator) {
    std::vector<std::vector<std::pair<size_t, double>>> nn_map;
    SearchNeighborhoodSet(pc, search_radius, nn_map);
    return Segment(pc, nn_map, evaluator);
}

std::vector<std::vector<size_t>> ProximityExtractor::Segment(
    const open3d::geometry::PointCloud &pc,
    const std::vector<std::vector<size_t>> &nn_indices,
    const BaseProximityEvaluator &evaluator) {
    if (pc.points_.size() != nn_indices.size()) {
        misc3d::LogError("The number of input data size are not equal!");
        std::vector<std::vector<size_t>> result;
        return result;
    }

    std::vector<std::vector<std::pair<size_t, double>>> nn_map;
    BuildNeighborhoodSet(pc, nn_indices, nn_map);
    return Segment(pc, nn_map, evaluator);
}

std::vector<std::vector<size_t>> ProximityExtractor::Segment(
    const open3d::geometry::PointCloud &pc,
    const std::vector<std::vector<std::pair<size_t, double>>> &nn_map,
    const BaseProximityEvaluator &evaluator) {
    const size_t unassigned = std::numeric_limits<size_t>::max();
    const size_t num = pc.points_.size();
    points_num_ = num;

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
    for (int i = 0; i < num; i++) {
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
                if (curr_lbl == i ||
                    evaluator(curr_seed, nn[j].first, nn[j].second)) {
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
        for (auto it = seeds_to_merge_with[i].begin();
             it != seeds_to_merge_with[i].end(); ++it) {
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
        if (segment_to_point_map_tmp[ind].size() <= max_cluster_size_) {
            segment_to_point_map_tmp[ind].emplace_back(i);
        }
    }

    std::vector<std::vector<size_t>> clustered_indices_map;
    for (size_t i = 0; i < segment_to_point_map_tmp.size(); i++) {
        if (segment_to_point_map_tmp[i].size() >= min_cluster_size_ &&
            segment_to_point_map_tmp[i].size() <= max_cluster_size_) {
            clustered_indices_map.emplace_back(
                std::move(segment_to_point_map_tmp[i]));
        }
    }

    std::sort(clustered_indices_map.begin(), clustered_indices_map.end(),
              [](std::vector<size_t> &indices1, std::vector<size_t> &indices2) {
                  return indices1.size() > indices2.size();
              });

    const size_t cluster_num = clustered_indices_map.size();
    clustered_indices_map_.assign(clustered_indices_map.begin(),
                                  clustered_indices_map.end());
    cluster_num_ = cluster_num;

    return clustered_indices_map;
}

size_t ProximityExtractor::GetClusterNum() {
    return cluster_num_;
}

std::vector<size_t> ProximityExtractor::GetClusterIndexMap() {
    // noise points will be assign largest label
    indices_map_.resize(points_num_, cluster_num_);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < clustered_indices_map_.size(); i++) {
        for (size_t j = 0; j < clustered_indices_map_[i].size(); j++) {
            indices_map_[clustered_indices_map_[i][j]] = i;
        }
    }
    return indices_map_;
}

}  // namespace segmentation
}  // namespace misc3d
