#include <float.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>

#include <misc3d/features/edge_detection.h>
#include <misc3d/logging.h>
#include <misc3d/pose_estimation/data_structure.h>
#include <misc3d/pose_estimation/ppf_estimation.h>
#include <misc3d/utils.h>
#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/KDTreeSearchParam.h>
#include <open3d/pipelines/registration/GeneralizedICP.h>
#include <open3d/pipelines/registration/Registration.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace misc3d {

namespace pose_estimation {

inline bool IsNormalizedVector(const PointXYZ &v) {
    const double norm = v.norm();
    if (norm < 1.0e-6) {
        return false;
    }
    return true;
}

class PPFEstimator::Impl {
public:
    Impl(const PPFEstimatorConfig &config) {
        config_ = config;
        dist_threshold_ = config.rel_dist_thresh_;
        // (calc_normal_relative_ = config.rel_sample_dist_ / 2) is usually a
        // good choise
        calc_normal_relative_ = config.training_param_.calc_normal_relative;
        enable_edge_support_ = (config.voting_param_.method ==
                                PPFEstimatorConfig::VotingMode::EdgePoints)
                                   ? true
                                   : false;
        invert_normal_ = config.training_param_.invert_model_normal;
    }

    ~Impl() {
        // if (hash_table_ != nullptr) {
        //     delete[] hash_table_;
        // }
        // if (hashtable_boundary_ != nullptr) {
        //     delete[] hashtable_boundary_;
        // }
    }

    bool SetConfig(const PPFEstimatorConfig &config);

    bool Train(const PointCloudPtr &pc);

    bool Estimate(const PointCloudPtr &pc, std::vector<Pose6D> &results);

private:
    void PreprocessTrain(const PointCloudPtr &pc,
                         open3d::geometry::PointCloud &pc_sample);

    void PreprocessEstimate(const PointCloudPtr &pc,
                            open3d::geometry::PointCloud &pc_sample);

    void CalcPPF(const PointXYZ &p0, const Normal &n0, const PointXYZ &p1,
                 const Normal &n1, double ppf[4]);

    void QuantizePPF(const double ppf[4], size_t quant_ppf[4]);

    size_t HashPPF(const size_t ppf[4]);

    size_t HashPPF(const double ppf[4]);

    Transformation CalcTNormal2RegionX(const PointXYZ &p, const Normal &n);

    double CalcAlpha(const PointXYZ &pt,
                     const Transformation t_normal_to_region);

    // use_flag: 1: use translation, 2: dsadse rotation, 3: both
    bool MatchPose(const Pose6D &src, const Pose6D &dst, const int use_flag);

    void SpreadPPF(const size_t ppf[4], size_t idx[81], int &n);

    void ClusterPoses(std::vector<Pose6D> &pose_list,
                      std::vector<std::vector<Pose6D>> &clustered_pose);

    void RefineSparsePose(
        const open3d::geometry::PointCloud &model_pcn,
        const open3d::geometry::PointCloud &scene_pcn,
        const std::vector<std::vector<Pose6D>> &clustered_pose,
        std::vector<Pose6D> &pose_list);

    void CalcPoseNeighbor(const std::vector<Pose6D> &pose_list,
                          const std::vector<int> &indices, const int use_flag,
                          std::vector<std::vector<int>> &neibor_idx);

    void GatherPose(const std::vector<Pose6D> &pose_list,
                    const std::vector<int> &indices,
                    const std::vector<std::vector<int>> &neibor_idx,
                    std::vector<PoseCluster> &pose_clusters);

    void PoseAverage(const std::vector<Pose6D> &pose_list,
                     const std::vector<int> &pose_idx, Pose6D &result_pose);

    void FlipNormal(Normal &normal) { normal *= -1; }

    void CalcModelNormalAndSampling(const PointCloudPtr &pc,
                                    const std::shared_ptr<KDTree> &kdtree,
                                    const double normal_r, const double step,
                                    const PointXYZ &view_pt,
                                    open3d::geometry::PointCloud &pc_sample);

    void CalcMeanAndCovarianceMatrix(const std::vector<PointXYZ> &pc,
                                     int *indices, const int indices_num,
                                     double cov_matrix[3][3],
                                     double centoid[3]);

    void GenerateLUT();

    void GenerateModelPCNeighbor(const open3d::geometry::PointCloud &pts,
                                 std::vector<std::vector<int>> &neighbor_table,
                                 const double radius);

    void CalcLocalMaximum(const size_t *accumulator, const size_t rows,
                          const size_t cols, const size_t vote_threshold,
                          const double relative_to_max,
                          std::vector<std::vector<size_t>> &max_in_pair,
                          std::vector<std::vector<int>> &neighbor_table);

    void ExtractEdges(const open3d::geometry::PointCloud &pc, double radius,
                      std::vector<size_t> &edge_ind);

    void CalcHashTable(const open3d::geometry::PointCloud &reference_pts,
                       const open3d::geometry::PointCloud &refered_pts,
                       std::vector<std::vector<PointPair>> &hashtable_ptr,
                       std::vector<Transformation> &tmg_ptr,
                       bool b_same_pointset);

    void DownSamplePCNormal(const open3d::geometry::PointCloud &pc,
                            const std::shared_ptr<KDTree> &kdtree,
                            const double radius, const size_t min_idx,
                            open3d::geometry::PointCloud &pc_normal);

    void VotingAndGetPose(
        const open3d::geometry::PointCloud &reference_pts,
        const open3d::geometry::PointCloud &refered_pts,
        const std::vector<std::vector<PointPair>> &hashtable_ptr,
        const std::vector<Transformation> &tmg_ptr,
        std::vector<Pose6D> &pose_list, size_t reference_num_in_model,
        size_t refered_model_num,
        std::vector<std::vector<int>> &neighbor_table);

public:
    std::vector<Pose6D> pose_list_;
    std::vector<std::vector<Pose6D>> pose_clustered_by_t_;
    double diameter_;

    bool enable_edge_support_;
    bool invert_normal_;
    open3d::geometry::PointCloud dense_model_sample_;
    open3d::geometry::PointCloud dense_scene_sample_;
    open3d::geometry::PointCloud model_sample_;
    open3d::geometry::PointCloud scene_sample_;
    std::vector<size_t> scene_edge_ind_;
    std::vector<size_t> model_edge_ind_;

private:
    double dist_step_;
    double dist_threshold_;
    open3d::geometry::PointCloud model_sample_centered_;
    size_t pc_num_;
    std::vector<std::vector<PointPair>> hash_table_;
    std::vector<std::vector<PointPair>> hashtable_boundary_;
    double r_min_, r_max_;

    size_t reference_num_;
    size_t refered_num_;
    size_t hash_table_size_;
    size_t angle_num_, dist_num_;

    std::vector<Transformation> tmg_ptr_;
    std::vector<Transformation> tmg_ptr_boundary_;

    bool trained_;
    double calc_normal_relative_;

    size_t angle_num_2_, angle_num_3_;

    PointXYZ centroid_;

    std::vector<int> alpha_lut_;
    std::vector<std::array<int, 3>> alpha_shift_lut_;
    std::vector<std::array<int, 3>> dist_shift_lut_;
    std::vector<std::array<int, 3>> alpha_2_shift_lut_;

    std::vector<std::vector<int>> pc_neighbor_;

    double dist_step_dense_;

    PPFEstimatorConfig config_;
};

void PPFEstimator::Impl::PreprocessTrain(
    const PointCloudPtr &pc, open3d::geometry::PointCloud &pc_sample) {
    // compute bounding box of model
    const open3d::geometry::OrientedBoundingBox bbox =
        pc->GetOrientedBoundingBox();
    const Eigen::Vector3d d = bbox.extent_;
    std::vector<double> d_ordered{d(0), d(1), d(2)};

    const double diameter = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    diameter_ = diameter;

    // set view point in above of bbox center
    PointXYZ view_pt = bbox.GetCenter();
    view_pt(2) += VIEW_POINT_Z_EXTEND * diameter;

    // set parameters
    r_max_ = diameter;
    std::sort(d_ordered.begin(), d_ordered.end());
    r_min_ = sqrt(d_ordered[0] * d_ordered[0] + d_ordered[1] * d_ordered[1]);
    dist_step_ = diameter * config_.training_param_.rel_sample_dist;
    dist_step_dense_ = diameter * config_.training_param_.rel_dense_sample_dist;
    dist_threshold_ *= diameter;
    const double normal_r = diameter * calc_normal_relative_;

    misc3d::LogInfo(
        "Model training params: normal search radius: {}, sample "
        "distance: {}, dense sample distance: {}",
        normal_r, dist_step_, dist_step_dense_);

    auto kdtree = std::make_shared<KDTree>(*pc);
    CalcModelNormalAndSampling(pc, kdtree, normal_r, dist_step_, view_pt,
                               pc_sample);

    misc3d::LogInfo("Model sample point number is {} | {} after preprocessing",
                    pc_sample.points_.size(), pc->points_.size());
}

void PPFEstimator::Impl::PreprocessEstimate(
    const PointCloudPtr &pc, open3d::geometry::PointCloud &pc_sample) {
    Timer timer;
    timer.Start();

    if (!pc->HasPoints()) {
        misc3d::LogError("There is no scene point clouds.");
        return;
    }

    const bool has_normals = pc->HasNormals();
    const double normal_radius = calc_normal_relative_ * diameter_;
    pc->RemoveNonFinitePoints();
    if (!has_normals) {
        pc->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
            normal_radius, NORMAL_CALC_NN));
    }

    NormalConsistent(*pc);
    pc->NormalizeNormals();

    const size_t num = pc->points_.size();
    pc_sample = *pc->VoxelDownSample(dist_step_);

    misc3d::LogInfo("Scene point number is {} | {} after preprocessing.",
                    pc_sample.points_.size(), num);

    if (enable_edge_support_) {
        dense_scene_sample_ = *pc->VoxelDownSample(dist_step_dense_);

        scene_edge_ind_.clear();
        const double radius = diameter_ * calc_normal_relative_;
        ExtractEdges(dense_scene_sample_, radius, scene_edge_ind_);
    }

    misc3d::LogInfo("Preprocess time cost for estimate stage: {}",
                    timer.Stop());
}

bool PPFEstimator::Impl::Estimate(const PointCloudPtr &pc,
                                  std::vector<Pose6D> &results) {
    PreprocessEstimate(pc, scene_sample_);

    results.clear();
    Timer timer;
    timer.Start();

    if (!trained_) {
        misc3d::LogError("Need training before estimating!");
        return false;
    }

    if (!scene_sample_.HasPoints()) {
        return false;
    }

    if (config_.voting_param_.method ==
        PPFEstimatorConfig::VotingMode::EdgePoints) {
        if (scene_edge_ind_.empty()) {
            return false;
        }
    }

    // currently only supporting key points random selection
    const size_t num = scene_sample_.points_.size();
    std::vector<size_t> indices(num);
    std::iota(indices.begin(), indices.end(), 0);

    RandomSampler<size_t> sampler;
    const size_t sample = config_.ref_param_.ratio * num;
    std::vector<size_t> keypoints_indices = sampler(indices, sample);

    const auto key_points = scene_sample_.SelectByIndex(keypoints_indices);
    std::vector<Pose6D> pose_list;
    if (config_.voting_param_.method ==
        PPFEstimatorConfig::VotingMode::SampledPoints) {
        VotingAndGetPose(*key_points, scene_sample_, hash_table_, tmg_ptr_,
                         pose_list, pc_num_, pc_num_, pc_neighbor_);
    }

    if (config_.voting_param_.method ==
        PPFEstimatorConfig::VotingMode::EdgePoints) {
        const auto scene_boundary_points =
            dense_scene_sample_.SelectByIndex(scene_edge_ind_);

        // we use sampled key points and edge points for point pair computation
        VotingAndGetPose(*key_points, *scene_boundary_points,
                         hashtable_boundary_, tmg_ptr_boundary_, pose_list,
                         pc_num_, model_edge_ind_.size(), pc_neighbor_);
    }

    if (pose_list.empty()) {
        return false;
    } else {
        misc3d::LogInfo("Find {} raw poses after voting.", pose_list.size());
    }

    std::vector<std::vector<Pose6D>> pose_cluster;
    ClusterPoses(pose_list, pose_cluster);

    RefineSparsePose(model_sample_centered_, scene_sample_, pose_cluster,
                     results);

    // perform center shifting
    auto TransformOrigin = [&](Pose6D &pose) {
        Transformation T = pose.pose_;
        for (int i = 0; i < 3; i++) {
            T(i, 3) -= centroid_(0) * T(i, 0) + centroid_(1) * T(i, 1) +
                       centroid_(2) * T(i, 2);
        }
        pose.UpdateByPose(T);
    };
    std::for_each(results.begin(), results.end(), TransformOrigin);

    // sort pose based on number of votes
    std::sort(results.begin(), results.end(), [](Pose6D &p0, Pose6D &p1) {
        return p0.num_votes_ > p1.num_votes_;
    });

    double expected_votes_num =
        config_.ref_param_.ratio * reference_num_ * refered_num_;

    if (config_.voting_param_.method ==
        PPFEstimatorConfig::VotingMode::SampledPoints) {
        expected_votes_num *= VOTES_NUM_REDUCTION_FACTOR;
    }

    for (size_t i = 0; i < results.size(); i++) {
        results[i].object_id_ = config_.object_id_;
        const double score =
            std::min(1.0, (double)results[i].num_votes_ / expected_votes_num);
        results[i].score_ = score;
        misc3d::LogInfo("Pose {} with score {}", i, score);
    }

    // remove pose which votes is less than threshold
    size_t valid_num = 0;
    for (; valid_num < results.size();) {
        if (results[valid_num].score_ < config_.score_thresh_) {
            break;
        }
        valid_num++;
    }
    results.erase(results.begin() + valid_num, results.end());
    misc3d::LogInfo("Find {} pose satisfy score threshold", valid_num);

    if (results.empty()) {
        return false;
    }

    // remove pose by number of output
    if (valid_num > config_.num_result_) {
        results.erase(results.begin() + config_.num_result_, results.end());
    }
    pose_list_.assign(results.begin(), results.end());

    misc3d::LogInfo("Estimating time cost: {}", timer.Stop());
    return true;
}

void PPFEstimator::Impl::VotingAndGetPose(
    const open3d::geometry::PointCloud &reference_pts,
    const open3d::geometry::PointCloud &refered_pts,
    const std::vector<std::vector<PointPair>> &hashtable_ptr,
    const std::vector<Transformation> &tmg_ptr, std::vector<Pose6D> &pose_list,
    size_t reference_model_num, size_t refered_model_num,
    std::vector<std::vector<int>> &neighbor_table) {
    pose_list.clear();
    const size_t reference_num = reference_pts.points_.size();
    const size_t refered_num = refered_pts.points_.size();
    misc3d::LogInfo("Reference num: {}, Refered num: {}", reference_num,
                    refered_num);

    const size_t votes_threshold = refered_model_num * VOTING_THRESHOLD_FACTOR;
    const int alpha_model_num = angle_num_ * 2 - 1;
    const double dist_thresh = config_.voting_param_.min_dist_thresh * r_min_;
    const double cos_angle_thresh = cos(config_.voting_param_.min_angle_thresh);
    const double vote_num_relative_to_max = VOTE_NUM_RATIO;

    const size_t acc_size = reference_model_num * alpha_model_num;

    KDTree kdtree(refered_pts);
#pragma omp parallel for
    for (size_t si = 0; si < reference_num; si++) {
        double ppf[4];
        size_t quantized_ppf[4], spread_idx[81];

        double alpha_scene_rad, alpha;
        size_t quantize_alpha, idx;
        int n_searched, alpha_index, n_ppfs;
        ulong alpha_scene_bit, t;

        Transformation model_pose, tsg_inv, tmg;

        const PointXYZ &p0_p = reference_pts.points_[si];
        const Normal &p0_n = reference_pts.normals_[si];

        size_t max_votes, corr_alpha_index, corr_model_index;

        std::vector<size_t> accumulator(acc_size, 0);
        std::vector<ulong> flags_b(hash_table_size_, 0);

        Transformation tsg = CalcTNormal2RegionX(p0_p, p0_n);

        std::vector<int> ret_indices;
        std::vector<double> out_dists_sqr;

        n_searched =
            kdtree.SearchRadius(p0_p, r_min_, ret_indices, out_dists_sqr);

        if (n_searched > votes_threshold) {
            for (int j = 0; j < n_searched; j++) {
                const PointXYZ &p1_p = refered_pts.points_[ret_indices[j]];
                const Normal &p1_n = refered_pts.normals_[ret_indices[j]];
                // filter out the angle between normal which less than
                // angle threshold
                if (sqrt(out_dists_sqr[j]) < dist_thresh &&
                    p1_n.dot(p0_n) > cos_angle_thresh) {
                    continue;
                }

                alpha_scene_rad = CalcAlpha(p1_p, tsg);
                quantize_alpha = round((alpha_scene_rad + M_PI) /
                                       config_.voting_param_.angle_step);
                alpha_scene_bit = 1 << quantize_alpha;
                CalcPPF(p0_p, p0_n, p1_p, p1_n, ppf);
                QuantizePPF(ppf, quantized_ppf);
                SpreadPPF(quantized_ppf, spread_idx, n_ppfs);

                for (int ni = 0; ni < n_ppfs; ni++) {
                    idx = spread_idx[ni];
                    t = flags_b[idx] | alpha_scene_bit;
                    if (t == flags_b[idx]) {
                        continue;
                    }

                    flags_b[idx] = t;
                    const int n_vals = hashtable_ptr[idx].size();
                    for (int hi = 0; hi < n_vals; hi++) {
                        const PointPair &pp = hashtable_ptr[idx][hi];
                        alpha_index =
                            alpha_lut_[pp.quant_alpha_ * alpha_model_num +
                                       quantize_alpha];
                        accumulator[pp.i_ * alpha_model_num + alpha_index]++;
                    }
                }
            }
            // 0: model id, 1: quantized angle id, 2: vote number
            std::vector<std::vector<size_t>> local_maximum;

            CalcLocalMaximum(accumulator.data(), reference_model_num,
                             alpha_model_num, votes_threshold,
                             vote_num_relative_to_max, local_maximum,
                             neighbor_table);

            if (local_maximum.size() > 0) {
                // CalcMatrixInverse(tsg, tsg_inv);
                tsg_inv = tsg.inverse();
            } else {
                continue;
            }

            for (int li = 0; li < local_maximum.size(); li++) {
                Pose6D pose;
                corr_model_index = local_maximum[li][0];
                corr_alpha_index = local_maximum[li][1];
                max_votes = local_maximum[li][2];

                alpha = corr_alpha_index * config_.voting_param_.angle_step;
                double ca = cos(alpha), sa = sin(alpha);
                Transformation tr;
                tr << 1, 0, 0, 0, 0, ca, -sa, 0, 0, sa, ca, 0, 0, 0, 0, 1;

                tmg = tmg_ptr[corr_model_index];

                model_pose = tsg_inv * tr * tmg;
                pose.UpdateByPose(model_pose);
                pose.num_votes_ = max_votes;
                pose.corr_mi_ = corr_model_index;
#pragma omp critical
                { pose_list.emplace_back(pose); }
            }
        }
    }
}

bool PPFEstimator::Impl::Train(const PointCloudPtr &pc) {
    Timer timer;
    timer.Start();
    if (!pc->HasPoints()) {
        misc3d::LogError("There is no input points");
        return false;
    }

    // preprocess input point and compute normal
    PreprocessTrain(pc, model_sample_);

    // set ppf look-up table parameters
    angle_num_ = round(M_PI / config_.voting_param_.angle_step) + 1;
    angle_num_2_ = angle_num_ * angle_num_;
    angle_num_3_ = angle_num_2_ * angle_num_;
    dist_num_ = round(1 / config_.training_param_.rel_sample_dist) + 1;
    hash_table_size_ = pow(angle_num_, 3) * dist_num_;
    pc_num_ = model_sample_.points_.size();

    if (pc_num_ == 0) {
        misc3d::LogError("There is no input points after preprocessing");
        return false;
    }

    // shift model origin to centroid for easy pose clustering
    centroid_.setZero();
    for (int i = 0; i < pc_num_; i++) {
        centroid_ += model_sample_.points_[i];
    }
    centroid_ /= static_cast<double>(pc_num_);
    const size_t data_length = sizeof(double) * 3;
    model_sample_centered_ = model_sample_;
    for (int i = 0; i < pc_num_; i++) {
        model_sample_centered_.points_[i] -= centroid_;
    }

    hash_table_.resize(hash_table_size_);
    CalcHashTable(model_sample_centered_, model_sample_centered_, hash_table_,
                  tmg_ptr_, true);

    GenerateModelPCNeighbor(model_sample_centered_, pc_neighbor_,
                            r_min_ * NEIGHBOR_RADIUS_FACTOR);

    reference_num_ = model_sample_centered_.points_.size();
    refered_num_ = model_sample_centered_.points_.size();

    if (enable_edge_support_) {
        open3d::geometry::PointCloud dense_model_sample_centered =
            dense_model_sample_;

        if (model_edge_ind_.empty()) {
            misc3d::LogError("Fail to extract edge points!");
            return false;
        }

        for (size_t i = 0; i < dense_model_sample_centered.points_.size();
             i++) {
            dense_model_sample_centered.points_[i] -= centroid_;
        }

        hashtable_boundary_.resize(hash_table_size_);

        const auto model_edge_pts =
            dense_model_sample_.SelectByIndex(model_edge_ind_);

        CalcHashTable(model_sample_centered_, *model_edge_pts,
                      hashtable_boundary_, tmg_ptr_boundary_, false);
    }

    // generate ppf look-up table
    GenerateLUT();
    trained_ = true;

    misc3d::LogInfo("Training time cost: {}", timer.Stop());
    return true;
}

void PPFEstimator::Impl::CalcHashTable(
    const open3d::geometry::PointCloud &reference_pts,
    const open3d::geometry::PointCloud &refered_pts,
    std::vector<std::vector<PointPair>> &hashtable_ptr,
    std::vector<Transformation> &tmg_ptr, bool b_same_pointset) {
    const size_t reference_num = reference_pts.points_.size();
    const size_t refered_num = refered_pts.points_.size();
    const int double_size = sizeof(double);
    tmg_ptr.resize(reference_num);

#pragma omp parallel for
    for (size_t i = 0; i < reference_num; i++) {
        size_t idx;
        const PointXYZ &p0_p = reference_pts.points_[i];
        const Normal &p0_n = reference_pts.normals_[i];

        const Transformation tmg = CalcTNormal2RegionX(p0_p, p0_n);
        tmg_ptr[i] = tmg;

        double ppf[4];
        for (size_t j = 0; j < refered_num; j++) {
            if ((!b_same_pointset) || (b_same_pointset && j != i)) {
                PointPair pp;
                const PointXYZ &p1_p = refered_pts.points_[j];
                const Normal &p1_n = refered_pts.normals_[j];
                CalcPPF(p0_p, p0_n, p1_p, p1_n, ppf);
                pp.i_ = i;
                pp.j_ = j;
                pp.alpha_ = CalcAlpha(p1_p, tmg);
                pp.quant_alpha_ = round((pp.alpha_ + M_PI) /
                                        config_.voting_param_.angle_step);
                idx = HashPPF(ppf);
#pragma omp critical
                { hashtable_ptr[idx].emplace_back(pp); }
            }
        }
    }
}

void PPFEstimator::Impl::CalcPPF(const PointXYZ &p0, const Normal &n0,
                                 const PointXYZ &p1, const Normal &n1,
                                 double ppf[4]) {
    Eigen::Vector3d d(p1(0) - p0(0), p1(1) - p0(1), p1(2) - p0(2));
    const double norm = d.norm();
    d.normalize();

    ppf[0] = acos(n0.dot(d));
    ppf[1] = acos(n1.dot(d));
    ppf[2] = acos(n0.dot(n1));
    ppf[3] = norm;
}

void PPFEstimator::Impl::QuantizePPF(const double ppf[4], size_t quant_ppf[4]) {
    for (int i = 0; i < 3; i++) {
        quant_ppf[i] = round(ppf[i] / config_.voting_param_.angle_step);
    }
    quant_ppf[3] = round(ppf[3] / dist_step_);
}

size_t PPFEstimator::Impl::HashPPF(const size_t ppf[4]) {
    return ppf[0] + ppf[1] * angle_num_ + ppf[2] * angle_num_2_ +
           ppf[3] * angle_num_3_;
}

size_t PPFEstimator::Impl::HashPPF(const double ppf[4]) {
    size_t quantized_ppf[4];
    QuantizePPF(ppf, quantized_ppf);
    return HashPPF(quantized_ppf);
}

Transformation PPFEstimator::Impl::CalcTNormal2RegionX(const PointXYZ &p,
                                                       const Normal &n) {
    PointXYZ u(0, n(2), -n(1));
    double angle_rad = acos(n(0));
    if (!IsNormalizedVector(u)) {
        u(1) = 1;
        u(2) = 0;
    }

    // quaternion to rotate matrix
    double half_alpha_rad = angle_rad / 2;
    const Eigen::Vector4d quat(cos(half_alpha_rad), 0,
                               sin(half_alpha_rad) * u(1),
                               sin(half_alpha_rad) * u(2));

    Rotation rotation;
    Pose6D::QuatToMat(quat, rotation);

    Transformation tg;
    tg.setIdentity();
    tg.block<3, 1>(0, 3) = -1 * rotation * p;
    tg.block<3, 3>(0, 0) = rotation;

    return tg;
}

double PPFEstimator::Impl::CalcAlpha(const PointXYZ &pt,
                                     const Transformation t_normal_to_region) {
    const auto tran_pt = t_normal_to_region.block<3, 3>(0, 0) * pt +
                         t_normal_to_region.block<3, 1>(0, 3);
    return atan2(-tran_pt(2), tran_pt(1));
}

void PPFEstimator::Impl::SpreadPPF(const size_t ppf[4], size_t idx[81],
                                   int &n) {
    int d, a0, a1, a2;
    n = 0;

    int k;

    if (config_.voting_param_.faster_mode) {
        k = 2;
    } else {
        k = 3;
    }

    for (int s0 = 0; s0 < 3; s0++) {
        d = dist_shift_lut_[ppf[3]][s0] * angle_num_3_;
        if (d < 0)
            continue;
        for (int s1 = 0; s1 < k; s1++) {
            a0 = alpha_shift_lut_[ppf[0]][s1];
            if (a0 < 0)
                continue;
            a0 += d;
            for (int s2 = 0; s2 < k; s2++) {
                a1 = alpha_shift_lut_[ppf[1]][s2] * angle_num_;
                if (a1 < 0)
                    continue;
                a1 += a0;
                for (int s3 = 0; s3 < k; s3++) {
                    a2 = alpha_shift_lut_[ppf[2]][s3] * angle_num_2_;
                    if (a2 < 0)
                        continue;
                    idx[n] = a2 + a1;
                    n++;
                }
            }
        }
    }
}

bool PPFEstimator::Impl::MatchPose(const Pose6D &src, const Pose6D &dst,
                                   const int use_flag) {
    const auto &src_pose = src.pose_;
    const auto &dst_pose = dst.pose_;
    bool matched = true;
    if (use_flag & 1) {
        const PointXYZ t =
            dst_pose.block<3, 1>(0, 3) - src_pose.block<3, 1>(0, 3);
        const double norm_2 = t.norm();
        matched = (norm_2 < dist_threshold_);
    }
    if (use_flag & 2 && matched) {
        double trace = 0;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                trace += src_pose(r, c) * dst_pose(r, c);
            }
        }

        double ca = (trace - 1) / 2;
        double angle_rad;
        if (ca > 1) {
            angle_rad = 0;
        } else if (ca < -1) {
            angle_rad = M_PI;
        } else {
            angle_rad = acos(ca);
        }
        matched = (angle_rad < config_.rel_angle_thresh_);
    }
    return matched;
}

void PPFEstimator::Impl::GatherPose(
    const std::vector<Pose6D> &pose_list, const std::vector<int> &indices,
    const std::vector<std::vector<int>> &neibor_idx,
    std::vector<PoseCluster> &pose_clusters) {
    pose_clusters.clear();
    const size_t n_pose = neibor_idx.size();
    if (n_pose == 0)
        return;
    // int min_num_pt = 10;  // double min_num_pt
    int min_num_pt = 6;  // double min_num_pt
    for (size_t i = 0, num; i < n_pose; i++) {
        num = neibor_idx[i].size();
        if (num > min_num_pt) {
            min_num_pt = num;
        }
    }
    min_num_pt /= 2.0;
    std::vector<bool> check_flags(pose_list.size(), false);
    size_t n_cluster = 0;
    size_t mi, ci;
    for (size_t pi = 0; pi < n_pose; pi++) {
        ci = pi;
        mi = indices[ci];
        if (check_flags[mi] || neibor_idx[ci].size() < min_num_pt)
            continue;
        pose_clusters.emplace_back(PoseCluster());
        std::stack<size_t> my_stack;
        my_stack.push(ci);
        while (!my_stack.empty()) {
            ci = my_stack.top();
            mi = indices[ci];
            my_stack.pop();
            if (check_flags[mi])
                continue;
            pose_clusters[n_cluster].pose_indices_.emplace_back(mi);
            pose_clusters[n_cluster].num_votes_ += pose_list[mi].num_votes_;
            check_flags[mi] = true;
            for (size_t ni : neibor_idx[ci]) {
                ci = ni;
                mi = indices[ci];
                if (check_flags[mi])
                    continue;
                if (neibor_idx[ci].size() >= min_num_pt) {
                    my_stack.push(ci);
                } else {
                    pose_clusters[n_cluster].pose_indices_.emplace_back(mi);
                    pose_clusters[n_cluster].num_votes_ +=
                        pose_list[mi].num_votes_;
                    check_flags[mi] = true;
                }
            }
        }
        n_cluster++;
    }
    sort(pose_clusters.begin(), pose_clusters.end(),
         [](PoseCluster &p0, PoseCluster &p1) {
             return p0.num_votes_ > p1.num_votes_;
         });
}

void PPFEstimator::Impl::CalcPoseNeighbor(
    const std::vector<Pose6D> &pose_list, const std::vector<int> &indices,
    const int use_flag, std::vector<std::vector<int>> &neibor_idx) {
    neibor_idx.clear();
    const size_t n_pose = indices.size();
    if (n_pose == 0)
        return;
    std::vector<std::vector<bool>> match_table(
        n_pose, std::vector<bool>(n_pose, false));
#pragma omp parallel for
    for (size_t i = 0; i < n_pose; i++) {
        bool matched;
        const Pose6D *center_pose, *cur_pose;
        center_pose = &pose_list[indices[i]];
        match_table[i][i] = true;
        for (size_t j = i + 1; j < n_pose; j++) {
            cur_pose = &pose_list[indices[j]];
            matched = MatchPose(*center_pose, *cur_pose, use_flag);
            match_table[i][j] = matched;
            match_table[j][i] = matched;
        }
    }
    neibor_idx.resize(n_pose);
#pragma omp parallel for
    for (size_t pi = 0; pi < n_pose; pi++) {
        for (size_t j = 0; j < n_pose; j++) {
            if (match_table[pi][j]) {
                neibor_idx[pi].emplace_back(j);
            }
        }
    }
}

void PPFEstimator::Impl::ClusterPoses(
    std::vector<Pose6D> &pose_list,
    std::vector<std::vector<Pose6D>> &clustered_pose) {
    clustered_pose.clear();
    std::sort(pose_list.begin(), pose_list.end(), [](Pose6D &p0, Pose6D &p1) {
        return p0.num_votes_ > p1.num_votes_;
    });
    const size_t n_pose = pose_list.size();

    const size_t num_votes_threshold = pose_list[0].num_votes_ * 0.5;
    size_t valid_pose = 0;
    for (; valid_pose < n_pose; valid_pose++) {
        if (pose_list[valid_pose].num_votes_ < num_votes_threshold) {
            break;
        }
    }
    // use kdtree to find neighbor
    open3d::geometry::PointCloud pose_ts;
    pose_ts.points_.resize(valid_pose);
    for (size_t i = 0; i < valid_pose; i++) {
        pose_ts.points_[i] = pose_list[i].t_;
    }

    KDTree kdtree(pose_ts);

    std::vector<int> ret_indices;
    std::vector<double> out_dists_sqr;
    int n_searched;
    std::vector<std::vector<int>> neighbor_indexs(valid_pose);
    std::vector<int> indices(valid_pose);
    for (int i = 0; i < valid_pose; i++) {
        indices[i] = i;
        const PointXYZ &temp = pose_ts.points_[i];
        n_searched = kdtree.SearchRadius(temp, dist_threshold_, ret_indices,
                                         out_dists_sqr);
        neighbor_indexs[i] = ret_indices;
    }
    std::vector<PoseCluster> pose_cluster;
    GatherPose(pose_list, indices, neighbor_indexs, pose_cluster);
    const size_t n_cluster = pose_cluster.size();
    clustered_pose.resize(n_cluster);
    pose_clustered_by_t_.resize(n_cluster);

#pragma omp parallel for
    for (size_t i = 0; i < n_cluster; i++) {
        std::vector<std::vector<int>> temp_neighbor_indices;

        for (size_t index : pose_cluster[i].pose_indices_) {
            pose_clustered_by_t_[i].emplace_back(pose_list[index]);
        }

        CalcPoseNeighbor(pose_list, pose_cluster[i].pose_indices_, 3,
                         temp_neighbor_indices);

        std::vector<PoseCluster> temp_pose_cluster;
        GatherPose(pose_list, pose_cluster[i].pose_indices_,
                   temp_neighbor_indices, temp_pose_cluster);

        clustered_pose[i].resize(temp_pose_cluster.size());
        for (int j = 0; j < temp_pose_cluster.size(); j++) {
            PoseAverage(pose_list, temp_pose_cluster[j].pose_indices_,
                        clustered_pose[i][j]);
        }
    }
}

void PPFEstimator::Impl::RefineSparsePose(
    const open3d::geometry::PointCloud &model_pcn,
    const open3d::geometry::PointCloud &scene_pcn,
    const std::vector<std::vector<Pose6D>> &clustered_pose,
    std::vector<Pose6D> &pose_list) {
    const double max_correspondence_distance =
        config_.refine_param_.rel_dist_sparse_thresh * dist_step_;
    const open3d::pipelines::registration::ICPConvergenceCriteria criteria(
        1e-6, 1e-6, SPARSE_REFINE_ICP_ITERATION);
#pragma omp parallel for
    for (size_t i = 0; i < clustered_pose.size(); i++) {
        if (clustered_pose[i].size() == 0) {
            continue;
        }

        const Pose6D max_votes_pose = *std::max_element(
            clustered_pose[i].begin(), clustered_pose[i].end(),
            [](const Pose6D &p1, const Pose6D &p2) {
                return p1.num_votes_ < p2.num_votes_;
            });

        const auto &init_pose = max_votes_pose.pose_;

        open3d::pipelines::registration::RegistrationResult res;
        if (config_.refine_param_.method ==
            PPFEstimatorConfig::RefineMethod::PointToPoint) {
            res = open3d::pipelines::registration::RegistrationICP(
                model_pcn, scene_pcn, max_correspondence_distance, init_pose,
                open3d::pipelines::registration::
                    TransformationEstimationPointToPoint(),
                criteria);
        } else if (config_.refine_param_.method ==
                   PPFEstimatorConfig::RefineMethod::PointToPlane) {
            // we use L1 loss for point to plane icp. See paper "Analysis of
            // Robust Functions for Registration Algorithms" for reference.
            auto loss =
                std::make_shared<open3d::pipelines::registration::L1Loss>();
            res = open3d::pipelines::registration::RegistrationICP(
                model_pcn, scene_pcn, max_correspondence_distance, init_pose,
                open3d::pipelines::registration::
                    TransformationEstimationPointToPlane(loss),
                criteria);
        }
#pragma omp critical
        {
            pose_list.emplace_back(max_votes_pose);
            pose_list[pose_list.size() - 1].UpdateByPose(res.transformation_);
        }
    }
}

void PPFEstimator::Impl::PoseAverage(const std::vector<Pose6D> &pose_list,
                                     const std::vector<int> &pose_idx,
                                     Pose6D &result_pose) {
    result_pose.corr_mi_ = -1;
    Eigen::Vector4d q_avg;
    Eigen::Matrix4d q_avg_mat = Eigen::Matrix4d::Zero();
    Eigen::Vector3d t_avg = Eigen::Vector3d::Zero();

    // Perform the final averaging
    const int curSize = (int)pose_idx.size();
    int cur_i;
    result_pose.num_votes_ = 0;

    for (int j = 0; j < curSize; j++) {
        cur_i = pose_idx[j];
        Eigen::Vector4d q_cv(pose_list[cur_i].q_);
        q_avg_mat += q_cv * q_cv.transpose();
        t_avg += Eigen::Vector3d(pose_list[cur_i].t_);
        result_pose.num_votes_ += pose_list[cur_i].num_votes_;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(q_avg_mat);
    q_avg = eigensolver.eigenvectors().col(3);
    t_avg *= 1.0 / double(curSize);
    result_pose.UpdateByQuat(q_avg, t_avg);
}

void PPFEstimator::Impl::DownSamplePCNormal(
    const open3d::geometry::PointCloud &pc,
    const std::shared_ptr<KDTree> &kdtree, const double radius,
    const size_t min_idx, open3d::geometry::PointCloud &pc_sample) {
    // This down sample method aims at sampling original point clouds by
    // distance interval, and hold the actual position of point itself. Because
    // voxel downsample method will change the position of points by averaging
    // the points inside a voxel.
    int cur_i, n_searched, near_i;
    double dist;
    const double search_r = radius * 2;
    const size_t n_pt = pc.points_.size();
    std::vector<bool> checked_flags(n_pt, false);
    std::queue<int> queue;
    queue.push((int)min_idx);
    std::vector<int> ret_indices;
    std::vector<double> out_dists_sqr;
    std::vector<size_t> selected_indices;
    while (!queue.empty()) {
        cur_i = queue.front();
        queue.pop();
        if (checked_flags[cur_i])
            continue;
        checked_flags[cur_i] = true;

        auto &p = pc.points_[cur_i];
        selected_indices.emplace_back(cur_i);
        n_searched =
            kdtree->SearchRadius(p, search_r, ret_indices, out_dists_sqr);

        for (int i = 0; i < n_searched; i++) {
            near_i = ret_indices[i];
            dist = sqrt(out_dists_sqr[i]);
            if (checked_flags[near_i])
                continue;
            if (dist < radius) {
                checked_flags[near_i] = true;
            } else {
                queue.push(near_i);
            }
        }
    }
    pc_sample = *pc.SelectByIndex(selected_indices);
}

void PPFEstimator::Impl::CalcModelNormalAndSampling(
    const PointCloudPtr &pc, const std::shared_ptr<KDTree> &kdtree,
    const double normal_r, const double step, const PointXYZ &view_pt,
    open3d::geometry::PointCloud &pc_sample) {
    pc_sample.Clear();
    const size_t n_pt = pc->points_.size();

    // calc normals
    if (!pc->HasNormals()) {
        pc->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(normal_r, NORMAL_CALC_NN),
            false);
    }
    pc->NormalizeNormals();

    // calc nearest point respect to view point
    std::vector<int> ret_indices(1);
    std::vector<double> out_dists_sqr(1);
    const int n = kdtree->SearchKNN(view_pt, 1, ret_indices, out_dists_sqr);
    const PointXYZ nearest = pc->points_[ret_indices[0]];
    const int nearest_idx = ret_indices[0];
    const PointXYZ pt_2_view(view_pt(0) - nearest(0), view_pt(1) - nearest(1),
                             view_pt(2) - nearest(2));

    // modify normal based on orientation
    if (pt_2_view.transpose() * pc->normals_[nearest_idx] < 0)
        FlipNormal(pc->normals_[nearest_idx]);

    if (invert_normal_)
        FlipNormal(pc->normals_[nearest_idx]);

    // normal consistent
    std::vector<bool> calc_flags(n_pt, false);
    auto cmp = [](std::pair<size_t, double> &q0,
                  std::pair<size_t, double> &q1) {
        return q0.second > q1.second;
    };

    std::priority_queue<std::pair<size_t, double>,
                        std::vector<std::pair<size_t, double>>, decltype(cmp)>
        pq(cmp);

    int cur_i = nearest_idx, near_i, n_searched;
    std::pair<size_t, double> cur_pair(nearest_idx, 0);
    pq.push(cur_pair);
    std::vector<int> ret_indices2;
    std::vector<double> out_dists_sqr2;
    while (!pq.empty()) {
        cur_pair = pq.top();
        cur_i = cur_pair.first;
        pq.pop();
        if (calc_flags[cur_i])
            continue;
        const PointXYZ &temp = pc->points_[cur_i];
        n_searched =
            kdtree->SearchRadius(temp, normal_r, ret_indices2, out_dists_sqr2);

        for (size_t i = 0; i < n_searched; i++) {
            near_i = ret_indices2[i];
            if (!calc_flags[near_i]) {
                pq.push(std::make_pair(size_t(ret_indices2[i]),
                                       sqrt(out_dists_sqr2[i])));
            }
        }
        for (size_t i = 0; i < n_searched; i++) {
            near_i = ret_indices2[i];
            if (calc_flags[near_i]) {
                auto &cur_n = pc->normals_[cur_i];
                auto &near_n = pc->normals_[near_i];
                if (cur_n.dot(near_n) < 0)
                    FlipNormal(cur_n);
                break;
            }
        }
        calc_flags[cur_i] = true;
    }

    DownSamplePCNormal(*pc, kdtree, step, nearest_idx, pc_sample);

    if (enable_edge_support_) {
        DownSamplePCNormal(*pc, kdtree, dist_step_dense_, nearest_idx,
                           dense_model_sample_);
        model_edge_ind_.clear();
        ExtractEdges(dense_model_sample_, normal_r, model_edge_ind_);
        misc3d::LogInfo("Extract {} edge points form model point clouds.",
                        model_edge_ind_.size());
    }
}

void PPFEstimator::Impl::ExtractEdges(const open3d::geometry::PointCloud &pc,
                                      double radius,
                                      std::vector<size_t> &edge_ind) {
    edge_ind = features::DetectEdgePoints(
        pc, open3d::geometry::KDTreeSearchParamHybrid(
                radius, config_.edge_param_.pts_num));
}

void PPFEstimator::Impl::CalcLocalMaximum(
    const size_t *accumulator, const size_t rows, const size_t cols,
    const size_t vote_threshold, const double relative_to_max,
    std::vector<std::vector<size_t>> &max_in_pair,
    std::vector<std::vector<int>> &neighbor_table) {
    max_in_pair.clear();
    size_t(*model_infos)[2] =
        new size_t[rows][2];  // 0: quantized angle, 1: votes number
    size_t final_idx = cols - 1;
    const size_t *acc_ptr = accumulator;
    for (int r = 0; r < rows; r++, acc_ptr += cols) {
        model_infos[r][1] = acc_ptr[cols - 1] + acc_ptr[0] + acc_ptr[1];
        model_infos[r][0] = 0;
        size_t temp;
        for (int c = 1; c < final_idx; c++) {
            temp = acc_ptr[c - 1] + acc_ptr[c] + acc_ptr[c + 1];
            if (temp > model_infos[r][1]) {
                model_infos[r][1] = temp;
                model_infos[r][0] = c;
            }
        }
        temp = acc_ptr[final_idx - 1] + acc_ptr[final_idx] + acc_ptr[0];
        if (temp > model_infos[r][1]) {
            model_infos[r][1] = temp;
            model_infos[r][0] = final_idx;
        }
    }
    size_t max_idx, max_votes = 0;
    for (int i = 0; i < rows; i++) {
        if (model_infos[i][1] > max_votes) {
            max_idx = i;
            max_votes = model_infos[i][1];
        }
    }
    if (max_votes < vote_threshold) {
        delete[] model_infos;
        return;
    }
    // 0: model index, 1: quantized angle, 2: votes number
    size_t n_neighbor;
    int *n_idx;
    const size_t votes_threshold = max_votes * relative_to_max;
    std::vector<bool> max_flags(rows, true);
    bool is_max;
    for (size_t i = 0; i < rows; i++) {
        if (model_infos[i][1] > votes_threshold && max_flags[i]) {
            is_max = true;
            n_neighbor = neighbor_table[i].size();
            n_idx = neighbor_table[i].data();
            for (int ni = 0; ni < n_neighbor; ni++, n_idx++) {
                if (model_infos[i][1] < model_infos[*n_idx][1]) {
                    is_max = false;
                } else {
                    max_flags[*n_idx] = false;
                }
            }
            max_flags[i] = is_max;
            if (is_max) {
                max_in_pair.push_back(
                    {i, model_infos[i][0], model_infos[i][1]});
            }
        }
    }
    delete[] model_infos;
}

void PPFEstimator::Impl::GenerateModelPCNeighbor(
    const open3d::geometry::PointCloud &pts,
    std::vector<std::vector<int>> &neighbor_table, const double radius) {
    const size_t pts_size = pts.points_.size();
    neighbor_table.resize(pts_size);
    KDTree kdtree(pts);

    int n_search;
#pragma omp parallel for
    for (size_t i = 0; i < pts_size; i++) {
        std::vector<int> ret_indices;
        std::vector<double> out_dists_sqr;
        const PointXYZ &temp = pts.points_[i];
        n_search =
            kdtree.SearchRadius(temp, radius, ret_indices, out_dists_sqr);
        neighbor_table[i] = ret_indices;
    }
}

void PPFEstimator::Impl::GenerateLUT() {
    int alpha_model_num = 2 * angle_num_ - 1;
    alpha_lut_.resize(alpha_model_num * alpha_model_num);
    int row_idx;
    for (int i = 0; i < alpha_model_num; i++) {
        row_idx = i * alpha_model_num;
        for (int j = 0; j < alpha_model_num; j++) {
            alpha_lut_[row_idx + j] =
                (i - j + alpha_model_num) % alpha_model_num;
        }
    }
    alpha_2_shift_lut_.resize(alpha_model_num);
    for (int i = 0; i < alpha_model_num; i++) {
        for (int j = 0; j < 3; j++) {
            alpha_2_shift_lut_[i][j] =
                (i + j - 1 + alpha_model_num) % alpha_model_num;
        }
    }
    alpha_shift_lut_.resize(angle_num_);
    int t;
    for (int i = 0; i < angle_num_; i++) {
        for (int j = 0; j < 3; j++) {
            t = i + j - 1;
            if (t < 0 || t >= angle_num_)
                t = -1;
            alpha_shift_lut_[i][j] = t;
        }
    }
    dist_shift_lut_.resize(dist_num_);
    for (int i = 0; i < dist_num_; i++) {
        for (int j = 0; j < 3; j++) {
            t = i + j - 1;
            if (t < 0 || t >= dist_num_)
                t = -1;
            dist_shift_lut_[i][j] = t;
        }
    }
}

void PPFEstimator::Impl::CalcMeanAndCovarianceMatrix(
    const std::vector<PointXYZ> &pc, int *indices, const int indices_num,
    double cov_matrix[3][3], double centoid[3]) {
    double Sxx = 0, Sxy = 0, Sxz = 0, Syy = 0, Syz = 0, Szz = 0, Sx = 0, Sy = 0,
           Sz = 0;
    int n_pt = indices_num;
    double scale = 1. / n_pt;
    const double *pt;
    for (int i = 0; i < n_pt; i++) {
        pt = pc[indices[i]].data();
        Sxx += pt[0] * pt[0];
        Sxy += pt[0] * pt[1];
        Sxz += pt[0] * pt[2];
        Syy += pt[1] * pt[1];
        Syz += pt[1] * pt[2];
        Szz += pt[2] * pt[2];
        Sx += pt[0];
        Sy += pt[1];
        Sz += pt[2];
    }
    Sxx *= scale;
    Sxy *= scale;
    Sxz *= scale;
    Syy *= scale;
    Syz *= scale;
    Szz *= scale;
    Sx *= scale;
    Sy *= scale;
    Sz *= scale;
    cov_matrix[0][0] = Sxx - Sx * Sx;
    cov_matrix[1][1] = Syy - Sy * Sy;
    cov_matrix[2][2] = Szz - Sz * Sz;
    cov_matrix[0][1] = Sxy - Sx * Sy;
    cov_matrix[0][2] = Sxz - Sx * Sz;
    cov_matrix[1][2] = Syz - Sy * Sz;
    cov_matrix[1][0] = cov_matrix[0][1];
    cov_matrix[2][0] = cov_matrix[0][2];
    cov_matrix[2][1] = cov_matrix[1][2];
    centoid[0] = Sx;
    centoid[1] = Sy;
    centoid[2] = Sz;
}

PPFEstimator::PPFEstimator() {
    impl_ptr_ = std::make_unique<Impl>(PPFEstimatorConfig());
}

PPFEstimator::PPFEstimator(const PPFEstimatorConfig &config) {
    if (CheckConfig(config))
        impl_ptr_ = std::make_unique<Impl>(config);
}

PPFEstimator::~PPFEstimator() = default;

bool PPFEstimator::Train(const PointCloudPtr &pc) {
    if (impl_ptr_ != nullptr) {
        return impl_ptr_->Train(pc);
    } else {
        misc3d::LogError("PPF Estimator not is initialized successfully");
        return false;
    }
}

bool PPFEstimator::Estimate(const PointCloudPtr &pc,
                            std::vector<Pose6D> &results) {
    if (impl_ptr_ != nullptr) {
        return impl_ptr_->Estimate(pc, results);
    } else {
        misc3d::LogError("PPF Estimator not is initialized successfully");
        return false;
    }
}

bool PPFEstimator::Impl::SetConfig(const PPFEstimatorConfig &config) {
    config_ = config;
    dist_threshold_ = config.rel_dist_thresh_;
    // (calc_normal_relative_ = config.rel_sample_dist_ / 2) is usually a
    // good choise
    calc_normal_relative_ = config.training_param_.calc_normal_relative;
    enable_edge_support_ = (config.voting_param_.method ==
                            PPFEstimatorConfig::VotingMode::EdgePoints)
                               ? true
                               : false;
    invert_normal_ = config.training_param_.invert_model_normal;
    return true;
}

bool PPFEstimator::SetConfig(const PPFEstimatorConfig &config) {
    if (CheckConfig(config)) {
        impl_ptr_->SetConfig(config);
        return true;
    } else {
        return false;
    }
}

double PPFEstimator::GetModelDiameter() {
    return impl_ptr_->diameter_;
}

std::vector<Pose6D> PPFEstimator::GetPose() {
    return impl_ptr_->pose_list_;
}

open3d::geometry::PointCloud PPFEstimator::GetSampledModel() {
    return impl_ptr_->model_sample_;
}

open3d::geometry::PointCloud PPFEstimator::GetSampledScene() {
    return impl_ptr_->scene_sample_;
}

open3d::geometry::PointCloud PPFEstimator::GetModelEdges() {
    open3d::geometry::PointCloud out;
    if (impl_ptr_->enable_edge_support_) {
        out = *impl_ptr_->dense_model_sample_.SelectByIndex(
            impl_ptr_->model_edge_ind_);
    } else {
        misc3d::LogWarning(
            "Can not get model edge points due to edge support is not "
            "enabled.");
    }

    return out;
}

open3d::geometry::PointCloud PPFEstimator::GetSceneEdges() {
    open3d::geometry::PointCloud out;
    if (impl_ptr_->enable_edge_support_) {
        out = *impl_ptr_->dense_scene_sample_.SelectByIndex(
            impl_ptr_->scene_edge_ind_);
    } else {
        misc3d::LogWarning(
            "Can not get scene edge points due to edge support is not "
            "enabled.");
    }

    return out;
}

// set default parameters
PPFEstimatorConfig::PPFEstimatorConfig() {
    training_param_ = {false, 0.05, 0.025, 0.01};
    ref_param_ = {ReferencePointSelection::Random, 0.6};
    voting_param_ = {VotingMode::SampledPoints, true, Deg2Rad<double>(12),
                     1.0 / 3, Deg2Rad<double>(30)};
    edge_param_ = {20};
    refine_param_ = {RefineMethod::PointToPlane, 5};

    rel_dist_thresh_ = 0.05;
    rel_angle_thresh_ = Deg2Rad<double>(12);
    score_thresh_ = 0.6;
    num_result_ = 10;
    object_id_ = 0;
}

PPFEstimatorConfig::~PPFEstimatorConfig() = default;

bool PPFEstimator::CheckConfig(const PPFEstimatorConfig &config) {
    if (config.training_param_.rel_dense_sample_dist >=
        config.training_param_.rel_sample_dist) {
        misc3d::LogError(
            "Dense_sample_dist should be smaller than sample_dist.");
        return false;
    }

    return true;
}

}  // namespace pose_estimation
}  // namespace misc3d