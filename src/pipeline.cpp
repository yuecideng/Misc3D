#include <thread>
#include <vector>

#include <misc3d/logging.h>
#include <misc3d/registration/transform_estimation.h>
#include <misc3d/resonstruction/pipeline.h>
#include <misc3d/utils.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/ImageIO.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/PoseGraphIO.h>
#include <open3d/pipelines/integration/ScalableTSDFVolume.h>
#include <open3d/pipelines/odometry/Odometry.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>

namespace misc3d {
namespace reconstruction {

PipelineConfig::PipelineConfig() {
    data_path_ = "";
    depth_scale_ = 1000.0;
    max_depth_ = 3.0;
    max_depth_diff_ = 0.05;
    voxel_size_ = 0.01;
    integration_voxel_size_ = 0.005;
    make_fragment_param_ = {100, 40, 0.2};
    local_refine_method_ = LocalRefineMethod::ColoredICP;
    global_registration_method_ = GlobalRegistrationMethod::TeaserPlusPlus;
    optimization_param_ = {0.1, 5.0};
}

ReconstructionPipeline::ReconstructionPipeline(const PipelineConfig& config)
    : config_(config) {
    // Init ORB detector.
    orb_detector_ =
        cv::ORB::create(config_.make_fragment_param_.orb_feature_num);

    if (config_.data_path_[config_.data_path_.size() - 1] != '/') {
        config_.data_path_ += "/";
    }
}

bool ReconstructionPipeline::ReadRGBDData() {
    const std::string& data_path = config_.data_path_;
    const std::string color_path = data_path + "/color";
    const std::string depth_path = data_path + "/depth";
    std::vector<std::string> color_files, depth_files;
    bool ret;

    misc3d::LogInfo("Reading RGBD data from {}", data_path.c_str());
    if (!open3d::utility::filesystem::ListFilesInDirectoryWithExtension(
            color_path, "png", color_files) ||
        !open3d::utility::filesystem::ListFilesInDirectoryWithExtension(
            depth_path, "png", depth_files)) {
        misc3d::LogWarning("Failed to read RGBD data.");
        return false;
    }

    if (color_files.size() != depth_files.size()) {
        misc3d::LogWarning(
            "Number of color {} and depth {} images are not equal.",
            color_files.size(), depth_files.size());
        return false;
    }

    misc3d::LogInfo("Found {} RGBD images.", color_files.size());
    rgbd_lists_.resize(color_files.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < color_files.size(); ++i) {
        // Read color image and depth image.
        open3d::geometry::Image color, depth;
        open3d::io::ReadImage(color_files[i], color);
        open3d::io::ReadImage(depth_files[i], depth);

        // Create RGBD image.
        const auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            color, depth, config_.depth_scale_, config_.max_depth_, true);
        rgbd_lists_[i] = *rgbd;
    }

    return true;
}

void ReconstructionPipeline::BuildSingleFragment(int fragment_id) {
    const int sid =
        fragment_id * config_.make_fragment_param_.n_frame_per_fragment;
    const int eid =
        std::min(sid + config_.make_fragment_param_.n_frame_per_fragment,
                 static_cast<int>(rgbd_lists_.size()));
    BuildPoseGraphForFragment(fragment_id, sid, eid);
    OptimizePoseGraphForFragment(fragment_id);
    IntegrateFragmentTSDF(fragment_id);
}

void ReconstructionPipeline::BuildPoseGraphForFragment(int fragment_id, int sid,
                                                       int eid) {
    open3d::pipelines::registration::PoseGraph pose_graph;
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.push_back(
        open3d::pipelines::registration::PoseGraphNode(trans_odometry));

    for (int s = sid; s < eid; ++s) {
        for (int t = s + 1; t < eid; ++t) {
            // Compute odometry.
            if (t == s + 1) {
                misc3d::LogInfo(
                    "Fragment {:03d} / {:03d} :: RGBD odometry between frame : "
                    "{} and {}",
                    fragment_id, n_fragments_ - 1, s, t);
                const auto result = RegisterRGBDPair(s, t);
                trans_odometry = std::get<1>(result) * trans_odometry;
                pose_graph.nodes_.push_back(
                    open3d::pipelines::registration::PoseGraphNode(
                        trans_odometry.inverse()));
                pose_graph.edges_.push_back(
                    open3d::pipelines::registration::PoseGraphEdge(
                        s - sid, t - sid, std::get<1>(result),
                        std::get<2>(result), false));
                // Keyframe loop closure.
            } else if (s % n_keyframes_per_n_frame_ == 0 &&
                       t % n_keyframes_per_n_frame_ == 0) {
                misc3d::LogInfo(
                    "Fragment {:03d} / {:03d} :: RGBD loop closure between "
                    "frame : "
                    "{} and {}",
                    fragment_id, n_fragments_ - 1, s, t);
                const auto result = RegisterRGBDPair(s, t);
                if (std::get<0>(result)) {
                    pose_graph.edges_.push_back(
                        open3d::pipelines::registration::PoseGraphEdge(
                            s - sid, t - sid, std::get<1>(result),
                            std::get<2>(result), true));
                }
            }
        }
    }
    fragment_pose_graphs_[fragment_id] = pose_graph;
}

void ReconstructionPipeline::OptimizePoseGraphForFragment(int fragment_id) {
    open3d::pipelines::registration::GlobalOptimizationOption option(
        config_.max_depth_diff_, 0.25,
        config_.optimization_param_.preference_loop_closure_odometry, 0);
    open3d::pipelines::registration::GlobalOptimization(
        fragment_pose_graphs_[fragment_id],
        open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
        open3d::pipelines::registration::
            GlobalOptimizationConvergenceCriteria(),
        option);
}

void ReconstructionPipeline::IntegrateFragmentTSDF(int fragment_id) {
    open3d::pipelines::integration::ScalableTSDFVolume volume(
        config_.integration_voxel_size_, 0.04,
        open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
    const auto& pose_graph = fragment_pose_graphs_[fragment_id];
    const size_t graph_num = pose_graph.nodes_.size();
    for (size_t i = 0; i < graph_num; ++i) {
        const int i_abs =
            fragment_id * config_.make_fragment_param_.n_frame_per_fragment + i;
        misc3d::LogInfo(
            "Fragment {:03d} / {:03d} :: Integrate rgbd frame {:d} ({:d} of "
            "{:d}.",
            fragment_id, n_fragments_ - 1, i_abs, i + 1, graph_num);
        const open3d::geometry::RGBDImage& rgbd = rgbd_lists_[i_abs];
        const auto color = rgbd.color_.CreateImageFromFloatImage<uint8_t>();
        open3d::geometry::RGBDImage new_rgbd(*color, rgbd.depth_);
        volume.Integrate(new_rgbd, config_.camera_intrinsic_,
                         pose_graph.nodes_[i].pose_.inverse());
    }
    auto mesh = volume.ExtractTriangleMesh();
    mesh->ComputeVertexNormals();

    // Create fragment point clouds.
    open3d::geometry::PointCloud pcd;
    pcd.points_ = mesh->vertices_;
    pcd.colors_ = mesh->vertex_colors_;
    pcd.normals_ = mesh->vertex_normals_;
    fragment_point_clouds_[fragment_id] = pcd;
}

void ReconstructionPipeline::SaveFragmentResults() {
    const std::string pose_graph_path = config_.data_path_ + "fragments/";
    // Save fragment pose graph and point clouds.
    for (int i = 0; i < n_fragments_; i++) {
        std::string id = std::to_string(i);
        if (id.size() != 3) {
            const int size = 3 - id.size();
            for (size_t i = 0; i < size; i++) {
                id = "0" + id;
            }
        }
        const std::string file_name = "fragment_" + id;
        open3d::io::WritePoseGraph(file_name + ".json",
                                   fragment_pose_graphs_[i]);
        open3d::io::WritePointCloud(file_name + ".ply",
                                    fragment_point_clouds_[i]);
    }
}

std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
ReconstructionPipeline::RegisterRGBDPair(int s, int t) {
    bool success;
    Eigen::Matrix4d odometry;
    Eigen::Matrix6d information_matrix;

    open3d::geometry::RGBDImage& rgbd_s = rgbd_lists_[s];
    open3d::geometry::RGBDImage& rgbd_t = rgbd_lists_[t];

    open3d::pipelines::odometry::OdometryOption option;
    option.max_depth_diff_ = config_.max_depth_diff_;
    if (abs(s - t) != 1) {
        Eigen::Matrix4d odo_init = PoseEstimation(rgbd_s, rgbd_t);
        if (!odo_init.isIdentity(1e-8)) {
            return open3d::pipelines::odometry::ComputeRGBDOdometry(
                rgbd_s, rgbd_t, config_.camera_intrinsic_, odo_init,
                open3d::pipelines::odometry::
                    RGBDOdometryJacobianFromHybridTerm(),
                option);
        } else {
            return std::make_tuple(false, Eigen::Matrix4d::Identity(),
                                   Eigen::Matrix6d::Identity());
        }
    } else {
        return open3d::pipelines::odometry::ComputeRGBDOdometry(
            rgbd_s, rgbd_t, config_.camera_intrinsic_,
            Eigen::Matrix4d::Identity(),
            open3d::pipelines::odometry::RGBDOdometryJacobianFromHybridTerm(),
            option);
    }
}

Eigen::Matrix4d ReconstructionPipeline::PoseEstimation(
    const open3d::geometry::RGBDImage& src,
    const open3d::geometry::RGBDImage& dst) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    // Create CV format images.
    const int width = config_.camera_intrinsic_.width_;
    const int heigth = config_.camera_intrinsic_.height_;
    const auto color_src = src.color_.CreateImageFromFloatImage<uint8_t>();
    const auto color_dst = dst.color_.CreateImageFromFloatImage<uint8_t>();
    cv::Mat cv_src(cv::Size(width, heigth), CV_8UC1, color_src->data_.data());
    cv::Mat cv_dst(cv::Size(width, heigth), CV_8UC1, color_dst->data_.data());

    // Compute ORB features.
    std::vector<cv::KeyPoint> kp_src, kp_dst;
    cv::Mat des_src, des_dst;
    orb_detector_->detectAndCompute(cv_src, cv::Mat(), kp_src, des_src);
    orb_detector_->detectAndCompute(cv_dst, cv::Mat(), kp_dst, des_dst);
    if (kp_src.size() == 0 || kp_dst.size() == 0) {
        return pose;
    }

    // Search correspondences.
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(des_src, des_dst, matches);

    std::vector<cv::Point2f> src_pts, dst_pts;
    for (auto& match : matches) {
        dst_pts.push_back(kp_dst[match.trainIdx].pt);
        src_pts.push_back(kp_src[match.queryIdx].pt);
    }

    const double focal_input =
        (config_.camera_intrinsic_.intrinsic_matrix_(0, 0) +
         config_.camera_intrinsic_.intrinsic_matrix_(1, 1)) /
        2.0;
    const cv::Point2d pp =
        cv::Point2d(config_.camera_intrinsic_.intrinsic_matrix_(0, 2),
                    config_.camera_intrinsic_.intrinsic_matrix_(1, 2));

    const size_t kp_size = src_pts.size();
    std::vector<cv::Point2i> src_pts_int, dst_pts_int;
    src_pts_int.reserve(kp_size);
    dst_pts_int.reserve(kp_size);
    for (size_t i = 0; i < kp_size; i++) {
        const auto& src_pt = src_pts[i];
        const auto& dst_pt = dst_pts[i];
        src_pts_int.push_back(cv::Point2i(src_pt.x + 0.5, src_pt.y + 0.5));
        dst_pts_int.push_back(cv::Point2i(dst_pt.x + 0.5, dst_pt.y + 0.5));
    }

    std::vector<int> mask;
    const cv::Mat em =
        cv::findEssentialMat(src_pts_int, dst_pts_int, focal_input, pp,
                             cv::RANSAC, 0.999, 1.0, 1000, mask);
    if (mask.empty()) {
        return pose;
    }

    // Create 3D corresponding points.
    Eigen::Matrix3Xd src_pts_eigen(3, kp_size);
    Eigen::Matrix3Xd dst_pts_eigen(3, kp_size);
    int count = 0;
    for (size_t i = 0; i < kp_size; i++) {
        if (mask[i] == 1) {
            src_pts_eigen.col(count) =
                GetXYZFromUVD(src_pts[i], src.depth_, pp.x, pp.y, focal_input);
            dst_pts_eigen.col(count) =
                GetXYZFromUVD(dst_pts[i], dst.depth_, pp.x, pp.y, focal_input);
            count++;
        }
    }
    src_pts_eigen.conservativeResize(Eigen::NoChange, count);
    dst_pts_eigen.conservativeResize(Eigen::NoChange, count);

    misc3d::registration::TeaserSolver teaser_solver(config_.max_depth_diff_);
    pose = teaser_solver.Solve(src_pts_eigen, dst_pts_eigen);

    return pose;
}

Eigen::Vector3d ReconstructionPipeline::GetXYZFromUVD(
    const cv::Point2f& uv, const open3d::geometry::Image& depth, double cx,
    double cy, double f) {
    const float& u = uv.x;
    const float& v = uv.y;
    const int u0 = (int)u;
    const int v0 = (int)v;
    const int& height = depth.height_;
    const int& width = depth.width_;
    if (u0 > 0 && u0 < width - 1 && v0 > 0 && v0 < height - 1) {
        const float up = u - u0;
        const float vp = v - v0;
        float* d0 = depth.PointerAt<float>(v0, u0);
        float* d1 = depth.PointerAt<float>(v0, u0 + 1);
        float* d2 = depth.PointerAt<float>(v0 + 1, u0);
        float* d3 = depth.PointerAt<float>(v0 + 1, u0 + 1);
        float d = (1.0 - vp) * ((*d1) * up + (*d0) * (1.0 - up)) +
                  vp * ((*d3) * up + (*d2) * (1.0 - up));
        return GetXYZFromUV(u, v, d, cx, cy, f);
    } else {
        return Eigen::Vector3d(0, 0, 0);
    }
}

Eigen::Vector3d ReconstructionPipeline::GetXYZFromUV(int u, int v, double depth,
                                                     double cx, double cy,
                                                     double f) {
    double x, y;
    if (f == 0) {
        x = 0;
        y = 0;
    } else {
        x = (u - cx) * depth / f;
        y = (v - cy) * depth / f;
    }
    return Eigen::Vector3d(x, y, depth);
}

void ReconstructionPipeline::MakeFragments() {
    // Clear and create folder to save results.
    const std::string pose_graph_path = config_.data_path_ + "fragments/";
    if (open3d::utility::filesystem::DirectoryExists(pose_graph_path)) {
        open3d::utility::filesystem::DeleteDirectory(pose_graph_path);
    }
    open3d::utility::filesystem::MakeDirectory(pose_graph_path);

    misc3d::Timer timer;
    timer.Start();
    misc3d::LogInfo("Start Make Fragment.");

    if (!ReadRGBDData()) {
        misc3d::LogInfo("End Make Fragment: {}.", timer.Stop());
        return;
    }

    const size_t num = rgbd_lists_.size();
    n_fragments_ = ceil(static_cast<float>(
        num / config_.make_fragment_param_.n_frame_per_fragment));
    n_keyframes_per_n_frame_ =
        1.0 / config_.make_fragment_param_.keyframe_ratio;
    fragment_pose_graphs_.resize(n_fragments_);
    fragment_point_clouds_.resize(n_fragments_);

    std::vector<std::thread> thread_list;
    thread_list.reserve(n_fragments_);
    for (int i = 0; i < n_fragments_; i++) {
        thread_list.push_back(
            std::thread(&ReconstructionPipeline::BuildSingleFragment, this, i));
    }
    for (auto& thread : thread_list) {
        thread.join();
    }

    misc3d::LogInfo("End Make Fragment: {}.", timer.Stop());
    SaveFragmentResults();
}

}  // namespace reconstruction
}  // namespace misc3d