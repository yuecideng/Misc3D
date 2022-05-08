#include <fstream>
#include <thread>
#include <vector>
#include "json.hpp"

#include <misc3d/logging.h>
#include <misc3d/registration/correspondence_matching.h>
#include <misc3d/registration/transform_estimation.h>
#include <misc3d/resonstruction/pipeline.h>
#include <misc3d/utils.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/ImageIO.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/PoseGraphIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/pipelines/integration/ScalableTSDFVolume.h>
#include <open3d/pipelines/odometry/Odometry.h>
#include <open3d/pipelines/registration/ColoredICP.h>
#include <open3d/pipelines/registration/GeneralizedICP.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/t/pipelines/slac/SLACOptimizer.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>

namespace {
void CVMatToEigenMat(const cv::Mat& src, Eigen::MatrixXd& dst) {
    dst.resize(src.cols, src.rows);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            dst(j, i) = src.at<double>(i, j);
        }
    }
}
}  // namespace

namespace misc3d {
namespace reconstruction {

using json = nlohmann::ordered_json;

bool OdometryTrajectory::WriteToJsonFile(const std::string& file_name) {
    json j;
    j["class_name"] = "SceneOdomtryTrajectory";
    const size_t num_poses = odomtry_list_.size();
    for (size_t i = 0; i < num_poses; i++) {
        const std::string id = std::to_string(i);
        std::array<double, 16> arr;
        EigenMat4x4ToArray<double>(odomtry_list_[i], arr);
        j[id] = arr;
    }

    try {
        std::ofstream file(file_name);
        file << j.dump(0) << std::endl;
    } catch (json::other_error& e) {
        misc3d::LogWarning("Failed to write json file: {}", file_name.c_str());
        return false;
    }
    return true;
}

bool OdometryTrajectory::ReadFromJsonFile(const std::string& file_name) {
    try {
        std::ifstream file(file_name);
        json j = json::parse(file);
        if (j["class_name"] != "SceneOdomtryTrajectory") {
            misc3d::LogWarning("Invalid json file: {}", file_name.c_str());
            return false;
        }
        odomtry_list_.clear();
        for (auto& it : j) {
            if (it.is_string())
                continue;

            const std::array<double, 16> arr = it.get<std::array<double, 16>>();
            Eigen::Matrix4d mat;
            ArrayToEigenMat4x4<double>(arr, mat);
            odomtry_list_.push_back(mat);
        }
    } catch (json::other_error& e) {
        misc3d::LogWarning("Failed to read json file: {}", file_name.c_str());
        return false;
    }
    return true;
}

PipelineConfig::PipelineConfig() {
    data_path_ = "";
    depth_scale_ = 1000.0;
    max_depth_ = 3.0;
    max_depth_diff_ = 0.07;
    voxel_size_ = 0.01;
    integration_voxel_size_ = 0.005;
    tsdf_integeation_ = false;
    enable_slac_ = false;
    make_fragment_param_ = {PipelineConfig::DescriptorType::ORB, 100, 40, 0.2};
    local_refine_method_ = LocalRefineMethod::ColoredICP;
    global_registration_method_ = GlobalRegistrationMethod::TeaserPlusPlus;
    optimization_param_ = {0.1, 5.0};
}

ReconstructionPipeline::ReconstructionPipeline(const PipelineConfig& config)
    : config_(config) {
    CheckConfig();
}

ReconstructionPipeline::ReconstructionPipeline(const std::string& file) {
    if (file.substr(file.find_last_of(".") + 1) != "json") {
        misc3d::LogError(
            "Invalid file format, only support json format configuration "
            "file.");
    }

    ReadJsonPipelineConfig(file);
    CheckConfig();
}

void ReconstructionPipeline::CheckConfig() {
    // Unify data path.
    if (config_.data_path_[config_.data_path_.size() - 1] != '/') {
        config_.data_path_ += "/";
    }

    // Validate camera intrinsics.
    if (config_.camera_intrinsic_.width_ <= 0 ||
        config_.camera_intrinsic_.height_ <= 0) {
        misc3d::LogError("Camera intrinsics must be valid.");
    }
}

void ReconstructionPipeline::ReadJsonPipelineConfig(
    const std::string& file_name) {
    try {
        std::ifstream file(file_name);
        json j = json::parse(file);

        config_.data_path_ = j["data_path"];

        // Load camera intrinsics.
        config_.depth_scale_ = j["camera"]["depth_scale"];
        const int width = j["camera"]["width"];
        const int height = j["camera"]["height"];
        const double fx = j["camera"]["fx"];
        const double fy = j["camera"]["fy"];
        const double cx = j["camera"]["cx"];
        const double cy = j["camera"]["cy"];
        config_.camera_intrinsic_ = open3d::camera::PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy);

        if (j.contains("make_fragments")) {
            if (j["make_fragments"].contains("descriptor_type")) {
                const std::string descriptor_type =
                    j["make_fragments"]["descriptor_type"];
                if (descriptor_type == "orb") {
                    config_.make_fragment_param_.descriptor_type =
                        PipelineConfig::DescriptorType::ORB;
                } else if (descriptor_type == "sift") {
                    config_.make_fragment_param_.descriptor_type =
                        PipelineConfig::DescriptorType::SIFT;
                } else {
                    misc3d::LogError("Invalid descriptor type: {}",
                                     descriptor_type.c_str());
                }
            }

            if (j["make_fragments"].contains("feature_num")) {
                config_.make_fragment_param_.feature_num =
                    j["make_fragments"]["feature_num"];
            }
            if (j["make_fragments"].contains("n_frame_per_fragment")) {
                config_.make_fragment_param_.n_frame_per_fragment =
                    j["make_fragments"]["n_frame_per_fragment"];
            }
            if (j["make_fragments"].contains("keyframe_ratio")) {
                config_.make_fragment_param_.keyframe_ratio =
                    j["make_fragments"]["keyframe_ratio"];
            }
        }

        if (j.contains("local_refine")) {
            if (j["local_refine"] == "point2point") {
                config_.local_refine_method_ =
                    PipelineConfig::LocalRefineMethod::Point2PointICP;
            } else if (j["local_refine"] == "point2plane") {
                config_.local_refine_method_ =
                    PipelineConfig::LocalRefineMethod::Point2PlaneICP;
            } else if (j["local_refine"] == "color") {
                config_.local_refine_method_ =
                    PipelineConfig::LocalRefineMethod::ColoredICP;
            } else if (j["local_refine"] == "generalized") {
                config_.local_refine_method_ =
                    PipelineConfig::LocalRefineMethod::GeneralizedICP;
            } else {
                misc3d::LogError("Invalid local refine method.");
            }
        }

        if (j.contains("global_registration")) {
            if (j["global_registration"] == "ransac") {
                config_.global_registration_method_ =
                    PipelineConfig::GlobalRegistrationMethod::Ransac;
            } else if (j["global_registration"] == "teaser") {
                config_.global_registration_method_ =
                    PipelineConfig::GlobalRegistrationMethod::TeaserPlusPlus;
            } else {
                misc3d::LogError("Invalid global registration method.");
            }
        }

        if (j.contains("optimization_param")) {
            if (j["optimization_param"].contains(
                    "preference_loop_closure_odometry")) {
                config_.optimization_param_.preference_loop_closure_odometry =
                    j["optimization_param"]["preference_loop_closure_odometry"];
            }
            if (j["optimization_param"].contains(
                    "preference_loop_closure_registration")) {
                config_.optimization_param_
                    .preference_loop_closure_registration =
                    j["optimization_param"]
                     ["preference_loop_closure_registration"];
            }
        }

        if (j.contains("voxel_size")) {
            config_.voxel_size_ = j["voxel_size"];
        }

        if (j.contains("max_depth")) {
            config_.max_depth_ = j["max_depth"];
        }

        if (j.contains("max_depth_diff")) {
            config_.max_depth_diff_ = j["max_depth_diff"];
        }

        if (j.contains("integration_voxel_size")) {
            config_.integration_voxel_size_ = j["integration_voxel_size"];
        }

        if (j.contains("tsdf_integeation")) {
            config_.tsdf_integeation_ = j["tsdf_integeation"];
        }

        if (j.contains("enable_slac")) {
            config_.enable_slac_ = j["enable_slac"];
        }

    } catch (json::other_error& e) {
        misc3d::LogError("Failed to read json file.");
    }
}

bool ReconstructionPipeline::ReadRGBDData() {
    const std::string& data_path = config_.data_path_;
    const std::string color_path = data_path + "color";
    const std::string depth_path = data_path + "depth";
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

    // Color image can be stored in `jpg` format.
    if (color_files.size() == 0) {
        open3d::utility::filesystem::ListFilesInDirectoryWithExtension(
            color_path, "jpg", color_files);
    }

    if (color_files.size() != depth_files.size()) {
        misc3d::LogWarning(
            "Number of color {} and depth {} images are not equal.",
            color_files.size(), depth_files.size());
        return false;
    }

    misc3d::LogInfo("Found {} RGBD images.", color_files.size());
    rgbd_lists_.resize(color_files.size());
    intensity_img_lists_.resize(color_files.size());
    kp_des_lists_.resize(color_files.size());
    const int& width = config_.camera_intrinsic_.width_;
    const int& heigth = config_.camera_intrinsic_.height_;

    // TODO: The 2D image keypoints and descriptors can be extended into
    // multiple methods selection scheme (SIFT, SURF, ORB, etc.).

#pragma omp parallel for schedule(static)
    for (int i = 0; i < color_files.size(); ++i) {
        // Read color image and depth image.
        open3d::geometry::Image color, depth;
        open3d::io::ReadImage(color_files[i], color);
        open3d::io::ReadImage(depth_files[i], depth);

        // Create RGBD image.
        const auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            color, depth, config_.depth_scale_, config_.max_depth_, false);
        rgbd_lists_[i] = *rgbd;
        const auto intensity_img = color.CreateFloatImage();
        intensity_img_lists_[i] = *intensity_img;

        // Detect ORB keypoints and descriptors.
        cv::Mat cv_img(cv::Size(width, heigth), CV_8UC3, color.data_.data());
        cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
        std::vector<cv::KeyPoint> kp;
        cv::Mat des;
        ComputeKeypointsAndDescriptors(cv_img, kp, des);
        kp_des_lists_[i] = std::make_pair(kp, des);
    }

    return true;
}

void ReconstructionPipeline::ComputeKeypointsAndDescriptors(
    const cv::Mat& img, std::vector<cv::KeyPoint>& kp, cv::Mat& des) {
    if (config_.make_fragment_param_.descriptor_type ==
        PipelineConfig::DescriptorType::ORB) {
        cv::Ptr<cv::ORB> orb_detector =
            cv::ORB::create(config_.make_fragment_param_.feature_num);
        orb_detector->detectAndCompute(img, cv::Mat(), kp, des);
    } else if (config_.make_fragment_param_.descriptor_type ==
               PipelineConfig::DescriptorType::SIFT) {
        cv::Ptr<cv::SIFT> sift_detector =
            cv::SIFT::create(config_.make_fragment_param_.feature_num);
        sift_detector->detectAndCompute(img, cv::Mat(), kp, des);
    } else {
        misc3d::LogError("Invalid descriptor type.");
    }
}

bool ReconstructionPipeline::ReadFragmentData() {
    const std::string& data_path = config_.data_path_;
    const std::string fragments_path = data_path + "fragments";

    misc3d::LogInfo("Reading Fragments data from {}", fragments_path.c_str());
    if (!open3d::utility::filesystem::DirectoryExists(fragments_path)) {
        misc3d::LogWarning("Fragment data path does not exist.");
        return false;
    }

    std::vector<std::string> fragment_files, pose_graph_files;
    if (!open3d::utility::filesystem::ListFilesInDirectoryWithExtension(
            fragments_path, "ply", fragment_files) ||
        !open3d::utility::filesystem::ListFilesInDirectoryWithExtension(
            fragments_path, "json", pose_graph_files)) {
        misc3d::LogWarning("Failed to read Fragments data.");
        return false;
    }

    n_fragments_ = fragment_files.size();
    // Read fragment point clouds.
    fragment_features_.resize(n_fragments_);
    preprocessed_fragment_lists_.resize(n_fragments_);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_fragments_; i++) {
        open3d::geometry::PointCloud pcd;
        open3d::io::ReadPointCloud(fragment_files[i], pcd);
        misc3d::LogInfo("Preprocessing fragment {}.", i);
        PreProcessFragments(pcd, i);
    }

    // Read fragment pose graph.
    fragment_pose_graphs_.resize(n_fragments_);
    for (size_t i = 0; i < n_fragments_; i++) {
        open3d::io::ReadPoseGraph(pose_graph_files[i],
                                  fragment_pose_graphs_[i]);
    }

    return true;
}

void ReconstructionPipeline::PreProcessFragments(
    open3d::geometry::PointCloud& pcd, int i) {
    if (!pcd.HasNormals()) {
        pcd.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
            config_.voxel_size_ * 2, 30));
    }
    pcd.OrientNormalsTowardsCameraLocation();

    const auto fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
        pcd, open3d::geometry::KDTreeSearchParamHybrid(
                       config_.voxel_size_ * 5, 100));
    if (fragment_features_.size() != n_fragments_ ||
        preprocessed_fragment_lists_.size() != n_fragments_) {
        misc3d::LogError(
            "Fragment features size {} or fragment lists size {} "
            "is not equal to n_fragments {}.",
            fragment_features_.size(), preprocessed_fragment_lists_.size(),
            n_fragments_);
        return;
    }
    fragment_features_[i] = *fpfh;
    preprocessed_fragment_lists_[i] = pcd;
}

void ReconstructionPipeline::BuildSingleFragment(int fragment_id) {
    const int sid =
        fragment_id * config_.make_fragment_param_.n_frame_per_fragment;
    const int eid =
        std::min(sid + config_.make_fragment_param_.n_frame_per_fragment,
                 static_cast<int>(rgbd_lists_.size()));
    BuildPoseGraphForFragment(fragment_id, sid, eid);
    OptimizePoseGraph(
        config_.max_depth_diff_,
        config_.optimization_param_.preference_loop_closure_odometry,
        fragment_pose_graphs_[fragment_id]);
    IntegrateFragmentRGBD(fragment_id);
}

void ReconstructionPipeline::BuildPoseGraphForScene() {
    Eigen::Matrix4d odom = Eigen::Matrix4d::Identity();
    scene_pose_graph_.nodes_.push_back(
        open3d::pipelines::registration::PoseGraphNode(odom));

    for (int i = 0; i < n_fragments_; i++) {
        for (int j = i + 1; j < n_fragments_; j++) {
            fragment_matching_results_.push_back(MatchingResult(i, j));
        }
    }

    const size_t num_pairs = fragment_matching_results_.size();
    std::vector<std::thread> thread_list;
    for (size_t i = 0; i < num_pairs; i++) {
        MatchingResult& matching_result = fragment_matching_results_[i];
        const int s = matching_result.s_;
        const int t = matching_result.t_;
        thread_list.push_back(
            std::thread(&ReconstructionPipeline::RegisterFragmentPair, this, s,
                        t, std::ref(matching_result)));
    }
    for (auto& thread : thread_list) {
        thread.join();
    }

    for (size_t i = 0; i < num_pairs; i++) {
        if (fragment_matching_results_[i].success_) {
            const int& t = fragment_matching_results_[i].t_;
            const int& s = fragment_matching_results_[i].s_;
            const Eigen::Matrix4d& pose =
                fragment_matching_results_[i].transformation_;
            const Eigen::Matrix6d info =
                fragment_matching_results_[i].information_;
            if (s + 1 == t) {
                odom = pose * odom;
                const Eigen::Matrix4d& odom_inv = odom.inverse();
                scene_pose_graph_.nodes_.push_back(
                    open3d::pipelines::registration::PoseGraphNode(odom_inv));
                scene_pose_graph_.edges_.push_back(
                    open3d::pipelines::registration::PoseGraphEdge(
                        s, t, pose, info, false));
            } else {
                scene_pose_graph_.edges_.push_back(
                    open3d::pipelines::registration::PoseGraphEdge(s, t, pose,
                                                                   info, true));
            }
        }
    }
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
                    "Fragment {:03d} / {:03d} :: RGBD odometry between "
                    "frame : "
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

void ReconstructionPipeline::OptimizePoseGraph(
    double max_correspondence_distance, double preference_loop_closure,
    open3d::pipelines::registration::PoseGraph& pose_graph) {
    open3d::pipelines::registration::GlobalOptimizationOption option(
        max_correspondence_distance, 0.25, preference_loop_closure, 0);

    open3d::pipelines::registration::GlobalOptimization(
        pose_graph,
        open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
        open3d::pipelines::registration::
            GlobalOptimizationConvergenceCriteria(),
        option);
}

void ReconstructionPipeline::IntegrateFragmentRGBD(int fragment_id) {
    open3d::geometry::PointCloud fragment;
    const auto& pose_graph = fragment_pose_graphs_[fragment_id];
    const size_t graph_num = pose_graph.nodes_.size();
#pragma omp parallel for
    for (int i = 0; i < int(graph_num); ++i) {
        const int i_abs =
            fragment_id * config_.make_fragment_param_.n_frame_per_fragment + i;
        misc3d::LogInfo(
            "Fragment {:03d} / {:03d} :: Integrate rgbd frame {:d} ({:d} "
            "of "
            "{:d}).",
            fragment_id, n_fragments_ - 1, i_abs, i + 1, graph_num);
        const open3d::geometry::RGBDImage& rgbd = rgbd_lists_[i_abs];
        auto pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(
            rgbd, config_.camera_intrinsic_, Eigen::Matrix4d::Identity(), true);
        pcd->Transform(pose_graph.nodes_[i].pose_);
#pragma omp critical
        { fragment += *pcd; }
    }

    fragment_point_clouds_[fragment_id] =
        *fragment.VoxelDownSample(config_.voxel_size_);
}

void ReconstructionPipeline::IntegrateSceneRGBDTSDF() {
    open3d::pipelines::integration::ScalableTSDFVolume volume(
        config_.integration_voxel_size_, 0.04,
        open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
    const size_t num = rgbd_lists_.size();
    for (size_t i = 0; i < num; i++) {
        misc3d::LogInfo("Scene :: Integrate rgbd frame {} | {}", i, num);
        volume.Integrate(rgbd_lists_[i], config_.camera_intrinsic_,
                         scene_odometry_trajectory_.odomtry_list_[i].inverse());
    }

    const auto mesh = volume.ExtractTriangleMesh();
    mesh->ComputeVertexNormals();

    open3d::io::WriteTriangleMesh(config_.data_path_ + "scene/integrated.ply",
                                  *mesh);
}

void ReconstructionPipeline::IntegrateSceneRGBD() {
    open3d::geometry::PointCloud scene;
    const size_t num = rgbd_lists_.size();
#pragma omp parallel for
    for (int i = 0; i < int(num); i++) {
        misc3d::LogInfo("Scene :: Integrate rgbd frame {} | {}", i, num);
        const open3d::geometry::RGBDImage& rgbd = rgbd_lists_[i];
        auto pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(
            rgbd, config_.camera_intrinsic_, Eigen::Matrix4d::Identity(), true);
        pcd->Transform(scene_odometry_trajectory_.odomtry_list_[i]);
#pragma omp critical
        { scene += *pcd; }
    }

    const auto scene_down =
        scene.VoxelDownSample(config_.integration_voxel_size_);
    open3d::io::WritePointCloud(config_.data_path_ + "scene/integrated.ply",
                                *scene_down);
}

void ReconstructionPipeline::RefineRegistration() {
    misc3d::LogInfo("Start Refine Registration.");
    misc3d::Timer timer;
    timer.Start();

    // Clear matching results.
    fragment_matching_results_.clear();

    for (auto& edge : scene_pose_graph_.edges_) {
        const int s = edge.source_node_id_;
        const int t = edge.target_node_id_;
        MatchingResult mr(s, t);
        mr.transformation_ = edge.transformation_;
        fragment_matching_results_.push_back(mr);
    }

    std::vector<std::thread> thread_list;
    for (size_t i = 0; i < fragment_matching_results_.size(); i++) {
        const int s = fragment_matching_results_[i].s_;
        const int t = fragment_matching_results_[i].t_;
        thread_list.push_back(
            std::thread(&ReconstructionPipeline::RefineFragmentPair, this, s, t,
                        std::ref(fragment_matching_results_[i])));
    }
    for (auto& thread : thread_list) {
        thread.join();
    }

    // Update scene pose graph.
    scene_pose_graph_.edges_.clear();
    scene_pose_graph_.nodes_.clear();
    Eigen::Matrix4d odom = Eigen::Matrix4d::Identity();
    scene_pose_graph_.nodes_.push_back(
        open3d::pipelines::registration::PoseGraphNode(odom));
    for (auto& result : fragment_matching_results_) {
        const int s = result.s_;
        const int t = result.t_;
        const Eigen::Matrix4d& pose = result.transformation_;
        const Eigen::Matrix6d& info = result.information_;

        if (s + 1 == t) {
            odom = pose * odom;
            scene_pose_graph_.nodes_.push_back(
                open3d::pipelines::registration::PoseGraphNode(odom.inverse()));
            scene_pose_graph_.edges_.push_back(
                open3d::pipelines::registration::PoseGraphEdge(s, t, pose, info,
                                                               false));
        } else {
            scene_pose_graph_.edges_.push_back(
                open3d::pipelines::registration::PoseGraphEdge(s, t, pose, info,
                                                               true));
        }
    }

    OptimizePoseGraph(
        config_.voxel_size_ * 1.4,
        config_.optimization_param_.preference_loop_closure_registration,
        scene_pose_graph_);

    if (config_.enable_slac_) {
        SLACOptimization();
    }

    time_cost_table_["RefineRegistration"] = timer.Stop();
    misc3d::LogInfo("End Refine Registration: {}",
                    time_cost_table_.at("RefineRegistration"));
}

void ReconstructionPipeline::SLACOptimization() {
    misc3d::LogInfo("Start SLAC Optimization.");
    misc3d::Timer timer;
    timer.Start();

    // Setup SLAC params.
    open3d::t::pipelines::slac::SLACOptimizerParams params(
        5, config_.voxel_size_, 0.07, 0.3, 1.0, open3d::core::Device("CPU:0"));
    open3d::t::pipelines::slac::SLACDebugOption option(false, 0);

    std::vector<std::string> fragment_names;
    if (!open3d::utility::filesystem::ListFilesInDirectoryWithExtension(
            config_.data_path_ + "fragments", "ply", fragment_names)) {
        misc3d::LogWarning("Failed to read Fragments data.");
        misc3d::LogInfo("End SLAC Optimization: {}", timer.Stop());
        return;
    }

    const auto result =
        open3d::t::pipelines::slac::RunSLACOptimizerForFragments(
            fragment_names, scene_pose_graph_, params, option);

    // Update scene pose graph.
    scene_pose_graph_ = std::get<0>(result);

    misc3d::LogInfo("End SLAC Optimization: {}", timer.Stop());
}

void ReconstructionPipeline::RefineFragmentPair(
    int s, int t, MatchingResult& matched_result) {
    const auto& pcd_s = preprocessed_fragment_lists_[s];
    const auto& pcd_t = preprocessed_fragment_lists_[t];
    const float voxel_size = config_.voxel_size_;
    const auto& init_trans = matched_result.transformation_;
    const auto result = MultiScaleICP(
        pcd_s, pcd_t, {voxel_size, voxel_size / 2, voxel_size / 4},
        {50, 30, 15}, init_trans);
    matched_result.transformation_ = std::get<0>(result);
    matched_result.information_ = std::get<1>(result);
}

void ReconstructionPipeline::SaveFragmentResults() {
    const std::string fragments_path = config_.data_path_ + "fragments/";
    // Save fragment pose graph and point clouds.
    for (int i = 0; i < n_fragments_; i++) {
        std::string id = std::to_string(i);
        if (id.size() != 3) {
            const int size = 3 - id.size();
            for (size_t i = 0; i < size; i++) {
                id = "0" + id;
            }
        }
        const std::string file_name = fragments_path + "fragment_" + id;
        open3d::io::WritePoseGraph(file_name + ".json",
                                   fragment_pose_graphs_[i]);
        open3d::io::WritePointCloud(file_name + ".ply",
                                    fragment_point_clouds_[i]);
    }
}

void ReconstructionPipeline::SaveSceneResults() {
    for (size_t i = 0; i < n_fragments_; i++) {
        const auto& fragment_pose_graph = fragment_pose_graphs_[i];
        for (size_t j = 0; j < fragment_pose_graph.nodes_.size(); j++) {
            const Eigen::Matrix4d odom = scene_pose_graph_.nodes_[i].pose_ *
                                         fragment_pose_graph.nodes_[j].pose_;
            scene_odometry_trajectory_.odomtry_list_.push_back(odom);
        }
    }
    bool ret = scene_odometry_trajectory_.WriteToJsonFile(
        config_.data_path_ + "scene/trajectory.json");
}

std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
ReconstructionPipeline::RegisterRGBDPair(int s, int t) {
    if (abs(s - t) != 1) {
        Eigen::Matrix4d odo_init = PoseEstimation(s, t);
        if (!odo_init.isIdentity(1e-8)) {
            return ComputeOdometry(s, t, odo_init);
        } else {
            return std::make_tuple(false, Eigen::Matrix4d::Identity(),
                                   Eigen::Matrix6d::Identity());
        }
    } else {
        return ComputeOdometry(s, t, Eigen::Matrix4d::Identity());
    }
}

void ReconstructionPipeline::RegisterFragmentPair(
    int s, int t, MatchingResult& matched_result) {
    const open3d::geometry::PointCloud& pcd_s = preprocessed_fragment_lists_[s];
    const open3d::geometry::PointCloud& pcd_t = preprocessed_fragment_lists_[t];
    Eigen::Matrix4d pose;
    Eigen::Matrix6d info;

    // Odometry estimation.
    if (s + 1 == t) {
        misc3d::LogInfo("Fragment odometry {} and {}", s, t);
        const auto& pose_graph_frag = fragment_pose_graphs_[s];
        const int n_nodes = pose_graph_frag.nodes_.size();
        const Eigen::Matrix4d init_trans =
            pose_graph_frag.nodes_[n_nodes - 1].pose_.inverse();
        const auto result = MultiScaleICP(pcd_s, pcd_t, {config_.voxel_size_},
                                          {50}, init_trans);
        pose = std::get<0>(result);
        info = std::get<1>(result);
    } else {
        // Loop closure estimation.
        misc3d::LogInfo("Fragment loop closure {} and {}", s, t);
        const auto result = GlobalRegistration(s, t);
        const bool success = std::get<0>(result);
        if (!success) {
            misc3d::LogWarning(
                "Global registration failed. Skip pair ({} | {}).", s, t);
            matched_result.success_ = false;
            matched_result.transformation_ = Eigen::Matrix4d::Identity();
            matched_result.information_ = Eigen::Matrix6d::Identity();
        } else {
            pose = std::get<1>(result);
            info = std::get<2>(result);
        }
    }

    matched_result.success_ = true;
    matched_result.transformation_ = pose;
    matched_result.information_ = info;
}

// TODO: Correspondence matching can also be done by using 2d features like
// ORB, SIFT and SURF. Just store these features in 'Make Fragment' stage
// for each RGBD and integrate them into the fragment point clouds.
std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
ReconstructionPipeline::GlobalRegistration(int s, int t) {
    const auto& pcd_s = preprocessed_fragment_lists_[s];
    const auto& pcd_t = preprocessed_fragment_lists_[t];
    const auto& fpfh_s = fragment_features_[s];
    const auto& fpfh_t = fragment_features_[t];
    const double max_dis = config_.voxel_size_ * 1.4;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    // Match correspondences.
    misc3d::registration::ANNMatcher matcher(
        misc3d::registration::MatchMethod::ANNOY);
    const auto matched_list = matcher.Match(fpfh_s, fpfh_t);

    if (config_.global_registration_method_ ==
        PipelineConfig::GlobalRegistrationMethod::Ransac) {
        misc3d::registration::RANSACSolver solver(max_dis);
        pose = solver.Solve(pcd_s, pcd_t, matched_list);
    } else if (config_.global_registration_method_ ==
               PipelineConfig::GlobalRegistrationMethod::TeaserPlusPlus) {
        misc3d::registration::TeaserSolver solver(config_.voxel_size_ * 3);
        const auto pcd_s_ = pcd_s.SelectByIndex(matched_list.first);
        const auto pcd_t_ = pcd_t.SelectByIndex(matched_list.second);
        pose = solver.Solve(*pcd_s_, *pcd_t_);
    }

    if (pose.isIdentity(1e-8)) {
        return std::make_tuple(true, pose, Eigen::Matrix6d::Identity());
    }

    const Eigen::Matrix6d info =
        open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
            pcd_s, pcd_t, max_dis, pose);
    if (info(5, 5) / std::min(pcd_s.points_.size(), pcd_t.points_.size()) <
        0.3) {
        return std::make_tuple(false, pose, Eigen::Matrix6d::Identity());
    }
    return std::make_tuple(true, pose, info);
}

std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
ReconstructionPipeline::ComputeOdometry(int s, int t,
                                        const Eigen::Matrix4d& init_trans) {
    open3d::geometry::RGBDImage& rgbd_s = rgbd_lists_[s];
    open3d::geometry::RGBDImage& rgbd_t = rgbd_lists_[t];

    open3d::geometry::RGBDImage new_rgbd_s(intensity_img_lists_[s],
                                           rgbd_s.depth_);
    open3d::geometry::RGBDImage new_rgbd_t(intensity_img_lists_[t],
                                           rgbd_t.depth_);

    open3d::pipelines::odometry::OdometryOption option;
    option.max_depth_diff_ = config_.max_depth_diff_;

    return open3d::pipelines::odometry::ComputeRGBDOdometry(
        new_rgbd_s, new_rgbd_t, config_.camera_intrinsic_, init_trans,
        open3d::pipelines::odometry::RGBDOdometryJacobianFromHybridTerm(),
        option);
}

Eigen::Matrix4d ReconstructionPipeline::PoseEstimation(int s, int t) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    // Obtain ORB features from stored lists.
    const auto& kp_src = std::get<0>(kp_des_lists_[s]);
    const auto& kp_dst = std::get<0>(kp_des_lists_[t]);
    auto& des_src = std::get<1>(kp_des_lists_[s]);
    auto& des_dst = std::get<1>(kp_des_lists_[t]);

    if (kp_src.size() == 0 || kp_dst.size() == 0) {
        return pose;
    }

    // Search correspondences.
    std::vector<cv::Point2f> src_pts, dst_pts;
    if (config_.make_fragment_param_.descriptor_type ==
        PipelineConfig::DescriptorType::ORB) {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(des_src, des_dst, matches);

        for (auto& match : matches) {
            dst_pts.push_back(kp_dst[match.trainIdx].pt);
            src_pts.push_back(kp_src[match.queryIdx].pt);
        }
    } else if (config_.make_fragment_param_.descriptor_type ==
               PipelineConfig::DescriptorType::SIFT) {
        // Convert mat to eigen matrix.
        Eigen::MatrixXd des_src_eigen, des_dst_eigen;
        CVMatToEigenMat(des_src, des_src_eigen);
        CVMatToEigenMat(des_dst, des_dst_eigen);

        misc3d::registration::ANNMatcher matcher(
            misc3d::registration::MatchMethod::ANNOY);

        const auto matches = matcher.Match(des_src_eigen, des_dst_eigen);
        const auto& index1 = matches.first;
        const auto& index2 = matches.second;
        for (size_t i = 0; i < index1.size(); i++) {
            src_pts.push_back(kp_src[index1[i]].pt);
            dst_pts.push_back(kp_dst[index2[i]].pt);
        }
    }

    // Number of correspondences must be greater than 3.
    if (dst_pts.size() < 3) {
        return pose;
    }

    const size_t kp_size = src_pts.size();

    const double focal_input =
        (config_.camera_intrinsic_.intrinsic_matrix_(0, 0) +
         config_.camera_intrinsic_.intrinsic_matrix_(1, 1)) /
        2.0;
    const cv::Point2d pp =
        cv::Point2d(config_.camera_intrinsic_.intrinsic_matrix_(0, 2),
                    config_.camera_intrinsic_.intrinsic_matrix_(1, 2));

    // Create 3D corresponding points.
    Eigen::Matrix3Xd src_pts_eigen(3, kp_size);
    Eigen::Matrix3Xd dst_pts_eigen(3, kp_size);
    const auto& src = rgbd_lists_[s];
    const auto& dst = rgbd_lists_[t];
    for (int i = 0; i < kp_size; i++) {
        src_pts_eigen.col(i) =
            GetXYZFromUVD(src_pts[i], src.depth_, pp.x, pp.y, focal_input);
        dst_pts_eigen.col(i) =
            GetXYZFromUVD(dst_pts[i], dst.depth_, pp.x, pp.y, focal_input);
    }

    misc3d::registration::TeaserSolver teaser_solver(config_.max_depth_diff_);
    pose = teaser_solver.Solve(src_pts_eigen, dst_pts_eigen);
    return pose;
}

std::tuple<Eigen::Matrix4d, Eigen::Matrix6d>
ReconstructionPipeline::MultiScaleICP(const open3d::geometry::PointCloud& src,
                                      const open3d::geometry::PointCloud& dst,
                                      const std::vector<float>& voxel_size,
                                      const std::vector<int>& max_iter,
                                      const Eigen::Matrix4d& init_trans) {
    Eigen::Matrix4d current = init_trans;
    Eigen::Matrix6d info;
    const size_t num_scale = voxel_size.size();
    for (size_t i = 0; i < num_scale; i++) {
        const double max_dis = config_.voxel_size_ * 1.4;
        const auto src_down = src.VoxelDownSample(voxel_size[i]);
        const auto dst_down = dst.VoxelDownSample(voxel_size[i]);
        const open3d::pipelines::registration::ICPConvergenceCriteria criteria(
            1e-6, 1e-6, max_iter[i]);
        open3d::pipelines::registration::RegistrationResult result;
        if (config_.local_refine_method_ ==
            PipelineConfig::LocalRefineMethod::Point2PointICP) {
            result = open3d::pipelines::registration::RegistrationICP(
                *src_down, *dst_down, max_dis, current,
                open3d::pipelines::registration::
                    TransformationEstimationPointToPoint(),
                criteria);
        } else if (config_.local_refine_method_ ==
                   PipelineConfig::LocalRefineMethod::Point2PlaneICP) {
            result = open3d::pipelines::registration::RegistrationICP(
                *src_down, *dst_down, max_dis, current,
                open3d::pipelines::registration::
                    TransformationEstimationPointToPlane(),
                criteria);
        } else if (config_.local_refine_method_ ==
                   PipelineConfig::LocalRefineMethod::ColoredICP) {
            result = open3d::pipelines::registration::RegistrationColoredICP(
                *src_down, *dst_down, max_dis, current,
                open3d::pipelines::registration::
                    TransformationEstimationForColoredICP(),
                criteria);
        } else if (config_.local_refine_method_ ==
                   PipelineConfig::LocalRefineMethod::GeneralizedICP) {
            result =
                open3d::pipelines::registration::RegistrationGeneralizedICP(
                    *src_down, *dst_down, max_dis, current,
                    open3d::pipelines::registration::
                        TransformationEstimationForGeneralizedICP(),
                    criteria);
        } else {
            misc3d::LogError("Unknown local refine method.");
        }
        current = result.transformation_;
        if (i == num_scale - 1) {
            info = open3d::pipelines::registration::
                GetInformationMatrixFromPointClouds(
                    src, dst, voxel_size[i] * 1.4, current);
        }
    }
    return std::make_tuple(current, info);
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
        float* d0 = depth.PointerAt<float>(u0, v0);
        float* d1 = depth.PointerAt<float>(u0, v0 + 1);
        float* d2 = depth.PointerAt<float>(u0 + 1, v0);
        float* d3 = depth.PointerAt<float>(u0 + 1, v0 + 1);
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
    const std::string fragments_path = config_.data_path_ + "fragments/";
    if (open3d::utility::filesystem::DirectoryExists(fragments_path)) {
        open3d::utility::filesystem::DeleteDirectory(fragments_path);
    }
    open3d::utility::filesystem::MakeDirectory(fragments_path);

    misc3d::Timer timer;
    timer.Start();
    misc3d::LogInfo("Start Make Fragment.");

    if (!ReadRGBDData()) {
        misc3d::LogInfo("End Make Fragment: {}.", timer.Stop());
        return;
    }

    const size_t num = rgbd_lists_.size();
    n_fragments_ =
        ceil(static_cast<float>(num) /
             (float)config_.make_fragment_param_.n_frame_per_fragment);
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

    time_cost_table_["MakeFragment"] = timer.Stop();
    misc3d::LogInfo("End Make Fragment: {}.",
                    time_cost_table_.at("MakeFragment"));
    SaveFragmentResults();
}

void ReconstructionPipeline::RegisterFragments() {
    // Clear and create folder to save results.
    const std::string scene_path = config_.data_path_ + "scene/";
    if (open3d::utility::filesystem::DirectoryExists(scene_path)) {
        open3d::utility::filesystem::DeleteDirectory(scene_path);
    }
    open3d::utility::filesystem::MakeDirectory(scene_path);

    misc3d::Timer timer;
    timer.Start();
    misc3d::LogInfo("Start Register Fragments.");

    if (!ReadFragmentData()) {
        misc3d::LogInfo("End Register Fragments: {}.", timer.Stop());
        return;
    }

    BuildPoseGraphForScene();

    OptimizePoseGraph(
        config_.voxel_size_ * 1.4,
        config_.optimization_param_.preference_loop_closure_registration,
        scene_pose_graph_);

    // Perform refinement.
    RefineRegistration();

    // Save optimal results.
    SaveSceneResults();

    time_cost_table_["RegisterFragments"] = timer.Stop();
    misc3d::LogInfo("End Register Fragments: {}",
                    time_cost_table_.at("RegisterFragments"));
}

void ReconstructionPipeline::IntegrateScene() {
    misc3d::Timer timer;
    timer.Start();
    misc3d::LogInfo("Start Integrate Scene.");

    // Read raw RGBD data.
    if (rgbd_lists_.size() == 0) {
        if (!ReadRGBDData()) {
            misc3d::LogInfo("End Integrate Scene: {}.", timer.Stop());
            return;
        }
    }

    // Read scene odometry.
    if (scene_odometry_trajectory_.odomtry_list_.size() == 0) {
        const std::string name = config_.data_path_ + "scene/trajectory.json";
        if (!scene_odometry_trajectory_.ReadFromJsonFile(name)) {
            misc3d::LogInfo("End Integrate Scene: {}.", timer.Stop());
            return;
        }
    }

    if (config_.tsdf_integeation_) {
        IntegrateSceneRGBDTSDF();
    } else {
        IntegrateSceneRGBD();
    }

    time_cost_table_["IntegrateScene"] = timer.Stop();
    misc3d::LogInfo("End Integrate Scene: {}",
                    time_cost_table_.at("IntegrateScene"));
}

void ReconstructionPipeline::RunSystem() {
    misc3d::LogInfo("Start Reconstruction Pipeline system.");
    MakeFragments();
    RegisterFragments();
    IntegrateScene();

    misc3d::LogInfo("End Reconstruction Pipeline system.");
    misc3d::LogInfo("----------------------------------");
    for (auto& item : time_cost_table_) {
        std::string time;
        time = std::to_string(int(item.second / 60)) + ":" +
               std::to_string(int(item.second) % 60);
        misc3d::LogInfo("{}: {}", item.first.c_str(), time);
    }
}

}  // namespace reconstruction
}  // namespace misc3d