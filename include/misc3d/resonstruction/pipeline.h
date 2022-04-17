#pragma once

#include <map>
#include <memory>

#include <misc3d/resonstruction/pipeline_config.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/RGBDImage.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/pipelines/registration/Feature.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <opencv2/opencv.hpp>


namespace misc3d {
namespace reconstruction {

class OdometryTrajectory {
public:
    OdometryTrajectory() {}
    ~OdometryTrajectory() {}

public:
    bool WriteToJsonFile(const std::string& file_name);
    bool ReadFromJsonFile(const std::string& file_name);

public:
    std::vector<Eigen::Matrix4d> odomtry_list_;
};

class MatchingResult {
public:
    explicit MatchingResult(int s, int t)
        : s_(s)
        , t_(t)
        , success_(false)
        , transformation_(Eigen::Matrix4d::Identity())
        , information_(Eigen::Matrix6d::Identity()) {}

    virtual ~MatchingResult() {}

public:
    int s_;
    int t_;
    bool success_;
    Eigen::Matrix4d transformation_;
    Eigen::Matrix6d information_;
};

class ReconstructionPipeline {
public:
    /**
     * @brief Construct a new Reconstruction Pipeline given config.
     *
     * @param config
     */
    explicit ReconstructionPipeline(const PipelineConfig& config);

    /**
     * @brief Construct a new Reconstruction Pipeline from file.
     *
     * @param config_file
     */
    ReconstructionPipeline(const std::string& config_file);

    virtual ~ReconstructionPipeline() {}

    /**
     * @brief Make fragments from raw RGBD images.
     * The output will be the fragment point clouds and fragment pose graph.
     *
     */
    void MakeFragments();

    /**
     * @brief Register fragments and compute global odometry.
     * The output will be the global odometry trajectory.
     *
     */
    void RegisterFragments();

    /**
     * @brief Integrate RGBD images with global odometry.
     *  The output will be the integrated triangle mesh of scene.
     */
    void IntegrateScene();

    /**
     * @brief Run the whole pipeline.
     *
     */
    void RunSystem();

    /**
     * @brief Get the Data Path 
     * 
     * @return std::string 
     */
    std::string GetDataPath() const { return config_.data_path_; }

private:
    void CheckConfig();

    bool ReadRGBDData();

    bool ReadFragmentData();

    void ReadJsonPipelineConfig(const std::string& file_name);

    void ComputeKeypointsAndDescriptors(const cv::Mat& img,
                                        std::vector<cv::KeyPoint>& kp,
                                        cv::Mat& des);

    void BuildSingleFragment(int fragment_id);

    void BuildPoseGraphForFragment(int fragment_id, int sid, int eid);

    void BuildPoseGraphForScene();

    void OptimizePoseGraph(
        double max_correspondence_distance, double preference_loop_closure,
        open3d::pipelines::registration::PoseGraph& pose_graph);

    void RefineRegistration();

    void IntegrateFragmentTSDF(int fragment_id);

    void IntegrateRGBDTSDF();

    void SaveFragmentResults();

    void SaveSceneResults();

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> RegisterRGBDPair(int s,
                                                                        int t);

    void RefineFragmentPair(int s, int t, MatchingResult& matched_result);

    void RegisterFragmentPair(int s, int t, MatchingResult& matched_result);

    Eigen::Vector3d GetXYZFromUVD(const cv::Point2f& uv,
                                  const open3d::geometry::Image& depth,
                                  double cx, double cy, double f);

    Eigen::Vector3d GetXYZFromUV(int u, int v, double depth, double cx,
                                 double cy, double f);

    Eigen::Matrix4d PoseEstimation(int s, int t);

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> GlobalRegistration(
        int s, int t);

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> ComputeOdometry(
        int s, int t, const Eigen::Matrix4d& init_trans);

    void PreProcessFragments(const open3d::geometry::PointCloud& pcd, int i);

    std::tuple<Eigen::Matrix4d, Eigen::Matrix6d> MultiScaleICP(
        const open3d::geometry::PointCloud& src,
        const open3d::geometry::PointCloud& dst,
        const std::vector<float>& voxel_size, const std::vector<int>& max_iter,
        const Eigen::Matrix4d& init_trans = Eigen::Matrix4d::Identity());

private:
    PipelineConfig config_;
    std::map<std::string, double> time_cost_table_;

    // Member variables for make fragments.
    std::vector<open3d::geometry::RGBDImage> rgbd_lists_;
    std::vector<open3d::geometry::Image> intensity_img_lists_;
    std::vector<std::tuple<std::vector<cv::KeyPoint>, cv::Mat>> kp_des_lists_;
    std::vector<open3d::pipelines::registration::PoseGraph>
        fragment_pose_graphs_;
    std::vector<open3d::geometry::PointCloud> fragment_point_clouds_;
    int n_fragments_;
    int n_keyframes_per_n_frame_;

    // Member variables for register fragments.
    std::vector<open3d::geometry::PointCloud> preprocessed_fragment_lists_;
    std::vector<open3d::pipelines::registration::Feature> fragment_features_;
    std::vector<MatchingResult> fragment_matching_results_;
    open3d::pipelines::registration::PoseGraph scene_pose_graph_;
    OdometryTrajectory scene_odometry_trajectory_;
};

}  // namespace reconstruction
}  // namespace misc3d