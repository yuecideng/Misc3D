#pragma once

#include <memory>

#include <misc3d/resonstruction/pipeline_config.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/RGBDImage.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


namespace misc3d {
namespace reconstruction {

class ReconstructionPipeline {
public:
    explicit ReconstructionPipeline(const PipelineConfig& config);
    virtual ~ReconstructionPipeline() {}

    void MakeFragments();

    void RegisterFragments();

    void Optimize();

    void Integrate();

    void RunSystem();

private:
    bool ReadRGBDData();

    void BuildSingleFragment(int fragment_id);

    void BuildPoseGraphForFragment(int fragment_id, int sid, int eid);

    void OptimizePoseGraphForFragment(int fragment_id);

    void IntegrateFragmentTSDF(int fragment_id);

    void SaveFragmentResults();

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> RegisterRGBDPair(int s,
                                                                        int t);

    Eigen::Vector3d GetXYZFromUVD(const cv::Point2f& uv,
                                  const open3d::geometry::Image& depth,
                                  double cx, double cy, double f);

    Eigen::Vector3d GetXYZFromUV(int u, int v, double depth, double cx,
                                 double cy, double f);

    Eigen::Matrix4d PoseEstimation(const open3d::geometry::RGBDImage& src,
                                   const open3d::geometry::RGBDImage& dst);

private:
    PipelineConfig config_;
    std::vector<open3d::geometry::RGBDImage> rgbd_lists_;
    std::vector<open3d::pipelines::registration::PoseGraph>
        fragment_pose_graphs_;
    std::vector<open3d::geometry::PointCloud> fragment_point_clouds_;
    int n_fragments_;
    int n_keyframes_per_n_frame_;

    cv::Ptr<cv::ORB> orb_detector_;
};

}  // namespace reconstruction
}  // namespace misc3d