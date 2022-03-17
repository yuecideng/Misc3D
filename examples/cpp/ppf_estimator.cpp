#include <iostream>
#include <memory>
#include <vector>

#include <misc3d/pose_estimation/ppf_estimation.h>
#include <misc3d/preprocessing/filter.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/Image.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/RGBDImage.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/ImageIO.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/pipelines/registration/Registration.h>

int main(int argc, char *argv[]) {
    misc3d::pose_estimation::PPFEstimatorConfig config;
    config.training_param_.rel_sample_dist = 0.04;
    config.score_thresh_ = 0.01;
    config.refine_param_.method =
        misc3d::pose_estimation::PPFEstimatorConfig::RefineMethod::PointToPoint;

    misc3d::pose_estimation::PPFEstimator estimator(config);

    bool ret;
    auto model = std::make_shared<open3d::geometry::PointCloud>();
    ret = open3d::io::ReadPointCloud(
        "../examples/data/pose_estimation/model/obj.ply", *model);
    model->Scale(0.001, Eigen::Vector3d::Zero());

    estimator.Train(model);

    open3d::geometry::Image depth, color;
    ret = open3d::io::ReadImage(
        "../examples/data/pose_estimation/scene/depth.png", depth);
    ret = open3d::io::ReadImage(
        "../examples/data/pose_estimation/scene/rgb.png", color);
    auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
        color, depth, 1000, 3.0, false);

    open3d::camera::PinholeCameraIntrinsic intrinsic(
        848, 480, 598.7568, 598.7568, 430.3443, 250.244);

    auto scene = open3d::geometry::PointCloud::CreateFromRGBDImage(
        *rgbd, intrinsic, Eigen::Matrix4d::Identity(), false);

    auto scene_crop = misc3d::preprocessing::CropROIPointCloud(
        *scene, {222, 296, 263, 340}, {640, 480});

    std::vector<misc3d::pose_estimation::Pose6D> results;
    ret = estimator.Estimate(scene_crop, results);

    Eigen::Matrix4d pose;
    if (ret) {
        // icp refine
        pose = results[0].pose_;
        auto sampled_model = estimator.GetSampledModel();

        auto res = open3d::pipelines::registration::RegistrationICP(
            sampled_model, *scene_crop, 0.01, pose,
            open3d::pipelines::registration::
                TransformationEstimationPointToPoint());
        pose = res.transformation_;
    } else {
        std::cout << "No matched" << std::endl;
    }

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("PPF estimation", 1920, 1200);

    open3d::geometry::TriangleMesh mesh;
    open3d::io::ReadTriangleMesh(
        "../examples/data/pose_estimation/model/obj.ply", mesh);
    mesh.Scale(0.001, Eigen::Vector3d::Zero());
    misc3d::vis::DrawPose(vis, Eigen::Matrix4d::Identity(), 0.1);
    misc3d::vis::DrawGeometry3D(vis, scene);
    if (ret) {
        auto bbox = std::make_shared<open3d::geometry::OrientedBoundingBox>(
            mesh.GetOrientedBoundingBox());
        auto mesh_vis = std::make_shared<open3d::geometry::TriangleMesh>(mesh);
        misc3d::vis::DrawGeometry3D(vis, mesh_vis, {0, 0, 0}, pose);
        misc3d::vis::DrawGeometry3D(vis, bbox, {0, 1, 0}, pose);
    }
    vis->Run();

    return 0;
}