#include <iostream>
#include <memory>

#include <misc3d/registration/correspondence_matching.h>
#include <misc3d/registration/transform_estimation.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/Image.h>
#include <open3d/geometry/KDTreeSearchParam.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/RGBDImage.h>
#include <open3d/io/ImageIO.h>
#include <open3d/pipelines/registration/Feature.h>
#include <open3d/pipelines/registration/Registration.h>

std::tuple<open3d::geometry::PointCloud,
           open3d::pipelines::registration::Feature>
PreprocessPointCloud(const open3d::geometry::PointCloud &pcd,
                     double voxel_size) {
    auto pcd_down = pcd.VoxelDownSample(voxel_size);
    const double radius = voxel_size * 2;
    pcd_down->EstimateNormals(
        open3d::geometry::KDTreeSearchParamHybrid(radius, 30));
    const double radius_feature = voxel_size * 5;
    auto fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
        *pcd_down,
        open3d::geometry::KDTreeSearchParamHybrid(radius_feature, 100));
    return std::make_tuple(*pcd_down, *fpfh);
}

int main(int argc, char *argv[]) {
    bool ret;
    open3d::geometry::Image depth, color;
    ret = open3d::io::ReadImage("../examples/data/indoor/depth/depth_0.png",
                                depth);
    ret = open3d::io::ReadImage("../examples/data/indoor/color/color_0.png",
                                color);

    auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
        color, depth, 1000, 3.0, false);

    open3d::camera::PinholeCameraIntrinsic intrinsic(
        848, 480, 598.7568, 598.7568, 430.3443, 250.244);

    auto pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(
        *rgbd, intrinsic, Eigen::Matrix4d::Identity(), true);

    ret = open3d::io::ReadImage("../examples/data/indoor/depth/depth_1.png",
                                depth);
    ret = open3d::io::ReadImage("../examples/data/indoor/color/color_1.png",
                                color);

    rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
        color, depth, 1000, 3.0, false);

    auto pcd_ = open3d::geometry::PointCloud::CreateFromRGBDImage(
        *rgbd, intrinsic, Eigen::Matrix4d::Identity(), true);

    auto res1 = PreprocessPointCloud(*pcd, 0.02);
    auto res2 = PreprocessPointCloud(*pcd_, 0.02);

    misc3d::Timer timer;
    misc3d::registration::ANNMatcher matcher(
        misc3d::registration::MatchMethod::ANNOY, 4);
    timer.Start();
    auto matched_list = matcher.Match(std::get<1>(res1), std::get<1>(res2));
    std::cout << "Corres num: " << std::get<0>(matched_list).size()
              << std::endl;
    std::cout << "Time cost for matching: " << timer.Stop() << std::endl;

    auto src_down = std::get<0>(res1);
    auto dst_down = std::get<0>(res2);

    misc3d::registration::RANSACSolver solver(0.03);
    timer.Start();
    auto pose = solver.Solve(src_down, dst_down, matched_list);

    auto res = open3d::pipelines::registration::RegistrationICP(
        src_down, dst_down, 0.02, pose,
        open3d::pipelines::registration::
            TransformationEstimationPointToPoint());
    pose = res.transformation_;
    std::cout << "Time cost for solving: " << timer.Stop() << std::endl;

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Before Registration", 1920, 1200);
    misc3d::vis::DrawGeometry3D(vis, pcd);
    misc3d::vis::DrawGeometry3D(vis, pcd_);
    vis->Run();
    
    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("After Registration", 1920, 1200);
    misc3d::vis::DrawGeometry3D(vis, pcd, {0, 0, 0}, pose);
    misc3d::vis::DrawGeometry3D(vis, pcd_);
    vis->Run();

    return 0;
}