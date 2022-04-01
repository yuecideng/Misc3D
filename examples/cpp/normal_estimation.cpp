#include <iostream>
#include <memory>

#include <misc3d/common/normal_estimation.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/Image.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/RGBDImage.h>
#include <open3d/io/ImageIO.h>

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
        *rgbd, intrinsic, Eigen::Matrix4d::Identity(), false);

    misc3d::Timer timer;
    timer.Start();
    misc3d::common::EstimateNormalsFromMap(pcd, {848, 480}, 3);
    std::cout << "Time cost: " << timer.Stop() << std::endl;

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Estimate normals form map", 1920, 1200);
    misc3d::vis::DrawPose(vis, Eigen::Matrix4d::Identity(), 0.1);
    misc3d::vis::DrawGeometry3D(vis, pcd);
    vis->Run();

    return 0;
}