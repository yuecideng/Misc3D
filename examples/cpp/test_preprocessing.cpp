#include <iostream>
#include <memory>

#include <misc3d/preprocessing/filter.h>
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
    auto pcd_roi = misc3d::preprocessing::CropROIPointCloud(
        *pcd, {500, 300, 600, 400}, {848, 480});
    auto pcd_plane = misc3d::preprocessing::ProjectIntoPlane(*pcd);
    std::cout << "Time cost: " << timer.Stop() << std::endl;

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Crop ROI", 1920, 1200);
    misc3d::vis::DrawPointCloud(vis, *pcd);
    misc3d::vis::DrawPointCloud(vis, *pcd_roi, {1, 0, 0});
    vis->Run();

    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Project into plane", 1920, 1200);
    misc3d::vis::DrawPointCloud(vis, *pcd_plane);
    vis->Run();

    return 0;
}