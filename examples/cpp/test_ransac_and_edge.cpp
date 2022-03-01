#include <iostream>
#include <memory>

#include <misc3d/common/ransac.h>
#include <misc3d/features/edge_detection.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/Image.h>
#include <open3d/geometry/KDTreeSearchParam.h>
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

    auto pcd_down = pcd->VoxelDownSample(0.005);

    misc3d::Timer timer;
    timer.Start();
    misc3d::common::RANSACPlane fit;
    fit.SetMaxIteration(400);
    fit.SetProbability(0.99);
    fit.SetParallel(true);
    misc3d::common::Plane plane;
    std::vector<size_t> inliers;
    fit.SetPointCloud(*pcd_down);
    ret = fit.FitModel(0.01, plane, inliers);
    std::cout << "Time cost for fitting: " << timer.Stop() << std::endl;

    auto pcd_plane = pcd_down->SelectByIndex(inliers);
    timer.Start();
    auto indices = misc3d::features::DetectEdgePoints(
        *pcd_plane, open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
    std::cout << "Time cost for edge detection: " << timer.Stop() << std::endl;

    auto pcd_edges = pcd_plane->SelectByIndex(indices);

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Ransac and edge detection", 1920, 1200);
    misc3d::vis::DrawPointCloud(vis, *pcd_down, {0.5, 0.5, 0.5});
    misc3d::vis::DrawPointCloud(vis, *pcd_plane);
    misc3d::vis::DrawPointCloud(vis, *pcd_edges, {1, 0, 0});
    vis->Run();

    return 0;
}