#include <iostream>
#include <memory>

#include <misc3d/segmentation/proximity_extraction.h>
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
        *rgbd, intrinsic, Eigen::Matrix4d::Identity(), true);

    misc3d::Timer timer;
    auto pcd_down = pcd->VoxelDownSample(0.01);
    pcd_down->EstimateNormals(
        open3d::geometry::KDTreeSearchParamHybrid(0.02, 15));
    pcd_down->OrientNormalsTowardsCameraLocation();

    timer.Start();
    misc3d::segmentation::ProximityExtractor extractor(100);
    misc3d::segmentation::DistanceNormalsProximityEvaluator evaluator(
        pcd_down->normals_, 0.02, 30);
    auto index_list = extractor.Segment(*pcd_down, 0.02, evaluator);
    std::cout << "Time cost: " << timer.Stop() << std::endl;

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Segmentation", 1920, 1200);
    for (size_t i = 0; i < index_list.size(); i++) {
        auto cluster = pcd_down->SelectByIndex(index_list[i]);
        const float r = (float)rand() / RAND_MAX;
        const float g = (float)rand() / RAND_MAX;
        const float b = (float)rand() / RAND_MAX;
        misc3d::vis::DrawPointCloud(vis, *cluster, {r, g, b});
    }
    vis->Run();

    return 0;
}