#include <iostream>
#include <memory>

#include <misc3d/segmentation/iterative_plane_segmentation.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>

int main(int argc, char *argv[]) {
    open3d::geometry::PointCloud pcd;
    bool ret = open3d::io::ReadPointCloud(
        "../examples/data/segmentation/test.ply", pcd);

    misc3d::Timer timer;
    timer.Start();
    const auto results =
        misc3d::segmentation::SegmentPlaneIterative(pcd, 0.01, 100, 0.1);
    std::cout << "Segmentation time: " << timer.Stop() << std::endl;

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Iterative Plane Segmentation", 1920, 1200);
    for (size_t i = 0; i < results.size(); i++) {
        auto cluster = results[i].second;
        const float r = (float)rand() / RAND_MAX;
        const float g = (float)rand() / RAND_MAX;
        const float b = (float)rand() / RAND_MAX;
        misc3d::vis::DrawGeometry3D(
            vis, std::make_shared<open3d::geometry::PointCloud>(cluster),
            {r, g, b});
    }
    vis->Run();

    return 0;
}