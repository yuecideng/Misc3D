#include <iostream>
#include <memory>

#include <misc3d/preprocessing/filter.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/visualization/visualizer/Visualizer.h>

int main(int argc, char *argv[]) {
    open3d::geometry::PointCloud pcd;
    bool ret = open3d::io::ReadPointCloud(
        "../examples/data/pose_estimation/model/obj.ply", pcd);

    std::cout << "Point cloud size before smapling: " << pcd.points_.size()
              << std::endl;

    misc3d::Timer timer;
    timer.Start();
    auto indices = misc3d::preprocessing::FarthestPointSampling(pcd, 1000);
    std::cout << "Time cost: " << timer.Stop() << std::endl;

    auto sample = pcd.SelectByIndex(indices);

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Farthest point sampling", 1920, 1200);
    misc3d::vis::DrawPointCloud(vis, pcd);
    misc3d::vis::DrawPointCloud(vis, *sample, {0, 1, 0},
                                Eigen::Matrix4d::Identity(), 5);
    vis->Run();

    return 0;
}