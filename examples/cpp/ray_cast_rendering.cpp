#include <iostream>
#include <memory>

#include <misc3d/pose_estimation/ray_cast_renderer.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

int main(int argc, char *argv[]) {
    const open3d::camera::PinholeCameraIntrinsic camera_intrinsic(
        640, 480, 572.4114, 573.5704, 325.2611, 242.0489);

    misc3d::pose_estimation::RayCastRenderer renderer(camera_intrinsic);

    open3d::geometry::TriangleMesh mesh;
    bool ret = open3d::io::ReadTriangleMesh(
        "../examples/data/pose_estimation/model/obj.ply", mesh);
    mesh.Scale(0.001, Eigen::Vector3d::Zero());
    const open3d::geometry::TriangleMesh mesh2 = mesh;

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> pose, pose2;
    pose << 0.29493218, 0.95551309, 0.00312103, -0.14527225, 0.89692822,
        -0.27572004, -0.34568516, 0.12533501, -0.32944616, 0.10475302,
        -0.93834537, 0.99371838, 0., 0., 0., 1.;
    pose2 << 0.29493218, 0.95551309, 0.00312103, -0.04527225, 0.89692822,
        -0.27572004, -0.34568516, 0.02533501, -0.32944616, 0.10475302,
        -0.93834537, 0.99371838, 0., 0., 0., 1.;

    misc3d::Timer timer;
    timer.Start();
    ret = renderer.CastRays({mesh, mesh2}, {pose, pose2});
    std::cout << "Ray cast time: " << timer.Stop() << std::endl;

    auto instance_pcds = renderer.GetInstancePointCloud();
    instance_pcds[0].PaintUniformColor({1, 0, 0});
    instance_pcds[1].PaintUniformColor({0, 1, 0});
    const auto axis = open3d::geometry::TriangleMesh::CreateCoordinateFrame(
        0.1, Eigen::Vector3d::Zero());

    open3d::visualization::DrawGeometries(
        {std::make_shared<const open3d::geometry::PointCloud>(instance_pcds[0]),
         std::make_shared<const open3d::geometry::PointCloud>(instance_pcds[1]),
         axis},
        "Ray Cast Rendering");

    return 0;
}