#include <memory>

#include <misc3d/logging.h>
#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>

namespace misc3d {

namespace vis {

void DrawPose(const std::shared_ptr<open3d::visualization::Visualizer> &vis,
              const Eigen::Matrix4d &pose, double size) {
    auto axis = open3d::geometry::TriangleMesh::CreateCoordinateFrame(size);
    axis->Transform(pose);
    vis->AddGeometry(axis);
}

void DrawPointCloud(
    const std::shared_ptr<open3d::visualization::Visualizer> &vis,
    const PointCloudPtr &pc, const std::array<float, 3> &color,
    const Eigen::Matrix4d &pose, float size) {
    pc->RemoveNonFinitePoints();
    pc->Transform(pose);
    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
        if (!pc->HasColors()) {
            pc->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
        }
    } else {
        pc->PaintUniformColor(Eigen::Vector3d(color[0], color[1], color[2]));
    }
    vis->AddGeometry(pc);

    auto &option = vis->GetRenderOption();
    option.point_size_ = (double)size;
}

void DrawTriangleMesh(
    const std::shared_ptr<open3d::visualization::Visualizer> &vis,
    const TriangleMeshPtr &mesh, const std::array<float, 3> &color,
    const Eigen::Matrix4d &pose) {
    mesh->Transform(pose);
    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
        if (!mesh->HasVertexColors() || mesh->HasTextures()) {
            mesh->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
        }
    } else {
        mesh->PaintUniformColor(Eigen::Vector3d(color[0], color[1], color[2]));
    }
    vis->AddGeometry(mesh);
}

void DrawGeometry3D(
    const std::shared_ptr<open3d::visualization::Visualizer> &vis,
    const std::shared_ptr<open3d::geometry::Geometry3D> &geometry,
    const std::array<float, 3> &color, const Eigen::Matrix4d &pose,
    float size) {
    if (geometry->GetGeometryType() ==
        open3d::geometry::Geometry::GeometryType::PointCloud) {
        auto pcd =
            std::dynamic_pointer_cast<open3d::geometry::PointCloud>(geometry);
        DrawPointCloud(vis, pcd, color, pose, size);
    } else if (geometry->GetGeometryType() ==
               open3d::geometry::Geometry::GeometryType::TriangleMesh) {
        auto mesh =
            std::dynamic_pointer_cast<open3d::geometry::TriangleMesh>(geometry);
        DrawTriangleMesh(vis, mesh, color, pose);
    } else if (
        geometry->GetGeometryType() ==
            open3d::geometry::Geometry::GeometryType::OrientedBoundingBox ||
        geometry->GetGeometryType() ==
            open3d::geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        auto bbox =
            std::dynamic_pointer_cast<open3d::geometry::OrientedBoundingBox>(
                geometry);
        const auto rotation = pose.block<3, 3>(0, 0);
        const auto translation = pose.block<3, 1>(0, 3);
        bbox->Rotate(rotation, bbox->GetCenter());
        bbox->Translate(translation, true);
        bbox->color_ = Eigen::Vector3d(color[0], color[1], color[2]);
        vis->AddGeometry(bbox);
    } else {
        misc3d::LogError("Unsupported geometry type.");
    }
}

}  // namespace vis

}  // namespace misc3d
