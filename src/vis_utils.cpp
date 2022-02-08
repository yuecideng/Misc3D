#include <memory>

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

void DrawPointCloud(const std::shared_ptr<open3d::visualization::Visualizer> &vis,
                    const open3d::geometry::PointCloud &pc,
                    const std::array<float, 3> &color, const Eigen::Matrix4d &pose,
                    float size) {
    auto pc_vis = std::make_shared<open3d::geometry::PointCloud>(pc);
    pc_vis->RemoveNonFinitePoints();
    pc_vis->Transform(pose);
    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
        if (!pc_vis->HasColors()) {
            pc_vis->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
        }
    } else {
        pc_vis->PaintUniformColor(Eigen::Vector3d(color[0], color[1], color[2]));
    }
    vis->AddGeometry(pc_vis);

    auto &option = vis->GetRenderOption();
    option.point_size_ = (double)size;
}

}  // namespace vis

}  // namespace misc3d
