#include <array>

#include <py_misc3d.h>
#include <misc3d/vis/vis_utils.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace vis {

void pybind_vis(py::module &m) {
    m.def("draw_pose", &DrawPose, py::arg("vis"),
          py::arg("pose") = Eigen::Matrix4d::Identity(), py::arg("size") = 0.1);
    m.def(
        "draw_point_cloud",
        [](const std::shared_ptr<open3d::visualization::Visualizer> &vis,
           const PointCloudPtr &pc,
           const std::array<float, 3> &color, const Eigen::Matrix4d &pose,
           float size) { DrawPointCloud(vis, *pc, color, pose, size); },
        py::arg("vis"), py::arg("pc"),
        py::arg("color") = std::array<float, 3>{0, 0, 0},
        py::arg("pose") = Eigen::Matrix4d::Identity(), py::arg("size") = 3.0);
}

}  // namespace vis
}  // namespace misc3d