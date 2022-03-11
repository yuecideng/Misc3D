#include <array>

#include <misc3d/utils.h>
#include <misc3d/vis/vis_utils.h>
#include <py_misc3d.h>

namespace misc3d {

namespace vis {

void pybind_vis(py::module &m) {
    m.def("draw_pose", &DrawPose, py::arg("vis"),
          py::arg("pose") = Eigen::Matrix4d::Identity(), py::arg("size") = 0.1);
    m.def("draw_geometry3d", &DrawGeometry3D, py::arg("vis"),
          py::arg("geometry"), py::arg("color") = std::array<float, 3>{0, 0, 0},
          py::arg("pose") = Eigen::Matrix4d::Identity(), py::arg("size") = 3.0);
}

}  // namespace vis
}  // namespace misc3d