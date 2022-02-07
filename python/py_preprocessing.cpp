#include <py_misc3d.h>

#include <misc3d/preprocessing/conversion.h>

namespace misc3d {

namespace preprocessing {

void pybind_preprocessing(py::module &m) {
    m.def("depth_roi_to_pointcloud", &DepthROIToPointCloud,
          "Convert depth roi to pointcloud", py::arg("depth"), py::arg("roi"),
          py::arg("intrinsic"), py::arg("depth_scale") = 1000.0,
          py::arg("depth_trunc") = 3.0, py::arg("stride") = 1);
    m.def("rgbd_roi_to_pointcloud", &RGBDROIToPointCloud,
          "Convert RGBD roi to pointcloud", py::arg("rgbd"), py::arg("roi"),
          py::arg("intrinsic"), py::arg("depth_scale") = 1000.0,
          py::arg("depth_trunc") = 3.0, py::arg("stride") = 1);
}

}  // namespace preprocessing
}  // namespace misc3d