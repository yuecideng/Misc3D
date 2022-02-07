#include "py_misc3d.h"

namespace misc3d {

PYBIND11_MODULE(py_misc3d, m) {
    py::object o3d_geometry =
        (py::object)py::module_::import("open3d").attr("geometry");

    py::object o3d_pipelines =
        (py::object)py::module_::import("open3d").attr("pipelines");

    py::object o3d_camera =
        (py::object)py::module_::import("open3d").attr("camera");

    py::object o3d_vis =
        (py::object)py::module_::import("open3d").attr("visualization");

    py::module m_submodule_common = m.def_submodule("common");
    common::pybind_common(m_submodule_common);

    py::module m_submodule_preprocessing = m.def_submodule("preprocessing");
    preprocessing::pybind_preprocessing(m_submodule_preprocessing);

    py::module m_submodule_segmentation = m.def_submodule("segmentation");
    segmentation::pybind_segmentation(m_submodule_segmentation);

    py::module m_submodule_features = m.def_submodule("features");
    features::pybind_features(m_submodule_features);

    py::module m_submodule_registration = m.def_submodule("registration");
    registration::pybind_registration(m_submodule_registration);

    py::module m_submodule_pose_estimation = m.def_submodule("pose_estimation");
    pose_estimation::pybind_pose_estimation(m_submodule_pose_estimation);

    py::module m_submodule_vis = m.def_submodule("vis");
    vis::pybind_vis(m_submodule_vis);
}
}  // namespace misc3d