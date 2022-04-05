#include "py_misc3d.h"
#include <misc3d/logging.h>

namespace misc3d {

PYBIND11_MODULE(py_misc3d, m) {
    py::object o3d_geometry =
        (py::object)py::module_::import("open3d").attr("geometry");

    py::object o3d_pipelines =
        (py::object)py::module_::import("open3d").attr("pipelines");

    py::object o3d_camera =
        (py::object)py::module_::import("open3d").attr("camera");

    py::object o3d_core =
        (py::object)py::module_::import("open3d").attr("core");

    py::object o3d_vis =
        (py::object)py::module_::import("open3d").attr("visualization");

    py::object o3d_utility =
        (py::object)py::module_::import("open3d").attr("utility");

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

    // logging level setting
    py::enum_<misc3d::VerbosityLevel> vl(m, "VerbosityLevel", py::arithmetic(),
                                 "VerbosityLevel");
    vl.value("Error", misc3d::VerbosityLevel::Error)
            .value("Warning", misc3d::VerbosityLevel::Warning)
            .value("Info", misc3d::VerbosityLevel::Info)
            .value("Debug", misc3d::VerbosityLevel::Debug)
            .export_values();
    m.def("set_verbosity_level", &misc3d::SetVerbosityLevel,
          "Set global verbosity level of Misc3D", py::arg("verbosity_level"));
    m.def("get_verbosity_level", &misc3d::GetVerbosityLevel,
          "Get global verbosity level of Misc3D");

}
}  // namespace misc3d