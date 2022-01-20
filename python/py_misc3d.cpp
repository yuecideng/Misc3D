#include "py_misc3d.h"

namespace misc3d {

PYBIND11_MODULE(py_misc3d, m) {
    py::object o3d_geometry =
        (py::object)py::module_::import("open3d").attr("geometry");

    py::module m_submodule_common = m.def_submodule("common");
    common::pybind_common(m_submodule_common);

    py::module m_submodule_segmentation = m.def_submodule("segmentation");
    segmentation::pybind_segmentation(m_submodule_segmentation);

}
}  // namespace misc3d