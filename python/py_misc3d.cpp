#include "py_misc3d.h"

namespace misc3d {

PYBIND11_MODULE(py_misc3d, m) {
    py::module m_submodule_common = m.def_submodule("common");
    common::pybind_common(m_submodule_common);

}  // namespace rv6d