#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace misc3d {

py::object o3d_geometry = (py::object)py::module_::import("open3d").attr("geometry");

namespace common {
void pybind_common(py::module &m);
}

}  // namespace misc3d