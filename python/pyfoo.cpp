// Copyright (c) RVBUST Inc. - All rights reserved.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <misc3d/foo.h>

namespace py = pybind11;

PYBIND11_MODULE(py_misc3d, m) {
    py::object o3d_geometry = (py::object)py::module_::import("open3d").attr("geometry");

    py::class_<foo>(m, "foo", o3d_geometry.attr("PointCloud"))
        .def(py::init<>())
        .def("receive", [](foo &self, const open3d::geometry::PointCloud &p) { self.receive(p); })
        .def_readwrite("test", &foo::test_);
}
