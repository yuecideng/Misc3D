// Copyright (c) RVBUST Inc. - All rights reserved.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace primitives_fitting {

namespace segmentation {
void pybind_segmentation(py::module &m);
}

namespace ransac {
void pybind_ransac(py::module &m);
}

}  // namespace primitives_fitting
