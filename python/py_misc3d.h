#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace misc3d {

namespace py = pybind11;

namespace common {
void pybind_common(py::module &m);
}

namespace segmentation {
void pybind_segmentation(py::module &m);
}

namespace features {
void pybind_features(py::module &m);
}

namespace registration {
void pybind_registration(py::module &m);
}

namespace vis {
void pybind_vis(py::module &m);
}

}  // namespace misc3d