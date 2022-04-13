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

namespace preprocessing {
void pybind_preprocessing(py::module &m);
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

namespace pose_estimation {
void pybind_pose_estimation(py::module &m);
}

#ifdef ENABLE_RECONSTRUCTION
namespace reconstruction {
void pybind_reconstruction(py::module &m);
}
#endif

namespace vis {
void pybind_vis(py::module &m);
}

}  // namespace misc3d