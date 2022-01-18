// Copyright (c) RVBUST Inc. - All rights reserved.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <primitives_fitting/Segmentation.h>
#include <primitives_fitting/Utils.h>
#include <py_primitives_fitting.h>

namespace py = pybind11;

namespace primitives_fitting {

namespace segmentation {

template <class ProximityEvaluator>
class PyProximityEvaluator : public ProximityEvaluator {
public:
    using ProximityEvaluator::ProximityEvaluator;
    bool operator()(size_t i, size_t j, double dist) const override {
        PYBIND11_OVERLOAD_PURE_NAME(bool, BaseProximityEvaluator, "__call__", i, j, dist);
    }
};

void pybind_segmentation(py::module &m) {
    py::class_<BaseProximityEvaluator, PyProximityEvaluator<BaseProximityEvaluator>>(
        m, "BaseProximityEvaluator")
        .def(py::init<>())
        .def("__call__", [](BaseProximityEvaluator &self, size_t i, size_t j, double dist) {
            return self(i, j, dist);
        });

    py::class_<DistanceProximityEvaluator, PyProximityEvaluator<DistanceProximityEvaluator>,
               BaseProximityEvaluator>(m, "DistanceProximityEvaluator")
        .def(py::init<double>())
        .def("__call__", [](DistanceProximityEvaluator &self, size_t i, size_t j, double dist) {
            return self(i, j, dist);
        });

    py::class_<NormalsProximityEvaluator, PyProximityEvaluator<NormalsProximityEvaluator>,
               BaseProximityEvaluator>(m, "NormalsProximityEvaluator")
        .def("__init__",
             [](NormalsProximityEvaluator &instance,
                const Eigen::Ref<const Eigen::MatrixX3d> &normals, double angle_thresh) {
                 std::vector<Eigen::Vector3d> normals_in;
                 utils::EigenMatrixToVector(normals, normals_in);
                 new (&instance) NormalsProximityEvaluator(normals_in, angle_thresh);
             })
        .def("__call__", [](NormalsProximityEvaluator &self, size_t i, size_t j, double dist) {
            return self(i, j, dist);
        });

    py::class_<DistanceNormalsProximityEvaluator,
               PyProximityEvaluator<DistanceNormalsProximityEvaluator>, BaseProximityEvaluator>(
        m, "DistanceNormalsProximityEvaluator")
        .def("__init__",
             [](DistanceNormalsProximityEvaluator &instance,
                const Eigen::Ref<const Eigen::MatrixX3d> &normals, double dist_thresh,
                double angle_thresh) {
                 std::vector<Eigen::Vector3d> normals_in;
                 utils::EigenMatrixToVector(normals, normals_in);
                 new (&instance)
                     DistanceNormalsProximityEvaluator(normals_in, dist_thresh, angle_thresh);
             })
        .def("__call__", [](DistanceNormalsProximityEvaluator &self, size_t i, size_t j,
                            double dist) { return self(i, j, dist); });

    py::class_<ProximityExtractor>(m, "ProximityExtractor")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def(py::init<size_t, size_t>())
        .def("segment",
             [](ProximityExtractor &self, const Eigen::Ref<const Eigen::MatrixX3d> &points,
                const std::vector<std::vector<size_t>> &nn_indices,
                const BaseProximityEvaluator &evaluator) {
                 std::vector<Eigen::Vector3d> points_in;
                 utils::EigenMatrixToVector(points, points_in);
                 return self.Segment(points_in, nn_indices, evaluator);
             })
        .def("segment",
             [](ProximityExtractor &self, const Eigen::Ref<const Eigen::MatrixX3d> &points,
                const double search_radius, const BaseProximityEvaluator &evaluator) {
                 std::vector<Eigen::Vector3d> points_in;
                 utils::EigenMatrixToVector(points, points_in);
                 return self.Segment(points_in, search_radius, evaluator);
             })
        .def("get_cluster_index_map", &ProximityExtractor::GetClusterIndexMap)
        .def("get_cluster_num", &ProximityExtractor::GetClusterNum);
}
}  // namespace segmentation
}  // namespace primitives_fitting
