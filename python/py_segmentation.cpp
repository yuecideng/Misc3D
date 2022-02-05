#include <py_misc3d.h>

#include <misc3d/segmentation/proximity_extraction.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace segmentation {

template <class ProximityEvaluator>
class PyProximityEvaluator : public ProximityEvaluator {
public:
    using ProximityEvaluator::ProximityEvaluator;
    bool operator()(size_t i, size_t j, double dist) const override {
        PYBIND11_OVERLOAD_PURE_NAME(bool, BaseProximityEvaluator, "__call__", i, j,
                                    dist);
    }
};

void pybind_segmentation(py::module &m) {
    py::class_<BaseProximityEvaluator, PyProximityEvaluator<BaseProximityEvaluator>>(
        m, "BaseProximityEvaluator")
        .def(py::init<>())
        .def("__call__", [](BaseProximityEvaluator &self, size_t i, size_t j,
                            double dist) { return self(i, j, dist); });

    py::class_<DistanceProximityEvaluator,
               PyProximityEvaluator<DistanceProximityEvaluator>,
               BaseProximityEvaluator>(m, "DistanceProximityEvaluator")
        .def(py::init<double>())
        .def("__call__", [](DistanceProximityEvaluator &self, size_t i, size_t j,
                            double dist) { return self(i, j, dist); });

    py::class_<NormalsProximityEvaluator,
               PyProximityEvaluator<NormalsProximityEvaluator>,
               BaseProximityEvaluator>(m, "NormalsProximityEvaluator")
        .def("__init__",
             [](NormalsProximityEvaluator &instance,
                const Eigen::Ref<const Eigen::MatrixX3d> &normals,
                double angle_thresh) {
                 std::vector<Eigen::Vector3d> normals_in;
                 EigenMatrixToVector<double>(normals, normals_in);
                 new (&instance) NormalsProximityEvaluator(normals_in, angle_thresh);
             })
        .def("__call__", [](NormalsProximityEvaluator &self, size_t i, size_t j,
                            double dist) { return self(i, j, dist); });

    py::class_<DistanceNormalsProximityEvaluator,
               PyProximityEvaluator<DistanceNormalsProximityEvaluator>,
               BaseProximityEvaluator>(m, "DistanceNormalsProximityEvaluator")
        .def("__init__",
             [](DistanceNormalsProximityEvaluator &instance,
                const Eigen::Ref<const Eigen::MatrixX3d> &normals,
                double dist_thresh, double angle_thresh) {
                 std::vector<Eigen::Vector3d> normals_in;
                 EigenMatrixToVector<double>(normals, normals_in);
                 new (&instance) DistanceNormalsProximityEvaluator(
                     normals_in, dist_thresh, angle_thresh);
             })
        .def("__call__", [](DistanceNormalsProximityEvaluator &self, size_t i,
                            size_t j, double dist) { return self(i, j, dist); });

    py::class_<ProximityExtractor>(m, "ProximityExtractor")
        .def(py::init<size_t, size_t>(), py::arg("min_size") = 1,
             py::arg("max_size") = std::numeric_limits<size_t>::max())
        .def(
            "segment",
            [](ProximityExtractor &self,
               const PointCloudPtr &pc,
               const std::vector<std::vector<size_t>> &nn_indices,
               const BaseProximityEvaluator &evaluator) {
                return self.Segment(*pc, nn_indices, evaluator);
            },
            "Segment point clouds with given nearest neighboor", py::arg("pc"),
            py::arg("nn_indices"), py::arg("evaluator"))
        .def(
            "segment",
            [](ProximityExtractor &self,
               const PointCloudPtr &pc,
               const double search_radius, const BaseProximityEvaluator &evaluator) {
                return self.Segment(*pc, search_radius, evaluator);
            },
            "Segment point clouds with radius for nearest neighboor searching",
            py::arg("pc"), py::arg("search_radius"), py::arg("evaluator"))
        .def("get_cluster_index_map", &ProximityExtractor::GetClusterIndexMap)
        .def("get_cluster_num", &ProximityExtractor::GetClusterNum);
}

}  // namespace segmentation

}  // namespace misc3d