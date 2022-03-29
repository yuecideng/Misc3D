#include <py_misc3d.h>

#include <misc3d/common/knn.h>
#include <misc3d/common/normal_estimation.h>
#include <misc3d/common/ransac.h>

namespace misc3d {

namespace common {

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitPlane(
    const PointCloudPtr &pc, double threshold, size_t max_iteration,
    double probability) {
    RANSACPlane fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    Plane plane;
    std::vector<size_t> inliers;
    fit.SetPointCloud(*pc);
    const bool ret = fit.FitModel(threshold, plane, inliers);
    if (!ret) {
        plane.parameters_.setZero(4);
        return std::make_tuple(plane.parameters_, inliers);
    } else {
        return std::make_tuple(plane.parameters_, inliers);
    }
}

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitSphere(
    const PointCloudPtr &pc, double threshold, size_t max_iteration,
    double probability) {
    RANSACShpere fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    Sphere sphere;
    std::vector<size_t> inliers;
    fit.SetPointCloud(*pc);
    const bool ret = fit.FitModel(threshold, sphere, inliers);
    if (!ret) {
        sphere.parameters_.setZero(4);
        return std::make_tuple(sphere.parameters_, inliers);
    } else {
        return std::make_tuple(sphere.parameters_, inliers);
    }
}

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitCylinder(
    const PointCloudPtr &pc, double threshold, size_t max_iteration,
    double probability) {
    if (!pc->HasNormals()) {
        misc3d::LogError("Fit cylinder requires normals.");
    }

    RANSACCylinder fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    Cylinder Cylinder;
    std::vector<size_t> inliers;
    fit.SetPointCloud(*pc);
    const bool ret = fit.FitModel(threshold, Cylinder, inliers);
    if (!ret) {
        Cylinder.parameters_.setZero(4);
        return std::make_tuple(Cylinder.parameters_, inliers);
    } else {
        return std::make_tuple(Cylinder.parameters_, inliers);
    }
}

void pybind_common(py::module &m) {
    m.def("fit_plane", &FitPlane, "Fit a plane from point clouds",
          py::arg("pc"), py::arg("threshold") = 0.01,
          py::arg("max_iteration") = 1000, py::arg("probability") = 0.9999);
    m.def("fit_sphere", &FitSphere, "Fit a sphere from point clouds",
          py::arg("pc"), py::arg("threshold") = 0.01,
          py::arg("max_iteration") = 1000, py::arg("probability") = 0.9999);
    m.def("fit_cylinder", &FitCylinder, "Fit a cylinder from point clouds",
          py::arg("pc"), py::arg("threshold") = 0.01,
          py::arg("max_iteration") = 1000, py::arg("probability") = 0.9999);
    m.def(
        "estimate_normals",
        [](const PointCloudPtr &pc, const std::tuple<int, int> shape, int k,
           const std::array<double, 3> &view_point) {
            EstimateNormalsFromMap(pc, shape, k, view_point);

            return pc;
        },
        "Estimate normals from pointmap structure", py::arg("pc"),
        py::arg("shape"), py::arg("k") = 5,
        py::arg("view_point") = std::array<double, 3>{0, 0, 0});

    py::class_<KNearestSearch>(m, "KNearestSearch")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init(
                 [](const Eigen::Ref<const Eigen::MatrixXd> &data,
                    int n_trees) { return new KNearestSearch(data, n_trees); }),
             py::arg("data"), py::arg("n_trees") = 10)
        .def(py::init<const open3d::geometry::PointCloud &, int>())
        .def(py::init<const open3d::pipelines::registration::Feature &, int>())

        .def(
            "set_mat_data",
            [](KNearestSearch &self,
               const Eigen::Ref<const Eigen::MatrixXd> &data) {
                return self.SetMatrixData(data);
            },
            "Set data from numpy array")
        .def(
            "set_geometry",
            [](KNearestSearch &self, const GeometryPtr &geometry) {
                return self.SetGeometry(*geometry);
            },
            "Set data from open3d geometry")
        .def(
            "set_feature",
            [](KNearestSearch &self, const FeaturePtr &feature) {
                return self.SetFeature(*feature);
            },
            "Set data from open3d feature")
        .def(
            "search",
            [](KNearestSearch &self, const Eigen::RowVectorXd &query,
               const open3d::geometry::KDTreeSearchParam &param) {
                std::vector<size_t> indices;
                std::vector<double> distances;
                int k =
                    self.Search(query.transpose(), param, indices, distances);
                return std::make_pair(indices, distances);
            },
            "Search knn with open3d kdtree param", py::arg("query"),
            py::arg("param"))
        .def(
            "search_knn",
            [](KNearestSearch &self, const Eigen::RowVectorXd &query, int knn) {
                std::vector<size_t> indices;
                std::vector<double> distances;
                int k =
                    self.SearchKNN(query.transpose(), knn, indices, distances);
                return std::make_pair(indices, distances);
            },
            "Search knn with given k", py::arg("query"), py::arg("knn"))
        .def(
            "search_hybrid",
            [](KNearestSearch &self, const Eigen::RowVectorXd &query,
               double radius, int knn) {
                std::vector<size_t> indices;
                std::vector<double> distances;
                int k = self.SearchHybrid(query.transpose(), radius, knn,
                                          indices, distances);
                return std::make_pair(indices, distances);
            },
            "Search knn with given raidus and k", py::arg("query"),
            py::arg("radius"), py::arg("knn"));
}

}  // namespace common
}  // namespace misc3d