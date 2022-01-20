#include <py_misc3d.h>

#include <misc3d/common/normal_estimation.h>
#include <misc3d/common/ransac.h>

// namespace py = pybind11;

namespace misc3d {

namespace common {

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitPlane(
    const open3d::geometry::PointCloud &pc, double threshold, size_t max_iteration,
    double probability, bool enable_parallel) {
    RANSACPlane fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    fit.SetParallel(enable_parallel);
    Plane plane;
    std::vector<size_t> inliers;
    fit.SetPointCloud(pc);
    const bool ret = fit.FitModel(threshold, plane, inliers);
    if (!ret) {
        plane.parameters_.setZero(4);
        return std::make_tuple(plane.parameters_, inliers);
    } else {
        return std::make_tuple(plane.parameters_, inliers);
    }
}

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitSphere(
    const open3d::geometry::PointCloud &pc, double threshold, size_t max_iteration,
    double probability, bool enable_parallel) {
    RANSACShpere fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    fit.SetParallel(enable_parallel);
    Sphere sphere;
    std::vector<size_t> inliers;
    fit.SetPointCloud(pc);
    const bool ret = fit.FitModel(threshold, sphere, inliers);
    if (!ret) {
        sphere.parameters_.setZero(4);
        return std::make_tuple(sphere.parameters_, inliers);
    } else {
        return std::make_tuple(sphere.parameters_, inliers);
    }
}

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitCylinder(
    const open3d::geometry::PointCloud &pc, double threshold, size_t max_iteration,
    double probability, bool enable_parallel) {
    RANSACShpere fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    fit.SetParallel(enable_parallel);
    Sphere sphere;
    std::vector<size_t> inliers;
    fit.SetPointCloud(pc);
    const bool ret = fit.FitModel(threshold, sphere, inliers);
    if (!ret) {
        sphere.parameters_.setZero(4);
        return std::make_tuple(sphere.parameters_, inliers);
    } else {
        return std::make_tuple(sphere.parameters_, inliers);
    }
}

void pybind_common(py::module &m) {
    m.def("fit_plane", &FitPlane, "Fit a plane from point clouds", py::arg("pc"),
          py::arg("threshold") = 0.01, py::arg("max_iteration") = 1000,
          py::arg("probability") = 0.99, py::arg("enable_parallel") = false);
    m.def("fit_sphere", &FitSphere, "Fit a sphere from point clouds", py::arg("pc"),
          py::arg("threshold") = 0.01, py::arg("max_iteration") = 1000,
          py::arg("probability") = 0.99, py::arg("enable_parallel") = false);
    m.def("fit_cylinder", &FitCylinder, "Fit a cylinder from point clouds",
          py::arg("pc"), py::arg("threshold") = 0.01,
          py::arg("max_iteration") = 1000, py::arg("probability") = 0.99,
          py::arg("enable_parallel") = false);

    m.def(
        "estimate_normals",
        [](const open3d::geometry::PointCloud &pc, int w, int h, int k) {
            open3d::geometry::PointCloud pc_ = pc;
            EstimateNormalsFromMap(pc_, w, h, k);
        
            return pc_;
        },
        "Estimate normals from pointmap structure", py::arg("pc"), py::arg("weight"),
        py::arg("height"), py::arg("k") = 5);
}

}  // namespace common
}  // namespace misc3d