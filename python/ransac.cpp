// Copyright (c) RVBUST Inc. - All rights reserved.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <primitives_fitting/Ransac.h>
#include <primitives_fitting/Utils.h>
#include <py_primitives_fitting.h>

namespace py = pybind11;

namespace primitives_fitting {

namespace ransac {

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitPlane(
    const std::vector<Eigen::Vector3d> &points, double threshold, size_t max_iteration,
    double probability, bool enable_parallel) {
    RANSACPlane fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    fit.SetParallel(enable_parallel);
    Plane plane;
    std::vector<size_t> inliers;
    fit.SetPointCloud(points);
    const bool ret = fit.FitModel(threshold, plane, inliers);
    if (!ret) {
        plane.m_parameters.setZero(4);
        return std::make_tuple(plane.m_parameters, inliers);
    } else {
        return std::make_tuple(plane.m_parameters, inliers);
    }
}

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitSphere(
    const std::vector<Eigen::Vector3d> &points, double threshold, size_t max_iteration,
    double probability, bool enable_parallel) {
    RANSACShpere fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    fit.SetParallel(enable_parallel);
    Sphere sphere;
    std::vector<size_t> inliers;
    fit.SetPointCloud(points);
    const bool ret = fit.FitModel(threshold, sphere, inliers);
    if (!ret) {
        sphere.m_parameters.setZero(4);
        return std::make_tuple(sphere.m_parameters, inliers);
    } else {
        return std::make_tuple(sphere.m_parameters, inliers);
    }
}

std::tuple<Eigen::VectorXd, std::vector<size_t>> FitCylinder(
    const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector3d> &normals,
    double threshold, size_t max_iteration, double probability, bool enable_parallel) {
    RANSACShpere fit;
    fit.SetMaxIteration(max_iteration);
    fit.SetProbability(probability);
    fit.SetParallel(enable_parallel);
    Sphere sphere;
    std::vector<size_t> inliers;
    fit.SetPointCloud(points);
    fit.SetNormals(normals);
    const bool ret = fit.FitModel(threshold, sphere, inliers);
    if (!ret) {
        sphere.m_parameters.setZero(4);
        return std::make_tuple(sphere.m_parameters, inliers);
    } else {
        return std::make_tuple(sphere.m_parameters, inliers);
    }
}

void pybind_ransac(py::module &m) {
    m.def("fit_plane", &FitPlane, "Fit a plane from point clouds", py::arg("points"),
          py::arg("threshold") = 0.01, py::arg("max_iteration") = 1000,
          py::arg("probability") = 0.99, py::arg("enable_parallel") = false);
    m.def("fit_sphere", &FitSphere, "Fit a sphere from point clouds", py::arg("points"),
          py::arg("threshold") = 0.01, py::arg("max_iteration") = 1000,
          py::arg("probability") = 0.99, py::arg("enable_parallel") = false);
    m.def("fit_cylinder", &FitCylinder, "Fit a cylinder from point clouds", py::arg("points"),
          py::arg("normals"), py::arg("threshold") = 0.01, py::arg("max_iteration") = 1000,
          py::arg("probability") = 0.99, py::arg("enable_parallel") = false);
}

}  // namespace ransac
}  // namespace primitives_fitting