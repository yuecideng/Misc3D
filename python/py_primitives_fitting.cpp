// Copyright (c) RVBUST Inc. - All rights reserved.

#include "py_primitives_fitting.h"
#include <primitives_fitting/PrimitivesFitting.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace primitives_fitting {

PYBIND11_MODULE(py_primitives_fitting, m) {
    py::module m_submodule_seg = m.def_submodule("segmentation");
    segmentation::pybind_segmentation(m_submodule_seg);

    py::module m_submodule_ransac = m.def_submodule("ransac");
    ransac::pybind_ransac(m_submodule_ransac);

    py::enum_<PrimitivesType>(m, "PrimitivesType")
        .value("plane", PrimitivesType::plane)
        .value("sphere", PrimitivesType::sphere)
        .value("cylinder", PrimitivesType::cylinder)
        .export_values();

    py::class_<PrimitivesDetectorConfig::ClusterParam>(m, "ClusterParam")
        .def(py::init<>())
        .def_readwrite("min_cluster_size",
                       &PrimitivesDetectorConfig::ClusterParam::min_cluster_size)
        .def_readwrite("max_cluster_size",
                       &PrimitivesDetectorConfig::ClusterParam::max_cluster_size)
        .def_readwrite("dist_thresh", &PrimitivesDetectorConfig::ClusterParam::dist_thresh)
        .def_readwrite("angle_thresh", &PrimitivesDetectorConfig::ClusterParam::angle_thresh);

    py::class_<PrimitivesDetectorConfig::FilteringParam>(m, "FilteringParam")
        .def(py::init<>())
        .def_readwrite("min_bound", &PrimitivesDetectorConfig::FilteringParam::min_bound)
        .def_readwrite("max_bound", &PrimitivesDetectorConfig::FilteringParam::max_bound);

    py::class_<PrimitivesDetectorConfig::FittingParam>(m, "FittingParam")
        .def(py::init<>())
        .def_readwrite("type", &PrimitivesDetectorConfig::FittingParam::type)
        .def_readwrite("enable_parallel", &PrimitivesDetectorConfig::FittingParam::enable_parallel)
        .def_readwrite("max_iteration", &PrimitivesDetectorConfig::FittingParam::max_iteration)
        .def_readwrite("threshold", &PrimitivesDetectorConfig::FittingParam::threshold);

    py::class_<PrimitivesDetectorConfig::PreProcessParam>(m, "PreProcessParam")
        .def(py::init<>())
        .def_readwrite("enable_smoothing",
                       &PrimitivesDetectorConfig::PreProcessParam::enable_smoothing)
        .def_readwrite("voxel_size", &PrimitivesDetectorConfig::PreProcessParam::voxel_size);

    py::class_<PrimitivesDetectorConfig>(m, "PrimitivesDetectorConfig")
        .def(py::init<>())
        .def_readwrite("m_preprocess_param", &PrimitivesDetectorConfig::m_preprocess_param)
        .def_readwrite("m_cluster_param", &PrimitivesDetectorConfig::m_cluster_param)
        .def_readwrite("m_filtering_param", &PrimitivesDetectorConfig::m_filtering_param)
        .def_readwrite("m_fitting_param", &PrimitivesDetectorConfig::m_fitting_param);

    py::class_<PrimitivesDetector>(m, "PrimitivesDetector")
        .def(py::init<>())
        .def(py::init<const PrimitivesDetectorConfig &>())
        .def("set_config", &PrimitivesDetector::SetConfiguration)
        .def("detect",
             [](PrimitivesDetector &self, const Eigen::Ref<const Eigen::MatrixX3d> &pc) {
                 std::vector<Eigen::Vector3d> pc_in;
                 utils::EigenMatrixToVector(pc, pc_in);
                 return self.Detect(pc_in);
             })
        .def("detect",
             [](PrimitivesDetector &self, const Eigen::Ref<const Eigen::MatrixX3d> &pc,
                const Eigen::Ref<const Eigen::MatrixX3d> &normals) {
                 std::vector<Eigen::Vector3d> pc_in;
                 std::vector<Eigen::Vector3d> normals_in;
                 utils::EigenMatrixToVector(pc, pc_in);
                 utils::EigenMatrixToVector(normals, normals_in);
                 return self.Detect(pc_in, normals_in);
             })
        .def("get_primitives",
             [](PrimitivesDetector &self) {
                 std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> primitives;
                 const auto result = self.GetPrimitives();
                 for (auto &primitive : result) {
                     primitives.push_back(primitive.m_parameters.transpose());
                 }

                 return primitives;
             })
        .def("get_poses", &PrimitivesDetector::GetPoses)
        .def("get_clusters", [](PrimitivesDetector &self) {
            auto cs = self.GetClusters();
            const size_t num = cs.size();
            std::vector<Eigen::MatrixX3d> cs_out(num);
            for (size_t i = 0; i < num; i++) {
                Eigen::MatrixX3d c;
                utils::VectorToEigenMatrix(cs[i], c);
                cs_out[i] = c;
            }
            return cs_out;
        });
}
}  // namespace primitives_fitting
