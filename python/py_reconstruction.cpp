#include <py_misc3d.h>

#include <misc3d/resonstruction/pipeline.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace reconstruction {

void pybind_reconstruction(py::module &m) {
    py::class_<PipelineConfig> config(m, "PipelineConfig");
    config.def(py::init<>());

    py::enum_<PipelineConfig::DescriptorType>(config, "DescriptorType")
        .value("ORB", PipelineConfig::DescriptorType::ORB)
        .value("SIFT", PipelineConfig::DescriptorType::SIFT)
        .export_values();

    py::class_<PipelineConfig::MakeFragmentParam>(config, "MakeFragmentParam")
        .def(py::init<>())
        .def_readwrite("descriptor_type",
                       &PipelineConfig::MakeFragmentParam::descriptor_type)
        .def_readwrite("feature_num",
                       &PipelineConfig::MakeFragmentParam::feature_num)
        .def_readwrite("n_frame_per_fragment",
                       &PipelineConfig::MakeFragmentParam::n_frame_per_fragment)
        .def_readwrite("keyframe_ratio",
                       &PipelineConfig::MakeFragmentParam::keyframe_ratio);

    py::class_<PipelineConfig::OptimizationParam>(config, "OptimizationParam")
        .def(py::init<>())
        .def_readwrite("preference_loop_closure_odometry",
                       &PipelineConfig::OptimizationParam::
                           preference_loop_closure_odometry)
        .def_readwrite("preference_loop_closure_registration",
                       &PipelineConfig::OptimizationParam::
                           preference_loop_closure_registration);

    py::enum_<PipelineConfig::LocalRefineMethod>(config, "LocalRefineMethod")
        .value("Point2PointICP",
               PipelineConfig::LocalRefineMethod::Point2PointICP)
        .value("Point2PlaneICP",
               PipelineConfig::LocalRefineMethod::Point2PlaneICP)
        .value("ColoredICP", PipelineConfig::LocalRefineMethod::ColoredICP)
        .value("GeneralizedICP",
               PipelineConfig::LocalRefineMethod::GeneralizedICP)
        .export_values();

    py::enum_<PipelineConfig::GlobalRegistrationMethod>(
        config, "GlobalRegistrationMethod")
        .value("Ransac", PipelineConfig::GlobalRegistrationMethod::Ransac)
        .value("TeaserPlusPlus",
               PipelineConfig::GlobalRegistrationMethod::TeaserPlusPlus)
        .export_values();

    config.def_readwrite("data_path", &PipelineConfig::data_path_);
    config.def_readwrite("camera_intrinsic",
                         &PipelineConfig::camera_intrinsic_);
    config.def_readwrite("depth_scale", &PipelineConfig::depth_scale_);
    config.def_readwrite("max_depth", &PipelineConfig::max_depth_);
    config.def_readwrite("max_depth_diff", &PipelineConfig::max_depth_diff_);
    config.def_readwrite("voxel_size", &PipelineConfig::voxel_size_);
    config.def_readwrite("enable_slac", &PipelineConfig::enable_slac_);
    config.def_readwrite("integration_voxel_size",
                         &PipelineConfig::integration_voxel_size_);
    config.def_readwrite("make_fragment_param",
                         &PipelineConfig::make_fragment_param_);
    config.def_readwrite("local_refine_method",
                         &PipelineConfig::local_refine_method_);
    config.def_readwrite("global_registration_method",
                         &PipelineConfig::global_registration_method_);
    config.def_readwrite("optimization_param",
                         &PipelineConfig::optimization_param_);

    py::class_<ReconstructionPipeline>(m, "ReconstructionPipeline")
        .def(py::init<const PipelineConfig &>())
        .def(py::init<const std::string &>())
        .def("get_data_path", &ReconstructionPipeline::GetDataPath)
        .def("make_fragments", &ReconstructionPipeline::MakeFragments)
        .def("register_fragments", &ReconstructionPipeline::RegisterFragments)
        .def("integrate_scene", &ReconstructionPipeline::IntegrateScene)
        .def("run_system", &ReconstructionPipeline::RunSystem);
}

}  // namespace reconstruction
}  // namespace misc3d