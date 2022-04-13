#include <py_misc3d.h>

#include <misc3d/resonstruction/pipeline.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace reconstruction {

void pybind_reconstruction(py::module& m) {
    py::class_<PipelineConfig> config(m, "PipelineConfig");
    config.def(py::init<>());

    py::class_<PipelineConfig::MakeFragmentParam>(config, "MakeFragmentParam")
        .def(py::init<>())
        .def_readwrite("orb_feature_num",
                       &PipelineConfig::MakeFragmentParam::orb_feature_num)
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

    py::enum_<PipelineConfig::GlobalRegistrationMethod>(config,
                                                        "GlobalRegistrationMethod")
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
        .def(py::init<const PipelineConfig&>())
        .def("make_fragments", &ReconstructionPipeline::MakeFragments);
}

}  // namespace reconstruction
}  // namespace misc3d