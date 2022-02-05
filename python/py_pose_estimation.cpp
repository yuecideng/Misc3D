#include <py_misc3d.h>

#include <misc3d/pose_estimation/ppf_estimation.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace pose_estimation {

void pybind_pose_estimation(py::module &m) {
    py::class_<PPFEstimator>(m, "PPFEstimator")
        .def(py::init<>())
        .def(py::init<const PPFEstimatorConfig &>())
        .def("set_config", &PPFEstimator::SetConfig)
        .def("train", &PPFEstimator::Train)
        .def("match",
             [](PPFEstimator &self, const PointCloudPtr &pc) {
                 std::vector<Pose6D> results;
                 bool ret = self.Estimate(pc, results);
                 return std::make_tuple(ret, results);
             })
        .def("get_model_diameter", &PPFEstimator::GetModelDiameter)
        .def("getl_pose", &PPFEstimator::GetPose)
        .def("get_sampled_model", &PPFEstimator::GetSampledModel)
        .def("get_sampled_scene", &PPFEstimator::GetSampledScene)
        .def("get_model_edges", &PPFEstimator::GetModelEdges)
        .def("get_scene_edges", &PPFEstimator::GetSceneEdges);

    py::class_<Pose6D>(m, "Pose6D")
        .def(py::init<>())
        .def_readwrite("pose", &Pose6D::pose_)
        .def_readwrite("q", &Pose6D::q_)
        .def_readwrite("t", &Pose6D::t_)
        .def_readwrite("num_votes", &Pose6D::num_votes_)
        .def_readwrite("score", &Pose6D::score_)
        .def_readwrite("object_id", &Pose6D::object_id_)
        .def_readwrite("corr_mi", &Pose6D::corr_mi_);

    py::class_<PPFEstimatorConfig> config(m, "PPFEstimatorConfig");
    config.def(py::init<>());

    py::class_<PPFEstimatorConfig::TrainingParam>(config, "TrainingParam")
        .def(py::init<>())
        .def_readwrite("invert_model_normal",
                       &PPFEstimatorConfig::TrainingParam::invert_model_normal)
        .def_readwrite("rel_sample_dist",
                       &PPFEstimatorConfig::TrainingParam::rel_sample_dist)
        .def_readwrite("calc_normal_relative",
                       &PPFEstimatorConfig::TrainingParam::calc_normal_relative)
        .def_readwrite(
            "rel_dense_sample_dist",
            &PPFEstimatorConfig::TrainingParam::rel_dense_sample_dist);

    py::enum_<PPFEstimatorConfig::ReferencePointSelection>(
        config, "ReferencePointSelection")
        .value("Random", PPFEstimatorConfig::ReferencePointSelection::Random)
        .export_values();

    py::enum_<PPFEstimatorConfig::RefineMethod>(config, "RefineMethod")
        .value("PointToPlane", PPFEstimatorConfig::RefineMethod::PointToPlane)
        .value("PointToPoint", PPFEstimatorConfig::RefineMethod::PointToPoint)
        .export_values();

    py::enum_<PPFEstimatorConfig::VotingMode>(config, "VotingMode")
        .value("SampledPoints", PPFEstimatorConfig::VotingMode::SampledPoints)
        .value("EdgePoints", PPFEstimatorConfig::VotingMode::EdgePoints)
        .export_values();

    py::class_<PPFEstimatorConfig::EdgeParam>(config, "EdgeParam")
        .def(py::init<>())
        .def_readwrite("pts_num", &PPFEstimatorConfig::EdgeParam::pts_num);

    py::class_<PPFEstimatorConfig::ReferenceParam>(config, "ReferenceParam")
        .def(py::init<>())
        .def_readwrite("method", &PPFEstimatorConfig::ReferenceParam::method)
        .def_readwrite("ratio", &PPFEstimatorConfig::ReferenceParam::ratio);

    py::class_<PPFEstimatorConfig::VotingParam>(config, "VotingParam")
        .def(py::init<>())
        .def_readwrite("method", &PPFEstimatorConfig::VotingParam::method)
        .def_readwrite("angle_step",
                       &PPFEstimatorConfig::VotingParam::angle_step)
        .def_readwrite("min_dist_thresh",
                       &PPFEstimatorConfig::VotingParam::min_dist_thresh)
        .def_readwrite("min_angle_thresh",
                       &PPFEstimatorConfig::VotingParam::min_angle_thresh);

    py::class_<PPFEstimatorConfig::RefineParam>(config, "RefineParam")
        .def(py::init<>())
        .def_readwrite("method", &PPFEstimatorConfig::RefineParam::method)
        .def_readwrite(
            "rel_dist_sparse_thresh",
            &PPFEstimatorConfig::RefineParam::rel_dist_sparse_thresh);

    config.def_readwrite("training_param",
                         &PPFEstimatorConfig::training_param_);
    config.def_readwrite("ref_param", &PPFEstimatorConfig::ref_param_);
    config.def_readwrite("voting_param", &PPFEstimatorConfig::voting_param_);
    config.def_readwrite("edge_param", &PPFEstimatorConfig::edge_param_);
    config.def_readwrite("refine_param", &PPFEstimatorConfig::refine_param_);
    config.def_readwrite("score_thresh", &PPFEstimatorConfig::score_thresh_);
    config.def_readwrite("num_result", &PPFEstimatorConfig::num_result_);
    config.def_readwrite("object_id", &PPFEstimatorConfig::object_id_);
}

}  // namespace pose_estimation
}  // namespace misc3d