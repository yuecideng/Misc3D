#include <py_misc3d.h>

#include <misc3d/features/edge_detection.h>
#include <misc3d/utils.h>

namespace misc3d {

namespace features {

void pybind_features(py::module& m) {
    m.def(
        "detect_edge_points",
        [](const PointCloudPtr& pc,
           const open3d::geometry::KDTreeSearchParam& param,
           double angle_threshold) {
            return DetectEdgePoints(*pc, param, angle_threshold);
        },
        py::arg("pc"),
        py::arg("param") = open3d::geometry::KDTreeSearchParamHybrid(0.01, 30),
        py::arg("angle_threshold") = 90.0);
}

}  // namespace features
}  // namespace misc3d