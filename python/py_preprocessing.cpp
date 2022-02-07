#include <py_misc3d.h>

#include <misc3d/preprocessing/conversion.h>

namespace misc3d {

namespace preprocessing {

void pybind_preprocessing(py::module &m) {
    m.def(
        "crop_roi_pointcloud",
        [](const PointCloudPtr &pc, const std::tuple<int, int, int, int> &roi,
           const std::tuple<int, int> &shape) {
            return CropROIPointCloud(*pc, roi, shape);
        },
        "Crop roi pointcloud", py::arg("pc"), py::arg("roi"), py::arg("shape"));
}

}  // namespace preprocessing
}  // namespace misc3d