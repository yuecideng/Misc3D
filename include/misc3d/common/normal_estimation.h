#pragma once

#include <memory>

#include <misc3d/utils.h>

namespace misc3d {

namespace common {

/**
 * @brief Estimate normals from point map structure
 *
 * @param pc
 * @param w width of RGBD data
 * @param h height of RGBD data
 * @param k pixel radius for neareast points, eg: if k = 5, t
 * @param view_point the normal will be oriented towards view point
 */
void EstimateNormalsFromMap(const PointCloudPtr &pc,
                            int w, int h, int k,
                            const std::array<double, 3> &view_point = {0, 0, 0});

}  // namespace common
}  // namespace misc3d