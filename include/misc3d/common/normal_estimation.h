#pragma once

#include <open3d/geometry/PointCloud.h>

namespace misc3d {

namespace common {

/**
 * @brief Estimate normals from point map structure
 * 
 * @param pc 
 * @param w width of RGBD data
 * @param h height of RGBD data
 * @param k pixel radius for neareast points, eg: if k = 5, t
 */
void EstimateNormalsFromMap(open3d::geometry::PointCloud &pc, int w, int h, int k);

}
}