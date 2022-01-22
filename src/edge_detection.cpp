#include <misc3d/features/edge_detection.h>
#include <misc3d/logger.h>

namespace misc3d {

namespace features {

Eigen::Vector4d ToEigenVector4(const Eigen::Vector3d& in) {
    Eigen::Vector4d out = Eigen::Vector4d::Zero();
    memcpy(out.data(), in.data(), sizeof(double) * 3);
    return out;
}

void GetCoordinateSystemOnPlane(const Eigen::Vector3d& query, Eigen::Vector4d& u,
                                Eigen::Vector4d& v) {
    const Eigen::Vector4d vector = ToEigenVector4(query);
    v = vector.unitOrthogonal();
    u = vector.cross3(v);
}

bool IsBoundaryPoint(const open3d::geometry::PointCloud& pc,
                     const Eigen::Vector3d& query, const std::vector<int> indices,
                     const Eigen::Vector4d& u, const Eigen::Vector4d& v,
                     double angle_threshold) {
    std::vector<double> angles;
    for (size_t i = 0; i < indices.size(); i++) {
        const Eigen::Vector4d delta =
            ToEigenVector4(pc.points_[indices[i]]) - ToEigenVector4(query);
        if (delta == Eigen::Vector4d::Zero()) {
            continue;
        }
        angles.push_back(atan2(v.dot(delta), u.dot(delta)));
    }

    if (angles.empty()) {
        return false;
    }

    std::sort(angles.begin(), angles.end());
    // Compute the maximal angle difference between two consecutive angles
    double dif;
    double max_dif = 0;
    for (size_t i = 0; i < angles.size() - 1; ++i) {
        dif = angles[i + 1] - angles[i];
        if (max_dif < dif) {
            max_dif = dif;
        }
    }

    // Get the angle difference between the last and the first
    dif = 2 * M_PI - angles[angles.size() - 1] + angles[0];
    if (max_dif < dif)
        max_dif = dif;

    // Check results
    if (max_dif > angle_threshold * M_PI / 180.0)
        return true;
    else
        return false;
}

std::vector<size_t> DetectEdgePoints(
    const open3d::geometry::PointCloud& pc,
    const open3d::geometry::KDTreeSearchParam& param, double angle_threshold) {
    std::vector<size_t> edge_indices;
    if (!pc.HasPoints()) {
        MISC3D_ERROR("No PointCloud data.");
        return edge_indices;
    }

    open3d::geometry::PointCloud pc_(pc);
    if (!pc.HasNormals()) {
        MISC3D_INFO("Computing normals.");
        pc_.EstimateNormals(param);
        pc_.OrientNormalsTowardsCameraLocation();
    }

    const size_t num = pc_.points_.size();
    open3d::geometry::KDTreeFlann kdtree(pc_);
#pragma omp parallel for
    for (size_t idx = 0; idx < num; idx++) {
        std::vector<int> ret_indices;
        std::vector<double> out_dists_sqr;
        if (kdtree.Search(pc_.points_[idx], param, ret_indices, out_dists_sqr) < 3) {
            continue;
        }
        Eigen::Vector4d u = Eigen::Vector4d::Zero(), v = Eigen::Vector4d::Zero();
        // Obtain a coordinate system on the plane
        GetCoordinateSystemOnPlane(pc_.normals_[idx], u, v);

        // Estimate whether the point is lying on a boundary surface or not
        if (IsBoundaryPoint(pc_, pc_.points_[idx], ret_indices, u, v,
                            angle_threshold)) {
#pragma omp critical
            { edge_indices.push_back(idx); }
        }
    }
    MISC3D_INFO("Found {} edge points from {} input points.", edge_indices.size(),
                num);

    return edge_indices;
}

}  // namespace features

}  // namespace misc3d
