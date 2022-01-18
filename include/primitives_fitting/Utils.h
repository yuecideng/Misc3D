#pragma once

#include <float.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <chrono>
#include <vector>

#include <open3d/geometry/PointCloud.h>
#include <Eigen/Core>

namespace primitives_fitting {

namespace utils {
/**
 * @brief perfoem normal consistent, here we have assumption that the point clouds are all
 *        in camera coordinate
 *
 * @param pc
 */
inline void NormalConsistent(open3d::geometry::PointCloud &pc) {
    if (!pc.HasNormals()) {
        std::cout << "(NormalConsistent) The target point cloud has no normals" << std::endl;
    } else {
        const int size = pc.points_.size();

#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            pc.normals_[i].normalize();
            if (pc.points_[i].dot(pc.normals_[i]) > 0) {
                pc.normals_[i] *= -1;
            }
        }
    }
}

/**
 * @brief extract data by indices
 *
 * @param src
 * @param index
 * @param dst
 */
template <typename T>
inline void GetVectorByIndex(const std::vector<T> &src, const std::vector<size_t> &index,
                             std::vector<T> &dst) {
    const size_t num = index.size();
    dst.resize(num);
    for (size_t i = 0; i < num; i++) {
        dst[i] = src[index[i]];
    }
}

/**
 * @brief Get Eigen matrix from vector
 *
 * @param src
 * @param index
 * @param dst
 */
inline void GetMatrixByIndex(const std::vector<Eigen::Vector3d> &src,
                             const std::vector<size_t> &index, Eigen::Matrix3Xd &dst) {
    const size_t num = index.size();
    dst.setZero(3, num);
    if (src.size() == 0) {
        return;
    }
#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        dst.col(i) = src[index[i]];
    }
}

/**
 * @brief convert degree to radian
 *
 * @param angle_deg
 * @return double
 */
inline double Deg2Rad(const double angle_deg) {
    return angle_deg / 180 * M_PI;
}

/**
 * @brief convert radian to degree
 *
 * @param angle_rad
 * @return double
 */
inline double Rad2Deg(const double angle_rad) {
    return angle_rad / M_PI * 180;
}

/**
 * @brief data conversion
 *
 * @param pc
 * @param new_pc
 */
inline void EigenMatrixToVector(const Eigen::Ref<const Eigen::MatrixX3d> &pc,
                                std::vector<Eigen::Vector3d> &new_pc) {
    const size_t num = pc.rows();
    const size_t data_length = sizeof(double) * 3;
    new_pc.resize(num);

#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        const Eigen::Vector3d &p = pc.row(i);
        memcpy(new_pc[i].data(), p.data(), data_length);
    }
}

/**
 * @brief data conversion
 *
 * @param pc
 * @param normal
 * @param new_pc
 */
inline void EigenMatrixToVector(const Eigen::Ref<const Eigen::MatrixX3d> &pc,
                                const Eigen::Ref<const Eigen::MatrixX3d> &normal,
                                std::vector<Eigen::Vector6d> &new_pc) {
    const size_t num = pc.rows();
    const size_t data_length = sizeof(double) * 3;
    new_pc.resize(num);

#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        const Eigen::Vector3d &p = pc.row(i);
        const Eigen::Vector3d &n = normal.row(i);
        memcpy(new_pc[i].data(), p.data(), data_length);
        memcpy(new_pc[i].data() + 3, n.data(), data_length);
    }
}

/**
 * @brief data conversion
 *
 * @param pc
 * @param new_pc
 */
inline void VectorToEigenMatrix(const std::vector<Eigen::Vector3d> &pc,
                                Eigen::Matrix<double, Eigen::Dynamic, 3> &new_pc) {
    const size_t num = pc.size();
    new_pc.setZero(num, 3);

#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        new_pc.row(i) = pc[i].transpose();
    }
}

/**
 * @brief data conversion
 *
 * @param pc
 * @param new_pc
 */
inline void VectorToEigenMatrix(const std::vector<Eigen::Vector6d> &pc,
                                Eigen::Matrix<double, Eigen::Dynamic, 6> &new_pc) {
    const size_t num = pc.size();
    new_pc.setZero(num, 6);

#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        new_pc.row(i) = pc[i].transpose();
    }
}

/**
 * @brief Compute the coordinate transformation between the target coordinate
 *        and origin coordinate
 *
 * @param x_head Point at target coordinate x-axis
 * @param origin Point at target coordinate origin
 * @param ref    Point at target coordinate x-y plane
 * @return Eigen::Matrix4d
 */
template <typename T>
Eigen::Matrix<T, 4, 4> CalcCoordinateTransform(const Eigen::Matrix<T, 3, 1> &x_head,
                                               const Eigen::Matrix<T, 3, 1> &origin,
                                               const Eigen::Matrix<T, 3, 1> &ref) {
    const Eigen::Matrix<T, 3, 1> x_axis = (x_head - origin) / (x_head - origin).norm();
    const Eigen::Matrix<T, 3, 1> tmp_axis = (ref - origin) / (ref - origin).norm();

    Eigen::Matrix<T, 3, 1> z_axis = x_axis.cross(tmp_axis);
    if (z_axis.dot(Eigen::Matrix<T, 3, 1>(0, 0, 1)) > 0) {
        z_axis /= z_axis.norm();
    } else {
        z_axis /= -z_axis.norm();
    }

    Eigen::Matrix<T, 3, 1> y_axis = z_axis.cross(x_axis);
    y_axis /= y_axis.norm();

    Eigen::Matrix<T, 4, 4> transform;
    transform << x_axis(0), y_axis(0), z_axis(0), origin(0), x_axis(1), y_axis(1), z_axis(1),
        origin(1), x_axis(2), y_axis(2), z_axis(2), origin(2), 0, 0, 0, 1;

    return transform;
}

/**
 * @brief compute point to line distance
 *
 * @tparam T
 * @param query
 * @param point1
 * @param point2
 * @return T
 */
template <typename T>
inline T CalcPoint2LineDistance(const Eigen::Matrix<T, 3, 1> &query,
                                const Eigen::Matrix<T, 3, 1> &point1,
                                const Eigen::Matrix<T, 3, 1> &point2) {
    const Eigen::Matrix<T, 3, 1> a = query - point1;
    const Eigen::Matrix<T, 3, 1> b = query - point2;
    const Eigen::Matrix<T, 3, 1> c = point2 - point1;

    return a.cross(b).norm() / c.norm();
}

}  // namespace utils
}  // namespace primitives_fitting
