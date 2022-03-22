#pragma once

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#else
#include <math.h>
#endif
#include <float.h>
#include <omp.h>
#include <string.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include <open3d/geometry/Geometry.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/pipelines/registration/Feature.h>
#include <Eigen/Dense>

namespace misc3d {

typedef std::shared_ptr<open3d::geometry::Geometry> GeometryPtr;
typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;
typedef std::shared_ptr<open3d::geometry::TriangleMesh> TriangleMeshPtr;
typedef std::shared_ptr<open3d::pipelines::registration::Feature> FeaturePtr;

/**
 * @brief Timer for duration measurement.
 *
 */
class Timer {
public:
    void Start() { t0_ = std::chrono::high_resolution_clock::now(); }
    double Stop() {
        const double timestamp =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0_)
                .count();
        return timestamp;
    }

private:
    std::chrono::high_resolution_clock::time_point t0_;
};

/**
 * @brief base sampler class
 *
 * @tparam T
 */
template <typename T>
class Sampler {
public:
    /**
     * @brief pure virtual operator, which define the I/O of this sampler
     *
     * @param sample_size
     * @return std::vector<T>
     */
    virtual std::vector<T> operator()(size_t sample_size) = 0;
};

/**
 * @brief Extract a random sample of given sample_size from the input indices
 *
 * @tparam T
 */
template <typename T>
class RandomSampler : public Sampler<T> {
public:
    explicit RandomSampler(const size_t size) : Sampler<T>(), size_(size) {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    // This operator is usually used in for loop and sample a small subset from
    // original indices
    std::vector<T> operator()(size_t sample_size) override {
        // Lock this operation when using OpenMP to ensure synchronization
        std::lock_guard<std::mutex> guard(mutex_);

        std::vector<T> sample;
        sample.reserve(sample_size);
        size_t valid_sample = 0;
        while (valid_sample < sample_size) {
            size_t idx = rng_() % size_;
            if (std::find(sample.begin(), sample.end(), idx) == sample.end()) {
                sample.push_back(idx);
                valid_sample++;
            }
        }

        return sample;
    }

    // This function is usually called once to sample more than half of original
    // indices
    std::vector<T> SampleWithoutDuplicate(size_t sample_size) {
        std::vector<T> indices(size_);
        std::iota(indices.begin(), indices.end(), 0);

        for (size_t i = 0; i < sample_size; ++i) {
            std::swap(indices[i], indices[rng_() % size_]);
        }

        std::vector<T> sample;
        sample.reserve(sample_size);
        for (int idx = 0; idx < sample_size; ++idx) {
            sample.push_back(indices[idx]);
        }

        return sample;
    }

private:
    size_t size_;
    std::mt19937 rng_;
    std::mutex mutex_;
};

/**
 * @brief perfoem normal consistent, here we have assumption that the point
 * clouds are all in camera coordinate
 *
 * @param pc
 */
inline void NormalConsistent(open3d::geometry::PointCloud &pc) {
    if (!pc.HasNormals()) {
        return;
    } else {
        const int size = pc.points_.size();

#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            if (pc.points_[i].dot(pc.normals_[i]) > 0) {
                pc.normals_[i] *= -1;
            }
            pc.normals_[i].normalize();
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
inline void GetVectorByIndex(const std::vector<T> &src,
                             const std::vector<size_t> &index,
                             std::vector<T> &dst) {
    const size_t num = index.size();
    dst.resize(num);
#pragma omp parallel for
    for (int i = 0; i < num; i++) {
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
template <typename T>
inline void GetMatrixByIndex(const std::vector<Eigen::Matrix<T, 3, 1>> &src,
                             const std::vector<size_t> &index,
                             Eigen::Matrix<T, 3, Eigen::Dynamic> &dst) {
    const size_t num = index.size();
    dst.setZero(3, num);
    if (src.size() == 0) {
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        dst.col(i) = src[index[i]];
    }
}

/**
 * @brief data conversion
 *
 * @param pc
 * @param new_pc
 */
template <typename T>
inline void EigenMatrixToVector(const Eigen::Matrix<T, Eigen::Dynamic, 3> &pc,
                                std::vector<Eigen::Matrix<T, 3, 1>> &new_pc) {
    const size_t num = pc.rows();
    const size_t data_length = sizeof(T) * 3;
    new_pc.resize(num);

#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        const Eigen::Matrix<T, 3, 1> &p = pc.row(i);
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
template <typename T>
inline void EigenMatrixToVector(
    const Eigen::Matrix<T, Eigen::Dynamic, 3> &pc,
    const Eigen::Matrix<T, Eigen::Dynamic, 3> &normal,
    std::vector<Eigen::Matrix<T, 6, 1>> &new_pc) {
    const size_t num = pc.rows();
    const size_t data_length = sizeof(T) * 3;
    new_pc.resize(num);

#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        const Eigen::Matrix<T, 3, 1> &p = pc.row(i);
        const Eigen::Matrix<T, 3, 1> &n = normal.row(i);
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
template <typename T>
inline void VectorToEigenMatrix(const std::vector<Eigen::Matrix<T, 3, 1>> &pc,
                                Eigen::Matrix<T, 3, Eigen::Dynamic> &new_pc) {
    const size_t num = pc.size();
    new_pc.setZero(3, num);

#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        new_pc.col(i) = pc[i];
    }
}

/**
 * @brief data conversion
 *
 * @param pc
 * @param new_pc
 */
template <typename T>
inline void VectorToEigenMatrix(const std::vector<Eigen::Matrix<T, 6, 1>> &pc,
                                Eigen::Matrix<T, 6, Eigen::Dynamic> &new_pc) {
    const size_t num = pc.size();
    new_pc.setZero(6, num);

#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        new_pc.col(i) = pc[i];
    }
}

/**
 * @brief Compute the coordinate transformation between the target coordinate
 *        and origin coordinate
 *
 * @param x_head Point at target coordinate x-axis
 * @param origin Point at target coordinate origin
 * @param ref    Point at target coordinate x-y plane
 * @return Eigen::Matrix<T, 4, 4>
 */
template <typename T>
inline Eigen::Matrix<T, 4, 4> CalcCoordinateTransform(
    const Eigen::Matrix<T, 3, 1> &x_head, const Eigen::Matrix<T, 3, 1> &origin,
    const Eigen::Matrix<T, 3, 1> &ref) {
    const Eigen::Matrix<T, 3, 1> x_axis =
        (x_head - origin) / (x_head - origin).norm();
    const Eigen::Matrix<T, 3, 1> tmp_axis =
        (ref - origin) / (ref - origin).norm();

    Eigen::Matrix<T, 3, 1> z_axis = x_axis.cross(tmp_axis);
    if (z_axis.dot(Eigen::Matrix<T, 3, 1>(0, 0, 1)) > 0) {
        z_axis /= z_axis.norm();
    } else {
        z_axis /= -z_axis.norm();
    }

    Eigen::Matrix<T, 3, 1> y_axis = z_axis.cross(x_axis);
    y_axis /= y_axis.norm();

    Eigen::Matrix<T, 4, 4> transform;
    transform << x_axis(0), y_axis(0), z_axis(0), origin(0), x_axis(1),
        y_axis(1), z_axis(1), origin(1), x_axis(2), y_axis(2), z_axis(2),
        origin(2), 0, 0, 0, 1;

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

/**
 * @brief convert degree to radian
 *
 * @param angle_deg
 * @return double
 */
template <typename T>
inline T Deg2Rad(const T angle_deg) {
    return angle_deg / 180 * M_PI;
}

/**
 * @brief convert radian to degree
 *
 * @param angle_rad
 * @return double
 */
template <typename T>
inline T Rad2Deg(const T angle_rad) {
    return angle_rad / M_PI * 180;
}

/**
 * @brief data type conversion
 *
 * @param pc
 * @param new_pc
 */
inline void VectorToO3dPointCloud(const std::vector<Eigen::Vector6d> &pc,
                                  open3d::geometry::PointCloud &new_pc) {
    const int n_pt = pc.size();
    const size_t data_length = sizeof(double) * 3;
    new_pc.points_.resize(n_pt);
    new_pc.normals_.resize(n_pt);

#pragma omp parallel for
    for (int i = 0; i < n_pt; i++) {
        memcpy(new_pc.points_[i].data(), pc[i].data(), data_length);
        memcpy(new_pc.normals_[i].data(), pc[i].data() + 3, data_length);
    }
}

}  // namespace misc3d