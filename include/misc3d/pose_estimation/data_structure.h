#pragma once

#include <vector>

#include <open3d/geometry/KDTreeFlann.h>
#include <Eigen/Dense>

namespace misc3d {

namespace pose_estimation {

typedef Eigen::Vector3d PointXYZ;
typedef Eigen::Vector3d Normal;
typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rotation;
typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Transformation;

typedef open3d::geometry::KDTreeFlann KDTree;

// point pair feature: [<n1, d>, <n2, d>, <n1, n2>, ||d||].T
struct PointPair {
    int i_, j_;
    double alpha_;
    int quant_alpha_;
};

struct PoseCluster {
    PoseCluster() : num_votes_(0) {}
    size_t num_votes_;
    std::vector<int> pose_indices_;
};

class Pose6D {
public:
    /**
     * @brief Convert quaternion to rotation matrix
     *
     * @param q
     * @param R
     */
    static void QuatToMat(const Eigen::Vector4d &q, Rotation &R) {
        R(0, 0) = 1 - 2 * (q(2) * q(2) + q(3) * q(3));
        R(0, 1) = 2 * (q(1) * q(2) - q(0) * q(3));
        R(0, 2) = 2 * (q(0) * q(2) + q(1) * q(3));
        R(1, 0) = 2 * (q(1) * q(2) + q(0) * q(3));
        R(1, 1) = 1 - 2 * (q(1) * q(1) + q(3) * q(3));
        R(1, 2) = 2 * (q(2) * q(3) - q(0) * q(1));
        R(2, 0) = 2 * (q(1) * q(3) - q(0) * q(2));
        R(2, 1) = 2 * (q(0) * q(1) + q(2) * q(3));
        R(2, 2) = 1 - 2 * (q(1) * q(1) + q(2) * q(2));
    }

    /**
     * @brief rotation matrix to quaternion
     *
     * @param R
     * @param q
     */
    static void MatToQuat(const Rotation &R, Eigen::Vector4d &q) {
        const double tr = R(0, 0) + R(1, 1) + R(2, 2);
        double s;
        if (tr > -1) {  // a != 0
            s = sqrt(tr + 1) * 2;
            q(0) = 0.25 * s;
            q(1) = (R(2, 1) - R(1, 2)) / s;
            q(2) = (R(0, 2) - R(2, 0)) / s;
            q(3) = (R(1, 0) - R(0, 1)) / s;
        } else if (R(0, 0) > R(1, 1) &&
                   R(0, 0) > R(2, 2)) {  // max(b, c, d) = b != 0
            s = sqrt(1 + R(0, 0) - R(1, 1) - R(2, 2)) * 2;
            q(0) = (R(2, 1) - R(1, 2)) / s;
            q(1) = 0.25 * s;
            q(2) = (R(0, 1) + R(1, 0)) / s;
            q(3) = (R(0, 2) + R(2, 0)) / s;
        } else if (R(1, 1) > R(2, 2)) {  // max(b, c, d) = c != 0
            s = sqrt(1 + R(1, 1) - R(0, 0) - R(2, 2)) * 2;
            q(0) = (R(0, 2) - R(2, 0)) / s;
            q(1) = (R(0, 1) + R(1, 0)) / s;
            q(2) = 0.25 * s;
            q(3) = (R(1, 2) + R(2, 1)) / s;
        } else {  // max(b, c, d) = d != 0
            s = sqrt(1 + R(2, 2) - R(0, 0) - R(1, 1)) * 2;
            q(0) = (R(1, 0) - R(0, 1)) / s;
            q(1) = (R(0, 2) + R(2, 0)) / s;
            q(2) = (R(1, 2) + R(2, 1)) / s;
            q(3) = 0.25 * s;
        }
    }

public:
    Pose6D() {
        pose_ = Transformation::Identity();
        q_ << 0, 0, 0, 1.0;
        t_ << 0, 0, 0;
        num_votes_ = 0;
        score_ = 0;
        corr_mi_ = 0;
        object_id_ = 0;
    }

    /**
     * @brief update pose value from given transformation matrix
     *
     * @param new_pose
     */
    void UpdateByPose(const Transformation new_pose) {
        pose_ = new_pose;
        const Rotation R = new_pose.block<3, 3>(0, 0);
        MatToQuat(R, q_);
        t_ = new_pose.block<3, 1>(0, 3);
    }

    /**
     * @brief update pose value by quaternion
     *
     * @param q
     * @param t
     */
    void UpdateByQuat(const Eigen::Vector4d &q, const Eigen::Vector3d &t) {
        Rotation R;
        QuatToMat(q, R);
        pose_.block<3, 3>(0, 0) = R;
        pose_.block<3, 1>(0, 3) = t;
        q_ = q;
        t_ = t;
    }

public:
    Transformation pose_;
    Eigen::Vector4d q_;
    Eigen::Vector3d t_;
    size_t num_votes_;
    double score_;
    int corr_mi_;
    size_t object_id_;
};

}  // namespace pose_estimation
}  // namespace misc3d