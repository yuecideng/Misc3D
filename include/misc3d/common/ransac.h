#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <misc3d/utils.h>
#include <open3d/geometry/PointCloud.h>
#include <Eigen/Core>

#define EPS 1.0e-8

namespace misc3d {

namespace common {

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
     * @param indices
     * @param sample_size
     * @return std::vector<T>
     */
    virtual std::vector<T> operator()(const std::vector<T> &indices,
                                      size_t sample_size) const = 0;
};

/**
 * @brief Extract a random sample of given sample_size from the input indices
 *
 * @tparam T
 */
template <typename T>
class RandomSampler : public Sampler<T> {
public:
    std::vector<T> operator()(const std::vector<T> &indices,
                              size_t sample_size) const override {
        std::random_device rd;
        std::mt19937 rng(rd());
        const size_t num = indices.size();
        std::vector<T> indices_ = indices;

        for (size_t i = 0; i < sample_size; ++i) {
            std::swap(indices_[i], indices_[rng() % num]);
        }

        std::vector<T> sample(sample_size);
        for (int idx = 0; idx < sample_size; ++idx) {
            sample[idx] = indices_[idx];
        }

        return sample;
    }
};

/**
 * @brief base primitives model
 *
 */
class Model {
public:
    Eigen::VectorXd parameters_;  // The parameters of the current model

    Model(const Eigen::VectorXd &parameters) : parameters_(parameters) {}
    Model &operator=(const Model &model) {
        parameters_ = model.parameters_;
        return *this;
    }

    Model() {}
};

/**
 * @brief the plane model is described as [a, b, c, d] => ax + by + cz + d = 0
 *
 */
class Plane : public Model {
public:
    Plane() : Model(Eigen::VectorXd(4)){};
    Plane(const Plane &model) { parameters_ = model.parameters_; }
    Plane &operator=(const Plane &model) {
        parameters_ = model.parameters_;
        return *this;
    }
};

/**
 * @brief the sphere is describe as [x, y, z, r], where the first three are center
 * and the last is radius
 *
 */
class Sphere : public Model {
public:
    Sphere() : Model(Eigen::VectorXd(4)){};
    Sphere(const Sphere &model) { parameters_ = model.parameters_; }
    Sphere &operator=(const Sphere &model) {
        parameters_ = model.parameters_;
        return *this;
    }
};

/**
 * @brief the cylinder is describe as [x, y, z, nx, ny, nz, r], where the first three
 * is a point on the cylinder axis and the second three are normal vector or called
 * direction vector, the last one is radius
 *
 */
class Cylinder : public Model {
public:
    Cylinder() : Model(Eigen::VectorXd(7)){};
    Cylinder(const Cylinder &model) { parameters_ = model.parameters_; }
    Cylinder &operator=(const Cylinder &model) {
        parameters_ = model.parameters_;
        return *this;
    }
};

class ModelEstimator {
protected:
    ModelEstimator(int minimal_sample) : minimal_sample_(minimal_sample) {}

    /**
     * @brief check whether number of input points meet the minimal requirement
     *
     * @param num
     * @return true
     * @return false
     */
    bool MinimalCheck(int num) const {
        return num >= minimal_sample_ ? true : false;
    }

public:
    /**
     * @brief fit model using least sample points
     *
     * @param pc
     * @param model
     * @return true
     * @return false
     */
    virtual bool MinimalFit(const open3d::geometry::PointCloud &pc,
                            Model &model) const = 0;

    /**
     * @brief fit model using least square method
     *
     * @param pc
     * @param model
     * @return true
     * @return false
     */
    virtual bool GeneralFit(const open3d::geometry::PointCloud &pc,
                            Model &model) const = 0;

    /**
     * @brief evaluate point distance to model to determine inlier&outlier
     *
     * @param query
     * @param model
     * @return double
     */
    virtual double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                            const Model &model) const = 0;

public:
    int minimal_sample_;
};

class PlaneEstimator : public ModelEstimator {
public:
    PlaneEstimator() : ModelEstimator(3) {}

    bool MinimalFit(const open3d::geometry::PointCloud &pc,
                    Model &model) const override {
        if (!MinimalCheck(pc.points_.size())) {
            return false;
        }

        const auto &points = pc.points_;

        const Eigen::Vector3d e0 = points[1] - points[0];
        const Eigen::Vector3d e1 = points[2] - points[0];
        Eigen::Vector3d abc = e0.cross(e1);
        const double norm = abc.norm();
        // if the three points are co-linear, return invalid plane
        if (norm < EPS) {
            return false;
        }
        abc /= abc.norm();
        const double d = -abc.dot(points[0]);
        model.parameters_(0) = abc(0);
        model.parameters_(1) = abc(1);
        model.parameters_(2) = abc(2);
        model.parameters_(3) = d;

        return true;
    }

    bool GeneralFit(const open3d::geometry::PointCloud &pc,
                    Model &model) const override {
        const size_t num = pc.points_.size();
        if (!MinimalCheck(num)) {
            return false;
        }

        const auto &points = pc.points_;
        Eigen::Vector3d mean(0, 0, 0);
        for (auto &p : points) {
            mean += p;
        }
        mean /= double(num);

        double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
#pragma omp parallel for reduction(+ : xx, xy, xz, yy, yz, zz)
        for (size_t i = 0; i < num; ++i) {
            const Eigen::Vector3d residual = points[i] - mean;
            xx += residual(0) * residual(0);
            xy += residual(0) * residual(1);
            xz += residual(0) * residual(2);
            yy += residual(1) * residual(1);
            yz += residual(1) * residual(2);
            zz += residual(2) * residual(2);
        }

        const double det_x = yy * zz - yz * yz;
        const double det_y = xx * zz - xz * xz;
        const double det_z = xx * yy - xy * xy;

        Eigen::Vector3d abc;
        if (det_x > det_y && det_x > det_z) {
            abc = Eigen::Vector3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
        } else if (det_y > det_z) {
            abc = Eigen::Vector3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
        } else {
            abc = Eigen::Vector3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
        }

        const double norm = abc.norm();
        if (norm < EPS) {
            return false;
        }

        abc /= norm;
        const double d = -abc.dot(mean);
        model.parameters_ = Eigen::Vector4d(abc(0), abc(1), abc(2), d);

        return true;
    }

    double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                    const Model &model) const override {
        const Eigen::Vector4d p(query(0), query(1), query(2), 1);
        return std::abs(model.parameters_.transpose() * p) /
               model.parameters_.head<3>().norm();
    }
};

class SphereEstimator : public ModelEstimator {
private:
    bool ValidationCheck(const open3d::geometry::PointCloud &pc) const {
        PlaneEstimator fit;
        Plane plane;
        const bool ret = fit.MinimalFit(pc, plane);
        if (!ret) {
            return false;
        }
        return fit.CalcPointToModelDistance(pc.points_[3], plane) < EPS ? false
                                                                        : true;
    }

public:
    SphereEstimator() : ModelEstimator(4) {}

    bool MinimalFit(const open3d::geometry::PointCloud &pc,
                    Model &model) const override {
        const auto &points = pc.points_;
        if (!MinimalCheck(points.size()) || !ValidationCheck(pc)) {
            return false;
        }

        Eigen::Matrix4d det_mat;
        det_mat.setOnes(4, 4);
        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points[i](0);
            det_mat(i, 1) = points[i](1);
            det_mat(i, 2) = points[i](2);
        }
        const double M11 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points[i].transpose() * points[i];
            det_mat(i, 1) = points[i](1);
            det_mat(i, 2) = points[i](2);
        }
        const double M12 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points[i].transpose() * points[i];
            det_mat(i, 1) = points[i](0);
            det_mat(i, 2) = points[i](2);
        }
        const double M13 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points[i].transpose() * points[i];
            det_mat(i, 1) = points[i](0);
            det_mat(i, 2) = points[i](1);
        }
        const double M14 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points[i].transpose() * points[i];
            det_mat(i, 1) = points[i](0);
            det_mat(i, 2) = points[i](1);
            det_mat(i, 3) = points[i](2);
        }
        const double M15 = det_mat.determinant();

        const Eigen::Vector3d center(0.5 * (M12 / M11), -0.5 * (M13 / M11),
                                     0.5 * (M14 / M11));
        const double radius = std::sqrt(center.transpose() * center - (M15 / M11));
        model.parameters_(0) = center(0);
        model.parameters_(1) = center(1);
        model.parameters_(2) = center(2);
        model.parameters_(3) = radius;

        return true;
    }

    bool GeneralFit(const open3d::geometry::PointCloud &pc,
                    Model &model) const override {
        const size_t num = pc.points_.size();
        if (!MinimalCheck(num)) {
            return false;
        }

        const auto &o3d_points = pc.points_;
        Eigen::Matrix<double, 3, Eigen::Dynamic> points;
        VectorToEigenMatrix<double>(o3d_points, points);

        Eigen::Matrix<double, Eigen::Dynamic, 4> A;
        A.setOnes(num, 4);
        A.col(0) = points.row(0).transpose() * 2;
        A.col(1) = points.row(1).transpose() * 2;
        A.col(2) = points.row(2).transpose() * 2;

        Eigen::VectorXd b =
            (points.row(0).array().pow(2) + points.row(1).array().pow(2) +
             points.row(2).array().pow(2))
                .matrix();

        // TODO: dangerous when b is very large, which need large memory to compute
        // v. should be improved.
        const Eigen::Vector4d w =
            A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
        const double radius = sqrt(w(0) * w(0) + w(1) * w(1) + w(2) * w(2) + w(3));
        model.parameters_(0) = w(0);
        model.parameters_(1) = w(1);
        model.parameters_(2) = w(2);
        model.parameters_(3) = radius;

        return true;
    }

    double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                    const Model &model) const override {
        const Eigen::Vector3d center = model.parameters_.head<3>();
        const double radius = model.parameters_(3);
        const double d = (query - center).norm();

        if (d <= radius) {
            return radius - d;
        } else {
            return d - radius;
        }
    }
};

/**
 * @brief Cylinder estimation reference from PCL implementation.
 *
 */
class CylinderEstimator : public ModelEstimator {
public:
    CylinderEstimator() : ModelEstimator(2) {}

    bool MinimalFit(const open3d::geometry::PointCloud &pc,
                    Model &model) const override {
        if (!pc.HasNormals()) {
            MISC3D_ERROR("Cylinder estimation requires normals.");
            return false;
        }

        if (!MinimalCheck(pc.points_.size())) {
            return false;
        }

        const auto &points = pc.points_;
        const auto &normals = pc.normals_;
        if (fabs(points[0](0) - points[1](0) <=
                     std::numeric_limits<double>::epsilon() &&
                 fabs(points[0](1) - points[1](1)) <=
                     std::numeric_limits<float>::epsilon() &&
                 fabs(points[0](2) - points[1](2)) <=
                     std::numeric_limits<float>::epsilon())) {
            return false;
        }

        const Eigen::Vector4d p1(points[0](0), points[0](1), points[0](2), 0);
        const Eigen::Vector4d p2(points[1](0), points[1](1), points[1](2), 0);

        const Eigen::Vector4d n1(normals[0](0), normals[0](1), normals[0](2), 0);
        const Eigen::Vector4d n2(normals[1](0), normals[1](1), normals[1](2), 0);
        const Eigen::Vector4d w = n1 + p1 - p2;

        const double a = n1.dot(n1);
        const double b = n1.dot(n2);
        const double c = n2.dot(n2);
        const double d = n1.dot(w);
        const double e = n2.dot(w);
        const double denominator = a * c - b * b;
        double sc, tc;

        if (denominator < 1e-8)  // The lines are almost parallel
        {
            sc = 0;
            tc = (b > c ? d / b : e / c);  // Use the largest denominator
        } else {
            sc = (b * e - c * d) / denominator;
            tc = (a * e - b * d) / denominator;
        }

        const Eigen::Vector4d line_pt = p1 + n1 + sc * n1;
        Eigen::Vector4d line_dir = p2 + tc * n2 - line_pt;
        line_dir.normalize();

        model.parameters_[0] = line_pt[0];
        model.parameters_[1] = line_pt[1];
        model.parameters_[2] = line_pt[2];
        model.parameters_[3] = line_dir[0];
        model.parameters_[4] = line_dir[1];
        model.parameters_[5] = line_dir[2];
        // cylinder radius
        model.parameters_[6] = CalcPoint2LineDistance<double>(
            points[0], line_pt.head<3>(), line_dir.head<3>());

        return true;
    }  // namespace ransac

    /**
     * @brief the general fit of clinder model is not implemented yet
     * TODO: 1. linear least square method. 2. nonlinear least square method.
     *
     * @param pc
     * @return true
     * @return false
     */
    bool GeneralFit(const open3d::geometry::PointCloud &pc, Model &model) const {
        // if (!MinimalCheck(points.cols())) {
        //     return false;
        // }
        return true;
    }

    double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                    const Model &model) const override {
        const Eigen::Matrix<double, 7, 1> w = model.parameters_;
        const Eigen::Vector3d n(w(3), w(4), w(5));
        const Eigen::Vector3d center(w(0), w(1), w(2));

        const Eigen::Vector3d ref(w(0) + w(3), w(1) + w(4), w(2) + w(5));
        double d = CalcPoint2LineDistance<double>(query, center, ref);

        return abs(d - w(6));
    }
};  // namespace misc3d

/**
 * @brief RANSAC class for model fitting.
 *
 * @tparam ModelEstimator
 * @tparam Model
 * @tparam Sampler
 */
template <class ModelEstimator, class Model, class Sampler>
class RANSAC {
public:
    RANSAC()
        : fitness_(0)
        , inlier_rmse_(0)
        , max_iteration_(1000)
        , probability_(0.99)
        , enable_parallel_(false) {}

    /**
     * @brief Set Point Cloud to be used for RANSAC
     *
     * @param points
     */
    void SetPointCloud(const open3d::geometry::PointCloud &pc) {
        if (!pc_.HasPoints()) {
            pc_.Clear();
        }

        pc_ = pc;
    }

    /**
     * @brief set probability to find the best model
     *
     * @param probability
     */
    void SetProbability(double probability) { probability_ = probability; }

    /**
     * @brief set maximum iteration, usually used if using parallel ransac fitting
     *
     * @param num
     */
    void SetMaxIteration(size_t num) { max_iteration_ = num; }

    /**
     * @brief enable parallel ransac fitting
     *
     * @param flag
     */
    void SetParallel(bool flag) { enable_parallel_ = flag; }

    /**
     * @brief fit model with given parameters
     *
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool FitModel(double threshold, Model &model,
                  std::vector<size_t> &inlier_indices) {
        Clear();
        const size_t num_points = pc_.points_.size();
        if (num_points < estimator_.minimal_sample_) {
            MISC3D_WARN("Can not fit model due to lack of points");
            return false;
        }

        if (enable_parallel_) {
            return FitModelParallel(threshold, model, inlier_indices);
        } else {
            return FitModelVanilla(threshold, model, inlier_indices);
        }
    }

private:
    void Clear() {
        fitness_ = 0;
        inlier_rmse_ = 0;
    }

    /**
     * @brief refine model using general fitting of estimator, usually is least
     * square method.
     *
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool RefineModel(double threshold, Model &model,
                     std::vector<size_t> &inlier_indices) {
        inlier_indices.clear();
        for (size_t i = 0; i < pc_.points_.size(); ++i) {
            const double d =
                estimator_.CalcPointToModelDistance(pc_.points_[i], model);
            if (d < threshold) {
                inlier_indices.emplace_back(i);
            }
        }

        // improve best model using general fitting
        const auto inliers_pc = pc_.SelectByIndex(inlier_indices);

        return estimator_.GeneralFit(*inliers_pc, model);
    }

    /**
     * @brief vanilla ransac fitting method, the iteration number is varying with
     * inlier number in each iteration
     *
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool FitModelVanilla(double threshold, Model &model,
                         std::vector<size_t> &inlier_indices) {
        const size_t num_points = pc_.points_.size();
        std::vector<size_t> indices_list(num_points);
        std::iota(std::begin(indices_list), std::end(indices_list), 0);

        Model best_model;
        size_t count = 0;
        size_t current_iteration = max_iteration_;
        for (size_t i = 0; i < current_iteration; ++i) {
            const std::vector<size_t> sample_indices =
                sampler_(indices_list, estimator_.minimal_sample_);
            const auto sample = pc_.SelectByIndex(sample_indices);

            bool ret;
            ret = estimator_.MinimalFit(*sample, model);

            if (!ret) {
                continue;
            }

            const auto result = EvaluateModel(pc_.points_, threshold, model);
            double fitness = std::get<0>(result);
            double inlier_rmse = std::get<1>(result);

            // update model if satisfy both fitness and rmse check
            if (fitness > fitness_ ||
                (fitness == fitness_ && inlier_rmse < inlier_rmse_)) {
                fitness_ = fitness;
                inlier_rmse_ = inlier_rmse;
                best_model = model;
                const size_t tmp =
                    log(1 - probability_) /
                    log(1 - pow(fitness, estimator_.minimal_sample_));
                if (tmp < 0) {
                    current_iteration = max_iteration_;
                } else {
                    current_iteration = std::min(tmp, max_iteration_);
                }
            }

            // break the loop if count larger than max iteration
            if (current_iteration > max_iteration_) {
                break;
            }
            count++;
        }
        MISC3D_INFO(
            "[vanilla ransac] find best model with {}% inliers and run {} "
            "iterations",
            100 * fitness_, count);

        const bool ret = RefineModel(threshold, best_model, inlier_indices);
        model = best_model;
        return ret;
    }

    /**
     * @brief parallel ransac fitting method, the iteration number is fixed and use
     * OpenMP to speed up. Usually used if you want more accurate result.
     *
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool FitModelParallel(double threshold, Model &model,
                          std::vector<size_t> &inlier_indices) {
        const size_t num_points = pc_.points_.size();

        std::vector<size_t> indices_list(num_points);
        std::iota(std::begin(indices_list), std::end(indices_list), 0);

        std::vector<std::tuple<double, double, Model>> result_list;
#pragma omp parallel for shared(indices_list)
        for (size_t i = 0; i < max_iteration_; ++i) {
            const std::vector<size_t> sample_indices =
                sampler_(indices_list, estimator_.minimal_sample_);
            const auto sample = pc_.SelectByIndex(sample_indices);
            
            Model model_trial;
            bool ret;
            ret = estimator_.MinimalFit(*sample, model_trial);

            if (!ret) {
                continue;
            }

            const auto result = EvaluateModel(pc_.points_, threshold, model_trial);
            double fitness = std::get<0>(result);
            double inlier_rmse = std::get<1>(result);
#pragma omp critical
            {
                result_list.emplace_back(
                    std::make_tuple(fitness, inlier_rmse, model_trial));
            }
        }

        // selection best result from stored patch
        double max_fitness = 0;
        double min_inlier_rmse = 1.0e+10;
        auto best_result = *std::max_element(
            result_list.begin(), result_list.end(),
            [](const std::tuple<double, double, Model> &result1,
               const std::tuple<double, double, Model> &result2) {
                const double fitness1 = std::get<0>(result1);
                const double fitness2 = std::get<0>(result2);
                const double inlier_rmse1 = std::get<1>(result1);
                const double inlier_rmse2 = std::get<1>(result2);
                return (fitness2 > fitness1 ||
                        (fitness1 == fitness2 && inlier_rmse2 < inlier_rmse1));
            });
        Model best_model;
        best_model = std::get<2>(best_result);

        MISC3D_INFO(
            "[parallel ransac] find best model with {}% inliers and run "
            "{} iterations",
            100 * std::get<0>(best_result), max_iteration_);

        const bool ret = RefineModel(threshold, best_model, inlier_indices);
        model = best_model;

        return ret;
    }

    std::tuple<double, double> EvaluateModel(
        const std::vector<Eigen::Vector3d> &points, double threshold,
        const Model &model) {
        size_t inlier_num = 0;
        double error = 0;

        for (size_t idx = 0; idx < points.size(); ++idx) {
            const double distance =
                estimator_.CalcPointToModelDistance(points[idx], model);

            if (distance < threshold) {
                error += distance;
                inlier_num++;
            }
        }

        double fitness;
        double inlier_rmse;

        if (inlier_num == 0) {
            fitness = 0;
            inlier_rmse = 1e+10;
        } else {
            fitness = (double)inlier_num / (double)points.size();
            inlier_rmse = error / std::sqrt((double)inlier_num);
        }

        return std::make_tuple(fitness, inlier_rmse);
    }

private:
    open3d::geometry::PointCloud pc_;

    double probability_;
    size_t max_iteration_;
    double fitness_;
    double inlier_rmse_;
    Sampler sampler_;
    ModelEstimator estimator_;
    bool enable_parallel_;
};

using RandomIndexSampler = RandomSampler<size_t>;
using RANSACPlane = RANSAC<PlaneEstimator, Plane, RandomIndexSampler>;
using RANSACShpere = RANSAC<SphereEstimator, Sphere, RandomIndexSampler>;
using RANSACCylinder = RANSAC<CylinderEstimator, Cylinder, RandomIndexSampler>;

}  // namespace common
}  // namespace misc3d