#include <bits/stdc++.h>
#include <iostream>
#include <limits>

#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/KDTreeFlann.h>
#include <primitives_fitting/PrimitivesFitting.h>
#include <primitives_fitting/Segmentation.h>
#include <primitives_fitting/Utils.h>
#include <Eigen/Eigenvalues>

namespace primitives_fitting {

std::tuple<Eigen::Matrix3d, Eigen::Vector3d> ComputeCovarianceMatrixAndMean(
    const Eigen::MatrixX3d &points) {
    const Eigen::Vector3d mean = points.colwise().mean().transpose();
    const Eigen::Matrix3Xd points_ = (points.rowwise() - mean.transpose()).transpose();
    return std::make_tuple((points_ * points_.transpose()) / double(points.rows()), mean);
}

PrimitivesDetectorConfig::PrimitivesDetectorConfig() {
    m_preprocess_param = {0.002, false};
    m_fitting_param = {PrimitivesType::plane, 0.005};
    m_cluster_param = {4, std::numeric_limits<size_t>::max(), 0.01, 15.0};
    m_filtering_param = {0.01, 0.1};
}

PrimitivesDetectorConfig::~PrimitivesDetectorConfig() = default;

PrimitivesDetector::PrimitivesDetector() : m_config(PrimitivesDetectorConfig()) {}

PrimitivesDetector::~PrimitivesDetector() = default;

void PrimitivesDetector::CheckConfiguration() {
    // check point cloud down sample voxel size and clustering point cloud distance
    if (m_config.m_preprocess_param.voxel_size > m_config.m_cluster_param.dist_thresh) {
        std::cout << "invalid param setting: the downsample voxel size must be smaller than "
                     "clustering distance threshold!"
                  << std::endl;
        std::cout
            << "automatically adjust voxel size value into half of clustering distance threshold"
            << std::endl;
        m_config.m_preprocess_param.voxel_size = m_config.m_cluster_param.dist_thresh / 2;
    }

    // check primitives filtering params
    if (m_config.m_filtering_param.min_bound > m_config.m_filtering_param.max_bound) {
        std::cout << "invalid param setting: the min bound of filtering param for sphere is larger "
                     "than max bound!"
                  << std::endl;
        std::cout << "automatically exchange value" << std::endl;
        std::swap(m_config.m_filtering_param.min_bound, m_config.m_filtering_param.max_bound);
    }
}

PrimitivesDetector::PrimitivesDetector(const PrimitivesDetectorConfig &config) {
    m_config = config;
    CheckConfiguration();
}

void PrimitivesDetector::SetConfiguration(const PrimitivesDetectorConfig &config) {
    m_config = config;
    CheckConfiguration();
}

std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<std::vector<size_t>>>
PrimitivesDetector::PreProcessPointClouds(const open3d::geometry::PointCloud &points) {
    // voxel downsample pointcloud
    auto o3d_pts_down = points.VoxelDownSample(m_config.m_preprocess_param.voxel_size);

    // set neighbours search radius equal to half of clustering distance threshold
    const double search_radius = m_config.m_cluster_param.dist_thresh / 2;

    std::vector<std::vector<size_t>> neighbours = FindNeighbourHood(*o3d_pts_down, search_radius);

    if (m_config.m_preprocess_param.enable_smoothing) {
        SmoothPointClouds(neighbours, *o3d_pts_down, search_radius);
    }

    if (!o3d_pts_down->HasNormals()) {
        o3d_pts_down->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(NORMAL_SEARCH_RADIUS));
    }

    o3d_pts_down->NormalizeNormals();
    utils::NormalConsistent(*o3d_pts_down);

    return std::make_tuple(o3d_pts_down, neighbours);
}

std::vector<std::vector<size_t>> PrimitivesDetector::FindNeighbourHood(
    const open3d::geometry::PointCloud &pc, double radius) {
    open3d::geometry::KDTreeFlann kdtree(pc);
    const size_t num = pc.points_.size();
    std::vector<std::vector<size_t>> neighbours(num);

#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        std::vector<int> ret_indices;
        std::vector<double> out_dists_sqr;
        const int nn = kdtree.SearchRadius(pc.points_[i], radius, ret_indices, out_dists_sqr);
        std::vector<size_t> ret(ret_indices.begin(), ret_indices.end());
        neighbours[i] = ret;
    }

    return neighbours;
}

void PrimitivesDetector::SmoothPointClouds(std::vector<std::vector<size_t>> &neighbours,
                                           open3d::geometry::PointCloud &pc, double tolerance) {
    const size_t num = pc.points_.size();

#pragma omp parallel for shared(neighbours, pc)
    for (size_t i = 0; i < num; ++i) {
        Eigen::Matrix3Xd cluster_raw;
        utils::GetMatrixByIndex(pc.points_, neighbours[i], cluster_raw);
        const Eigen::Matrix3Xd cluster_smooth = ZAxisRegression(cluster_raw);

        if ((pc.points_[i] - cluster_smooth.col(0)).norm() < tolerance) {
            pc.points_[i] = cluster_smooth.col(0);
        } else {
            pc.points_[i] = Eigen::Vector3d(0, 0, 0);
        }
    }
}

Eigen::Matrix3Xd PrimitivesDetector::ZAxisRegression(const Eigen::Matrix3Xd &points) {
    const int num = points.cols();

    Eigen::VectorXd target = points.row(2);
    Eigen::Matrix3Xd data = Eigen::Matrix3Xd::Ones(3, num);

    data.row(0) = points.row(0);
    data.row(1) = points.row(1);

    Eigen::Vector3d weight = (data * data.transpose()).inverse() * data * target;

    Eigen::VectorXd new_target = data.transpose() * weight;
    data.row(2) = new_target.transpose();

    return data;
}

std::vector<Eigen::Matrix4d> PrimitivesDetector::FilterClusterAndCalcPose(
    const Clusters &clusters) {
    switch (m_config.m_fitting_param.type) {
    case PrimitivesType::plane:
        return FilterClusterAndCalcPoseUsingPlaneModel(clusters);

    case PrimitivesType::sphere:
        return FilterClusterAndCalcPoseUsingSphereModel(clusters);

    case PrimitivesType::cylinder:
        return FilterClusterAndCalcPoseUsingCylinderModel(clusters);

    default:
        std::cout << "use wrong primitives type, use plane instead" << std::endl;
        return FilterClusterAndCalcPoseUsingPlaneModel(clusters);
    }
}

std::vector<Eigen::Matrix4d> PrimitivesDetector::FilterClusterAndCalcPoseUsingPlaneModel(
    const Clusters &clusters) {
    std::vector<Eigen::Matrix4d> pose_out;

    const double min_bound = m_config.m_filtering_param.min_bound;
    const double max_bound = m_config.m_filtering_param.max_bound;
#pragma omp parallel for shared(clusters)
    for (size_t i = 0; i < clusters.size(); ++i) {
        ransac::RANSACPlane plane_finder;
        ransac::Plane plane_param;
        std::vector<size_t> indices;
        plane_finder.SetParallel(m_config.m_fitting_param.enable_parallel);
        plane_finder.SetMaxIteration(m_config.m_fitting_param.max_iteration);
        plane_finder.SetPointCloud(clusters[i].points_);
        const bool ret = plane_finder.FitModel(m_config.m_fitting_param.threshold,
                                               plane_param, indices);
        if (!ret) {
            continue;
        }

        std::vector<Eigen::Vector3d> inliers;
        utils::GetVectorByIndex(clusters[i].points_, indices, inliers);

        const open3d::geometry::PointCloud o3d(inliers);
        const open3d::geometry::OrientedBoundingBox bbox = o3d.GetOrientedBoundingBox();
        auto extent = bbox.extent_;
#pragma omp critical
        {
            if (extent(0) > min_bound && extent(0) < max_bound && extent(1) > min_bound &&
                extent(1) < max_bound) {
                m_clusters.emplace_back(inliers);

                Eigen::MatrixX3d cluster_mat;
                utils::VectorToEigenMatrix(inliers, cluster_mat);
                auto result = ComputeCovarianceMatrixAndMean(cluster_mat);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(std::get<0>(result));
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose.block<3, 1>(0, 3) = std::get<1>(result);
                pose.block<3, 1>(0, 0) = solver.eigenvectors().col(2);
                pose.block<3, 1>(0, 1) = solver.eigenvectors().col(1);
                pose.block<3, 1>(0, 2) =
                    solver.eigenvectors().col(2).cross(solver.eigenvectors().col(1));

                // orient normal direction to origin coordinate
                if (pose(2, 2) > 0) {
                    pose.block<3, 1>(0, 2) *= -1;
                    pose.block<3, 1>(0, 0) *= -1;
                }

                pose_out.emplace_back(pose);
                m_primitives.emplace_back(plane_param);
            }
        }
    }
    return pose_out;
}

std::vector<Eigen::Matrix4d> PrimitivesDetector::FilterClusterAndCalcPoseUsingSphereModel(
    const Clusters &clusters) {
    std::vector<Eigen::Matrix4d> pose_out;

    const double min_bound = m_config.m_filtering_param.min_bound;
    const double max_bound = m_config.m_filtering_param.max_bound;
#pragma omp parallel for shared(clusters)
    for (size_t i = 0; i < clusters.size(); ++i) {
        ransac::RANSACShpere sphere_finder;
        ransac::Sphere sphere_param;
        std::vector<size_t> indices;
        bool ret;
        // sphere fitter has error in FitModel function when in some special case, use exception to
        // prevent from broken
        try {
            sphere_finder.SetPointCloud(clusters[i].points_);
            sphere_finder.SetParallel(m_config.m_fitting_param.enable_parallel);
            sphere_finder.SetMaxIteration(m_config.m_fitting_param.max_iteration);
            ret = sphere_finder.FitModel(m_config.m_fitting_param.threshold,
                                         sphere_param, indices);
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            ret = false;
        }

        if (!ret) {
            continue;
        }

        std::vector<Eigen::Vector3d> inliers;
        utils::GetVectorByIndex(clusters[i].points_, indices, inliers);
#pragma omp critical
        {
            if (sphere_param.m_parameters(3) > min_bound &&
                sphere_param.m_parameters(3) < max_bound) {
                m_clusters.emplace_back(inliers);

                Eigen::Vector3d direction =
                    Eigen::Vector3d(0, 0, 0) - sphere_param.m_parameters.head<3>();
                direction /= direction.norm();
                const Eigen::Vector3d translation =
                    sphere_param.m_parameters.head<3>() + direction * sphere_param.m_parameters(3);

                Eigen::Vector3d y_axis = direction.cross(Eigen::Vector3d(1, 0, 0));
                Eigen::Vector3d x_axis = y_axis.cross(direction);
                y_axis = direction.cross(x_axis);

                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose.block<3, 1>(0, 3) = translation;
                pose.block<3, 1>(0, 0) = x_axis;
                pose.block<3, 1>(0, 1) = y_axis;
                pose.block<3, 1>(0, 2) = direction;
                pose_out.emplace_back(pose);
                m_primitives.emplace_back(sphere_param);
            }
        }
    }
    return pose_out;
}

std::vector<Eigen::Matrix4d> PrimitivesDetector::FilterClusterAndCalcPoseUsingCylinderModel(
    const Clusters &clusters) {
    std::vector<Eigen::Matrix4d> pose_out;

    const double min_bound = m_config.m_filtering_param.min_bound;
    const double max_bound = m_config.m_filtering_param.max_bound;
#pragma omp parallel for shared(clusters)
    for (size_t i = 0; i < clusters.size(); ++i) {
        ransac::RANSACCylinder cylinder_finder;
        ransac::Cylinder cylinder_param;
        std::vector<size_t> indices;
        bool ret;
        try {
            cylinder_finder.SetParallel(m_config.m_fitting_param.enable_parallel);
            cylinder_finder.SetMaxIteration(m_config.m_fitting_param.max_iteration);
            cylinder_finder.SetPointCloud(clusters[i].points_);
            cylinder_finder.SetNormals(clusters[i].normals_);
            ret = cylinder_finder.FitModel(m_config.m_fitting_param.threshold,
                                           cylinder_param, indices);
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            ret = false;
        }

        if (!ret) {
            continue;
        }

        std::vector<Eigen::Vector3d> inliers;
        utils::GetVectorByIndex(clusters[i].points_, indices, inliers);
#pragma omp critical
        {
            if (cylinder_param.m_parameters(6) > min_bound &&
                cylinder_param.m_parameters(6) < max_bound) {
                m_clusters.emplace_back(inliers);

                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose.block<3, 1>(0, 3) = cylinder_param.m_parameters.head<3>();
                const Eigen::Vector3d x_axis = cylinder_param.m_parameters.segment<3>(3);
                const Eigen::Vector3d z_axis = Eigen::Vector3d(0, 0, 0) - cylinder_param.m_parameters.head<3>();
                Eigen::Vector3d y_axis = z_axis.cross(x_axis);
                pose.block<3, 1>(0, 0) = x_axis;
                pose.block<3, 1>(0, 1) = y_axis;
                pose.block<3, 1>(0, 2) = z_axis;

                pose_out.emplace_back(pose);
                m_primitives.emplace_back(cylinder_param);
            }
        }
    }
    return pose_out;
}

bool PrimitivesDetector::Detect(const std::vector<Eigen::Vector3d> &points) {
    const open3d::geometry::PointCloud o3d_pc(points);
    return Detect(o3d_pc);
}

bool PrimitivesDetector::Detect(const std::vector<Eigen::Vector3d> &points,
                                const std::vector<Eigen::Vector3d> &normals) {
    open3d::geometry::PointCloud o3d_pc(points);
    o3d_pc.normals_ = normals;
    return Detect(o3d_pc);
}

bool PrimitivesDetector::Detect(const open3d::geometry::PointCloud &points) {
    const size_t points_num = points.points_.size();

    // clear previous results
    m_clusters.clear();
    m_primitives.clear();
    m_poses.clear();

    if (points_num == 0) {
        std::cout << "there is no input point clouds" << std::endl;
        return false;
    }

    auto processed_data = PreProcessPointClouds(points);
    auto pcd = std::get<0>(processed_data);
    auto neighbours = std::get<1>(processed_data);

    // init proximity extractor
    segmentation::ProximityExtractor extractor(m_config.m_cluster_param.min_cluster_size,
                                               m_config.m_cluster_param.max_cluster_size);
    segmentation::DistanceNormalsProximityEvaluator evaluator(
        pcd->normals_, m_config.m_cluster_param.dist_thresh, m_config.m_cluster_param.angle_thresh);

    std::vector<std::vector<size_t>> cluster_indices =
        extractor.Segment(pcd->points_, neighbours, evaluator);

    Clusters clusters(cluster_indices.size());
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        clusters[i] = *pcd->SelectByIndex(cluster_indices[i]);
    }

    m_poses = FilterClusterAndCalcPose(clusters);

    return true;
}

std::vector<std::vector<Eigen::Vector3d>> PrimitivesDetector::GetClusters() {
    std::vector<std::vector<Eigen::Vector3d>> clusters;
    for (auto &c : m_clusters) {
        clusters.push_back(c.points_);
    }

    return clusters;
}

std::vector<ransac::Model> PrimitivesDetector::GetPrimitives() {
    return m_primitives;
}

std::vector<Eigen::Matrix4d> PrimitivesDetector::GetPoses() {
    return m_poses;
}

}  // namespace primitives_fitting
