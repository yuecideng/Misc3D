#pragma once

#include <memory>
#include <vector>

#include <open3d/geometry/PointCloud.h>
#include <primitives_fitting/Ransac.h>
#include <Eigen/Core>

namespace primitives_fitting {

typedef std::vector<open3d::geometry::PointCloud> Clusters;

enum PrimitivesType : uint8_t {
    // plane usually refer to single face of a box
    plane = 0,
    sphere = 1,
    // TODO: cylinder model pose detection is not valid due to clinder fitting is not well
    cylinder = 2,
};

class PrimitivesDetectorConfig {
public:
    PrimitivesDetectorConfig();
    ~PrimitivesDetectorConfig();

    struct PreProcessParam {
        // voxel size for point cloud downsampling
        double voxel_size;
        // flag to control whether to smooth the point clouds, useful when point clouds quality is
        // bad
        bool enable_smoothing;
    };

    /**
     * @brief fitting param to find the key points from cluster,
     *
     */
    struct FittingParam {
        PrimitivesType type;
        // L2 measurement for inlier selection of ransac algorithm
        double threshold;
        // if enable parallel, the parallel ransac will be used and max iteration should be set
        bool enable_parallel;
        int max_iteration;
    };

    /**
     * @brief clustering param to segment point clouds into indivitual clusters
     *
     */
    struct ClusterParam {
        size_t min_cluster_size;
        size_t max_cluster_size;
        // point to point distance for dividing clusters
        double dist_thresh;
        // noraml angle difference for dividing clusters
        double angle_thresh;
    };

    struct FilteringParam {
        // if primitives is plane, the min&max bound are refer to size of boundingbox of plannar
        // points
        // if primitives is sphere, the min&max bound are refer to radius of sphere
        // if primitives is cylinder, the min&max bound are refer to radius of cylinder
        double min_bound;
        double max_bound;
    };

    PreProcessParam m_preprocess_param;
    FittingParam m_fitting_param;
    ClusterParam m_cluster_param;
    FilteringParam m_filtering_param;
};

class PrimitivesDetector {
private:
    /**
     * @brief check whether the configuration is valid, otherwise auto change the invalid param
     *
     */
    void CheckConfiguration();

    /**
     * @brief preprocess point clouds, including downsample, neighbours finding and normal
     * estimation
     *
     * @param points
     * @return std::tuple<std::shared_ptr<open3d::geometry::PointCloud>,
     * std::vector<std::vector<size_t>>>
     */
    std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<std::vector<size_t>>>
    PreProcessPointClouds(const open3d::geometry::PointCloud &points);

    /**
     * @brief find neighbours of each input point clouds
     *
     * @param pc
     * @param radius
     * @return std::vector<std::vector<size_t>>
     */
    std::vector<std::vector<size_t>> FindNeighbourHood(const open3d::geometry::PointCloud &pc,
                                                       double radius);

    /**
     * @brief smooth point clouds using least square regression
     *
     * @param neighbours
     * @param pc
     * @param tolerance
     */
    void SmoothPointClouds(std::vector<std::vector<size_t>> &neighbours,
                           open3d::geometry::PointCloud &pc, double tolerance);

    /**
     * @brief least square regression of z, the equation is ax + by + c = z.
     *
     * @param points
     * @return Eigen::Matrix3Xd
     */
    Eigen::Matrix3Xd ZAxisRegression(const Eigen::Matrix3Xd &points);

    /**
     * @brief filter cluster then compute pose
     *
     * @param clusters
     * @return std::vector<Eigen::Matrix4d>
     */
    std::vector<Eigen::Matrix4d> FilterClusterAndCalcPose(const Clusters &clusters);

    /**
     * @brief filter cluster then compute pose using plane model
     *
     * @param clusters
     * @return std::vector<Eigen::Matrix4d>
     */
    std::vector<Eigen::Matrix4d> FilterClusterAndCalcPoseUsingPlaneModel(const Clusters &clusters);

    /**
     * @brief filter cluster then compute pose using sphere model
     *
     * @param clusters
     * @return std::vector<Eigen::Matrix4d>
     */
    std::vector<Eigen::Matrix4d> FilterClusterAndCalcPoseUsingSphereModel(const Clusters &clusters);

    /**
     * @brief filter cluster then compute pose using cylinder model
     * 
     * @param clusters 
     * @return std::vector<Eigen::Matrix4d> 
     */
    std::vector<Eigen::Matrix4d> FilterClusterAndCalcPoseUsingCylinderModel(
        const Clusters &clusters);

public:
    PrimitivesDetector();
    PrimitivesDetector(const PrimitivesDetectorConfig &config);
    ~PrimitivesDetector();

    /**
     * @brief set detection configuration param
     *
     * @param config
     */
    void SetConfiguration(const PrimitivesDetectorConfig &config);

    /**
     * @brief Detect primitives from point clouds
     *
     * @param points
     * @return true
     * @return false
     */
    bool Detect(const std::vector<Eigen::Vector3d> &points);

    /**
     * @brief Detect primitives from point clouds with normals
     *
     * @param points
     * @param normals
     * @return true
     * @return false
     */
    bool Detect(const std::vector<Eigen::Vector3d> &points,
                const std::vector<Eigen::Vector3d> &normals);

    /**
     * @brief Detect primitives from open3d point clouds
     *
     * @param points
     * @return std::vector<Eigen::Matrix4d>
     */
    bool Detect(const open3d::geometry::PointCloud &points);

    /**
     * @brief retuen filtered clusters
     *
     * @return std::vector<Eigen::Vector3d>
     */
    std::vector<std::vector<Eigen::Vector3d>> GetClusters();

    /**
     * @brief Get the Primitives parameters
     *
     * @return std::vector<ransac::Model>
     */
    std::vector<ransac::Model> GetPrimitives();

    /**
     * @brief Get the Poses of primitives
     *
     * If primitives is plane, the translation is the center of plane and the z axis is normal to
     * the plane surface
     *
     * if primitives is sphere, the translation is the center of sphere and the z axis is orient to
     * origin
     *
     * @return std::vector<Eigen::Matrix4d>
     */
    std::vector<Eigen::Matrix4d> GetPoses();

private:
    PrimitivesDetectorConfig m_config;
    Clusters m_clusters;
    std::vector<ransac::Model> m_primitives;
    std::vector<Eigen::Matrix4d> m_poses;
};

}  // namespace primitives_fitting
