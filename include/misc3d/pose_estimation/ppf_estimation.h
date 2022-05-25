#pragma once

#include <misc3d/pose_estimation/data_structure.h>
#include <misc3d/utils.h>

// view point position factor
#define VIEW_POINT_Z_EXTEND 2
// maximum nn points of kdtree search for normal computing
#define NORMAL_CALC_NN 30
// maximum iteration of sparse pose refine for clustered pose
#define SPARSE_REFINE_ICP_ITERATION 30
#define NEIGHBOR_RADIUS_FACTOR 0.5
#define VOTING_THRESHOLD_FACTOR 0.2
#define VOTE_NUM_RATIO 0.8
#define VOTES_NUM_REDUCTION_FACTOR 0.25

namespace misc3d {

namespace pose_estimation {

class PPFEstimatorConfig {
public:
    PPFEstimatorConfig();
    ~PPFEstimatorConfig();

    // param to control key points selection
    // TODO: can be extended more key points seletection methods
    enum class ReferencePointSelection {
        Random = 0,
    };

    // voting mode of PPF.
    // using_all_sampled_points: usually for most of case, in which the object has
    // surface metrics using_edge_points: for object which has flat shape and has
    // boundary metrics.
    // TODO: this mode still need to improved.
    enum class VotingMode {
        SampledPoints = 0,
        EdgePoints = 1,
    };

    // sparse pose refine methods
    enum class RefineMethod {
        NoRefine = 0,
        PointToPoint = 1,
        PointToPlane = 2,
    };

    struct TrainingParam {
        // flag to control whether invert model normal, usually set to true of input
        // point clouds are not in camera coordinate (RGBD camera), or set to false
        // as default.
        bool invert_model_normal;

        // whether use external pre-computed normals for training.
        bool use_external_normal;

        // relative variable of point cloud sample distance
        double rel_sample_dist;
        // relative variable of normal search radius, usually is set as half of
        // rel_sample_dist
        double calc_normal_relative;
        // relative variable of dense point cloud sample distance, only used in edge
        // mode
        double rel_dense_sample_dist;
    };

    struct ReferenceParam {
        ReferencePointSelection method;
        // the ratio of point clouds would be selected
        double ratio;
    };

    struct VotingParam {
        VotingMode method;
        // if in faster mode, the spread of ppf hash table will be reduced
        bool faster_mode;
        // ppf quantization resulotion
        double angle_step;
        // minimum distance and angle threshold for point pair filtering
        double min_dist_thresh;
        double min_angle_thresh;
    };

    struct EdgeParam {
        // number of nearest points for kdtree searching
        size_t pts_num;
    };

    struct RefineParam {
        RefineMethod method;
        // icp refine distance thresh, the actual D = rel_dist_sparse_thresh *
        // diameter * rel_sample_dist.
        double rel_dist_sparse_thresh;
    };

    TrainingParam training_param_;
    ReferenceParam ref_param_;
    VotingParam voting_param_;
    EdgeParam edge_param_;
    RefineParam refine_param_;

    // relative variable to control pose clustering distance threshold
    double rel_dist_thresh_;
    // threshold to decide a valid estimated pose
    double rel_angle_thresh_;
    // score threshold for outlier pose removal
    double score_thresh_;
    // control number of output pose
    size_t num_result_;
    size_t object_id_;
};

class PPFEstimator {
public:
    PPFEstimator();

    PPFEstimator(const PPFEstimatorConfig &config);

    ~PPFEstimator();

    /**
     * @brief Set Configuration
     *
     * @param config
     * @return true
     * @return false
     */
    bool SetConfig(const PPFEstimatorConfig &config);

    /**
     * @brief Train PPF estimator with given model point clouds
     *
     * @param pc
     * @return true
     * @return false
     */
    bool Train(const PointCloudPtr &pc);

    /**
     * @brief Estimate 6d pose from scene point clouds
     *
     * @param pc
     * @param results
     * @return true
     * @return false
     */
    bool Estimate(const PointCloudPtr &pc,
                  std::vector<Pose6D> &results);

    /**
     * @brief Get Model Diameter
     *
     * @return double
     */
    double GetModelDiameter();

    /**
     * @brief Get all matched results
     *
     * @return std::vector<Pose6D>
     */
    std::vector<Pose6D> GetPose();

    /**
     * @brief Get sampled model point clouds
     *
     * @return open3d::geometry::PointCloud
     */
    open3d::geometry::PointCloud GetSampledModel();

    /**
     * @brief Get sampled scene point clouds
     *
     * @return open3d::geometry::PointCloud
     */
    open3d::geometry::PointCloud GetSampledScene();

    /**
     * @brief Get model edge points if using edge voting method
     *
     * @return open3d::geometry::PointCloud
     */
   open3d::geometry::PointCloud GetModelEdges();

    /**
     * @brief Get scene edge point if using edge voting method
     *
     * @return open3d::geometry::PointCloud
     */
    open3d::geometry::PointCloud GetSceneEdges();

private:
    class Impl;
    std::unique_ptr<Impl> impl_ptr_;
    bool CheckConfig(const PPFEstimatorConfig &config);
};

}  // namespace pose_estimation
}  // namespace misc3d