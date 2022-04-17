#pragma once

#include <open3d/camera/PinholeCameraIntrinsic.h>

namespace misc3d {
namespace reconstruction {

class PipelineConfig {
public:
    PipelineConfig();
    virtual ~PipelineConfig() {}

public:
    enum class DescriptorType {
        ORB = 0,
        SIFT = 1
    };
    
    struct MakeFragmentParam {
        DescriptorType descriptor_type;
        int feature_num;
        int n_frame_per_fragment;
        float keyframe_ratio;
    };

    enum class LocalRefineMethod {
        Point2PointICP = 0,
        Point2PlaneICP = 1,
        ColoredICP = 2,
        GeneralizedICP = 3
    };

    enum class GlobalRegistrationMethod { Ransac = 0, TeaserPlusPlus = 1 };

    struct OptimizationParam {
        float preference_loop_closure_odometry;
        float preference_loop_closure_registration;
    };

    // path to data stored folder
    std::string data_path_;
    open3d::camera::PinholeCameraIntrinsic camera_intrinsic_;
    float depth_scale_;
    float max_depth_;
    float max_depth_diff_;
    float voxel_size_;
    float integration_voxel_size_;
    MakeFragmentParam make_fragment_param_;
    LocalRefineMethod local_refine_method_;
    GlobalRegistrationMethod global_registration_method_;
    OptimizationParam optimization_param_;
};

}  // namespace reconstruction
}  // namespace misc3d