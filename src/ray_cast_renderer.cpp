#include <misc3d/logging.h>
#include <misc3d/pose_estimation/ray_cast_renderer.h>
#include <iostream>

namespace {
open3d::geometry::PointCloud GetPointCloudFromRayCastResults(
    const open3d::core::Tensor &rays,
    const std::unordered_map<std::string, open3d::core::Tensor>
        &ray_cast_results,
    const open3d::core::TensorKey &hit_key) {
    // Equal to numpy operation: points = rays[hit_key][:,
    // 3:] * ray_cast_results['t_hit'][hit_key].reshape((-1,1))
    const open3d::core::Tensor points =
        rays.GetItem(hit_key).GetItem(
            {open3d::core::TensorKey::Slice(
                 open3d::core::None, open3d::core::None, open3d::core::None),
             open3d::core::TensorKey::Slice(3, 6, 1)}) *
        ray_cast_results.at("t_hit").GetItem(hit_key).Reshape({-1, 1});

    const open3d::core::Tensor normals =
        ray_cast_results.at("primitive_normals").GetItem(hit_key);

    std::unordered_map<std::string, open3d::core::Tensor> attributes;
    attributes["normals"] = normals;
    attributes["positions"] = points;

    const open3d::t::geometry::PointCloud pcd(attributes);
    return pcd.ToLegacy();
}
}  // namespace

namespace misc3d {

namespace pose_estimation {

RayCastRenderer::RayCastRenderer(
    const open3d::camera::PinholeCameraIntrinsic &intrinsic) {
    width_ = intrinsic.width_;
    height_ = intrinsic.height_;

    // Create inrinsic matrix.
    open3d::core::Tensor intrinsic_tensor(intrinsic.intrinsic_matrix_.data(),
                                          {3, 3}, open3d::core::Float64,
                                          open3d::core::Device("CPU:0"));

    // Create a identity extrinsic matrix.
    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
    open3d::core::Tensor extrinsic_tensor(extrinsic.data(), {4, 4},
                                          open3d::core::Float64,
                                          open3d::core::Device("CPU:0"));

    // Create a ray cast tensor for each pixel.
    if (intrinsic.intrinsic_matrix_.IsRowMajor()) {
        rays_ = open3d::t::geometry::RaycastingScene::CreateRaysPinhole(
            intrinsic_tensor, extrinsic_tensor.T(), width_, height_);
    } else {
        rays_ = open3d::t::geometry::RaycastingScene::CreateRaysPinhole(
            intrinsic_tensor.T(), extrinsic_tensor.T(), width_, height_);
    }
}

bool RayCastRenderer::CastRays(
    const std::vector<open3d::geometry::TriangleMesh> &mesh_list,
    const std::vector<Eigen::Matrix4d> &pose_list) {
    const size_t num = mesh_list.size();
    if (num == 0) {
        misc3d::LogWarning("No mesh is provided.");
        return false;
    } else if (mesh_list.size() != pose_list.size()) {
        misc3d::LogError("The number of meshes and poses are not matched.");
    }

    open3d::t::geometry::RaycastingScene scene;

    // Add mesh into the scene with given poses.
    for (size_t i = 0; i < num; i++) {
        open3d::t::geometry::TriangleMesh mesh =
            open3d::t::geometry::TriangleMesh::FromLegacy(mesh_list[i]);

        open3d::core::Tensor pose;
        const auto &mat = pose_list[i];
        if (mat.IsRowMajor()) {
            pose = open3d::core::Tensor(pose_list[i].data(), {4, 4},
                                        open3d::core::Float64,
                                        open3d::core::Device("CPU:0"));
        } else {
            pose = open3d::core::Tensor(pose_list[i].data(), {4, 4},
                                        open3d::core::Float64,
                                        open3d::core::Device("CPU:0"))
                       .T();
        }

        mesh.Transform(pose);
        const uint32_t id = scene.AddTriangles(mesh);
    }

    ray_cast_results_ = scene.CastRays(rays_);
    num_instance_ = num;
    return true;
}

open3d::core::Tensor RayCastRenderer::GetDepthMap() const {
    if (ray_cast_results_.size() == 0) {
        misc3d::LogWarning("No ray cast result is available.");
        return open3d::core::Tensor();
    }

    return ray_cast_results_.at("t_hit");
}

open3d::core::Tensor RayCastRenderer::GetInstanceMap() const {
    if (ray_cast_results_.size() == 0) {
        misc3d::LogWarning("No ray cast result is available.");
        return open3d::core::Tensor();
    }

    return ray_cast_results_.at("geometry_ids");
}

open3d::geometry::PointCloud RayCastRenderer::GetPointCloud() const {
    if (ray_cast_results_.size() == 0) {
        misc3d::LogWarning("No ray cast result is available.");
        return open3d::geometry::PointCloud();
    }

    // Create hit tensorkey index.
    const open3d::core::TensorKey hit_key =
        open3d::core::TensorKey::IndexTensor(
            ray_cast_results_.at("t_hit").IsFinite());

    return GetPointCloudFromRayCastResults(rays_, ray_cast_results_, hit_key);
}

std::vector<open3d::geometry::PointCloud>
RayCastRenderer::GetInstancePointCloud() const {
    std::vector<open3d::geometry::PointCloud> instance_point_clouds;
    if (ray_cast_results_.size() == 0) {
        misc3d::LogWarning("No ray cast result is available.");
        return instance_point_clouds;
    }

    instance_point_clouds.reserve(num_instance_);
    for (size_t i = 0; i < num_instance_; i++) {
        const open3d::core::TensorKey hit_key =
            open3d::core::TensorKey::IndexTensor(
                ray_cast_results_.at("geometry_ids").Eq(i));

        instance_point_clouds.push_back(
            GetPointCloudFromRayCastResults(rays_, ray_cast_results_, hit_key));
    }

    return instance_point_clouds;
}

}  // namespace pose_estimation
}  // namespace misc3d