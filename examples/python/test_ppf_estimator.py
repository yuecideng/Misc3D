#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import time
import misc3d as m3d


# init ppf config
config = m3d.pose_estimation.PPFEstimatorConfig()
# init training param
config.training_param.rel_sample_dist = 0.04
config.score_thresh = 0.05
config.refine_param.method = m3d.pose_estimation.PPFEstimatorConfig.PointToPlane

# init ppf detector
ppf = m3d.pose_estimation.PPFEstimator(config)

model = o3d.io.read_point_cloud('../data/pose_estimation/model/obj.ply')

# convert mm to m
model = model.scale(0.001, np.array([0, 0, 0]))

# train ppf detector
ret = ppf.train(model)

if ret is False:
    print('train fail')
    exit()

depth = o3d.io.read_image('../data/pose_estimation/scene/depth.png')
color = o3d.io.read_image('../data/pose_estimation/scene/rgb.png')

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640, 480, 572.4114, 573.57043, 325.2611, 242.04899)
scene = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=False)

# crop roi from point clouds using given bbox of image
scene_crop = m3d.preprocessing.crop_roi_pointcloud(
    scene, (222, 296, 41 + 222, 44 + 296), (640, 480))

# mathch scene points
ret, results = ppf.estimate(scene_crop)

if ret is False:
    print('No matched')
else:
    pose = results[0].pose
    sampled_model = ppf.get_sampled_model()
    reg_result = o3d.pipelines.registration.registration_icp(
        sampled_model, scene_crop, 0.01, pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pose = reg_result.transformation

vis = o3d.visualization.Visualizer()
vis.create_window("Pose estimation", 1920, 1200)

mesh = o3d.io.read_triangle_mesh('../data/pose_estimation/model/obj.ply')
mesh = mesh.scale(0.001, np.array([0, 0, 0]))
m3d.vis.draw_point_cloud(vis, scene)
if ret:
    m3d.vis.draw_triangle_mesh(vis, mesh, pose=pose)
m3d.vis.draw_pose(vis, size=0.1)
vis.run()