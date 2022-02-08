#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import time

import misc3d as m3d
from IPython import embed

# init ppf config
config = m3d.pose_estimation.PPFEstimatorConfig()
# init training param
config.training_param.rel_sample_dist = 0.06
# use half of rel_sample_dist is usually a good choice
# config.training_param.calc_normal_relative = args.rel_sample_dist
config.score_thresh = 0.2
config.refine_param.method = m3d.pose_estimation.PPFEstimatorConfig.PointToPlane

# init ppf detector
ppf = m3d.pose_estimation.PPFEstimator(config)

model_o3d = o3d.io.read_point_cloud(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/examples/data/pose_estimation/model/triangle.ply'
)

# train ppf detector
ret = ppf.train(model_o3d)

if ret is False:
    print('train fail')
    exit()

scene = o3d.io.read_point_cloud(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/examples/data/pose_estimation/scene/triangles.ply'
)

# mathch scene points
ret, results = ppf.match(scene)

sampled_scene = ppf.get_sampled_scene()

pose_list = [p.pose for p in results]
# icp refine
t0 = time.time()
for i, pose in enumerate(pose_list):
    reg_result = o3d.pipelines.registration.registration_icp(
        model_o3d, sampled_scene, 0.01, pose,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    pose_list[i] = reg_result.transformation
print('time cost for icp refine: {}'.format(time.time() - t0))

vis = o3d.visualization.Visualizer()
vis.create_window("Pose estimation", 1920, 1200)

m3d.vis.draw_point_cloud(vis, scene)

# # draw mathced pose
for i, p in enumerate(pose_list):
    m3d.vis.draw_point_cloud(vis, model_o3d, [0, 1, 0], p, 5)
    m3d.vis.draw_pose(vis, p, size=0.05)

vis.run()