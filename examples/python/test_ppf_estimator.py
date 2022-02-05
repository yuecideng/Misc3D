#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import argparse
import time

import misc3d as m3d
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='foo', type=str)
parser.add_argument('--scene_path', default='foo', type=str)
parser.add_argument('--invert_model_normal', default=False, type=bool)
parser.add_argument('--voting_mode', default="SampledPoints", type=str)
parser.add_argument('--rel_sample_dist', default=0.055, type=float)
parser.add_argument('--score_thresh', default=0.1, type=float)
parser.add_argument('--enable_icp', default=True, type=bool)
args = parser.parse_args()

# init ppf config
config = m3d.pose_estimation.PPFEstimatorConfig()
# init training param
config.training_param.invert_model_normal = args.invert_model_normal
config.training_param.rel_sample_dist = args.rel_sample_dist
# use half of rel_sample_dist is usually a good choice
config.training_param.calc_normal_relative = args.rel_sample_dist

if args.voting_mode == 'EdgePoints':
    config.voting_param.method = m3d.pose_estimation.PPFEstimatorConfig.EdgePoints
    config.ref_param.ratio = 1.0

config.score_thresh = args.score_thresh
config.num_result = 20
config.refine_param.method = m3d.pose_estimation.PPFEstimatorConfig.PointToPlane
# init ppf detector
ppf = m3d.pose_estimation.PPFEstimator(config)
try:
    model_o3d = o3d.io.read_point_cloud(args.model_path)
except:
    print('model path error')
    exit()

# train ppf detector
ret = ppf.train(model_o3d)

if ret is False:
    print('train fail')
    exit()

sampled_model = ppf.get_sampled_model()

#----------------------------------------------#
try:
    scene = o3d.io.read_point_cloud(args.scene_path)
except:
    print('scene path error')
    exit()

# mathch scene points
ret, results = ppf.match(scene)

sampled_scene = ppf.get_sampled_scene()

pose_list = [p.pose for p in results]
# icp refine
if args.enable_icp:
    for i, pose in enumerate(pose_list):
        reg_result = o3d.pipelines.registration.registration_icp(
            model_o3d, sampled_scene, 0.01, pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        pose_list[i] = reg_result.transformation

vis = o3d.visualization.Visualizer()
vis.create_window("Pose estimation", 1920, 1200)

#m3d.vis.draw_point_cloud(vis, sampled_model, size=6);
m3d.vis.draw_point_cloud(vis, scene)
# m3d.vis.draw_pose(vis, size=0.1)

# # draw mathced pose
for i, p in enumerate(pose_list):
    m3d.vis.draw_point_cloud(vis, model_o3d, [0, 1, 0], p, 5)
    m3d.vis.draw_pose(vis, p, size=0.05)

vis.run()