#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import cv2

import misc3d as m3d
from utils import np2o3d
from IPython import embed

vis = o3d.visualization.Visualizer()
vis.create_window("Segmentation", 1920, 1200)

color_img = cv2.imread(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/examples/data/indoor/color/color_0.png'
)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

depth_img = cv2.imread(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/examples/data/indoor/depth/depth_0.png',
    cv2.IMREAD_ANYDEPTH)

depth = o3d.geometry.Image(depth_img)
color = o3d.geometry.Image(color_img)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    848, 480, 598.7568, 598.7568, 430.3443, 250.244)
pc = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=True)

print('Point size before sampling', pc)
t0 = time.time()
pc = pc.voxel_down_sample(0.01)
print('Point size after sampling', pc)

pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.02, 15))
pc.orient_normals_towards_camera_location()

normals = np.asarray(pc.normals)

pe = m3d.segmentation.ProximityExtractor(100)
ev = m3d.segmentation.DistanceNormalsProximityEvaluator(normals, 0.02, 30)
#ev = m3d.segmentation.DistanceProximityEvaluator(0.02)

index_list = pe.segment(pc, 0.02, ev)

print('Segmentation time: %.3f' % (time.time() - t0))

pc_render = o3d.geometry.PointCloud()
for index in index_list:
    c = pc.select_by_index(index)
    m3d.vis.draw_point_cloud(vis, c, [random(), random(), random()], size=3.0)

m3d.vis.draw_pose(vis, size=0.1)

vis.run()
