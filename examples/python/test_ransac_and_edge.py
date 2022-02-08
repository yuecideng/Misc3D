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
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=True)

print('Point size before sampling', pcd)
pcd = pcd.voxel_down_sample(0.005)
print('Point size after sampling', pcd)

t0 = time.time()
w, index = m3d.common.fit_plane(pcd, 0.01, 100, enable_parallel=True)
print('Plan fitting time: %.3f' % (time.time() - t0))

plane = pcd.select_by_index(index)

t1 = time.time()
index = m3d.features.detect_edge_points(
    plane, o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))
print('Edges detection time: %.3f' % (time.time() - t1))

edges = plane.select_by_index(index)

# scene points painted with gray
m3d.vis.draw_point_cloud(vis, pcd, color=(0.5, 0.5, 0.5))
m3d.vis.draw_point_cloud(vis, plane)
m3d.vis.draw_point_cloud(vis, edges, color=(1, 0, 0), size=5)

vis.run()

