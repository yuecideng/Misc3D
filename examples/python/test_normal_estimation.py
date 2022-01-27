#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import cv2
import argparse

import misc3d as m3d
from utils import np2o3d
from IPython import embed

vis = o3d.visualization.Visualizer()
vis.create_window("Segmentation", 1920, 1200)

# img = cv2.imread(
#     '/home/yuecideng/WorkSpace/Sources/Misc3D/scripts/output/color/color_0.png'
# )
# depth = cv2.imread(
#     '/home/yuecideng/WorkSpace/Sources/Misc3D/scripts/output/depth/depth_0.png'
# ) * 1000

# depth = o3d.geometry.Image(depth)
# color = o3d.geometry.Image(img)

# pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
#     848, 480, 598.7568359375, 598.7568969726562, 430.3443298339844,
#     250.24488830566406)

# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color, depth, convert_rgb_to_intensity=False)

# create point cloud
pcd = o3d.io.read_point_cloud(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/scripts/output/pcd/pcd0.ply',
    remove_nan_points=False)

t0 = time.time()
pcd = m3d.common.estimate_normals(pcd, 848, 480, 3)
print('time cost: {}'.format(time.time() - t0))
pcd.remove_non_finite_points()
pcd = pcd.voxel_down_sample(0.02)

m3d.vis.draw_pose(vis, size=0.1)

m3d.vis.draw_point_cloud(vis, pcd)

# m3d.vis.draw_pose(vis, size=0.1)
vis.run()
