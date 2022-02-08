#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import cv2
import copy

import misc3d as m3d
from utils import np2o3d
from IPython import embed

vis = o3d.visualization.Visualizer()
vis.create_window("Preprocessing", 1920, 1200)

color_img = cv2.imread(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/scripts/output/color/color_0.png'
)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

depth_img = cv2.imread(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/scripts/output/depth/depth_0.png',
    cv2.IMREAD_ANYDEPTH)

depth = o3d.geometry.Image(depth_img)
color = o3d.geometry.Image(color_img)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    848, 480, 598.7568, 598.7568, 430.3443, 250.244)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=False)

t1 = time.time()
# pcd = m3d.preprocessing.crop_roi_pointcloud(pcd, (200, 50, 600, 400), (848, 480))
pcd = m3d.preprocessing.project_into_plane(pcd)
print('time cost: {}'.format(time.time() - t1))

# cv2.rectangle(color_img, (200, 50), (600, 400), (0, 255, 0), 3)
# cv2.imshow('color', color_img)
# cv2.waitKey(1000)
m3d.vis.draw_point_cloud(vis, pcd)
vis.run()
