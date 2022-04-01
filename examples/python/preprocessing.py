#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import cv2
import copy
import misc3d as m3d

vis = o3d.visualization.Visualizer()
vis.create_window("Crop ROI", 1920, 1200)

color_img = cv2.imread('../data/indoor/color/color_0.png')
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

depth_img = cv2.imread('../data/indoor/depth/depth_0.png', cv2.IMREAD_ANYDEPTH)

depth = o3d.geometry.Image(depth_img)
color = o3d.geometry.Image(color_img)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    848, 480, 598.7568, 598.7568, 430.3443, 250.244)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=False)

t1 = time.time()
pcd_roi = m3d.preprocessing.crop_roi_pointcloud(pcd, (500, 300, 600, 400),
                                                (848, 480))
pcd_plane = m3d.preprocessing.project_into_plane(pcd)
print('time cost: {}'.format(time.time() - t1))

cv2.rectangle(color_img, (500, 300), (600, 400), (0, 255, 0), 3)
cv2.imshow('color', color_img)
cv2.waitKey(100)

m3d.vis.draw_geometry3d(vis, pcd)
m3d.vis.draw_geometry3d(vis, pcd_roi, color=(1, 0, 0))
vis.run()

vis = o3d.visualization.Visualizer()
vis.create_window("Project into plane", 1920, 1200)
m3d.vis.draw_geometry3d(vis, pcd_plane)
vis.run()
