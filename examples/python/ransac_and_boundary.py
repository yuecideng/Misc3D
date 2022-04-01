#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import misc3d as m3d

vis = o3d.visualization.Visualizer()
vis.create_window("Ransac and Bounary Detection", 1920, 1200)

depth = o3d.io.read_image('../data/indoor/depth/depth_0.png')
color = o3d.io.read_image('../data/indoor/color/color_0.png')

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
w, index = m3d.common.fit_plane(pcd, 0.01, 1000)
print('Plan fitting time: %.3f' % (time.time() - t0))

plane = pcd.select_by_index(index)

t1 = time.time()
index = m3d.features.detect_boundary_points(
    plane, o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))
print('Boundary detection time: %.3f' % (time.time() - t1))

boundary = plane.select_by_index(index)

# scene points painted with gray
bbox = plane.get_oriented_bounding_box()
m3d.vis.draw_geometry3d(vis, pcd, color=(0.5, 0.5, 0.5))
m3d.vis.draw_geometry3d(vis, bbox, color=(0, 1, 0))
m3d.vis.draw_geometry3d(vis, plane)
m3d.vis.draw_geometry3d(vis, boundary, color=(1, 0, 0), size=5)

vis.run()
