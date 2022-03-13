#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import open3d as o3d
import misc3d as m3d

vis = o3d.visualization.Visualizer()
vis.create_window("Estimate normals", 1920, 1200)

depth = o3d.io.read_image('../data/indoor/depth/depth_0.png')
color = o3d.io.read_image('../data/indoor/color/color_0.png')

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    848, 480, 598.7568, 598.7568, 430.3443, 250.244)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=False)

t0 = time.time()
m3d.common.estimate_normals(pcd, (848, 480), 3)
print('time cost: {}'.format(time.time() - t0))

m3d.vis.draw_pose(vis, size=0.1)
m3d.vis.draw_geometry3d(vis, pcd)
vis.run()
