#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import copy
from random import random
import open3d as o3d
import misc3d as m3d


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_down.orient_normals_towards_camera_location()

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                             max_nn=100))
    return pcd_down, pcd_fpfh


vis = o3d.visualization.Visualizer()
vis.create_window("Segmentation", 1920, 1200)

depth = o3d.io.read_image('../data/indoor/depth/depth_0.png')
color = o3d.io.read_image('../data/indoor/color/color_0.png')

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    848, 480, 598.7568, 598.7568, 430.3443, 250.244)
pc = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=True)

depth = o3d.io.read_image('../data/indoor/depth/depth_1.png')
color = o3d.io.read_image('../data/indoor/color/color_1.png')

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pc_ = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=True)

print('Point size before sampling', pc)
t0 = time.time()
pc_src, fpfh_src = preprocess_point_cloud(pc, 0.02)
pc_dst, fpfh_dst = preprocess_point_cloud(pc_, 0.02)

ts = time.time()
index1, index2 = m3d.registration.match_correspondence(fpfh_src, fpfh_dst)
print('Matching time: %f' % (time.time() - ts))
print("corres num: {}".format(len(index1)))

t1 = time.time()
pose = m3d.registration.compute_transformation_ransac(pc_src, pc_dst,
                                                      (index1, index2), 0.03,
                                                      100000)

pose = o3d.pipelines.registration.registration_icp(
    pc_src, pc_dst, 0.02, pose,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30)).transformation

print('Transformation estimation time: %.3f' % (time.time() - t1))

m3d.vis.draw_point_cloud(vis, pc_src, [1, 0, 0], pose)
m3d.vis.draw_point_cloud(vis, pc_dst, [0, 1, 0])

vis.run()