#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import copy
from random import random
import open3d as o3d
import cv2
import misc3d as m3d
from IPython import embed


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

color_img = cv2.imread(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/examples/data/indoor/color/color_1.png'
)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

depth_img = cv2.imread(
    '/home/yuecideng/WorkSpace/Sources/Misc3D/examples/data/indoor/depth/depth_1.png',
    cv2.IMREAD_ANYDEPTH)

depth = o3d.geometry.Image(depth_img)
color = o3d.geometry.Image(color_img)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False)

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    848, 480, 598.7568, 598.7568, 430.3443, 250.244)
pc_ = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, pinhole_camera_intrinsic, project_valid_depth_only=True)

print('Point size before sampling', pc)
t0 = time.time()
pc_src, fpfh_src = preprocess_point_cloud(pc, 0.02)
pc_dst, fpfh_dst = preprocess_point_cloud(pc_, 0.02)

index1, index2 = m3d.registration.match_correspondence(fpfh_src, fpfh_dst,
                                                       True)

src_ = pc_src.select_by_index(index1)
dst_ = pc_dst.select_by_index(index2)

print("corres num: {}".format(len(index1)))

print('Matching time: %.3f' % (time.time() - t0))

t1 = time.time()
pose = m3d.registration.compute_transformation_ransac(pc_src, pc_dst,
                                                      (index1, index2), 0.03,
                                                      100000)

pose = o3d.pipelines.registration.registration_icp(
    pc_src, pc_dst, 0.01, pose,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30)).transformation

print('Transformation estimation time: %.3f' % (time.time() - t1))

m3d.vis.draw_point_cloud(vis, pc_src, [1, 0, 0], pose)
m3d.vis.draw_point_cloud(vis, pc_dst, [0, 1, 0])

vis.run()