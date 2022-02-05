#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import copy
from random import random
import open3d as o3d
import argparse

import misc3d as m3d
from utils import np2o3d
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
pc = o3d.io.read_point_cloud(
    '/home/yuecideng/WorkSpace/Install/calib/scripts/output/pcd/pcd0.ply')
pc_ = o3d.io.read_point_cloud(
    '/home/yuecideng/WorkSpace/Install/calib/scripts/output/pcd/pcd1.ply')

print('Point size before sampling', pc)
t0 = time.time()
pc_src, fpfh_src = preprocess_point_cloud(pc, 0.02)
pc_dst, fpfh_dst = preprocess_point_cloud(pc_, 0.02)

index1, index2 = m3d.registration.match_correspondence(fpfh_src, fpfh_dst,
                                                       True)

src_ = pc_src.select_by_index(index1)
dst_ = pc_dst.select_by_index(index2)

print("corres num: {}".format(len(index1)))
# corres = o3d.utility.Vector2iVector(np.c_[np.array(index1).T,
#                                           np.array(index2).T])

# embed()
print('Matching time: %.3f' % (time.time() - t0))
# print("coress num: {} {}".format(src_, dst_))
# pose = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
#     src_, dst_, corres, 0.03,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
#     [
#         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#             0.03)
#     ],
#     o3d.pipelines.registration.RANSACConvergenceCriteria(100000,
#                                                          0.999)).transformation

#pose = execute_global_registration(pc_src, pc_dst, fpfh_src, fpfh_dst, 0.02).transformation
pose = m3d.registration.compute_transformation_ransac(pc_src, pc_dst,
                                                      (index1, index2), 0.03,
                                                      100000)
#pose = m3d.registration.compute_transformation_teaser(src_, dst_, 0.02)

pose = o3d.pipelines.registration.registration_icp(
    pc_src, pc_dst, 0.01, pose,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30)).transformation

print(pose)
# index = pc.cluster_dbscan(0.01, 100)
print('Transformation estimation time: %.3f' % (time.time() - t0))

# reg_p2l = o3d.pipelines.registration.registration_icp(
#     pc_src, pc_dst, 0.02, pose,
#     o3d.pipelines.registration.TransformationEstimationPointToPlane())
# pose = reg_p2l.transformation

m3d.vis.draw_point_cloud(vis, pc_src, [1, 0, 0], pose)
m3d.vis.draw_point_cloud(vis, pc_dst, [0, 1, 0])

vis.run()
embed()