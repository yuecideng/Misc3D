#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import open3d as o3d
import cv2
import torch

import misc3d as m3d
from IPython import embed


""" numpy implementation of farthest point sampling """
def farthest_point_sampling_numpy(xyz, npoint):
    N = xyz.shape[0]
    indices = [0] * npoint
    distance = np.ones((N, )) * 1e10
    farthest = 0
    for i in range(npoint):
        indices[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return indices


pcd = o3d.io.read_point_cloud('../data/pose_estimation/model/obj.ply')
print('before smapling: {}'.format(pcd))

points = np.asarray(pcd.points)

t0 = time.time()
indices = m3d.preprocessing.farthest_point_sampling(pcd, 1000)
print('time cost for misc3d: {}'.format(time.time() - t0))
sample = pcd.select_by_index(indices)

t0 = time.time()
indices = farthest_point_sampling_numpy(points, 1000)
print('time cost for numpy: {}'.format(time.time() - t0))
sample_numpy = pcd.select_by_index(indices)

vis = o3d.visualization.Visualizer()
vis.create_window("Farest point sampling", 1920, 1200)
m3d.vis.draw_point_cloud(vis, pcd)
m3d.vis.draw_point_cloud(vis, sample, color=(0, 1, 0), size=5)
vis.run()