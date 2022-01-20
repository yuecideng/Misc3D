#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import argparse

import misc3d as m3d
from utils import np2o3d
from IPython import embed

vis = o3d.visualization.Visualizer()
vis.create_window("Segmentation", 1920, 1200)

pc = o3d.io.read_point_cloud(
    '/home/yuecideng/WorkSpace/Install/calib/scripts/output/pcd/pcd0.ply')

print('Point size before sampling', pc)
t0 = time.time()
pc = pc.voxel_down_sample(0.01)
print('Point size after sampling', pc)
pc.estimate_normals()
pc.orient_normals_towards_camera_location()

normals = np.asarray(pc.normals)

pe = m3d.segmentation.ProximityExtractor(100)
# ev = m3d.segmentation.DistanceProximityEvaluator(0.01)
ev = m3d.segmentation.DistanceNormalsProximityEvaluator(normals, 0.01, 20)

index_list = pe.segment(pc, 0.01, ev)

# index = pc.cluster_dbscan(0.01, 100)
print('Segmentation time: %.3f' % (time.time() - t0))

pc_render = o3d.geometry.PointCloud()
for index in index_list:
    c = pc.select_by_index(index)
    c.paint_uniform_color([random(), random(), random()])
    pc_render += c

vis.add_geometry(pc_render)
op = vis.get_render_option()
op.point_size = 3.0



try:
    while True:
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
except:
    print('Force exit')

