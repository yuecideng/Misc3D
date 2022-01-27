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
pc = pc.voxel_down_sample(0.005)
print('Point size after sampling', pc)

t0 = time.time()
w, index = m3d.common.fit_plane(pc, 0.01, 1000, enable_parallel=True)

# index = pc.cluster_dbscan(0.01, 100)
print('Plan fitting time: %.3f' % (time.time() - t0))

plane = pc.select_by_index(index)

index = m3d.features.detect_edge_points(
    plane, o3d.geometry.KDTreeSearchParamHybrid(0.02, 15))

edge = plane.select_by_index(index)
edge.paint_uniform_color([1, 0, 0])

bbox = plane.get_oriented_bounding_box()

vis.add_geometry(bbox)
vis.add_geometry(plane)
vis.add_geometry(edge)
op = vis.get_render_option()
op.point_size = 3.0
op.background_color = np.array([0, 0, 0])

vis.run()
# try:
#     while True:
#         vis.poll_events()
#         vis.update_renderer()
# except:
#     print('Force exit')
