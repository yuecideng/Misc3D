#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import misc3d as m3d

pcd = o3d.io.read_point_cloud('../data/segmentation/test.ply')

t0 = time.time()
results = m3d.segmentation.segment_plane_iterative(pcd, 0.01, 100, 0.1)
print('Segmentation time: %.3f' % (time.time() - t0))

show = []
for cluster in results:
    w, c = cluster
    c.paint_uniform_color(np.array([random(), random(), random()]))
    show.append(c)

o3d.visualization.draw_geometries(show, 'Iterative Plane Segmentation')