#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import open3d as o3d
import misc3d as m3d
import numpy as np

try:
    import cv2
except:
    print("OpenCV is not installed, can not show color and depth images.")


config_path = 'config.json'

pipeline = m3d.reconstruction.ReconstructionPipeline(config_path)
pipeline.run_system()

# vis input and output
data_path = pipeline.get_data_path()
colors = None
depths = None
for i in range(5):
    color = cv2.imread(os.path.join(data_path, 'color', '%06d.png' % i))
    depth = cv2.imread(os.path.join(data_path, 'depth',
                       '%06d.png' % i), cv2.IMREAD_UNCHANGED)
    depth = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)

    colors = np.hstack((colors, color)) if colors is not None else color
    depths = np.hstack((depths, depth)) if depths is not None else depth

show = np.vstack((colors, depths))
show = cv2.resize(show, (1400, 420))
cv2.imshow('RGBD', show)
cv2.waitKey(0)

scene = o3d.io.read_triangle_mesh(
    '../data/reconstruction/data/scene/integrated.ply')
o3d.visualization.draw_geometries([scene])
