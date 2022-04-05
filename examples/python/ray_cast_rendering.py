#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from torch import embedding
import misc3d as m3d
from matplotlib import pyplot as plt
import time


# Create a ray cast renderer.
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640, 480, 572.4114, 573.5704, 325.2611, 242.0489)
renderer = m3d.pose_estimation.RayCastRenderer(camera_intrinsic)

# Load mesh and create a copy.
mesh = o3d.io.read_triangle_mesh('../data/pose_estimation/model/obj.ply')
mesh = mesh.scale(0.001, np.array([0, 0, 0]))
mesh2 = mesh

# Create mesh pose relative to camera.
pose = np.array([[0.29493218,  0.95551309,  0.00312103, -0.14527225],
                 [0.89692822, -0.27572004, -0.34568516,  0.12533501],
                 [-0.32944616,  0.10475302, -0.93834537,  0.99371838],
                 [0.,          0.,          0.,          1.]])
pose2 = np.array([[0.29493218,  0.95551309,  0.00312103, -0.04527225],
                 [0.89692822, -0.27572004, -0.34568516,  0.02533501],
                 [-0.32944616,  0.10475302, -0.93834537,  0.99371838],
                 [0.,          0.,          0.,          1.]])

# Cast rays for meshes in the scene.
t0 = time.time()
ret = renderer.cast_rays([mesh, mesh2], [pose, pose2])
print('cast_rays:', time.time() - t0)

# Visualize the depth and instance map. Both of them are open3d.core.Tensor type, 
# which can be converted into numpy array.
depth = renderer.get_depth_map().numpy()
instance_map = renderer.get_instance_map().numpy()
instance_pcds = renderer.get_instance_point_cloud()
instance_pcds[0].paint_uniform_color([1, 0, 0])
instance_pcds[1].paint_uniform_color([0, 1, 0])

plt.subplot(1, 2, 1)
plt.imshow(depth)
plt.subplot(1, 2, 2)
plt.imshow(instance_map, vmax=2)
plt.show()

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries(
    [instance_pcds[0], instance_pcds[1], axis], "Ray Cast Rendering")
