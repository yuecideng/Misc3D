#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import argparse

import primitives_fitting as pf
from utils import draw_result

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='foo', type=str)
parser.add_argument('--primitives_type', default='plane', type=str)
# ransac fitting distance threshold
parser.add_argument('--threshold', default=0.005, type=float)
parser.add_argument('--enable_parallel', default=False, type=bool)
parser.add_argument('--max_iteration', default=200, type=int)
# segmentation distance and angle threshold
parser.add_argument('--dist_thresh', default=0.01, type=float)
parser.add_argument('--angle_thresh', default=10, type=float)
# pre-processing parameters
parser.add_argument('--voxel_size', default=0.002, type=float)
parser.add_argument('--enable_smoothing', default=False, type=bool)
# filtering parameters
parser.add_argument('--min_bound', default=0.01, type=float)
parser.add_argument('--max_bound', default=0.05, type=float)

args = parser.parse_args()
primitives_type = args.primitives_type
assert (primitives_type == 'plane' or primitives_type == 'sphere'
        or primitives_type == 'cylinder')

try:
    o3d_pts = o3d.io.read_point_cloud(args.file_path)
except:
    print('path error')
    exit()

config = pf.PrimitivesDetectorConfig()
config.m_cluster_param.dist_thresh = args.dist_thresh
config.m_cluster_param.angle_thresh = args.angle_thresh
config.m_cluster_param.min_cluster_size = 100

if primitives_type == 'plane':
    config.m_fitting_param.type = pf.PrimitivesType.plane
elif primitives_type == 'sphere':
    config.m_fitting_param.type = pf.PrimitivesType.sphere
elif primitives_type == 'cylinder':
    config.m_fitting_param.type = pf.PrimitivesType.cylinder
config.m_fitting_param.threshold = args.threshold
config.m_fitting_param.enable_parallel = args.enable_parallel
config.m_fitting_param.max_iteration = args.max_iteration
config.m_preprocess_param.voxel_size = args.voxel_size
config.m_preprocess_param.enable_smoothing = args.enable_smoothing

config.m_filtering_param.min_bound = args.min_bound
config.m_filtering_param.max_bound = args.max_bound

detector = pf.PrimitivesDetector(config)

start = time.time()
pc = np.asarray(o3d_pts.points)
if o3d_pts.has_normals():
    normals = np.asarray(o3d_pts.normals)
    ret = detector.detect(pc, normals)
else:
    ret = detector.detect(pc)
print('time cost: {}'.format(time.time() - start))
if ret is False:
    print('no primitives detected')
    exit()

clusters = detector.get_clusters()
params = detector.get_primitives()

primitives_list = []
colors = []
for i in range(len(params)):
    print('primitives {} param: {}'.format(i, params[i]))

    r = random()
    g = random()
    b = random()

    color = np.zeros((clusters[i].shape[0], clusters[i].shape[1]))
    color[:, 0] = r
    color[:, 1] = g
    color[:, 2] = b

    primitives_list.append(clusters[i])
    colors.append(color)

primitives_list = np.concatenate(primitives_list, axis=0)
colors = np.concatenate(colors, axis=0)
draw_result(primitives_list, colors)
