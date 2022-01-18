#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

def transformation(points, transform):
    """ Transform point clouds 
    Args:
        points (ndarray): N x 3 point clouds
        transform (ndarray): 4 x 4 transform matrix
    Returns:
        [ndarray]: N x 3 transformed point clouds
    """

    points = np.concatenate((points, np.ones(len(points)).reshape(-1, 1)),
                            axis=1)

    points_ = transform.dot(points.T).T

    return points_[:, :3]


def np2o3d(xyz, normals=None):
    """ Convert numpy ndarray to open3D point cloud 

    Args:
        xyz ([np.ndarray]): [description]
        normals ([np.ndarray], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def draw_result(points, colors=None):
    """ Draw point clouds

    Args:
        points ([ndarray]): N x 3 array
        colors ([ndarray]): N x 3 array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])