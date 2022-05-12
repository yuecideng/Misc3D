#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import open3d as o3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('geometry', help='path to geometry file (.ply)')
    parser.add_argument("--pcd", action='store_true',
                        help="vis point cloud")
    args = parser.parse_args()

    path_to_geometry = args.geometry
    if args.pcd:
        scene = o3d.io.read_point_cloud(path_to_geometry)
    else:
        scene = o3d.io.read_triangle_mesh(path_to_geometry)
        if scene.has_triangles() == False:
            scene = o3d.io.read_point_cloud(path_to_geometry)

    # create axis in world frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0])

    geometries = [scene, axis]
    o3d.visualization.draw_geometries(geometries)
