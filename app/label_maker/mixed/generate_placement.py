#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import open3d as o3d
import misc3d as m3d
import json
import argparse
import os
import shutil
from scipy.spatial.transform import Rotation

from helper import ModelSampler, OffScreenRenderer
from utils import rgbd_to_pointcloud, mask_to_bbox


class PlacementAgent:
    def __init__(self, model_num, grid_size=0.3, filter_threshold=0.4, camera=None, vis=False):
        self.model_num = model_num
        self.camera = camera
        self.normal = None
        self.plane_segment_threshold = 0.01
        self.voxel_size = grid_size
        self.cam_to_plane = np.identity(4)
        self.filter_threshold = filter_threshold
        self.vis = vis

    def generate(self, rgbd, max_num=None):
        color, depth = rgbd
        intrin = (self.camera.intrinsic_matrix[0, 0],
                  self.camera.intrinsic_matrix[1, 1],
                  self.camera.intrinsic_matrix[0, 2],
                  self.camera.intrinsic_matrix[1, 2])
        pcd = rgbd_to_pointcloud(color, depth, intrin, depth_trunc=1.5,
                                 project_valid_depth=True)
        pcd = pcd.voxel_down_sample(self.plane_segment_threshold)
        plane_pcd = self._segment_plane(pcd)

        centers = self._build_proposal_positions(plane_pcd)
        print('Generated {} proposals'.format(len(centers)))

        if max_num is not None:
            num_model = min(max_num, len(centers))
        else:
            num_model = len(centers)

        index_list = np.random.choice(len(centers), num_model, replace=False)

        model_id = np.random.choice(self.model_num, num_model) + 1
        placement_pose = {}

        for i, index in enumerate(index_list):
            pose = np.identity(4)
            pose[:3, 3] = centers[index]
            z_random_rotate = np.random.random() * 180
            pose[:3, :3] = Rotation.from_euler(
                'z', z_random_rotate).as_matrix()
            pose = self.cam_to_plane @ pose
            key = str(model_id[i])
            if key not in placement_pose:
                placement_pose[key] = [pose]
            else:
                placement_pose[key].append(pose)

        return placement_pose

    def _segment_plane(self, pcd):
        w, indices = pcd.segment_plane(self.plane_segment_threshold, 3, 1000)
        pcd_plane = pcd.select_by_index(indices)
        pcd_plane_projected = m3d.preprocessing.project_into_plane(pcd_plane)
        w, _ = pcd_plane_projected.segment_plane(
            self.plane_segment_threshold, 3, 1000)
        self.normal = np.array(w[:3])
        if np.linalg.norm(np.cross(self.normal, [0, 0, 1])) > 0:
            self.normal = -self.normal

        return pcd_plane

    def _build_proposal_positions(self, pcd):
        bbox = pcd.get_oriented_bounding_box()
        box_points = bbox.get_box_points()
        R = np.identity(3)

        # find nearest point to the camera position
        target = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.array([[0, 0, 0]])))
        distances = pcd.compute_point_cloud_distance(target)
        close_point = pcd.points[np.argmin(distances)]

        self.cam_to_plane[:3, 3] = close_point
        R[:3, 2] = self.normal
        # R[:3, 1] = np.array([0, 1, 0])
        R[:3, 0] = np.array([1, 0, 0])
        R[:3, 1] = np.cross(R[:3, 0], R[:3, 2])
        self.cam_to_plane[:3, :3] = R
        pcd.transform(np.linalg.inv(self.cam_to_plane))

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, self.voxel_size)

        voxel_dict = {}
        max_size = 0
        for i, point in enumerate(pcd.points):
            index = voxel_grid.get_voxel(point)
            index = str(index.tolist())
            if index not in voxel_dict:
                voxel_dict[index] = [i]
            else:
                voxel_dict[index].append(i)

            max_size = max(max_size, len(voxel_dict[index]))

        indices = []
        voxel_centers = []
        for key, value in voxel_dict.items():
            voxel_points = pcd.select_by_index(value)
            if len(value) > self.filter_threshold * max_size:
                indices.extend(value)
                center = np.mean(voxel_points.points, axis=0)
                center[2] = np.max(voxel_points.points, axis=0)[2]
                voxel_centers.append(center)

        if self.vis:
            pcd_filtered = pcd.select_by_index(indices)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2, [
                0, 0, 0])
            voxel_centers = np.array(voxel_centers)
            target = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(voxel_centers))
            target.paint_uniform_color([1, 0, 0])

            # axis.transform(camera_to_plane)
            o3d.visualization.draw_geometries(
                [axis, target, pcd_filtered, voxel_grid])

        return voxel_centers


def remove_and_create_dir(dir_path):
    remove_and_create_dir_with_name(dir_path, 'mask')
    remove_and_create_dir_with_name(dir_path, 'color')
    remove_and_create_dir_with_name(dir_path, 'depth')


def remove_and_create_dir_with_name(dir_path, name):
    mask_path = os.path.join(dir_path, name)
    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)
    os.makedirs(mask_path)


def read_rgbd_and_name(path):
    rgbds = []
    names = []

    color_files = os.listdir(os.path.join(path, 'color'))
    depth_files = os.listdir(os.path.join(path, 'depth'))
    for i in range(len(color_files)):
        color = cv2.imread(os.path.join(path, 'color', color_files[i]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(
            path, 'depth', depth_files[i]), cv2.IMREAD_UNCHANGED)
        rgbds.append((color, depth))
        names.append(os.path.splitext(color_files[i])[0])

    return rgbds, names


def read_camera_intrinsic(path):
    f = open(os.path.join(path, 'camera_intrinsic.json'), 'r')
    data = json.load(f)
    camera = o3d.camera.PinholeCameraIntrinsic(
        data['width'], data['height'], data['fx'], data['fy'], data['cx'], data['cy'])
    return camera


def save_color_and_depth(color, depth, dir_path, save_count):
    cv2.imwrite(os.path.join(dir_path, 'color',
                '%06d.png' % save_count), color)
    cv2.imwrite(os.path.join(dir_path, 'depth',
                '%06d.png' % save_count), depth)


def generate_label_and_save_mask(data_path, instance_map, init_poses, name):
    # create new instance map and save it
    instance_mask = np.zeros(instance_map.shape, dtype=np.uint16)
    labels = []

    instance = 0
    for key, value in init_poses.items():
        for i, pose in enumerate(value):
            label = {}
            label['obj_id'] = int(key)
            label['instance_id'] = instance
            label['cam_R_m2c'] = pose[:3, :3].tolist()
            label['cam_t_m2c'] = pose[:3, 3].tolist()
            mask_255 = np.zeros(instance_map.shape, dtype=np.uint8)
            mask_255[instance_map == instance] = 255
            bbox = mask_to_bbox(mask_255)
            label['bbox'] = bbox

            instance_value = int(key) * 1000 + instance
            instance_mask[instance_map == instance] = instance_value
            instance += 1
            labels.append(label)

    cv2.imwrite(os.path.join(data_path, 'mask',
                '%06d.png' % name), instance_mask)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to CAD model')
    parser.add_argument("--model_unit", default='m', type=str,
                        help="unit of CAD model")
    parser.add_argument("--max_instances", default=4, type=int,
                        help="maximal number of instances rendered in the scene")
    parser.add_argument("--data_path", default='dataset',
                        help="path to RGBD data set")
    parser.add_argument("--grid_size", default=0.3, type=float,
                        help="voxel grid size of placement division")
    parser.add_argument("--filter_threshold", default=0.4, type=float,
                        help="threshold to filter voxel grid")
    parser.add_argument("--light_intensity", default=100000, type=int,
                        help="light intensity for renderer")
    parser.add_argument("--vis", action='store_true',
                        help="visualize the rendering results")
    args = parser.parse_args()

    save_path = args.data_path + '_mixed'
    remove_and_create_dir(save_path)

    # copy camera intrinsic
    shutil.copyfile(os.path.join(args.data_path, 'camera_intrinsic.json'),
                    os.path.join(save_path, 'camera_intrinsic.json'))

    # init model sampler
    sampler = ModelSampler(args.model_path, unit=args.model_unit)
    agent = PlacementAgent(sampler.get_model_num(), args.grid_size, args.filter_threshold,
                           read_camera_intrinsic(args.data_path), args.vis)
    rgbds, names = read_rgbd_and_name(args.data_path)
    camera_intrinsic = read_camera_intrinsic(args.data_path)
    renderer = OffScreenRenderer(
        camera_intrinsic, sampler, args.light_intensity)

    intrin = (camera_intrinsic.intrinsic_matrix[0, 0], camera_intrinsic.intrinsic_matrix[1, 1],
              camera_intrinsic.intrinsic_matrix[0, 2], camera_intrinsic.intrinsic_matrix[1, 2])

    for i, rgbd in enumerate(rgbds):
        place_poses = agent.generate(rgbd, args.max_instances)
        color, depth, mask = renderer.rendering(place_poses, rgbd)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        save_color_and_depth(color, depth, save_path, i)
        generate_label_and_save_mask(save_path, mask, place_poses, i)

        if args.vis:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((color, depth_colormap))
            cv2.imshow('RGBD', images)
            cv2.waitKey(0)

