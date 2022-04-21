#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import argparse
import os
import cv2
import shutil
import numpy as np
import open3d as o3d
import misc3d as m3d
import json
import sys
sys.path.append('../')
from utils import Colors
from utils import mask_to_bbox, rgbd_to_pointcloud

from IPython import embed


def remove_and_create_dir(dir_path):
    mask_path = os.path.join(dir_path, 'mask')
    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)
    os.makedirs(mask_path)


def read_model_and_init_poses(model_path, data_path):
    with open(os.path.join(data_path, 'init_poses.json'), 'r') as f:
        init_poses = json.load(f)

    models = {}
    for file_name in os.listdir(model_path):
        if file_name.endswith('.ply'):
            name = os.path.splitext(file_name)[0]
            if name in init_poses:
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(model_path, file_name))
                models[name] = mesh

    return (models, init_poses)


def read_rgbd_and_name(path):
    rgbds = []
    names = []

    color_files = os.listdir(os.path.join(path, 'color'))
    depth_files = os.listdir(os.path.join(path, 'depth'))
    for i in range(len(color_files)):
        color = cv2.imread(os.path.join(path, 'color', color_files[i]))
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


def read_odometry(path):
    odometrys = []
    f = open(os.path.join(path, 'scene/trajectory.json'), 'r')
    data = json.load(f)

    for key, value in data.items():
        if key == 'class_name':
            continue
        odometrys.append(np.array(value).reshape((4, 4)))

    return odometrys


def refine_local_pose(model, color, depth, camera, init_pose, threshold=0.005):
    intrin = (camera.intrinsic_matrix[0, 0],
              camera.intrinsic_matrix[1, 1], camera.intrinsic_matrix[0, 2], camera.intrinsic_matrix[1, 2])
    scene = rgbd_to_pointcloud(color, depth, intrin, 1000.0, 3.0, True)
    scene = scene.voxel_down_sample(voxel_size=0.01)

    model = o3d.geometry.PointCloud(model.vertices)
    bbox = model.get_oriented_bounding_box()
    bbox.rotate(init_pose[:3, :3], bbox.center)
    bbox.translate(init_pose[:3, 3])
    bbox.scale(1.2, bbox.center)
    scene = scene.crop(bbox)

    result = o3d.pipelines.registration.registration_icp(model, scene, threshold, init_pose,
                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pose = result.transformation

    return pose


def generate_label_and_save_mask(data_path, instance_map, init_poses, pose_list, name):
    # create new instance map and save it
    instance_mask = np.zeros(instance_map.shape, dtype=np.uint16)
    labels = []

    instance = 0
    for key, value in init_poses.items():
        for i in range(len(value)):
            label = {}
            label['obj_id'] = int(key)
            label['instance_id'] = instance
            label['cam_R_m2c'] = pose_list[instance][:3, :3].tolist()
            label['cam_t_m2c'] = pose_list[instance][:3, 3].tolist()
            mask_255 = np.zeros(instance_map.shape, dtype=np.uint8)
            mask_255[instance_map == instance] = 255
            bbox = mask_to_bbox(mask_255)
            label['bbox'] = bbox

            instance_value = int(key) * 1000 + instance
            instance_map[instance_map == instance] = instance_value
            instance_mask[instance_map == instance] = 0
            instance += 1
            labels.append(label)

    cv2.imwrite(os.path.join(data_path, 'mask', name + '.png'), instance_mask)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to CAD model')
    parser.add_argument("--data_path", default='dataset',
                        help="path to RGBD data set")
    parser.add_argument("--local_refine", action='store_true',
                        help="use icp the refine model to the scene")
    parser.add_argument("--vis", action='store_true',
                        help="visualize the rendering results")
    args = parser.parse_args()

    remove_and_create_dir(args.data_path)

    models, init_poses = read_model_and_init_poses(
        args.model_path, args.data_path)
    rgbds, file_names = read_rgbd_and_name(args.data_path)
    camera = read_camera_intrinsic(args.data_path)
    odometrys = read_odometry(args.data_path)

    render = m3d.pose_estimation.RayCastRenderer(camera)

    t0 = time.time()
    data_labels = {}
    for i in range(len(rgbds)):
        render_mesh = []
        mesh_pose = []
        odom = odometrys[i]

        for key, value in init_poses.items():
            for arr in value:
                pose = np.array(arr).reshape((4, 4))
                render_mesh.append(models[key])

                pose = np.linalg.inv(odom) @ pose
                if args.local_refine:
                    pose = refine_local_pose(
                        models[key], rgbds[i][0], rgbds[i][1], camera, pose)

                mesh_pose.append(pose)

        ret = render.cast_rays(render_mesh, mesh_pose)

        # rendering instance map
        instance_map = render.get_instance_map().numpy()

        label = generate_label_and_save_mask(
            args.data_path, instance_map, init_poses, mesh_pose, file_names[i])
        data_labels[file_names[i]] = label

        # visualization
        if args.vis:
            mask = np.zeros(
                (instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
            index = np.zeros(
                (instance_map.shape[0], instance_map.shape[1]), dtype=np.bool_)
            color = rgbds[i][0]

            for j in range(len(render_mesh)):
                mask[instance_map == j] = Colors()(j, True)
                index[instance_map == j] = True
                color[index] = cv2.addWeighted(color, 0.5, mask, 0.5, 0)[index]

            cv2.namedWindow('Instance Mask Rendering', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Instance Mask Rendering', color)
            cv2.imwrite(os.path.join(args.data_path, 'mask',
                        file_names[i] + '_vis.png'), color)
            key = cv2.waitKey(0)

    print('Time:', time.time() - t0)
    # save reuslts inside data path
    with open(os.path.join(args.data_path, 'labels.json'), 'w') as f:
        json.dump(data_labels, f, indent=4)
