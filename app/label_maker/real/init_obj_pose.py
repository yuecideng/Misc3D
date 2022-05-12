#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import open3d as o3d
import misc3d as m3d


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def read_model(path):
    try:
        pcd = o3d.io.read_point_cloud(path)
    except:
        print("Error: cannot read file: {}".format(path))
        return False, None

    if pcd.has_points() == False:
        return False, None

    return True, pcd


def RegistrationBySVD(model, scene, index1, index2, threshold=0.02):
    src = model.select_by_index(index1)
    dst = scene.select_by_index(index2)
    pose = m3d.registration.compute_transformation_least_square(src, dst)

    # icp refine
    result = o3d.pipelines.registration.registration_icp(
        model, scene, threshold, pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pose = result.transformation
    return pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to CAD model')
    parser.add_argument("--data_path", default='dataset',
                        help="path to RGBD data set")
    args = parser.parse_args()

    # read integrated scene
    scene = o3d.io.read_point_cloud(
        os.path.join(args.data_path, 'scene/integrated.ply'))

    init_poses = {}
    while True:
        file_name = input('Please enter the name of model CAD file: ')
        ret, model = read_model(os.path.join(
            args.model_path, file_name + '.ply'))
        if ret is False:
            print('Read model fail, please enter the correct file name.')
            continue

        model_index = pick_points(model)
        scene_index = pick_points(scene)

        if len(model_index) < 3 or len(scene_index) < 3 or len(model_index) != len(scene_index):
            print('Please pick at least three correspondences.')
            continue

        pose = RegistrationBySVD(model, scene, model_index, scene_index)
        model.transform(pose)
        model.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([model, scene])

        save = input('Save pose? (y/n): ')
        if save == 'y':
            if file_name not in init_poses:
                init_poses[file_name] = [pose.flatten().tolist()]
            else:
                init_poses[file_name].append(pose.flatten().tolist())

        end = input('Whether to continue? (y/n): ')
        if end == 'n':
            break

    # save reuslts inside data path
    with open(os.path.join(args.data_path, 'init_poses.json'), 'w') as f:
        json.dump(init_poses, f, indent=4)
