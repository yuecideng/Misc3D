#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import open3d as o3d
import misc3d as m3d


def crop_geometry(geometry):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'F' to switch to freeview mode")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

    geo = vis.get_cropped_geometry()
    return geo


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


def init_by_least_square(model, scene, index1, index2, threshold=0.02):
    src = model.select_by_index(index1)
    dst = scene.select_by_index(index2)
    pose = m3d.registration.compute_transformation_least_square(src, dst)

    # icp refine
    result = o3d.pipelines.registration.registration_icp(
        model, scene, threshold, pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pose = result.transformation
    return pose


def init_by_ppf(model, scene):
    # init ppf 
    config = m3d.pose_estimation.PPFEstimatorConfig()
    config.training_param.rel_sample_dist = 0.05
    config.score_thresh = 0.01
    config.refine_param.method = m3d.pose_estimation.PPFEstimatorConfig.PointToPoint
    config.training_param.use_external_normal = True
    config.ref_param.ratio = 1.0

    ppf = m3d.pose_estimation.PPFEstimator(config)
    ret = ppf.train(model)
    ret, results = ppf.estimate(scene)

    pose = results[0].pose
    reg_result = o3d.pipelines.registration.registration_icp(
        model, scene, 0.01, pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pose = reg_result.transformation
    return pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to CAD model')
    parser.add_argument("--data_path", default='dataset',
                        help="path to RGBD data set")
    parser.add_argument("--enable_ppf",
                        action='store_true',
                        help="whether to use ppf to init obj pose")
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

        if args.enable_ppf:
            cropped_pcd = crop_geometry(scene)
            pose = init_by_ppf(model, cropped_pcd)

        # Use manually selected points to init obj pose
        else:
            model_index = pick_points(model)
            scene_index = pick_points(scene)

            if len(model_index) < 3 or len(scene_index) < 3 or len(model_index) != len(scene_index):
                print('Please pick at least three correspondences.')
                continue

            pose = init_by_least_square(
                model, scene, model_index, scene_index)

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
