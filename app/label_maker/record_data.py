#!/usr/bin/python3
# -*- coding: utf-8 -*-

import shutil
import argparse
import numpy as np
import os
import time
import json
import cv2

from camera_manager import RealSenseManager, RealSenseFPS, RealSenseResolution


def remove_and_create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    os.makedirs(os.path.join(dir_path, 'depth'))
    os.makedirs(os.path.join(dir_path, 'color'))


def save_color_and_depth(color, depth, dir_path):
    global save_count
    cv2.imwrite(os.path.join(dir_path, 'color', '%06d.png' % save_count), color)
    cv2.imwrite(os.path.join(dir_path, 'depth', '%06d.png' % save_count), depth)


def save_camera_intrinsic(camera_intrinsic, resolution, dir_path):
    with open(os.path.join(dir_path, 'camera_intrinsic.json'), 'w') as outfile:
        obj = json.dump(
            {
                'width':
                resolution[0],
                'height':
                resolution[1],
                'fx': round(camera_intrinsic[0], 4),
                'fy': round(camera_intrinsic[1], 4),
                'cx': round(camera_intrinsic[2], 4),
                'cy': round(camera_intrinsic[3], 4),
                'depth_scale': 1000.0
            },
            outfile,
            indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="RGBD camera Recorder. Please select one of the optional arguments")
    parser.add_argument("--dataset",
                        default='dataset',
                        type=str,
                        help="name of dataset folder.")
    parser.add_argument("--record_mode",
                        default='d',
                        type=str,
                        help="mode of data capture. discrete (d) or continuous (c).")
    parser.add_argument("--frame_interval",
                        default='5',
                        type=int,
                        help="interval of frames to be recorded.")
    parser.add_argument("--maximum_frame",
                        default='200',
                        type=int,
                        help="the maximum number of frames to be recorded.")
    args = parser.parse_args()

    # create dataset folder
    remove_and_create_dir(args.dataset)

    camera = RealSenseManager(
        resolution=RealSenseResolution.Medium, fps=RealSenseFPS.High)
    camera.open()

    # save camera intrinsic
    intrinsic, _ = camera.get_intrinsics()
    resolution = camera.get_resolution()
    save_camera_intrinsic(intrinsic, resolution, args.dataset)

    count = 0
    save_count = 0
    try:
        while True:
            t0 = time.time()

            color, depth = camera.get_data()
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            color_cv = color.copy()
            depth_3d = np.dstack((depth, depth, depth))
            bg_removed = np.where((depth_3d <= 0), 127, color_cv)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)

            fps = 'FPS: ' + str(int(1 / (time.time() - t0)))
            cv2.putText(bg_removed, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('RGBD Recorder', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGBD Recorder', images)
            key = cv2.waitKey(1)

            count += 1
            if args.record_mode == 'd':
                if key == ord('s'):
                    save_color_and_depth(color, depth, args.dataset)
                    save_count += 1
            else:
                if count % args.frame_interval == 0:
                    save_color_and_depth(color, depth, args.dataset)
                    save_count += 1

            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                break

            if save_count >= args.maximum_frame:
                print('Maximum number of frames reached.')
                break

    except KeyboardInterrupt:
        camera.close()
    finally:
        print('Exit')
