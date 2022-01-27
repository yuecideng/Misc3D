import pyrealsense2 as rs
import numpy as np
import cv2
import time
import open3d as o3d
import os
import sys
import shutil

import misc3d as m3d 
from IPython import embed

view_ind = 0
break_loopFlag = 0
backgroundColorFlag = 1


# save image, depth image and point clouds in
def save_data(vis):
    global view_ind, depth_image, color_image1, pcd
    if not os.path.exists('./output/'):
        os.makedirs('./output/color/')
        os.makedirs('./output/depth/')
        os.makedirs('./output/pcd/')
    cv2.imwrite('./output/depth/depth_' + str(view_ind) + '.png', depth_image)
    cv2.imwrite('./output//color/color_' + str(view_ind) + '.png',
                color_image1)
    o3d.io.write_point_cloud('./output/pcd/pcd' + str(view_ind) + '.ply', pcd)
    view_ind += 1


def clear_saved_data(vis):
    if os.path.exists('./output/'):
        shutil.rmtree('./output/')


# exit
def break_loop(vis):
    global break_loopFlag
    break_loopFlag += 1
    cv2.destroyAllWindows()
    vis.destroy_window()

    sys.exit()


# change background color between white and black
def change_background_color(vis):
    global backgroundColorFlag
    opt = vis.get_render_option()
    if backgroundColorFlag:
        opt.background_color = np.asarray([0, 0, 0])
        backgroundColorFlag = 0
    else:
        opt.background_color = np.asarray([1, 1, 1])
        backgroundColorFlag = 1


if __name__ == "__main__":
    # align rgb and depth
    align = rs.align(rs.stream.color)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

    # create a pipeline handle
    pipeline = rs.pipeline()

    profile = pipeline.start(config)

    device = profile.get_device()
    try:
        depth_sensor = device.query_sensors()[0]
    except:
        print('Can not find sensor!')
        exit(0)

    # set defaulr laser power
    depth_sensor.set_option(rs.option.laser_power, 200)

    # get camera intrinsics
    intr = profile.get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()
    print('fx: {}, fy: {}, cx: {}, cy: {}'.format(intr.fx, intr.fy, intr.ppx,
                                                  intr.ppy))
    print('k1: {}, k2: {}, p1: {}, p2: {}, k3: {}'.format(
        intr.coeffs[0], intr.coeffs[1], intr.coeffs[2], intr.coeffs[3],
        intr.coeffs[4]))

    # create pinhole camera mode
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    # create vis object with callback functions
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # create open3d vis window
    vis.create_window("Pointcloud", 1200, 800)
    # create point cloud onject for vis
    pointcloud = o3d.geometry.PointCloud()

    # register callback functions
    vis.register_key_callback(ord("S"), save_data)
    vis.register_key_callback(ord("C"), clear_saved_data)
    vis.register_key_callback(ord("Q"), break_loop)
    vis.register_key_callback(ord("K"), change_background_color)

    try:
        while True:
            time_start = time.time()

            # clear point cloud
            pointcloud.clear()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # get color data
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # get depth data
            depth_frame = aligned_frames.get_depth_frame()

            # post processing
            depth_frame = rs.decimation_filter(1).process(depth_frame)

            # convert data
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            depth_image = np.asanyarray(depth_frame.get_data())
            # convert to opencv form
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # convert to open3d data format
            depth = o3d.geometry.Image(depth_image)
            color = o3d.geometry.Image(color_image)

            # create RGBD object
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, convert_rgb_to_intensity=False)
            # create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, pinhole_camera_intrinsic, project_valid_depth_only=False)

            pointcloud += pcd  

            if not geometrie_added:
                # add point cloud to vis
                vis.add_geometry(pointcloud)
                geometrie_added = True

            # update geometry
            vis.update_geometry(pointcloud)

            vis.poll_events()
            # udate render
            vis.update_renderer()

            # frame rate
            time_end = time.time()
            # print("FPS = {0}".format(int(1 / (time_end - time_start))))

            # show image and depth map
            img_show = color_image1.copy()
            fps = 'FPS: ' + str(int(1 / (time_end - time_start)))
            cv2.putText(img_show, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', img_show)
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_color_image)
            cv2.waitKey(1)

    finally:
        pipeline.stop()
