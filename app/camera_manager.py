#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pyrealsense2 as rs
from enum import IntEnum

class RealSenseResolution(IntEnum):
    Low = 0  # 480X270
    Medium = 1  # 640X480
    High = 2  # 848X480
    Max = 3  # 1280X720


class RealSenseFPS(IntEnum):
    Low = 0  # 6
    Medium = 1  # 15
    High = 2  # 30
    Max = 3  # 60


class RealSenseManager:

    def __init__(self,
                 resolution=RealSenseResolution.High,
                 fps=RealSenseFPS.High):
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()

        self.resolution = None
        if resolution == RealSenseResolution.Low:
            self.resolution = (480, 270)
        elif resolution == RealSenseResolution.Medium:
            self.resolution = (640, 480)
        elif resolution == RealSenseResolution.High:
            self.resolution = (848, 480)
        else:
            self.resolution = (1280, 720)

        fps_ = None
        if fps == RealSenseFPS.Low:
            fps_ = 6
        elif fps == RealSenseFPS.Medium:
            fps_ = 15
        elif fps == RealSenseFPS.High:
            fps_ = 30
        else:
            fps_ = 60

        self.config.enable_stream(rs.stream.depth, self.resolution[0],
                                  self.resolution[1], rs.format.z16, fps_)
        fps_color = 30 if fps_ > 30 else fps_
        self.config.enable_stream(rs.stream.color, self.resolution[0],
                                  self.resolution[1], rs.format.rgb8, fps_color)
        self.pipeline = rs.pipeline()

        self.profile = None
        self.depth_profile = None
        self.color_profile = None
        self.sensor = None

    def get_intrinsics(self, type='color'):
        """ Get intrinsics of the RGB camera or IR camera, which are varied with resolution 

        Args:
            power ([string]): color or depth

        Returns:
            [tuple[List, List]]: K and dist
        """
        if self.sensor is None:
            print('Sensor not opened!')
            return None
        intrin = None
        if type == 'color':
            intrin = self.profile.get_stream(
                rs.stream.color).as_video_stream_profile().get_intrinsics()
        else:
            intrin = self.profile.get_stream(
                rs.stream.depth).as_video_stream_profile().get_intrinsics()
        K = [intrin.fx, intrin.fy, intrin.ppx, intrin.ppy]
        dist = [
            intrin.coeffs[0], intrin.coeffs[1], intrin.coeffs[2],
            intrin.coeffs[3], intrin.coeffs[4]
        ]
        return (K, dist)

    def get_extrinsics(self):
        """ Get extrinsics from IR camera to RGB camera

        Returns:
            [np.ndarray(4 X 4)]: 
        """

        if self.sensor is None:
            print('Sensor not opened!')
            return None

        res = self.depth_profile.get_extrinsics_to(self.color_profile)
        rotation = np.array(res.rotation).reshape((3, 3))
        translation = np.array(res.translation)
        T = np.identity(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation

        return T

    def get_resolution(self):
        """ Get resolution of the camera

        Returns:
            tuple: (width, height)
        """
        return self.resolution

    def set_laser_power(self, power):
        """ Set laser power within range[10, 360])

        Args:
            power ([int]): laser power
        """
        power = max(10, min(360, power))
        if self.sensor is None:
            print('Sensor not opened!')
            return
        self.sensor.set_option(rs.option.laser_power, power)

    def get_data(self, hole_fill=False):
        """ Get data from the camera

        Args:
            hole_fill ([bool]): whether to fill the hole
            vis ([bool]): whether to show the image

        Returns:
            [tuple[pyrealsense2.frame, pyrealsense2.frame]]: color and depth frames
        """
        if self.sensor is None:
            print('Sensor not opened!')
            return None

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print('Can not get data from realsense!')
            return None
        # set fill = 2 will use 4 pixels neighbor for hole filling
        fill = 2 if hole_fill else 0
        depth_frame = rs.spatial_filter(0.5, 20, 2, fill).process(depth_frame)

        # convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return (color_image, depth_image)

    def open(self):
        """ Open the camera

        Returns:
            [bool]: 
        """
        self.profile = self.pipeline.start(self.config)
        self.depth_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.depth))
        self.color_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color))
        device = self.profile.get_device()
        if device.query_sensors().__len__() == 0:
            print('Can not find realsense sensor!')
            return False
        else:
            self.sensor = device.query_sensors()[0]
            # set default laser power
            self.sensor.set_option(rs.option.laser_power, 200)
            # set to high density mode
            self.sensor.set_option(rs.option.visual_preset, 4)
            return True

    def close(self):
        self.pipeline.stop()
        self.sensor = None
        self.profile = None
        self.depth_profile = None
        self.color_profile = None


if __name__ == '__main__':
    import cv2
    import numpy as np

    camera = RealSenseManager(resolution=RealSenseResolution.High,
                              fps=RealSenseFPS.High)
    camera.open()
    try:
        while True:
            t0 = time.time()
            color, depth = camera.get_data(hole_fill=False)
            color_image = np.asanyarray(color.get_data())
            depth_color_frame = rs.colorizer().colorize(depth)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            t1 = time.time()
            fps = 'FPS: ' + str(int(1 / (t1 - t0)))
            cv2.putText(color_image_bgr, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', color_image_bgr)
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_color_image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        camera.close()
        cv2.destroyAllWindows()
    finally:
        print('Exit')