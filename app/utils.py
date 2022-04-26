#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import open3d as o3d

from scipy.spatial.transform import Rotation


def mat_to_euler(m_list, mode='xyz', unit='m'):
    if type(m_list) != list:
        m_list = [m_list]
    out = []
    if isinstance(m_list, list) is False:
        m_list = [m_list]
    for m in m_list:
        r = m[:3, :3]
        euler = Rotation.from_matrix(r).as_euler(mode, degrees=True)

        if unit == 'mm':
            pose = np.r_[m[:3, 3] / 1000, euler].tolist()
        else:
            pose = np.r_[m[:3, 3], euler].tolist()
        out.extend(pose)

    return out


def mask_to_bbox(mask):
    """ Convert mask to bounding box (x, y, w, h).

    Args:
        mask (np.ndarray): maks with with 0 and 255.

    Returns:
        List: [x, y, w, h]
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def rgbd_to_pointcloud(rgb,
                       depth,
                       intrinsic,
                       depth_scale=1000.0,
                       depth_trunc=3.0,
                       project_valid_depth=False):
    """ Convert RGBD images to point cloud.

    Args:
        rgb ([np.ndarray]): rgb image.
        depth ([np.ndarray]): depth image.
        intrinsic ([tuple]): (fx, fy, cx, cy).
        depth_scale (float, optional): [description]. Defaults to 1000.0.
        depth_trunc (float, optional): [description]. Defaults to 3.0.
        project_valid_depth (bool, optional): [description]. Defaults to False.

    Returns:
        [open3d.geometry.PointCloud]: [description]
    """

    shape = depth.shape
    if shape[0] != rgb.shape[0] or shape[1] != rgb.shape[1]:
        print('Shape of depth and rgb image do not match.')
        return o3d.geometry.PointCloud()

    depth = o3d.geometry.Image(depth)
    color = o3d.geometry.Image(rgb)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity=False)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        shape[0], shape[1], intrinsic[0], intrinsic[1], intrinsic[2],
        intrinsic[3])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        pinhole_camera_intrinsic,
        project_valid_depth_only=project_valid_depth)

    return pcd


def depth_to_pointcloud(depth,
                        intrinsic,
                        depth_scale=1000.0,
                        depth_trunc=3.0,
                        project_valid_depth=False):
    """ Convert depth image to point cloud.

    Args:
        depth ([np.ndarray]): depth image.
        intrinsic ([tuple]): (fx, fy, cx, cy).
        depth_scale (float, optional): [description]. Defaults to 1000.0.
        depth_trunc (float, optional): [description]. Defaults to 3.0.
        project_valid_depth (bool, optional): [description]. Defaults to False.

    Returns:
        [open3d.geometry.PointCloud]: [description]
    """

    shape = depth.shape
    depth = o3d.geometry.Image(depth)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        shape[0], shape[1], intrinsic[0], intrinsic[1], intrinsic[2],
        intrinsic[3])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth,
        pinhole_camera_intrinsic,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        project_valid_depth_only=project_valid_depth)

    return pcd


def try_except(func):
    """
    Decorator to catch exceptions.
    """

    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def plot_bboxes(bboxes, img, map_class_name=None):
    """
    description: Plots bounding boxes on an image,
    param: 
        x (np.ndarray): bounding boxes with confidence score and class id
        img (np.ndarray): a image object
        map_class_name: (dict): a map from class id to class name
    """
    if bboxes is None:
        return 
    # generate color for unique class
    color = Colors()

    for bbox in bboxes:
        class_id = int(bbox[5])
        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img,
                      c1,
                      c2,
                      color(class_id, True),
                      thickness=2,
                      lineType=cv2.LINE_AA)
        if map_class_name is not None:
            class_name = map_class_name[class_id]
        else:
            class_name = str(class_id)
        text = 'class: ' + class_name + ' conf: {:.2f}'.format(bbox[4])
        t_size = cv2.getTextSize(text, 0, fontScale=0.5, thickness=1)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color(class_id, True), -1, cv2.LINE_AA)
        cv2.putText(
            img,
            text,
            (c1[0], c1[1] - 2),
            0,
            0.5,
            [225, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
