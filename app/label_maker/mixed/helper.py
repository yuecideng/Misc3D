#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import misc3d as m3d
import os
import copy
import cv2

from IPython import embed


def adjust_gamma(image, gamma=1.0):
    """ Adjust gamma of an image.

    Args:
        image (np.ndarray): input image
        gamma (float, optional): gamma value. Defaults to 1.0.

    Returns:
        np.ndarray: output image
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def o3d_image_to_numpy(o3d_image):
    """ Convert open3d legacy image to numpy array.

    Args:
        o3d_image (open3d.geometry.Image): open3d image

    Returns:
        np.ndarray: numpy image
    """

    o3d_t = o3d.t.geometry.Image.from_legacy(o3d_image)
    return o3d_t.as_tensor().numpy()


class ModelSampler:
    def __init__(self, path, unit='m'):
        self.path = path
        self.model_name = {}
        self.__load_model_file()
        self.unit = unit

    def __load_model_file(self):
        for file_name in os.listdir(self.path):
            if file_name.endswith('.ply'):
                name = os.path.splitext(file_name)[0]
                self.model_name[name] = os.path.join(self.path, file_name)

    def __load_single_model(self, name):
        mesh_model = o3d.io.read_triangle_model(name)
        mesh = o3d.io.read_triangle_mesh(name)
        if self.unit == 'mm':
            mesh.scale(0.001, [0, 0, 0])
            mesh_model.meshes[0].mesh.scale(0.001, [0, 0, 0])

        return (mesh, mesh_model)

    def get_model_num(self):
        return len(self.model_name)

    def get_model(self, name):
        if name not in self.model_name:
            print('Model {} not found.'.format(name))
            return None
        return self.__load_single_model(self.model_name[name])


# TODO: 1. Add noise for the rendered model
#       2. Add lighting and shader for the rendered model
class OffScreenRenderer:
    def __init__(self, camera, model_sampler, light_intensity=100000) -> None:
        self.camera = camera
        self.width = camera.width
        self.height = camera.height
        self.model_sampler = model_sampler
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.width, self.height)
        self.ray_caster = m3d.pose_estimation.RayCastRenderer(self.camera)

        # setup camera
        self.renderer.setup_camera(self.camera, np.identity(4))
        # set background to black
        self.renderer.scene.set_background_color([0, 0, 0, 1.0])
        # set light intensity
        # self.renderer.scene.scene.set_indirect_light(os.path.join(
        #     o3d.__path__[0], 'resources', 'crossroads'))
        self.renderer.scene.scene.set_indirect_light_intensity(light_intensity)

        self.name_list = []

    def rendering(self, pose_dict, rgbd=None):
        """ Render rgb image, depth image and instance map given poses and model.

        Args:
            pose_dict (dict): {name: [np.ndarray]}
            rgbd (tuple, optional): rgbd image. Defaults to None.

        Returns:
            (rgb, depth, instance_map): 
        """
        if rgbd is None:
            color, depth = None, None
        else:
            color, depth = rgbd
        mesh_list = []
        mesh_pose = []
        instance_count = 0
        for key, value in pose_dict.items():
            for pose in value:
                # get model mesh by key
                mesh_info = self.model_sampler.get_model(key)
                mesh, mesh_render = mesh_info

                mesh_list.append(mesh)
                mesh_pose.append(pose)

                mesh_render.meshes[0].mesh.transform(pose)
                # material = o3d.visualization.rendering.MaterialRecord()
                geometry_name = str(instance_count)
                self.renderer.scene.add_model(
                    geometry_name, mesh_render)
                self.name_list.append(geometry_name)
                instance_count += 1

        # rendering rgb image
        img = self.renderer.render_to_image()
        self.__remove_geometries()

        # ray casting depth and instance map
        ret = self.ray_caster.cast_rays(mesh_list, mesh_pose)
        instance_map = self.ray_caster.get_instance_map().numpy()
        depth_render = self.ray_caster.get_depth_map().numpy()
        depth_render = (depth_render * 1000).astype(np.uint16)

        img = o3d_image_to_numpy(img)

        color_merged = self.__merge_image(color, img, depth_render)
        depth_merged = self.__merge_depth(depth, depth_render, img)
        instance_mask = self.__modify_instance_map(instance_map, pose_dict)

        return (color_merged, depth_merged, instance_mask)

    def __remove_geometries(self):
        for name in self.name_list:
            self.renderer.scene.remove_geometry(name)
        self.name_list = []

    def __merge_image(self, color, img, depth_render):
        if color is None:
            return img

        color_bk = color.copy()
        index = np.zeros(
            (img.shape[0], img.shape[1]), dtype=np.bool_)
        index[depth_render != 0] = True
        color_bk[index] = cv2.addWeighted(color_bk, 0, img, 1.0, 0)[index]
        return color_bk

    def __merge_depth(self, depth, depth_render, img):
        if depth is None:
            return depth_render

        depth[depth_render != 0] = 0
        return depth_render + depth

    def __modify_instance_map(self, instance_map, pose_dict):
        instance_mask = np.zeros(instance_map.shape, dtype=np.uint16)
        instance = 0
        for key, value in pose_dict.items():
            for _ in value:
                instance_value = int(key) * 1000 + instance
                instance_mask[instance_map == instance] = instance_value
        return instance_mask
