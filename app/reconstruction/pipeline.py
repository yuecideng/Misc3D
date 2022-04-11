#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import misc3d as m3d


class ReconstructionConfig:
    camera = o3d.camera.PinholeCameraIntrinsic()