#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from random import random
import open3d as o3d
import argparse

import misc3d as m3d 
from utils import np2o3d
from IPython import embed


points = np.random.randint(0, 10, size=(100, 3))


pc = np2o3d(points)

w, index = m3d.common.fit_plane(pc)
