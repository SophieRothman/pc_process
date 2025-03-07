# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:32:20 2025

@author: Sophie
"""
import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2
import matplotlib.pyplot as plt
# import cloudComPy as cc
# cc.initCC()

# note need to be in the cloudcompy310 environment
#pathname
cloud='F:/nz_data/05_03_25_faro/test/Mangarere_FARO_20170314_13.e57'
#name='Mangarere_FARO_20170314_01.e57'

#Compute normal oriented towards the sensor 10 cm? 2- cm? not sure yet its a compri
cc.octree_normals(cloud, 0.1, orient='PLUS_ORIGIN', model='QUADRIC',  fmt='BIN', silent=False, verbose=False, global_shift='AUTO', cc='C:\\Program Files\\CloudCompare\\CloudCompare.exe')

#Compute sensor range


# compute scattering angle


# create station scalar field populated with the station number 

# create a scan number


# merge all scans