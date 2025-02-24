# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:00:20 2025

@author: sophie.rothman
"""

import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2

parameters="F:/nz_data/05_02_25/Rangi2009_3dmasc_PC.txt"
parameters_train="F:/nz_data/05_02_25/Rangi2009_3dmasc_PC_train.txt"
parameters_test="F:/nz_data/05_02_25/Rangi2009_3dmasc_PC_test.txt"

pc1="F:/nz_data/05_02_25/sitescombined_2009_clone.bin"
core="F:/nz_data/05_02_25/sitescombined_subsampled.bin"
training="F:/nz_data/05_02_25/sitescombined2009.ta_rl.bin"
testing="F:/nz_data/05_02_25/sitescombined2009.tes_rl.bin"
ctx="F:/nz_data/05_02_25/riversurface.bin"


#calculating features on point cloud

clouds = (pc1, training,ctx)  # pc1, pc2 and core are full paths to clouds
training_ft = cc.q3dmasc(clouds, parameters_train, only_features=True, verbose=True, fmt='sbf')


#load the different point clouds into python LABELS MUST BE SET TO TRUE
train_wft=classification.cc_3dmasc.load_sbf_features("F:/nz_data/05_02_25/sitescombined2009.ta_rl_WITH_FEATURES.sbf",
                                                       parameters_train, labels=True, coords=True)