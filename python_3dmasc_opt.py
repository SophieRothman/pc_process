# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:50:25 2025

@author: sophie.rothman
"""

#setting up

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification 
import lidar_platform.classification.feature_selection
import cv2

#First compute features on your training point cloud using the command line
parameters="D:/nz_data/05_02_25/4cats_classifier.txt"
pc1="D:/nz_data/05_02_25/sitescombined2009_allsfs.sbf"
core="D:/nz_data/05_02_25/sitescombined2009_cpallsfs.sbf"
training="D:/nz_data/05_02_25/sitescombined_training.sbf"
testing="D:/nz_data/05_02_25/sitescombined_testing.sbf"

clouds = (pc1, core)  # pc1, pc2 and core are full paths to clouds
core_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')  #creating features on the core points

clouds = (pc1, training)  # pc1, pc2 and core are full paths to clouds
training_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')

clouds = (pc1, testing)  # pc1, pc2 and core are full paths to clouds
testing_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')

#load the different point clouds into python
train_wft=classification.cc_3dmasc.load_sbf_features("D:/nz_data/05_02_25/sitescombined_training_WITH_FEATURES.sbf",
                                                       parameters)
test_wft=classification.cc_3dmasc.load_sbf_features("D:/nz_data/05_02_25/sitescombined_testing_WITH_FEATURES.sbf",
                                                       parameters)
n_scales=8
n_features=50
eval_sc=0.6
dictopt_param=lidar_platform.classification.feature_selection.rf_ft_selection(train_wft,
                  test_wft, n_scales, n_features, eval_sc, threshold=0.85, step=1)