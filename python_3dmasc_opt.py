# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:50:25 2025

@author: sophie.rothman
"""

#setting up
import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2

#First compute features on your training point cloud using the command line
parameters="F:/nz_data/05_02_25/messing_with_python/Rangi2009_3dmasc_PC_ex.txt"
pc1="F:/nz_data/05_02_25/messing_with_python/sitescombined_2009_clone.bin"
core="F:/nz_data/05_02_25/messing_with_python/sitescombined_subsampled.bin"
training="F:/nz_data/05_02_25/messing_with_python/sitescombined2009.ta_rl.bin"
testing="F:/nz_data/05_02_25/messing_with_python/sitescombined2009.tes_rl.bin"

#I think this calculates all the features on a point cloud

clouds = (pc1, core)  # pc1, pc2 and core are full paths to clouds
core_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')  #creating features on the core points

clouds = (pc1, training)  # pc1, pc2 and core are full paths to clouds
training_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')

clouds = (pc1, testing)  # pc1, pc2 and core are full paths to clouds
testing_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')

#load the different point clouds into python LABELS MUST BE SET TO TRUE
train_wft=classification.cc_3dmasc.load_sbf_features("F:/nz_data/05_02_25/messing_with_python/sitescombined2009.ta_rl_WITH_FEATURES.sbf",
                                                       parameters, labels=True, coords=True)
#train_wft['labels']=np.array(['Cliff', 'Bedrock', 'Debris/Cobbles', 'Vegetation', 'Water'])
test_wft=classification.cc_3dmasc.load_sbf_features("F:/nz_data/05_02_25/messing_with_python/sitescombined2009.tes_rl_WITH_FEATURES.sbf",
                                                       parameters,  labels=True, coords=True)
#test_wft['labels']=np.array(['Cliff', 'Bedrock', 'Debris/Cobbles', 'Vegetation', 'Water'])

n_scales=4
n_features=50
eval_sc=0.4 #needs to be one of the scales of analysis

#lidar
#scales, names, ds_names = lidar_platform.classification.feature_selection.get_scales_feats(trads)

#search_set = np.array(np.where(scales == eval_sc)[0].tolist())
dictopt_param=lidar_platform.classification.feature_selection.rf_ft_selection(train_wft,
                  test_wft, n_scales, n_features, eval_sc, threshold=0.85, step=1)


# %% trying to do with a dz2 feature of the river surface




parameters="F:/nz_data/05_02_25/Rangi2009_3dmasc_PCv2.txt"
parameters_train="F:/nz_data/05_02_25/Rangi2009_3dmasc_PC_train.txt"
parameters_test="F:/nz_data/05_02_25/Rangi2009_3dmasc_PC_test.txt"

pc1="F:/nz_data/05_02_25/sitescombined_2009_clone.bin"
core="F:/nz_data/05_02_25/sitescombined_subsampled.bin"
training="F:/nz_data/05_02_25/sitescombined2009.ta_rl2.bin"
testing="F:/nz_data/05_02_25/sitescombined2009.tes_rl.bin"
ctx="F:/nz_data/05_02_25/riversurface.bin"

# clouds = (pc1, core)  # pc1, pc2 and core are full paths to clouds
# core_ft = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True, fmt='sbf')  #creating features on the core points

clouds = (pc1, training,ctx)  # pc1, pc2 and core are full paths to clouds
training_ft = cc.q3dmasc(clouds, parameters_train, only_features=True, verbose=True, fmt='sbf')

clouds = (pc1,  testing, ctx)  # pc1, pc2 and core are full paths to clouds
testing_ft = cc.q3dmasc(clouds, parameters_test, only_features=True, verbose=True, fmt='sbf')

#load the different point clouds into python LABELS MUST BE SET TO TRUE
train_wft=classification.cc_3dmasc.load_sbf_features("F:/nz_data/05_02_25/sitescombined2009.ta_rl_WITH_FEATURES.sbf",
                                                       parameters_train, labels=True, coords=True)
#train_wft['labels']=np.array(['Cliff', 'Bedrock', 'Debris/Cobbles', 'Vegetation', 'Water'])
test_wft=classification.cc_3dmasc.load_sbf_features("F:/nz_data/05_02_25/sitescombined2009.tes_rl_WITH_FEATURES.sbf",
                                                       parameters_test,  labels=True, coords=True)
#test_wft['labels']=np.array(['Cliff', 'Bedrock', 'Debris/Cobbles', 'Vegetation', 'Water'])

n_scales=4
n_features=50
eval_sc=0.4 #needs to be one of the scales of analysis

#lidar
#scales, names, ds_names = lidar_platform.classification.feature_selection.get_scales_feats(trads)

#search_set = np.array(np.where(scales == eval_sc)[0].tolist())
dictopt_param=lidar_platform.classification.feature_selection.rf_ft_selection(train_wft,
                  test_wft, n_scales, n_features, eval_sc, threshold=0.85, step=1)

# dictopt_getn=lidar_platform.classification.feature_selection.get_n_optimal_sc_ft(train_wft,
#                   test_wft, n_scales, n_features, eval_sc, threshold=0.85)

wait=2
threshold=0.02
best_ft=lidar_platform.classification.feature_selection.get_best_rf_select_iter(dictopt_param, train_wft, test_wft, wait, threshold )
#cc_3dmasc.get_shap_expl()
#best_ft.save('F:/nz_data/05_02_25/python_classifier.yaml')
# %% trying to classify it

cp='F:/nz_data/05_03_25_faro/test/slice20cm.bin'
allp='F:/nz_data/05_03_25_faro/test/slice1cm.bin'
ws='F:/nz_data/05_03_25_faro/inner_outer/Mangarere_2017039_ws.bin'
clouds=(allp, cp, ws)
out=cc.q3dmasc(clouds, best_ft[1])
classification.cc_3dmasc.test()

