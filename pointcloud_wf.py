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
import matplotlib.pyplot as plt

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
# %% indexing stuff  in the dictionary



train_wft['labels'][train_wft['labels']==9]  #selects all the point cloud labels which are equal to 9
train_wft['features'][train_wft['labels']==9, train_wft['names']=='DZ_CTX_5@kNN=2'] #gets the value of feature 'DZ_CTX_5@kNN=2' for all points with label 9

np.average(train_wft['features'][train_wft['labels']==9, train_wft['names']=='DZ_CTX_5@kNN=2'])
np.nanmean(train_wft['features'][train_wft['labels']==3, train_wft['names']=='DZ_CTX_5@kNN=2'])


np.nanmean(train_wft['features'][train_wft['labels']==3, train_wft['names']=='DZ_CTX_5@kNN=2'])
# %%   make a histogram of how curvature for each label
labels_unique=np.unique(train_wft['labels'])

fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
count=0
for i in labels_unique:
    axs[count].hist(train_wft['features'][train_wft['labels']==i, train_wft['names']=='DZ_CTX_5@kNN=2'])
    axs[count].set_xlabel('Elevation above ground')
    axs[count].set_title(str(i))

    count=count+1


# %% can i load random point cloud data

sbf_filepath='F:/nz_data/05_02_25/sitescombined2009.riversurface.sbf'

sbf_data = sbf.read(sbf_filepath)
sf_dict = sbf_data.get_name_index_dict()
