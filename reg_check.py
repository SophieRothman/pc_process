# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:38:40 2025

@author: Sophie
"""

import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2
import matplotlib.pyplot as plt
params='F:/nz_data/m3c2_params.txt'

txtfile='F:/nz_data/05_03_25_faro/faro_list.txt'
cliff_cp='F:/nz_data/05_03_25_faro/test/Mangarere_2017.cliffcorepoints.bin'
filelist=np.array([])
with open(txtfile) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
        parsed=item.split('.')
        item='F:/nz_data/05_03_25_faro/'+parsed[0]+ '.'+parsed[1]
        #outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]

        filelist=np.append(filelist, item, axis=None)
#get all combinations
count=0
numcom=(len(filelist)*(len(filelist)-1))/2
combinations=np.zeros((int(numcom), 2))
#combinations=np.atleast_2d(combinations).T

for i in np.arange(0, len(filelist)):
    togo=len(filelist)-i-1
    col1=np.ones([togo, 1])*i
    col2=np.arange((i+1), len(filelist))
    col2=np.atleast_2d(col2).T
    col1=np.append(col1, col2, axis=1)
    combinations[count:(count+len(filelist)-i-1), :]=col1
    count=count+len(filelist)-i-1

for i in range(0, 1): #finally this will read 0, numcom
    results=cc.m3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, fmt='SBF',
             silent=True)
    #results=cc.icpm3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, silent=False, fmt='BIN', verbose=True)

    cloud_m3c2=sbf.read_sbf(results)
