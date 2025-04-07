# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:24:50 2025

@author: Sophie
"""

import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2

#First compute features on your training point cloud using the command line
parameter="F:/nz_data/Rangi2009_3dmasc_PC_analysis.txt"
pc1="F:/nz_data/05_03_25_leica/PBs5_feb11_1cm.bin"
core="F:/nz_data/05_03_25_leica/M3C2_feb09_feb11_m3c2_unc10cm.bin"
ctx="F:/nz_data/05_03_25_leica/line_feb11.bin"
a_points="F:/nz_data/rerepoints.txt"


pointlistx=[]
pointlisty=[]
with open(a_points) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
        parsed=item.split(' ')
        parsed1=parsed[0].split('.')
        #item=prefix+item#parsed[0]+ '.'+parsed[1]
        #outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]
        #fitem=[1, 1]
        #fitem=np.empty([1, 2])
        pointlistx.append(float(parsed[0]))
        pointlisty.append(float(parsed[1]))
        #fitem[1]=float(parsed[1])
        #pointlist=np.append(pointlist, fitem, axis=1)

#I think this calculates all the features on a point cloud

clouds = (pc1, core,ctx)  # pc1, pc2 and core are full paths to clouds
features = cc.q3dmasc(clouds, parameter, only_features=True, verbose=True, fmt='sbf')

#load the different point clouds into python LABELS MUST BE SET TO TRUE
train_wft=classification.cc_3dmasc.load_sbf_features(features, parameter, labels=True, coords=True)

#put into local coordinates

pointlistx=np.array(pointlistx)
pointlistx=pointlistx-395000
pointlisty=np.array(pointlisty)
pointlisty=pointlisty-5590000


coords=train_wft['coords']
dz=train_wft['features'][:, 0]
mask=dz>0.3 & dz<0.35
index=np.empty([len(pointlistx), 1])
for i in range(0,len(pointlistx)):
    leftside=coords[:, 0]<pointlistx[i]
    
