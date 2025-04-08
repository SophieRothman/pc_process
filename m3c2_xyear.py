# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:49:03 2025

@author: Sophie
"""

import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2
import matplotlib.pyplot as plt
import os

pc2014='F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm.bin'
pc2015='F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
pc2017='F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_1cm.bin'
pc2023=''
cp='F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_20cm.bin'
params='F:/nz_data/m3c2_params2.txt'



# pc2014='F:/nz_data/faro/2014/Mangarere_2014000_1cm.bin'
# pc2015='F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
# pc2017='F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_1cm.bin'
# cp='F:/nz_data/m3c2test/Mangarere_2014000_20cm.bin'
# resultsarray=np.empty((int(numcom), 3))

pc1=cc.to_sbf(pc2014, silent=True, debug=False)
    
results=cc.m3c2(pc1, pc2017, params, core=cp, fmt='BIN',
         silent=True, debug=True)# cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')

# def m3c2(pc1, pc2, params, core=None, fmt='SBF',
#          silent=True,  debug=False, global_shift='AUTO', cc=cc_exe):

os.rename(results, 'F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm_M3C2_14_15.sbf')
os.rename('F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm_M3C2.sbf.data', 'F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm_M3C2_14_15.sbf.data')

results2=cc.m3c2(pc2014, pc2017, params, core=cp, fmt='SBF',
         silent=False)
os.rename(results, 'F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm_M3C2_14_172.sbf')
os.rename('F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm_M3C2.sbf.data', 'F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm_M3C2_14_172.sbf.data')
    #results=cc.icpm3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, silent=False, fmt='BIN', verbose=True)
    # pc_m3c2=sbf.read(results)
    # xyz=pc_m3c2.xyz
    # nn=pc_m3c2.sf_names
    # sfs=pc_m3c2.sf
    # sfnames=pc_m3c2.sf_names
    # #avg m3c2distance
    # resultsarray[i, 0]=np.nanmean(sfs[:, 6])
    # #stdev of m3c2 distance
    # resultsarray[i, 1]=np.nanstd(sfs[:, 6])
    # #fraction significant change
    # resultsarray[i, 2]=len(sfs[sfs[:, 4]==True, 4])/len(sfs[:, 4])
    
#%% trying to 3dmasc across different datasets
pc_full=['F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm.bin', 'F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin', 'F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_1cm.bin', 'F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_1cm_clean10.bin']
pc_cp=['F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_20cm.bin', 'F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin', 'F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_20cm.bin', 'F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_20cm_clean10.bin']
ws_pcs=['F:/nz_data/ws_pcs/Mangarere_2014000_ws.bin', 'F:/nz_data/ws_pcs/Mangarere_2015000_wsv2.bin', 'F:/nz_data/ws_pcs/Mangarere_2017039_ws.bin', 'F:/nz_data/ws_pcs/Mangarere_ZF_20230222_ws.bin']
classifier='F:/nz_data/05_02_25/5class_4scale_dz2_93v3.txt'
output_3dmasc=[]

# ef q3dmasc(clouds, training_file, only_features=False, keep_attributes=False,
#             silent=True, verbose=False, global_shift='AUTO', cc_exe=cc_exe, fmt='sbf'):
for i in range(1, 4): #len(pc_full)):
    clouds=(ws_pcs[i], pc_full[i], pc_cp[i])
    out = cc.q3dmasc(clouds, classifier, keep_attributes=True, verbose=True)
    output_3dmasc.append(out)
    print(i)
    print('************* is done')
output_3dmasc=np.array(output_3dmasc)
np.save('F:/nz_data/classified_file_names.txt', output_3dmasc)