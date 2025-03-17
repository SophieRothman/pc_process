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
import os
# import cloudComPy as cc
# cc.initCC()

# note need to be in the cloudcompy310 environment
#pathname
cloud='F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_01.e57'
cloud2='F:/nz_data/05_03_25_faro/test/Mangarere_2017_partof13.e57'
cloud3='F:/nz_data/05_03_25_faro/inner_outer/Mangarere_FARO_20170314_01out.e57'
test='F:/nz_data/05_03_25_faro/test/Mangarere_2017052_little.bin'
txtfile='F:/nz_data/05_03_25_faro/faro_list.txt'



#%%
filelist=np.array([])
filelist_final=np.array([])
with open(txtfile) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
        parsed=item.split('.')
        item='F:/nz_data/05_03_25_faro/'+parsed[0]+ '.'+parsed[1]
        #outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]

        filelist=np.append(filelist, item, axis=None)

for i in range(0, len(filelist)):
 
    #name='Mangarere_FARO_20170314_01.e57'
    
    #Compute normal oriented towards the sensor 10 cm? 2- cm? not sure yet its a compri
    cc.octree_normals(filelist[i], 0.1,  with_grids=True,  angle=1, orient='WITH_GRIDS',  verbose=False, fmt='bin', global_shift='AUTO', cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    parsed=filelist[i].split('.')
    merged_name1=parsed[0]+'_WITH_NORMALS.bin'        
    
    # #Compute sensor range and scattering angle
    #             #cc.scattering_angles(cloud,silent=False, verbose=True, cc_exe='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    
    cc.distances_from_sensor_and_scattering_angles(merged_name1, degrees=True, verbose=True,  global_shift='AUTO',  cc_exe='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    parsed=merged_name1.split('.')
    merged_name2=parsed[0]+'_RANGES_ANGLES.bin'      
    
    
    # # create a scan number
    
    # # create station scalar field populated with the station number 
    cc.sf_add_const(merged_name2, ('STATION', (i+1)),  verbose=True, fmt='bin' )
    

    # #invert normals
    
    parsed=merged_name2.split('.')
    merged_name3=parsed[0]+'_SF_ADD_CONST.bin'          
    filelist_final=np.append(filelist_final, merged_name3, axis=None)
    print(i)

print('**********iterations complete')
cc.merge(filelist_final, fmt='bin',  silent=True, debug=False, cc='C:\\Program Files\\CloudCompare\\CloudCompare.exe')
parsed=filelist_final[0].split('.')
merged_name_out=parsed[0]+'_MERGED.bin'
print('*********merge done')
# SUBSAMPLE TO  2 mm
cc.ss(merged_name_out, method='SPATIAL', parameter=0.002,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
parsed=merged_name_out.split('.')
old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
new_name='F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_2mm.bin'
os.rename(old_name, new_name)
print('********subsample 1 done')
cc.ss(new_name, method='SPATIAL', parameter=0.01,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #

parsed=new_name.split('.')
old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
new_name='F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_1cm.bin'
os.rename(old_name, new_name)
#%%
# for item in files:
#     #print(item)
#     item=item.rstrip('\n')
#     parsed=item.split('.')
#     initem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'in.'+parsed[1]
#     outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]

#     filelistin=np.append(filelistin, initem, axis=None)
#     filelistout=np.append(filelistout, outitem, axis=None)

# merge all scans
txtfile='F:/nz_data/05_03_25_faro/faro_list.txt'
filelistin=np.array([])
filelistout=np.array([])
with open(txtfile) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
        parsed=item.split('.')
        initem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'in.'+parsed[1]
        outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]

        filelistin=np.append(filelistin, initem, axis=None)
        filelistout=np.append(filelistout, outitem, axis=None)

cc.merge(filelistout, fmt='bin', silent=False, debug=False, cc='C:\\Program Files\\CloudCompare\\CloudCompare.exe')
parsed=filelistout[0].split('.')
merged_name_out=parsed[0]+'_MERGED.bin'

# SUBSAMPLE TO  2 mm
cc.ss(merged_name_out, method='SPATIAL', parameter=0.002,  fmt='BIN', silent=False, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #


cc.merge(filelistin, fmt='bin', silent=False, debug=False, cc='C:\\Program Files\\CloudCompare\\CloudCompare.exe')
parsed=filelistin[0].split('.')
merged_name_in=parsed[0]+'_MERGED.bin'
cc.ss(merged_name_in, method='SPATIAL', parameter=0.002,  fmt='BIN', silent=False, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
