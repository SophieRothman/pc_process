# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:48:47 2025

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

txtfile='F:/nz_data/faro/2014/faro_2014_list.txt'
newname1='F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_2mm.bin' 
newname2='F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm.bin'
#%%
filelist=np.array([])
filelist_final=np.array([])
with open(txtfile) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
        #parsed=item.split('.')
        item='F:/nz_data/faro/2014/'+item#parsed[0]+ '.'+parsed[1]
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
    if i<5:
        cc.sf_add_const(merged_name2, ('STATION', (i)),  verbose=True, fmt='bin' )
    if i>=5:
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
cc.ss(merged_name_out, method='SPATIAL', silent=True, parameter=0.002,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
parsed=merged_name_out.split('.')
old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
new_name=newname1 ##NEED TO MANUALLY CHANGE FOR NEW SET
os.rename(old_name, new_name)
print('********subsample 1 done')
cc.ss(new_name, method='SPATIAL', parameter=0.01,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #

parsed=new_name.split('.')
old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
new_name=newname2
os.rename(old_name, new_name)

#%% NEED TO FIGURE OUT HOW TO CLASSIFY POINT CLOUD OR OBTAIN CLIFF CORE POINTS

#%% 
txtfile='F:/nz_data/faro/2014/faro_2014_list.txt'
params='F:/nz_data/m3c2_params.txt'

cliff_cp='F:/nz_data/05_03_25_faro/test/Mangarere_2017.cliffcorepoints.bin'
filelist=np.array([])
with open(txtfile) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
        parsed=item.split('.')
        item='F:/nz_data/faro/2014/'+parsed[0]+ '.'+parsed[1]
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
# %%
resultsarray=np.empty((int(numcom), 3))

for i in range(0, int(numcom)): #finally this will read 0, numcom
    results=cc.m3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, fmt='SBF',
             silent=True)
    #results=cc.icpm3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, silent=False, fmt='BIN', verbose=True)
    pc_m3c2=sbf.read(results)
    xyz=pc_m3c2.xyz
    nn=pc_m3c2.sf_names
    sfs=pc_m3c2.sf
    sfnames=pc_m3c2.sf_names
    #avg m3c2distance
    resultsarray[i, 0]=np.nanmean(sfs[:, 6])
    #stdev of m3c2 distance
    resultsarray[i, 1]=np.nanstd(sfs[:, 6])
    #fraction significant change
    resultsarray[i, 2]=len(sfs[sfs[:, 4]==True, 4])/len(sfs[:, 4])
    
# %%
#now average for each scan
results_perscan=np.empty((len(filelist), 4))
for i in range(0, len(filelist)):
    index_col1=np.where(combinations[:, 0]==i)[0]
    index_col2=np.where(combinations[:, 1]==i)[0]
    for j in range(0, len(index_col2)):
        index_col1=np.append(index_col1, index_col2[j], axis=None )
    #for 
    results_perscan[i, 0]=np.nanmean((resultsarray[index_col1, 0]))
    results_perscan[i, 1]=np.nanmean(np.absolute(resultsarray[index_col1, 0]))
    results_perscan[i, 2]=np.nanmean(resultsarray[index_col1, 1])
    results_perscan[i, 3]=np.nanmean(resultsarray[index_col1, 2])

fig, (ax1, ax2, ax3, ax4)=plt.subplots(4, 1, figsize=(12, 14), layout='tight')
ax1.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 0])
ax2.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 1])
ax3.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 2])
ax4.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 3])

ax3.set_xlabel('Scan number')
ax2.set_xlabel('Scan number')
ax1.set_xlabel('Scan number')
ax4.set_xlabel('Scan number')


ax1.set_ylabel('Avg m3c2 distance')
ax2.set_ylabel('Avg absolute m3c2 distance')
ax3.set_ylabel('Avg m3c2 stdev')
ax4.set_ylabel('Avg fraction significant change')

plt.show()
parsed=txtfile.split('_l')
figname=parsed[0]+ '_regresults.png'
fig.savefig(figname)
arrayname=parsed[0]+'_resultsarray'
np.save(arrayname, resultsarray)
