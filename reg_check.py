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
ax1.bar(np.arange(0, 13), results_perscan[:, 0])
ax2.bar(np.arange(0, 13), results_perscan[:, 1])
ax3.bar(np.arange(0, 13), results_perscan[:, 2])
ax4.bar(np.arange(0, 13), results_perscan[:, 3])

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

#resultsarray=np.load('F:/nz_data/05_03_25_faro/resultsarray_regcheck_faro2017data.npy')
