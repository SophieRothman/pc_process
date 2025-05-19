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
    
#%% trying to 3dmasc across different datasets I SHOULD REALLY DO THIS ON THE 1 CM FILE BUT FOR NOW THIS IS SUFFICIIENT
pc_full=['F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_1cm.bin', 'F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin', 'F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_1cm.bin', 'F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_1cm_clean10.bin']
pc_cp=['F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_20cm.bin', 'F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin', 'F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_20cm.bin', 'F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_20cm_clean10.bin']
ws_pcs=['F:/nz_data/ws_pcs/Mangarere_2014000_ws.bin', 'F:/nz_data/ws_pcs/Mangarere_2015000_wsv2.bin', 'F:/nz_data/ws_pcs/Mangarere_2017039_ws.bin', 'F:/nz_data/ws_pcs/Mangarere_ZF_20230222_ws.bin']
classifier='F:/nz_data/05_02_25/5class_4scale_dz2_93v3_opt.txt'
#classifier='F:/nz_data/05_02_25/python_classifier_opt.yaml'
output_3dmasc=[]

# ef q3dmasc(clouds, training_file, only_features=False, keep_attributes=False,
#             silent=True, verbose=False, global_shift='AUTO', cc_exe=cc_exe, fmt='sbf'):
for i in range(1, 4): #len(pc_full)):
    clouds=(ws_pcs[i], pc_full[i], pc_cp[i])
    out = cc.q3dmasc(clouds, classifier, keep_attributes=True, verbose=True, silent=False)
    output_3dmasc.append(out)
    print(i)
    print('************* is done')
output_3dmasc=np.array(output_3dmasc)
# output_3dmasc=['F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_20cm_CLASSIFIED.sbf','F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_20cm_CLASSIFIED.sbf',
#        'F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_20cm_CLASSIFIED.sbf',
#        'F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_20cm_clean10_CLASSIFIED.sbf']
np.save('F:/nz_data/classified_file_names', output_3dmasc)
#%% Remove everythin gthat is vegetation from these files    NEED TO THINK OF HOW TO KEEP NORMALS ON IT
file='F:/nz_data/classified_file_names.npy'
names=np.load(file)
written_files=[]
for i in range(0, len(names)):
    fname=names[i]
    dfname=cc.density(fname, .5, 'VOLUME', verbose=True)
    classified=sbf.read(fname)
    dclassified=sbf.read("F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_20cm_clean10_CLASSIFIED_part.sbf")#dfname)"
    xyz=np.array(classified.xyz)
    nn=np.array(dclassified.sf_names)
    sfs=np.array(dclassified.sf)
    #allsfs=np.empty((len(sfs[:, 0]), len(sfs[0, :]+1)))
    #allsfs[:, 0:31]=xyzc2
    num_sf=len(sfs[1, :])
    iclass=np.where(nn=='Classification')
    mask=sfs[:, iclass]!=3
    np.count_nonzero(mask)/len(mask)
    xyzc=np.copy(xyz[mask[:, 0, 0], :])
    sfsc=np.copy(sfs[mask[:, 0, 0], :])
    iclass2=np.where(nn=='"DZ_CTX_9@kNN=2"')
    mask2=sfsc[:, iclass2]>(-.1)
    xyzc2=np.copy(xyzc[mask2[:, 0, 0], :])
    sfsc2=np.copy(sfsc[mask2[:, 0, 0], :])
    parsed=fname.split('_CLASSIFIED')
    newname=parsed[0]+'_NONVEG.sbf'
    # mask3=sfsc2[:, -2]>(0.6)
    # xyzc3=np.copy(xyzc2[mask3, :])
    # sfsc3=np.copy(sfsc2[mask3, :])
    
    written=sbf.write(newname,  xyzc2, sfsc2)
    written_files.append(newname)

np.save('F:/nz_data/vegonly_file_names', written_files)


#%% run m3c2 across years
#REMEMBER TO CHANGE INTERYEAR SEARCH SCALE TO 0.2 AND CHANGE THE NAME WHEN YOU MOVE TO A 1 CM scale
clouds=np.load('F:/nz_data/vegonly_file_names.npy')
cp='F:/nz_data/ws_pcs/Mangarere_2014000_20cm5mnormals.bin'
params='F:/nz_data/m3c2_params_interyear.txt'

years=[2014, 2015, 2017, 2023]
filenames=[]
for i in range(0, len(clouds)-1):
    resultsm3c2=cc.m3c2(clouds[i], clouds[i+1], params, core=cp, fmt='BIN',
              debug=True, cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    parsed=clouds[0].split('_FARO')
    parsed=parsed[0]+ '20cm_M3C2'+ '_'+str(years[i])+'_'+ str(years[i+1])+'.bin'
    
    os.rename(resultsm3c2, parsed)
    filenames.append(parsed)
np.save('F:/nz_data/m3c2_file_names', filenames)    
#%% Now combine all into one file
clouds=np.load('F:/nz_data/m3c2_file_names.npy')
tosbf=cc.to_sbf(clouds[0], silent=True, debug=False)
cld=sbf.read(tosbf)
xyz2=np.copy(np.array(cld.xyz))
sfs2=np.empty((len(xyz2[:, 0]), len(clouds)))*np.nan
#sfs2[:, 0]=np.copy(cld.sf[:, -1])
for i in range(0, len(clouds)):
    tosbf=cc.to_sbf(clouds[i], silent=True, debug=False)
    cld=sbf.read(tosbf)

    sfs2[:, i]=np.copy(cld.sf[:, -1])


mask=~np.isnan(sfs2[:, 0]) | ~np.isnan(sfs2[:, 1]) | ~np.isnan(sfs2[:, 2]) 
#np.count_nonzero(mask)
xyz3=xyz2[mask, :]
sfs3=sfs2[mask, :]
parse=clouds[0].split('_2014')    
fn=parse[0]+'_20cm_combined.sbf'
written=sbf.write(fn, xyz3, sfs3)
written=np.array(written)
np.save('F:/nz_data/combined_file_name', written)    

#%% compute features on each years data using also the 2014 core points
cp='F:/nz_data/ws_pcs/Mangarere_2014000_20cm5mnormals.bin'

#fname=np.load('F:/nz_data/combined_file_name.npy')   
classifier='F:/nz_data/05_02_25/features_notfor3dmasc.txt'
clouds=(ws_pcs[0], pc_full[0], cp)   
out = cc.q3dmasc(clouds, classifier, only_features=True, keep_attributes=True, verbose=True)
cl_ft=sbf.read(out)
names=cl_ft.sf_names
sf_dip1=np.empty((len(cl_ft.sf[:, 0]), len(pc_full)));
idz=np.where()
idip1=np.where()
sf_dz=np.copy(cl_ft.sf[:, idz])

xyz_feat=np.copy(cl_ft.xyz);
sf_dip1[:, 0]=np.copy(cl_ft.sf[:, idip1])

for i in range (1, len(pc_full)):
    clouds=(ws_pcs[0], pc_full[0], 'F:/nz_data/faro/2014/Mangarere_M3C2_combined.sbf')   
    out = cc.q3dmasc(clouds, classifier, only_features=True, keep_attributes=True, verbose=True)
    #'F:/nz_data/faro/2014/Mangarere_M3C2_combined_WITH_FEATURES.sbf'
m3_wft=sbf.read('F:/nz_data/m3c2results/Mangarere_M3C2_comb_ft_cl.sbf')
    
    
sfs5=m3_wft.sf   
    
    
    
    
    