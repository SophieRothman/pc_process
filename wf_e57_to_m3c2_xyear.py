# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:48:47 2025

@author: Sophie
"""
#this is the version that only does one scanper pair
import numpy as np

import lidar_platform
from lidar_platform import cc, las, sbf
import lidar_platform.classification  as classification
import lidar_platform.classification.feature_selection
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
#%%
for j in range(1, 6):
    if j==0:
        #txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
        txtfile='F:/nz_data/Registration_V2/2009_02/leica_2009_02_list.txt'
        newname1='F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_RegV2_ALL_STATIONS_2mm.bin'
        newname2='F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_RegV2_ALL_STATIONS_1cm.bin'
        newname3='F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_RegV2_ALL_STATIONS_20cm.bin'
        #newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
        #newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
        #newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'
        
        #prefix='F:/nz_data/faro/2014/'
        prefix='F:/nz_data/Registration_V2/2009_02/'
        
    if j==1:
        #txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
        txtfile='F:/nz_data/Registration_V2/2010_02/leica_2010_02_list.txt'
        newname1='F:/nz_data/Registration_V2/2010_02/Mangarere_20100203_RegV2_ALL_STATIONS_2mm.bin'
        newname2='F:/nz_data/Registration_V2/2010_02/Mangarere_20100203_RegV2_ALL_STATIONS_1cm.bin'
        newname3='F:/nz_data/Registration_V2/2010_02/Mangarere_20100203_RegV2_ALL_STATIONS_20cm.bin'
        #newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
        #newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
        #newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'
        
        #prefix='F:/nz_data/faro/2014/'
        prefix='F:/nz_data/Registration_V2/2010_02/'
        
    if j==2:
        #txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
        txtfile='F:/nz_data/Registration_V2/2010_12/leica_2010_12_list.txt'
        newname1='F:/nz_data/Registration_V2/2010_12/Mangarere_20101218_RegV2_ALL_STATIONS_2mm.bin'
        newname2='F:/nz_data/Registration_V2/2010_12/Mangarere_20101218_RegV2_ALL_STATIONS_1cm.bin'
        newname3='F:/nz_data/Registration_V2/2010_12/Mangarere_20101218_RegV2_ALL_STATIONS_20cm.bin'
        #newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
        #newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
        #newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'
        
        #prefix='F:/nz_data/faro/2014/'
        prefix='F:/nz_data/Registration_V2/2010_12/'

    if j==3:
        #txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
        txtfile='F:/nz_data/Registration_V2/2011_02/leica_2011_02_list.txt'
        newname1='F:/nz_data/Registration_V2/2011_02/Mangarere_20110224_RegV2_ALL_STATIONS_2mm.bin'
        newname2='F:/nz_data/Registration_V2/2011_02/Mangarere_20110224_RegV2_ALL_STATIONS_1cm.bin'
        newname3='F:/nz_data/Registration_V2/2011_02/Mangarere_20110224_RegV2_ALL_STATIONS_20cm.bin'
        #newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
        #newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
        #newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'
        
        #prefix='F:/nz_data/faro/2014/'
        prefix='F:/nz_data/Registration_V2/2011_02/'

    if j==4:
        #txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
        txtfile='F:/nz_data/Registration_V2/2011_12/leica_2011_12_list.txt'
        newname1='F:/nz_data/Registration_V2/2011_12/Mangarere_20111201_RegV2_ALL_STATIONS_2mm.bin'
        newname2='F:/nz_data/Registration_V2/2011_12/Mangarere_20111201_RegV2_ALL_STATIONS_1cm.bin'
        newname3='F:/nz_data/Registration_V2/2011_12/Mangarere_20111201_RegV2_ALL_STATIONS_20cm.bin'
        #newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
        #newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
        #newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'
        
        #prefix='F:/nz_data/faro/2014/'
        prefix='F:/nz_data/Registration_V2/2011_12/'
        
    if j==5:
        #txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
        txtfile='F:/nz_data/Registration_V2/2012_02/leica_2012_02_list.txt'
        newname1='F:/nz_data/Registration_V2/2012_02/Mangarere_20120227_RegV2_ALL_STATIONS_2mm.bin'
        newname2='F:/nz_data/Registration_V2/2012_02/Mangarere_20120227_RegV2_ALL_STATIONS_1cm.bin'
        newname3='F:/nz_data/Registration_V2/2012_02/Mangarere_20120227_RegV2_ALL_STATIONS_20cm.bin'
        #newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
        #newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
        #newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'
        
        #prefix='F:/nz_data/faro/2014/'
        prefix='F:/nz_data/Registration_V2/2012_02/'

    #REDONE FOR FUSED TO REMOVE SCALAR FIELD AND SCATTERIN GANGLE
    filelist=np.array([])
    filelist_final=np.array([])
    with open(txtfile) as files:
        for item in files:
            #print(item)
            item=item.rstrip('\n')
            #parsed=item.split('.')
            item=prefix+item#parsed[0]+ '.'+parsed[1]
            #outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]
    
            filelist=np.append(filelist, item, axis=None)
    
    for i in range(0, len(filelist)):
    
        #name='Mangarere_FARO_20170314_01.e57'
        
        #Compute normal oriented towards the sensor 10 cm? 2- cm? not sure yet its a compri
        nam1=cc.octree_normals(filelist[i], 0.1,  with_grids=False,  angle=1, orient='WITH_SENSOR',  verbose=True, fmt='bin', global_shift='AUTO', cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe', all_at_once=True)
        parsed=filelist[i].split('.')
        merged_name1=parsed[0]+'_WITH_NORMALS.bin'   
        os.rename(nam1, merged_name1)     
        
        # #Compute sensor range and scattering angle
        #             #cc.scattering_angles(cloud,silent=False, verbose=True, cc_exe='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
        
        # commented for fused cc.distances_from_sensor_and_scattering_angles(merged_name1, degrees=True, verbose=True,  global_shift='AUTO',  cc_exe='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe') # need to change back to merged_name1
        #parsed=merged_name1.split('.')
        #merged_name2=parsed[0]+'_RANGES_ANGLES.bin'      
        
        
        # # create a scan number
        
        # # create station scalar field populated with the station number 
        #if i<5:
       # commented for fused cc.sf_add_const(merged_name2, ('STATION', (i)),  verbose=True, fmt='bin' )
        # if i>=5:
        #    cc.sf_add_const(merged_name2, ('STATION', (i+1)),  verbose=True, fmt='bin' )
        
    
        # #invert normals
        
        #parsed=merged_name2.split('.')
        #merged_name3=parsed[0]+'_SF_ADD_CONST.bin'          
        filelist_final=np.append(filelist_final, nam1, axis=None)
        print(i)
    
    filelist_final=np.array(filelist_final)
    np.save('F:/nz_data/fname_range_angles_norm_sf', filelist_final)
    
    print('**********i iterations complete')
    fmerged=cc.merge(filelist_final, fmt='bin',  silent=True, debug=True, cc='C:\\Program Files\\CloudCompare\\CloudCompare.exe')
    parsed=fmerged.split('.')
    if j>1:
        merged_name_out=parsed[0]+'_0.bin'

    if j<2:
        merged_name_out=parsed[0]+'.bin'
    print('*********merge done')
    # SUBSAMPLE TO  2 mm
    cc.ss(merged_name_out, method='SPATIAL', silent=True, parameter=0.002,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
    parsed=merged_name_out.split('.')
    old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
    os.rename(old_name, newname1)
    print('********subsample 1 done')
    cc.ss(newname1, method='SPATIAL', parameter=0.01,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
    
    parsed=newname1.split('.')
    old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
    os.rename(old_name, newname2)
    
    cc.ss(newname2, method='SPATIAL', parameter=0.2,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
    #cc.ss(g, method='SPATIAL', parameter=0.2,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
    
    #g='F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_1cm.bin'
    parsed=newname2.split('.')
    old_name=parsed[0]+'_SPATIAL_SUBSAMPLED.bin'
    os.rename(old_name, newname3)
    print('*****************j iteration finished ')
    print(j)

# trying to 3dmasc across different datasets I SHOULD REALLY DO THIS ON THE 1 CM FILE BUT FOR NOW THIS IS SUFFICIIENT
pc_full=['F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_RegV2_ALL_STATIONS_1cm.bin', 
         'F:/nz_data/Registration_V2/2010_02/Mangarere_20100203_RegV2_ALL_STATIONS_1cm.bin', 
         'F:/nz_data/Registration_V2/2010_12/Mangarere_20101218_RegV2_ALL_STATIONS_1cm.bin', 
         'F:/nz_data/Registration_V2/2011_02/Mangarere_20110224_RegV2_ALL_STATIONS_1cm.bin', 
         'F:/nz_data/Registration_V2/2011_12/Mangarere_20111201_RegV2_ALL_STATIONS_1cm.bin', 
         'F:/nz_data/Registration_V2/2012_02/Mangarere_20120227_RegV2_ALL_STATIONS_1cm.bin', ]

pc_cp=['F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_RegV2_ALL_STATIONS_20cm.bin', 
         'F:/nz_data/Registration_V2/2010_02/Mangarere_20100203_RegV2_ALL_STATIONS_20cm.bin', 
         'F:/nz_data/Registration_V2/2010_12/Mangarere_20101218_RegV2_ALL_STATIONS_20cm.bin', 
         'F:/nz_data/Registration_V2/2011_02/Mangarere_20110224_RegV2_ALL_STATIONS_20cm.bin', 
         'F:/nz_data/Registration_V2/2011_12/Mangarere_20111201_RegV2_ALL_STATIONS_20cm.bin', 
         'F:/nz_data/Registration_V2/2012_02/Mangarere_20120227_RegV2_ALL_STATIONS_20cm.bin', ]

ws_pcs=['F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_ws.bin',
        'F:/nz_data/Registration_V2/2010_02/Mangarere_20100203_ws.bin',
        'F:/nz_data/Registration_V2/2010_12/Mangarere_20101218_ws.bin', 
        'F:/nz_data/Registration_V2/2011_02/Mangarere_20110224_ws.bin',
        'F:/nz_data/Registration_V2/2011_12/Mangarere_20111201_ws.bin',        
        'F:/nz_data/Registration_V2/2012_02/Mangarere_20120227_ws.bin']
classifier='F:/nz_data/05_02_25/newclass_85vnodz2.txt'
#classifier='F:/nz_data/05_02_25/python_classifier_opt.yaml'
output_3dmasc=[]
#%%
# ef q3dmasc(clouds, training_file, only_features=False, keep_attributes=False,
#             silent=True, verbose=False, global_shift='AUTO', cc_exe=cc_exe, fmt='sbf'):
for i in range(0, len(pc_full)):
    clouds=(pc_full[i], pc_cp[i])
    out = cc.q3dmasc(clouds, classifier, keep_attributes=True, verbose=True, silent=True, fmt='bin')
    output_3dmasc=np.array(output_3dmasc)
    output_3dmasc=np.append(output_3dmasc, out)
    print(i)
    print('************* is done')
# output_3dmasc=['F:/nz_data/faro/2014/Mangarere_FARO_20140220_ALL_STATIONS_20cm_CLASSIFIED.sbf','F:/nz_data/faro/2015/Mangarere_FARO_20150324_ALL_STATIONS_20cm_CLASSIFIED.sbf',
#        'F:/nz_data/05_03_25_faro/Mangarere_FARO_20170314_ALL_STATIONS_20cm_CLASSIFIED.sbf',
#        'F:/nz_data/zf/Mangarere_ZF_20230222_ALL_STATIONS_20cm_clean10_CLASSIFIED.sbf']
np.save('F:/nz_data/classified_file_names', output_3dmasc)
#%% change normals to 5m onclassified 20cm data
norm5_class=[]
norm5_class=np.array(norm5_class)
cfiles=np.load('F:/nz_data/classified_file_names.npy')
for i in range(0, len(cfiles)):
    out=cc.octree_normals(cfiles[i], 5, with_grids=False, angle=1, orient='PREVIOUS', fmt='bin', verbose=True)
    norm5_class=np.append(norm5_class, out)
np.save('F:/nz_data/classified_file_names_5mnorm', norm5_class)

#HERE I MANUALLY TOOK THE EARLIEST TIME STEP CLASSIFIED AND RENORMALIZED FILE AND REMOVED THE VEG AND WATER IN THE GUI TO PRESERVE THE NORMALS
#%% RUNNING m3c2 while moving the core points back each year
og_cp='F:/nz_data/Registration_V2/2009_02/Mangarere_20090229_RegV2_class_5m_deb_cliff.bin'
params_ogyear='F:/nz_data/m3c2_params_ogyear.txt'
params_nextyear='F:/nz_data/m3c2_params_nextyear.txt'

cfiles=np.load('F:/nz_data/classified_file_names_5mnorm.npy')
m3c2_files=[]
m3c2_files=np.array(m3c2_files)
new_cp=[]
new_cp=np.array(new_cp)
for i in range(0, len(pc_full)-1):
    if i==0 :
        corep=og_cp
    if i>0:
        corep=new_cp[i-1]
    out=cc.m3c2(pc_full[i], pc_full[i+1], params_ogyear, core=corep, fmt='BIN',
              verbose=True, cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    parsed=out.split('M3C2.')
    parsed=parsed[0]+'M3C2_ogyear.bin'
    if os.path.isfile(parsed):
        os.remove(parsed)
    os.rename(out, parsed)

    m3c2_files=np.append(m3c2_files, parsed)
    
    #create new core points by projecting previous core points on to the newer time slice
    out2=cc.m3c2(pc_full[i], pc_full[i+1], params_nextyear, core=corep, fmt='BIN',
              verbose=True, cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    #recalculated the normals at 5 m using previous orientation
    clouds=(pc_full[i+1], out2)
    out3 = cc.q3dmasc(clouds, classifier, keep_attributes=True, verbose=True, silent=True, fmt='bin')

    out4=cc.octree_normals(out3, 5, with_grids=False, angle=1, orient='PREVIOUS', fmt='bin', verbose=True)

    parsed=out4.split('M3C2')
    parsed=parsed[0]+'newcorep.bin'
    if os.path.isfile(parsed):
        os.remove(parsed)
    os.rename(out4, parsed)

    new_cp=np.append(new_cp, parsed)
    
np.save('F:/nz_data/m3c2_resultnames', m3c2_files)
np.save('F:/nz_data/m3c2_newcpnames', new_cp)
#%% analyze results
#box1
xs=[]
ys=[]
zs=[]
dip=[]
class_from_2009=[]
erosion=[]
xs=np.array(xs)
ys=np.array(ys)
zs=np.array(zs)
dip=np.array(dip)
class_from_2009=np.array(class_from_2009)
erosion=np.array(erosion)
m3files=np.load('F:/nz_data/m3c2_resultnames.npy')
for i in range(0, len(m3files)):
    parsed=m3files[i].split('.')
    parsed=parsed[0]+'_in.bin'
    sbf_clas=cc.to_sbf(parsed) #change this input to change which files are analyzed
    cld=sbf.read(sbf_clas)
    cld_sf=cld.sf
    cld_n=cld.sf_names
    cld_xyz=cld.xyz    
    im3c2=cld_n.index('M3C2 distance')
    im3c2_unc=cld_n.index('distance uncertainty')
    im3c2_sig=cld_n.index('significant change')

    if i>0:
        idip6=cld_n.index('Dip_PC1@0.4',) #MADE A MISTAKE HERE
    iclass=cld_n.index('Classification')

    if i==0:
        xs=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan
        ys=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan
        zs=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan
        dip=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan
        class_from_2009=np.zeros((len(cld_sf[:, 0]), ))*np.nan
        erosion=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan
        erosion_unc=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan
        erosion_sig=np.zeros((len(cld_sf[:, 0]), len(m3files)))*np.nan


        class_from_2009[:]=cld_sf[:, iclass]
        idip6=cld_n.index('Dip_PC1@0.6',) #WHOOPS I MESSED UP THE CLASSIDIER FOR THE FIRST ROUND IS NOT THE SAME AS FOR ALL THE OTHERS NEED TO FIX THIS
    print(i)
    xs[:, i]=cld_xyz[:, 0]
    ys[:, i]=cld_xyz[:, 1]
    zs[:, i]=cld_xyz[:, 2]
    dip[:, i]=cld_sf[:, idip6]
    erosion[:, i]=cld_sf[:, im3c2]  
    erosion_unc[:, i]=cld_sf[:, im3c2_unc]  
    erosion_sig[:, i]=cld_sf[:, im3c2_sig]  

np.savez('F:/nz_data/m3c2_resultarrays.npz', xs=xs, ys=ys, zs=zs, dip=dip, class_from_2009=class_from_2009, erosion=erosion)
#%% if we want to restrict to only a box
x1=777
x2=990
y1=625
y2=762
for i in range(0, len(m3files)):
    mask=np.logical_and(np.logical_and(xs[:, i]>x1, xs[:, i]<x2) , np.logical_and(ys[:, i]>y1, ys[:, i]<y2))
    erosion[mask==False, :]=np.nan
#%% now produce figures of these results
#first just histograms of erosion rates
legends=['2009_02-2010_02', '2010_02-2010_12', '2010_12-2011_02', '2011_02-2011_12', '2011_12-2012_02']
from matplotlib import cm
fig, (ax1) = plt.subplots()
viridis = mpl.colormaps['viridis'].resampled(len(m3files)+1)#cm.get_cmap('bone', len(m3files)+1)
for i in range(0, len(m3files)):
    numbins=400
    sigma=1
    mu=0 
    n, bins, _=ax1.hist(erosion[erosion_sig[:, i]==1, i], numbins, histtype='step', color=viridis(i/(len(m3files)+1)), label=legends[i])
ax1.legend()
ax1.set_xlim((-1, 1))
ax1.set_xlabel('M3C2 Distance')
ax1.set_ylabel('Frequency')    

fig, (ax1) = plt.subplots(figsize=(10, 6))
sig_erosion=np.copy((erosion))
for i in range(0, len(m3files)):
    sig_erosion[erosion_sig[:, i]==0, i]=np.nan
averages=np.nanmean(sig_erosion, axis=0)
ax1.bar(legends, averages)
ax1.set_ylabel('Average Erosion (Cliff and Debris)')
#%% dip v erosion rate
numbins=20
dip_avg=np.empty((20, len(m3files)))
erosion_avg=np.empty((20, len(m3files)))

dip_bins=np.linspace(0, 90, numbins+1)
for j in range(0, len(m3files)):
    for i in range(0, numbins):
        mask=np.logical_and((np.abs(dip[:, j]>dip_bins[i])) , (np.abs(dip[:, j]<dip_bins[i+1])))
        dip_avg[i, j]=np.nanmean(dip[mask, j])
        erosion_avg[i, j]=np.nanmean(sig_erosion[mask, j])
        
fig, (ax1) = plt.subplots( figsize=(10, 6))
for i in range(0, len(m3files)):
    ax1.plot(dip_avg[:, i], erosion_avg[:, i], color=viridis(i/(len(m3files)+1)),  label=legends[i])
    
ax1.set_xlabel('Dip (0.4 m)')
ax1.set_ylabel('M3C2 avg')
ax1.legend()
ax1.axhline(0)

#ax2.scatter(np.abs(dip[class_from_2009==24]), sig_erosion[class_from_2009==24])

    #%%
#height vs erosion rate
numbins=20
e_avg=np.empty((numbins, len(m3files)))
z_avg=np.empty((numbins, len(m3files)))

elev_bins=np.linspace(280, 400, numbins+1)
for j in range(0, len(m3files)):
    for i in range(0, numbins):
        mask=np.logical_and((zs[:, j]>elev_bins[i]) , (zs[:, j]<elev_bins[i+1]))
        e_avg[i, j]=np.nanmean(sig_erosion[mask, j])
        z_avg[i, j]=np.nanmean(zs[mask, j])

fig, (ax1, ax2, ax3, ax4, ax5)=plt.subplots(5, 1, figsize=(10, 20), layout='tight')
ax1.plot(z_avg[:, 0], e_avg[:, 0], '-o')
ax1.set_xlim([280, 420])
#ax1.set_ylim([-1, 0.4])
ax1.set_title(legends[0])
ax1.axhline(0)

ax2.plot(z_avg[:, 1], e_avg[:, 1], '-o')
ax2.set_xlim([280, 420])
ax2.set_title(legends[1])
#ax2.set_ylim([-1, 0.4])
ax2.axhline(0)


ax3.plot(z_avg[:, 2], e_avg[:, 2], '-o')
ax3.set_xlim([280, 420])
ax3.set_title(legends[2])
#ax3.set_ylim([-1, 0.4])
ax3.axhline(0)


ax4.plot(z_avg[:, 3], e_avg[:, 3], '-o')
ax4.set_xlim([280, 420])
ax4.set_title(legends[3])
#ax4.set_ylim([-1, 0.4])
ax4.axhline(0)

ax5.plot(z_avg[:, 4], e_avg[:, 4], '-o')
ax5.set_xlim([280, 420])
ax5.set_title(legends[4])
ax5.axhline(0)
#ax5.set_ylim([-1, 0.4])

#%% Select only the soil/cliff parts,
cfiles=np.load('F:/nz_data/classified_file_names.npy')
nonveg_files=np.array([])
xyz=[]
for i in range(0, len(cfiles)):
    if i==0 :
        
        sbf_clas=cc.to_sbf(cfiles[i])
        cld=sbf.read(sbf_clas)
    else: 
        cld=sbf.read(cfiles[i])
    cld_sf=cld.sf
    cld_n=cld.sf_names
    cld_xyz=cld.xyz
    
    #separate the part of the cloud which is cliff and debris with over 0.6 certainty
    iclass=-1#np.where(cld_n=='Classification')
    iclassc=-2#np.where(cld_n=='Classification_confidence')
    mask_cl_deb=np.logical_and(np.logical_or((cld_sf[:, iclass]==2) , (cld_sf[:, iclass]==24)) , (cld_sf[:, iclassc]>0.6))
    cld_sf=cld_sf[mask_cl_deb, :]
    cld_xyz=cld_xyz[mask_cl_deb, :]
    
    #make a file which is only cliff and debris
    parsed=cfiles[i].split('_CLASSIFIED')
    parsed=parsed[0]+'clif_deb.sbf'
    sbf.write(parsed, cld_xyz, cld_sf)
    #save said file
    nonveg_files=np.append(nonveg_files, parsed, axis=None)
    
    #Run m3c2 using the 
    
    
np.save('F:/nz_data/nonveg_file_names', nonveg_files)

    #grab values from this including dip, curvature, xyz, 
#%% GETTING ALL THE COMBINATIONS OF FILES
txtfile='F:/nz_data/faro/2014/faro_2014_list.txt'
params='F:/nz_data/m3c2_paramsv7.txt'
filearray='F:/nz_data/fname_range_angles.npy'
cliff_cp='F:/nz_data/ws_pcs/Mangarere_2014000_20cm_5mn_clasclif_uc75.bin'
filelist=np.load(filearray)#np.array([])
# with open(txtfile) as files:
#     for item in files:
#         #print(item)
#         item=item.rstrip('\n')
#         parsed=item.split('.')
#         item='F:/nz_data/faro/2014/'+parsed[0]+ '.'+parsed[1]
#         #outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]

#         filelist=np.append(filelist, item, axis=None)
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
    
#%% get normals as sf - NOT NEEDED RElly
cpc='F:/nz_data/ws_pcs/Mangarere_2014000_20cm_5mn_clasclif_uc75_normsf.sbf'
cpc_sbf=sbf.read(cpc)
sfs_av=cpc_sbf.sf_names
ft=cpc_sbf.sf
nxyz=ft[:, len(ft[0, :])-3:len(ft[0, :])]

# %%
# cpc_sbf=pc_cc=cc.to_sbf(cliff_cp,  silent=True, debug=False)
# cpc_sbf=sbf.read(cpc_sbf)
# coords=cpc_sbf.xyz

results=cc.m3c2(filelist[int(combinations[0, 0])], filelist[int(combinations[0, 1])], params, core=cliff_cp, fmt='SBF',
             silent=True, debug=False)
pc_m3c2=sbf.read(results)

xyz=pc_m3c2.xyz
len_cloud=len(xyz[:, 1])
nn=pc_m3c2.sf_names
sfs=pc_m3c2.sf
resultsarray=np.empty((int(numcom), 3))
resultsnames=[]
results_m3c2=np.empty((len_cloud, int(numcom)))
results_dist_unc=np.empty((len_cloud, int(numcom)))


for i in range(0, int(numcom)): #finally this will read 0, numcom
    file1=filelist[int(combinations[i, 0])]
    
    results=cc.m3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, fmt='SBF',
             silent=True, debug=False)
    #results=cc.icpm3c2(filelist[int(combinations[i, 0])], filelist[int(combinations[i, 1])], params, core=cliff_cp, silent=False, fmt='BIN', verbose=True)
    pc_m3c2=sbf.read(results)
    # parsed=results.split('.')
    # file2=filelist[int(combinations[i, 1])].split('.')[0]
    # file2=file2.split('_')[-1]
    # newname3=parsed[0]+'_'+file2+'.sbf'
    # newname4=parsed[0]+'_'+file2+'.sbf.data'
    # oldname1=results+'.data'
    # os.rename(results, newname3)
    # os.rename(oldname1, newname4)
    # resultsnames.append(newname3)
    #pc_m3c2=sbf.read(resultsnames[i])
    
    
    #xyz=pc_m3c2.xyz
    nn=pc_m3c2.sf_names
    nn
    sfs=pc_m3c2.sf
    results_m3c2[:, i]=sfs[:, 6]
    results_dist_unc[:, i]=sfs[:, 5]

    mask=~np.isnan(sfs[:, 5])
    sfnames=pc_m3c2.sf_names
    #avg m3c2distance
    #m3c2i=np.where(nn='M3C2 distance')
    resultsarray[i, 0]=np.nanmean(sfs[mask, 6])
    #stdev of m3c2 distance
    #isd=np.where(nn='significant change')
    resultsarray[i, 1]=np.nanstd(sfs[mask, 6])
    #fraction overlap
    resultsarray[i, 2]=len(sfs[mask, 5])/len(sfs[:, 6])
    
# parsed=txtfile.split('_l')
# arrayname=parsed[0]+'_m3c2_array'    
# np.save( arrayname, results_m3c2)   
# arrayname=parsed[0]+'_dist_unc_array'    
# np.save( arrayname, results_dist_unc) 
# %%

#now average for each scan
m3c2dict={}
dist_uncdict={}
results_perscan=np.empty((len(filelist), 4))
weights_perscan=np.empty((len(filelist), len(filelist)))
a_scan=[]
a_weight=[]
a_names=[]
vectors=np.empty((len(filelist), 3))
vc_poin=np.empty((len_cloud, 3))
vectors[:]=np.nan
vc_poin[:]=np.nan
mag_vec=np.empty((len(filelist)))

for i in range(0, len(filelist)):
    results=np.empty((len_cloud, len(filelist)+5))
    results[:]=np.nan
    results[:, 0:3]=xyz
    length=np.empty((len(filelist)))
    length[:]=np.nan
    index_col1=np.where(combinations[:, 0]==i)[0]
    index_col2=np.where(combinations[:, 1]==i)[0]
    
    for j in range(0, len(index_col1)):
        iscan=int(combinations[index_col1[j], 1])+3 #finds the scan number of the comparison scan to put it in the proper order
        iscan
        results[ :, iscan]=results_m3c2[:,index_col1[j] ]
        #length[j]=np.count_nonzero(np.isnan(results[ :, j]))
        length[iscan-3]=len(results[~np.isnan(results[ :, iscan]), j])/len_cloud

    for k in range(0, len(index_col2)):
        iscan=int(combinations[index_col2[k], 0])+3

#        index_col1=np.append(index_col1, index_col2[j], axis=None )
        results[ :, iscan]=-results_m3c2[:,index_col2[k] ] #MAke sure to make those from the second column negative because the position of the cloud hs changed
        length[iscan-3]=len(results[~np.isnan(results[ :, iscan]),iscan])/len_cloud

    #for 
    #results_perscan[i, 3]=np.nanmean(results)
    for l in range(0, len_cloud):
        if np.count_nonzero(~np.isnan(results[l, 3:(len(filelist)+2)]))>0:
            results[l, -1]=np.nanmean(results[l, 3:(len(filelist)+2)])
            results[l, -2]=np.count_nonzero(~np.isnan(results[l, 3:(len(filelist)+2)]))+1
            vc_poin[l, :]=nxyz[l, :]*results[l, -1]
            
    weights_perscan[:, i]=np.copy(length)
    results_perscan[i, 0]=np.nanmean(np.copy(results[:, 3:(len(filelist)+2)]))
    results_perscan[i, 1]=np.nanmean(np.absolute(np.copy(results[:, 3:(len(filelist)+2)])))

    #results_perscan[i, 2]=np.nanmean(np.absolute(resultsarray[index_col1, 0]*weights))
    results_perscan[i, 2]=np.nanstd(np.copy(results[:, 3:(len(filelist)+2)]))
    
    a_scan.append(np.copy(results))
    #a_weight.append(np.copy(weights_perscan))
    if i<5:
        arrayname=parsed[0]+'_scan_' + str(i)+'_results.sbf'
    if i>4:    
        arrayname=parsed[0]+'_scan_' + str(i+1)+'_results.sbf'
    written=sbf.write(arrayname, results[:, 0:3], results[:, 3:])

    arrayname=arrayname+'.txt'
    np.savetxt(arrayname, np.copy(results)) #, header='x y z 1 2 3 4 6 7 8 9 10 11 12 #scans avgm3c2'
    
    a_names.append(np.copy(arrayname))
    vectors[i, 0]=np.nansum(np.copy(vc_poin[:, 0]))
    vectors[i, 1]=np.nansum(np.copy(vc_poin[:, 1]))
    vectors[i, 2]=np.nansum(np.copy(vc_poin[:, 2]))
    vectors[i, :]=vectors[i, :]/np.count_nonzero(~np.isnan(results[:, -1]))
    mag_vec[i]=np.sqrt(np.square(vectors[i, 0]) + np.square(vectors[i, 1]) + np.square(vectors[i, 2]))


# %%
#import matplotlib
#r=cc.to_sbf(a_names[-1])
fig, (ax1, ax2, ax3)=plt.subplots(3, 1, figsize=(12, 10), layout='tight')
ax1.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 0])
ax2.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 1])
ax3.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 2])
#ax4.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], results_perscan[:, 3])

ax3.set_xlabel('Scan number')
ax2.set_xlabel('Scan number')
ax1.set_xlabel('Scan number')
#ax4.set_xlabel('Scan number')


ax1.set_ylabel('Avg m3c2 distance')
ax2.set_ylabel('Avg absolute m3c2 distance')
ax3.set_ylabel('Avg m3c2 stdev')
#ax4.set_ylabel('Avg fraction significant change')

plt.show()
parsed=txtfile.split('_l')
figname=parsed[0]+ '_regresults.png'
fig.savefig(figname)
arrayname=parsed[0]+'_resultsarray'
np.save(arrayname, resultsarray)
resultsnames=np.array(resultsnames)
np.save(arrayname, resultsnames)


#fig, (ax1)=plt.subplots()
#pos=ax1.scatter(combinations[:, 0], combinations[:, 1], marker='o', s=50, c=resultsarray[:, 2], cmap=plt.cm.coolwarm, norm=matplotlib.colors.LogNorm())
#cbar=fig.colorbar(pos, ax=ax1)
#ax1.set_xlabel('Scan 1')
#ax1.set_ylabel('Scan 2')
#ax1.set_title('note that there is no scan 5 so 5-11 are known as scans 6-12')
#cbar.set_label('Fraction of core points for which the scans overlap')
#%%
m3c2_list=np.array([])
txt='F:/nz_data/faro/2014/m3c2list.txt'
with open(txt) as files:
    for item in files:
        #print(item)
        item=item.rstrip('\n')
       # parsed=item.split('.')
        item='F:/nz_data/faro/2014/'+item
        #outitem='F:/nz_data/05_03_25_faro/inner_outer/'+parsed[0]+'out.'+parsed[1]

        m3c2_list=np.append(m3c2_list, item, axis=None)


for i in range(0, len(m3c2_list)):
    results=sbf.read(m3c2_list[i])
    xyz=results.xyz
    nn=results.sf_names
    sfs=results.sf