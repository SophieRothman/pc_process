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
import matplotlib

txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
newname1='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_2mm.bin' 
newname2='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_1cm.bin'
newname3='F:/nz_data/faro/2014/testnorm/Mangarere_FARO_20150324_ALL_STATIONS_20cm.bin'


#%% FOR ONLY DOING RANGES AND ANGLES
txtfile='F:/nz_data/faro/2014/testnorm/faro_2014_list.txt'
prefix='F:/nz_data/faro/2014/'

filelist=np.array([])
filelist_final=np.array([])
filelist_ss=np.array([])
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
    wnorms=cc.octree_normals(filelist[i], 0.1,  with_grids=True,  angle=1, orient='WITH_GRIDS',  verbose=True, fmt='bin', global_shift='AUTO', cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
 
    
    # #Compute sensor range and scattering angle
    #             #cc.scattering_angles(cloud,silent=False, verbose=True, cc_exe='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    
    wangles=cc.distances_from_sensor_and_scattering_angles(wnorms, degrees=True, verbose=True,  global_shift='AUTO',  cc_exe='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe') # need to change back to merged_name1

    
    
    # # create a scan number
    
    # # create station scalar field populated with the station number 
    #COMMENTED FOR DIFFERENT USEif i<5:
     #COMMENTED FOR DIFFERENT USE   cc.sf_add_const(merged_name2, ('STATION', (i)),  verbose=True, fmt='bin' )
    #COMMENTED FOR DIFFERENT USEif i>=5:
     #COMMENTED FOR DIFFERENT USE  cc.sf_add_const(merged_name2, ('STATION', (i+1)),  verbose=True, fmt='bin' )
    

    # #invert normals
    
    #merged_name3=parsed[0]+'_SF_ADD_CONST.bin'          
    filelist_final=np.append(filelist_final, wangles, axis=None)
    ss_newnorms=cc.octree_normals(wangles, 5,  with_grids=True,  angle=1, orient='PREVIOUS',  verbose=True, fmt='bin', global_shift='AUTO', cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    
    ss_wangles=cc.ss(ss_newnorms, method='SPATIAL', silent=True, parameter=0.2,  fmt='BIN', verbose=True, cc_exe='C:\\Program Files\\CloudCompare\\CloudCompare.exe') #
    parsed=ss_wangles.split('WITH')
    newname=parsed[0]+'20cm_5mnorm_r_ang.bin'
    os.rename(ss_wangles, newname)

    filelist_ss=np.append(filelist_ss, ss_wangles)
    print(i)

filelist_final=np.array(filelist_final)
np.save('F:/nz_data/fname_range_angles', filelist_final)
np.save('F:/nz_data/fname_range_angles_ss', filelist_ss)

#np.savetxt('F:/nz_data/fname_range_angles.txt', filelist_final) #, header='x y z 1 2 3 4 6 7 8 9 10 11 12 #scans avgm3c2'




#%% NEED TO FIGURE OUT HOW TO CLASSIFY POINT CLOUD OR OBTAIN CLIFF CORE POINTS

#%% GETTING ALL THE COMBINATIONS OF FILES
#txtfile='F:/nz_data/faro/2014/faro_2014_list.txt'
params='F:/nz_data/m3c2_paramsv8.txt'
#filearray='F:/nz_data/fname_range_angles.npy'
#cliff_cp='F:/nz_data/ws_pcs/Mangarere_2014000_20cm_5mn_clasclif_uc75.bin'


# cpc_sbf=pc_cc=cc.to_sbf(cliff_cp,  silent=True, debug=False)
# cpc_sbf=sbf.read(cpc_sbf)
# coords=cpc_sbf.xyz
fullcloud=np.load('F:/nz_data/fname_range_angles.npy')
cp=np.load('F:/nz_data/fname_range_angles_ss.npy')
resultsnames=[]
array_resultsarray=np.empty((len(fullcloud), ))
for j in range(0, len(fullcloud)):
    parsed=fullcloud[j].split('WITH')[0]
    corep=parsed+'20cm_5mnorm_r_ang.bin'

    results=cc.m3c2(fullcloud[j], fullcloud[1], params, core=corep, fmt='SBF',
                 silent=True,cc='F:\\cc_vspecial\\CloudCompare\\CloudCompare.exe')
    pc_m3c2=sbf.read(results)

    xyz=np.copy(pc_m3c2.xyz)
    len_cloud=len(xyz[:, 1])
    nn=pc_m3c2.sf_names
    sfs=pc_m3c2.sf
    resultsarray=np.empty((len(fullcloud), 3))

    results_m3c2=np.empty((len_cloud,len(fullcloud)))


    for i in range(0, len(fullcloud)): #finally this will read 0, numcom
        
        results=cc.m3c2(filelist[j], filelist[i], params, core=corep, fmt='SBF',
                 silent=True, verbose=True)
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
        results_m3c2[:, i]=np.copy(sfs[:, 9])

        dist_unc=sfs[:, 8]
        mask=dist_unc<0.01
        
        results_m3c2[~mask, i]=np.nan
        
        
    #get the average m3c2 results
    avg_m3c2=np.empty((len(sfs[:, 1]), 1))
    num_scans=np.empty((len(sfs[:, 1]), 1))
    maskj=np.arange(0, len(fullcloud))!=j
    for jj in range(0, len(sfs[:, 1])):
        avg_m3c2[jj]=np.nanmean(results_m3c2[jj, maskj])
        num_scans[jj]=np.sum(~np.isnan(results_m3c2[jj, maskj])) #NEED TO DO A SPOT CHECK TO MAKE SURE THIS IS WORKING
    parsed=fullcloud[j].split('WITH')[0]
    parsed=parsed+'20cm_5mnorm_r_ang.bin'
    resbfed=cc.to_sbf(parsed)
    cp_cloud=sbf.read(resbfed)
    sfs_cp=cp_cloud.sf
    sfs_n=cp_cloud.sf_names
    results_m3c2=np.append(results_m3c2, sfs_cp[:, 1:], axis=1)#This creates a file where the first n columns are m3c2 runs then ranges and then scattering angles then the last two are # scans then avg m3c2 
    
    results_m3c2=np.append(results_m3c2, num_scans, axis=1)
    results_m3c2=np.append(results_m3c2, avg_m3c2, axis=1)

    parsed=parsed.split('_20cm')
    arrayname=parsed[0]+'_20cm_m3c2_rsang.sbf'
    written=sbf.write(arrayname, xyz, sf=results_m3c2)
    resultsnames.append(arrayname)

   # # mask=~np.isnan(sfs[:, 5])
   #  sfnames=pc_m3c2.sf_names
   #  #avg m3c2distance
   #  m3c2i=np.where(nn='M3C2 distance')
   #  resultsarray[i, 0]=np.nanmean(sfs[mask, m3c2i])
   #  #stdev of m3c2 distance
   #  #isd=np.where(nn='significant change')
   #  resultsarray[i, 1]=np.nanstd(sfs[mask, m3c2i])
   #  #fraction overlap
   #  resultsarray[i, 2]=len(sfs[mask, m3c2i])/len(sfs[:, m3c2i])
    
# parsed=txtfile.split('_l')
# arrayname=parsed[0]+'_m3c2_array'    
# np.save( arrayname, results_m3c2)   
parsed=fullcloud[0].split('0_')
newname6=parsed[0]+'combined_file_names'
resultsnames=np.array(resultsnames)
#arrayname=parsed[0]+'_dist_unc_array'    
np.save(newname6, resultsnames) 
# %%
arrayname=newname6+'.npy'
fnames=np.load(arrayname)
for j in range(0, len(fullcloud)):
    avg_error=np.empty((len(fullcloud)))
    overlap=np.empty((len(fullcloud)))
    cl=sbf.read(fnames[j])
    sf_tot=cl.sf
    for i in range(0, len(fullcloud)):

        avg_error[i]=np.nanmean(np.abs(sf_tot[:, i]))
        this_scan=sf_tot[:, i]
        exists=~np.isnan(this_scan)
        
        overlap[i]=np.count_nonzero(exists)
    overlap[j]=0
    avg_error[j]=0
    avgm3c2=sf_tot[:, -1]
    num_scan=sf_tot[:, -2]
    scat_a=sf_tot[:, -3]
    ranges=sf_tot[:, -4]

    figure, (ax1, ax2, ax3, ax4, ax5)=plt.subplots(5, 1, figsize=(12,18), layout='tight')
    
    ax1.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], avg_error)
    title='scan' + str(j)
    ax1.set_title(title)
    ax1.set_xlabel('Scan #')
    ax1.set_ylabel('Avg error')
    ax2.bar([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], overlap)
    ax2.set_xlabel('Scan #')
    ax2.set_ylabel('# shared points with less than 1 cm error')
    
    ax3.scatter(avgm3c2, num_scan, alpha=0.1)
    ax3.set_xlabel('Avg m3c2 eror at that point')
    ax3.set_ylabel('# of scans at that point')
    
    ax4.scatter(avgm3c2, scat_a, alpha=0.1)
    ax4.set_xlabel('Avg m3c2 eror at that point')
    ax4.set_ylabel('Scattering angle')
    
    ax5.scatter(avgm3c2, ranges, alpha=0.1)
    ax5.set_xlabel('Avg m3c2 eror at that point')
    ax5.set_ylabel('Distance from scanner')


    



#%%
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