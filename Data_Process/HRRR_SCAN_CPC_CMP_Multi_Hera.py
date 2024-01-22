#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:41:33 2021

@author: petermarinescu
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import pickle
import copy
import sys
import os.path
sys.path.append("/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/python/code/") 
from Read_SCAN import read_SCAN
from netCDF4 import Dataset
from sklearn import linear_model
regr = linear_model.LinearRegression()

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n*1)

start_date = datetime(2020,7,1)
end_date = datetime(2021,1,1)

savepath = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/python/save/ISM_scat_fin/'
ism_txt = ''

for t in daterange(start_date,end_date):

    a_date = t
    
    # Convert date to desired datestrs
    YYYY = a_date.strftime("%Y")
    YY = a_date.strftime("%y")
    YY = a_date.strftime("%y")
    MM = a_date.strftime("%m")
    DD = a_date.strftime("%d")
    DOY = a_date.strftime("%j")
    print(DOY) 
    # Load preprocessed HRRR data
    pickpath = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/hrrr/pickle_fin/'
    filename = 'HRRR_ISM_'+YYYY+MM+DD+'_lm.p'
    if os.path.exists(pickpath+filename):
#        [ism_pjm,ism_ts1,m_lvls,z_lvls,nx,ny,lons,lats,nslvls,an_date] = pickle.load(open( pickpath+filename, "rb" ))
        [ism_fin,m_lvls,z_lvls,nx,ny,lons,lats,nslvls,an_date] = pickle.load(open( pickpath+filename, "rb" ))
    else:
        continue
    
    if os.path.exists(savepath+'sc_scatter_values_'+YYYY+MM+DD+'_'+ism_txt+'_lowqc.p'):
        continue


    # Load CPC data
    openpath = '/scratch1/RDARCH/rda-ghpcs/Kyle.Hilburn/datasets/'
    filename = 'GOES_CPC_soil_moisture_dataset.nc'
    data = Dataset(openpath+filename, "r")
    cYYYY = data.variables['YEAR'][:]
    cMM = data.variables['MONTH'][:]
    cDOM = data.variables['DAYOFMONTH'][:]
    clat = data.variables['XLAT_M'][:]
    clon = data.variables['XLONG_M'][:]
    
    id1 = np.array(np.where(cYYYY==int(YYYY)))
    id2 = np.array(np.where(cMM==int(MM)))
    id3 = np.array(np.where(cDOM==int(DD)))
    idf = np.intersect1d(np.intersect1d(id1,id2),id3)
    
    if np.size(idf) == 0:
        print(a_date)
        print('No GOES or CPC Data')
        continue
    
    cpc = data.variables['CPC'][idf[0],:,:]
    
    # Read in SCAN data
    data_uscrn = read_SCAN(YYYY+MM+DD)
    #print(np.shape(data_uscrn))
    #print('QCing steps next.')
    # Only include CONUS
    data_uscrn = data_uscrn[data_uscrn['LATITUDE']<55]
    data_uscrn = data_uscrn[data_uscrn['LATITUDE']>23]
    
    #Eliminate Coastal Stations That cannot be Resolved in HRRR
    #data_uscrn = data_uscrn[data_uscrn['WBANNO']!=53152] # Santa Barbara, CA
    #data_uscrn = data_uscrn[data_uscrn['WBANNO']!=63856] # Brunswick, GA
    #data_uscrn = data_uscrn[data_uscrn['WBANNO']!=93245] # Bodega, CA

    data_uscrn_i = data_uscrn[data_uscrn['ism_1p6']>0]
    data_uscrn_i = data_uscrn_i[data_uscrn_i['ism_1p6']<1200]
    
    data_uscrn_v5 = data_uscrn[data_uscrn['SOIL_MOISTURE_5_DAILY']<80]
    data_uscrn_v10 = data_uscrn[data_uscrn['SOIL_MOISTURE_10_DAILY']<80]
    data_uscrn_v100 = data_uscrn[data_uscrn['SOIL_MOISTURE_100_DAILY']<80]

    #print(np.shape(data_uscrn))
    #print(np.shape(data_uscrn_v5))
    #print(np.shape(data_uscrn_v10))
    #print(np.shape(data_uscrn_v100))    

    # #
    # # Smooth field for easier comparison with CPC data
    # ss = 15
    # ism_smo = np.zeros((ny,nx))
    # for ii in np.arange(ss,ny-ss):
    #     for jj in np.arange(ss,nx-ss):
    #         ism_smo[ii,jj] = np.nanmean(ism[ii-ss:ii+ss,jj-ss:jj+ss])
    
    cmap_cpc = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","honeydew","lightgreen","lime","limegreen","forestgreen","green"])
    matplotlib.rcParams.update({'font.size': 16})
    fs = 10
    
    anls = [ism_fin]
    txts = ['']
    
    cnt = 0
    hrrr_int = OrderedDict()
    from scipy.interpolate import griddata      
    import scipy
    #from mpl_toolkits.basemap import Basemap
    for anl_cho in anls:
        # Choose ISM Analysis
        ism_anl = copy.deepcopy(anl_cho)
        ism_txt = txts[cnt]
      
        # Prepare data for interpolation
        ism_f = ism_anl.flatten()
        lon_f = lons.flatten()
        lat_f = lats.flatten()
        
        lon_f = lon_f[ism_f>0]
        lat_f = lat_f[ism_f>0]
        ism_f = ism_f[ism_f>0]
        
        points = (lon_f,lat_f)
        values = ism_f
        hrrr_int[ism_txt] = griddata(points,values,(data_uscrn_i['LONGITUDE'].values+360,data_uscrn_i['LATITUDE'].values))        
        ism_usc = data_uscrn_i['ism_1p6'].values
        
        # clvls = np.arange(0,701,25)
        # fig,ax = plt.subplots(1,1,figsize=[12.5,7])
        # # Plot data on map
        # m = Basemap(width=5500000,height=3300000,
        #             resolution='l',projection='eqdc',\
        #             lat_1=45.,lat_2=55,lat_0=40,lon_0=-95.,ax=ax)
        # x, y = m(lons, lats) # compute map proj coordinates.
        # x2,y2 = m(data_uscrn['LONGITUDE'].values,data_uscrn['LATITUDE'].values)
        # #plt.contourf(lon,lat,obs)
        # # draw coastlines, state and country boundaries, edge of map.
        # m.drawcoastlines()
        # m.drawstates()
        # m.drawcountries()
        # a = m.contourf(x,y,ism_anl,levels=clvls,cmap=cmap_cpc)
        # b = m.scatter(x2,y2,s=25,c=ism_usc,vmin=np.min(clvls),vmax=np.max(clvls),cmap=cmap_cpc,edgecolors='magenta')
        # #b = m.scatter(x2,y2,s=10,c=hrrr_int,vmin=np.min(clvls),vmax=np.max(clvls),cmap=plt.cm.Greens,edgecolors='c')
        # cbar = plt.colorbar(a,ax=ax)
        # cbar.ax.set_ylabel('1.6m ISM (mm)')
        # ax.set_title('ISM: '+YY+MM+DD)
        # plt.tight_layout()
        # plt.savefig(savepath+'HRRRISM_SMap_USCRN'+YYYY+MM+DD+'_'+ism_txt+'_1200.png')
        # plt.close()
        
        ism_anl = copy.deepcopy(anl_cho)
        ism_txt = txts[cnt]
      
        # Prepare data for interpolation
        cpc_f = cpc.flatten()
        lon_f = lons.flatten()
        lat_f = lats.flatten()
        
        lon_f = lon_f[cpc_f>0]
        lat_f = lat_f[cpc_f>0]
        cpc_f = cpc_f[cpc_f>0]
        
        points = (lon_f,lat_f)
        values = cpc_f
        cpc_int = griddata(points,values,(data_uscrn_i['LONGITUDE'].values+360,data_uscrn_i['LATITUDE'].values))        
        
        clvls = np.arange(0,701,25)
        #fig,ax = plt.subplots(1,1,figsize=[12.5,7])
        # Plot data on map
        #m = Basemap(width=5500000,height=3300000,
        #             resolution='l',projection='eqdc',\
        #             lat_1=45.,lat_2=55,lat_0=40,lon_0=-95.,ax=ax)
        #x, y = m(lons, lats) # compute map proj coordinates.
        #x2,y2 = m(data_uscrn['LONGITUDE'].values,data_uscrn['LATITUDE'].values)
        #plt.contourf(lon,lat,obs)
        ## draw coastlines, state and country boundaries, edge of map.
        #m.drawcoastlines()
        #m.drawstates()
        #m.drawcountries()
        #a = m.contourf(x,y,cpc,levels=clvls,cmap=cmap_cpc)
        #b = m.scatter(x2,y2,s=25,c=ism_usc,vmin=np.min(clvls),vmax=np.max(clvls),cmap=cmap_cpc,edgecolors='magenta')
        #b = m.scatter(x2,y2,s=10,c=hrrr_int,vmin=np.min(clvls),vmax=np.max(clvls),cmap=plt.cm.Greens,edgecolors='c')
        #cbar = plt.colorbar(a,ax=ax)
        #cbar.ax.set_ylabel('1.6m ISM (mm)')
        #ax.set_title('ISM: '+YY+MM+DD)
        #plt.tight_layout()
        #plt.savefig(savepath+'CPCISM_SMap_SCAN'+YYYY+MM+DD+'_'+ism_txt+'.png')
        #plt.close()

        # ##############################
    
        # clvls = np.arange(-600,601,25)
        # fig,ax = plt.subplots(1,1,figsize=[12.5,7])
        # # Plot data on map
        # m = Basemap(width=5500000,height=3300000,
        #             resolution='l',projection='eqdc',\
        #             lat_1=45.,lat_2=55,lat_0=40,lon_0=-95.,ax=ax)
        # x, y = m(lons, lats) # compute map proj coordinates.
        # #plt.contourf(lon,lat,obs)
        # # draw coastlines, state and country boundaries, edge of map.
        # m.drawcoastlines()
        # m.drawstates()
        # m.drawcountries()
        # pltval = ism_anl-cpc
        # pltval[np.abs(pltval) > 950] = np.nan
        # a = m.contourf(x,y,pltval,levels=clvls,extend='both',cmap=plt.cm.bwr_r)
        # #b = m.scatter(x2,y2,s=25,c=ism_usc,vmin=np.min(clvls),vmax=np.max(clvls),cmap=cmap_cpc,edgecolors='magenta')
        # #b = m.scatter(x2,y2,s=10,c=hrrr_int,vmin=np.min(clvls),vmax=np.max(clvls),cmap=plt.cm.Greens,edgecolors='c')
        # cbar = plt.colorbar(a,ax=ax)
        # cbar.ax.set_ylabel('Difference in 1.6m ISM (mm)')
        # ax.set_title('HRRR - CPC ISM: '+YY+MM+DD)
        # plt.tight_layout()
        # plt.savefig(savepath+'f_Diff_CPC_Map_HRRR_'+YYYY+MM+DD+'_'+ism_txt+'.png')
        # plt.close()
        
        ###############################
    
        cpc_f = cpc.flatten()
        ism_f = ism_anl.flatten()
    
        cpc_f = cpc_f[ism_f>0]
        ism_f = ism_f[ism_f>0]
    
        ism_f = ism_f[cpc_f>0]
        cpc_f = cpc_f[cpc_f>0]
    
        lims = [0,1200]
        fig,ax = plt.subplots(1,1,figsize=[6,5])
        plt.plot(np.arange(0,1200),np.arange(0,1200),'-k')
    
        plt.scatter(cpc_f,ism_f,c='m')
    
        x_arr = np.arange(0,1201,100)
        lr = scipy.stats.linregress(cpc_f,ism_f)
        #print(lr.slope)
        plt.plot(x_arr,lr.intercept+lr.slope*x_arr,'-m')
        plt.text(600,1100,'y = '+str(np.round(lr.slope,2))+'x + '+str(np.round(lr.intercept,2)),c='m',fontsize=fs)
    
        plt.xlabel('CPC ISM (1.6m)')
        plt.ylabel('HRRR ISM (1.6m)')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.grid()
        plt.tight_layout()
        plt.savefig(savepath+'sc_CPC_scatter_ISM_HRRR_CMP_'+YYYY+MM+DD+'_'+ism_txt+'.png')
        plt.close()
    
        # bins = np.arange(0,1201,50)
        # H, xedges, yedges = np.histogram2d(cpc_f,ism_f,bins)
        # H = H.T
        # X, Y = np.meshgrid(xedges, yedges)
    
        # fig,ax = plt.subplots(1,1,figsize=[6,5])
        # a = ax.pcolormesh(X, Y, np.log10(H),cmap=plt.cm.Reds)
        # cbar = plt.colorbar(a,ax=ax)
        # cbar.ax.set_ylabel('# of HRRR Grid Points')
        # tix = np.array([1,2,3,4,5])
        # cbar.set_ticks(tix)
        # cbar.set_ticklabels(np.power(10,tix))
        
        # plt.plot(x_arr,lr.intercept+lr.slope*x_arr,'-r')
        # plt.text(600,1100,'y = '+str(np.round(lr.slope,2))+'x + '+str(np.round(lr.intercept,2)),c='r',fontsize=fs)
        # plt.plot(np.arange(0,1200),np.arange(0,1200),'-k')
    
        # plt.xlabel('CPC 1.6m ISM (mm)')
        # plt.ylabel('HRRR 1.6m ISM (mm)')
        # plt.xlim(lims)
        # plt.ylim(lims)
        # plt.grid()
        # plt.tight_layout()
        # plt.savefig(savepath+'CPC_2DHist_ISM_HRRR_CMP_'+YYYY+MM+DD+'_'+ism_txt+'.png')
        # plt.close()
    
        ##############################   
        lims = [0,1200]
        fig,ax = plt.subplots(1,1,figsize=[6,5])
        plt.plot(np.arange(0,1200),np.arange(0,1200),'-k')
    
        plt.scatter(ism_usc,hrrr_int[ism_txt],c='r')
    
        x_arr = np.arange(0,1201,100)
        lr = scipy.stats.linregress(ism_usc,hrrr_int[ism_txt])
        #print(lr.slope)
        plt.plot(x_arr,lr.intercept+lr.slope*x_arr,'-r')
        plt.text(600,1100,'HRRR: y = '+str(np.round(lr.slope,2))+'x + '+str(np.round(lr.intercept,2)),c='r',fontsize=fs)
    
    
        plt.scatter(ism_usc,cpc_int,c='b')
    
        x_arr = np.arange(0,1201,100)
        lr = scipy.stats.linregress(ism_usc,cpc_int)
        #print(lr.slope)
        plt.plot(x_arr,lr.intercept+lr.slope*x_arr,'-b')
        plt.text(600,1050,'CPC: y = '+str(np.round(lr.slope,2))+'x + '+str(np.round(lr.intercept,2)),color='b',fontsize=fs)
    
        plt.text(600,1000,'Number of Points: '+str(len(ism_usc)),color='k',fontsize=fs)
    
    
        plt.xlabel('USCRN ISM (1.6m)')
        plt.ylabel('HRRR / CPC ISM (1.6m)')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.grid()
        plt.tight_layout()
        plt.savefig(savepath+'sc_CPC_scatter_ISM_HRRR_USCRN_CMP_'+YYYY+MM+DD+'_'+ism_txt+'.png')
        plt.close()
        
        cnt = cnt + 1   
    
    usc = OrderedDict()
    vsm = OrderedDict()
   # slon = OrderedDict()
   # slat = OrderedDict()
       
    m_ids = [0,1,10,3,6]
    v_ids = [0,1,5,10,100]
    usc_v = ['5','5','5','10','100']
    for i in np.arange(0,len(m_ids)):
        
        if m_ids[i] == 10:
            slope = (m_lvls[:,:,3]-m_lvls[:,:,2])/6
            mlvls_5cm = m_lvls[:,:,2]+(slope*1)
            ism_f = mlvls_5cm.flatten()
        else:
            ism_f = m_lvls[:,:,m_ids[i]].flatten()
    
        lon_f = lons.flatten()
        lat_f = lats.flatten()
        
        lon_f = lon_f[ism_f>0]
        lat_f = lat_f[ism_f>0]
        ism_f = ism_f[ism_f>0]
        
        points = (lon_f,lat_f)
        values = ism_f.flatten() # 10 cm VSM
       
        if usc_v[i] == '5':
           vsm[v_ids[i]] = griddata(points,values,(data_uscrn_v5['LONGITUDE'].values+360,data_uscrn_v5['LATITUDE'].values))        
           usc[usc_v[i]] = data_uscrn_v5['SOIL_MOISTURE_'+usc_v[i]+'_DAILY'].values/100.0
   
        elif usc_v[i] == '10':
           vsm[v_ids[i]] = griddata(points,values,(data_uscrn_v10['LONGITUDE'].values+360,data_uscrn_v10['LATITUDE'].values))        
           usc[usc_v[i]] = data_uscrn_v10['SOIL_MOISTURE_'+usc_v[i]+'_DAILY'].values/100.0

        elif usc_v[i] == '100': 
           vsm[v_ids[i]] = griddata(points,values,(data_uscrn_v100['LONGITUDE'].values+360,data_uscrn_v100['LATITUDE'].values))
           usc[usc_v[i]] = data_uscrn_v100['SOIL_MOISTURE_'+usc_v[i]+'_DAILY'].values/100.0
         
        #print(i)
        #print(np.shape(usc[usc_v[i]]))
        #print(np.shape(vsm[v_ids[i]]))


    pickle.dump([usc,vsm,ism_usc,hrrr_int,cpc_int,a_date],open( savepath+'sc_scatter_values_'+YYYY+MM+DD+'_'+ism_txt+'_lowqc.p', "wb" ))
