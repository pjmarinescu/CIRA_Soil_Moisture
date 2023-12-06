#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:21:13 2020
@author: Peter Marinescu
Code: To make Map Plots of CPC and HRRR Soil Moisture
"""

# Import Python Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from collections import OrderedDict
import os
import glob
from netCDF4 import Dataset
import numpy.ma as ma
import pandas as pd
import pickle
import copy
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from datetime import datetime, timedelta
import pandas


savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/WAF/Revision/RevFigSubmit/'
avg = OrderedDict()
std = OrderedDict()

savename2 = '3Y-15_0.08'
monsave = 'ALLMONTHS'

# Shapefilepath for plotting shapefiles
shp_path = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/python/code/Map_Plots/shp_files/'
shp_path = '/Volumes/PassBack/CIRA/GSL/python/code/Map_Plots/shp_files/'
#shp_path = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Code/Shapefiles/'


# Read in saved plot data from pickle file
pickledatapath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Code/git/fig2data.p'
if os.path.exists(pickledatapath):

    with open(pickledatapath,'rb') as f:    
        [lons,lats,avg,std,df_s,df_std] = pickle.load(f)

# Calculate plot data from original data (takes a couple hours)
else:
    
    #month_str = ['01','02','03','04','05','06','07','08','09','10','11','12']
    #month_str = ['06','07','08','09','10','11']
    #month_str = ['01','02','03','04','05','11','12']
    #month_str = ['06','07','08','09','10']
    
    #month_str = ['01','02','03','04','05','06','07','08','12']
    #monsave = 'SON'
    
    #month_str = ['01','02','03','04','05','09','10','11','12']
    #monsave = 'JJA'
    
    #month_str = ['01','02','06','07','08','09','10','11','12']
    #monsave = 'MAM'
    
    #month_str = ['03','04','05','06','07','08','09','10','11']
    #monsave = 'DJF'
    
    month_str = ['13']
    monsave = 'ALLMONTHS'
    
    #monsave = 'JJASO'
    nx = 1059
    ny = 1799
    
    start_date = datetime(2018,7,12)
    end_date = datetime(2020,12,2)
    
    sids = np.arange(0,1240,1)
    
    
    # File path with data
    openpath = '/scratch1/RDARCH/rda-ghpcs/Kyle.Hilburn/datasets/'
    openpath = '/Volumes/PassBack/CIRA/GSL/data/cpc/datasets/'
    filename = 'GOES_CPC_soil_moisture_dataset.nc'
    data = Dataset(openpath+filename, "r")
    cpc = data.variables['CPC'][sids,:,:]
    cpc[cpc <= 0] = np.nan
    cpc[cpc > 800] = np.nan
    print('CPC Data Stats')
    print(np.nanmean(cpc))
    print(np.nanstd(cpc))
    
    for i in np.arange(0,len(sids)):
       for m in np.arange(0,len(month_str)):
          if data['MONTH'][i] == int(month_str[m]):
               cpc[i,:,:] = np.nan
    
    avg[1] = np.nanmean(cpc,axis=0)
    std[1] = np.nanstd(cpc,axis=0)
    print('CPC Data Stats')
    print(np.nanmean(cpc))
    print(np.nanstd(cpc))
    data.close()
    del(cpc)
    
    print('Finished Reading in CPC Data')
    
    # File path with data
    #pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_Data/pickle/pickle/'
    pickpath = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/hrrr/pickle_fin/'
    pickpath = '/Volumes/PassBack/CIRA/GSL/data/hrrr/pickle_fin/'
    files = sorted(glob.glob(pickpath+'*.p'))
        
    hrr = np.zeros((len(files),nx,ny))
    # File path with data
    #for f in sids:
    for f in np.arange(0,len(files)):
        print(f)
        #for f in np.arange(0,len(files)):
    
        with open(files[f], 'rb') as handle:
            [ism_fin,m_lvls,z_lvls,nx,ny,lons,lats,nslvls,an_date] = pickle.load(handle)
        
        for m in np.arange(0,len(month_str)):
            if an_date.strftime("%m") == month_str[m]:
                hrr[f,:,:] = np.nan
            else:   
                hrr[f,:,:] = copy.deepcopy(ism_fin)
    
    print(np.nanmax(hrr))   
    print(np.nanmean(hrr))   
    print(np.nanmin(hrr))
    print(np.min(hrr))    
    hrr[hrr <= 0] = np.nan  
    hrr[hrr > 800] = np.nan  
    avg[0] = np.nanmean(hrr,axis=0)
    std[0] = np.nanstd(hrr,axis=0)
    print('HRRR Data Stats')
    print(np.nanmean(avg[0]))
    print(np.nanstd(std[0]))
    print(np.nanmax(avg[0]))
    print(np.shape(avg[0]))
    del(hrr)
    
    # Get In Situ Station Data
    usc_i = []
    net_i = []
    hrr_i = []
    cpc_i = []
    usc_p = []
    hrr_p = []
    lon_p = []
    lat_p = []
    nstat = []
    
    start_date = datetime(2018,7,12)
    end_date = datetime(2020,12,2)
    ism_txt = ''
    openpaths = ['/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/python/save/ISM_scat_fin/lowqc/',
                '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/python/save/ISM_scat_fin_sc/']
    openpaths = ['/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_lqc/',
                '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_sqc/']
    
    
    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n*1)
    
    for t in daterange(start_date,end_date):
    
        a_date = t
        
        # Convert date to desired datestrs
        YYYY = a_date.strftime("%Y")
        YY = a_date.strftime("%y")
        YY = a_date.strftime("%y")
        MM = a_date.strftime("%m")
        DD = a_date.strftime("%d")
        DOY = a_date.strftime("%j")
        datestr = MM+DD
    
        mon_test = 0
        for m in np.arange(0,len(month_str)):
            print(MM,month_str[m])
            if MM == month_str[m]:
                print(datestr)
                mon_test = 1
    
        if mon_test == 1:
            continue   
    
        print('Test',MM)
    
        nlon = 0
        for o in np.arange(0,2):
            
            if o == 0:
                filename = 'f_scatter_values_'+YYYY+MM+DD+'_'+ism_txt+'_lowqc.p'
                lfilename = 'l_scatter_values_'+YYYY+MM+DD+'_'+ism_txt+'_lowqc.p'
                if os.path.exists(openpaths[o]+filename):
                    [usc,vsm,ism_usc,hrrr_int,cpc_int,a_date] = pickle.load(open( openpaths[o]+filename, "rb" ))
                    [lat,lon] = pickle.load(open( openpaths[o]+lfilename, "rb" ))
                    # lat, lon, 'ism', 'v5','v10','v100'
                else:
                    print('No Data @ '+str(t))
                    continue
             
                latn = lat['ism']
                lonn = lon['ism']          
                
                lat_o = copy.deepcopy(latn)
                #Eliminate data from Selma, AL station (often very high VSM values (>0.6) at 100 cm depth)   
                ism_usc = ism_usc[(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                cpc_int = cpc_int[(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                hrrr_int[ism_txt] = hrrr_int[ism_txt][(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                latn = latn[(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                lonn = lonn[(lat_o<32.45) | (lat_o > 32.47) | (lonn<-87.25) | (lonn>-87.23)]    
                net = np.ones(len(latn))*1
    
            elif o == 1:
                filename = 'sc_scatter_values_'+YYYY+MM+DD+'_'+ism_txt+'_lowqc.p'
                lfilename = 'scl_scatter_values_'+YYYY+MM+DD+'_'+ism_txt+'_lowqc.p'
                if os.path.exists(openpaths[o]+filename):
                    [usc,vsm,ism_usc,hrrr_int,cpc_int,a_date] = pickle.load(open( openpaths[o]+filename, "rb" ))
                    [lat,lon] = pickle.load(open( openpaths[o]+lfilename, "rb" ))
                    # lat, lon, 'ism', 'v5','v10','v100'
                else:
                    print('No Data @ '+str(t))
                    continue             
                latn = lat['ism']
                lonn = lon['ism']
                net = np.ones(len(latn))*2
    
            nlon = nlon+len(lonn)
        
            usc_i = np.append(usc_i,ism_usc)
            hrr_i = np.append(hrr_i,hrrr_int[ism_txt])
            cpc_i = np.append(cpc_i,cpc_int)    
            net_i = np.append(net_i,net)    
            lon_p = np.append(lon_p,lonn)
            lat_p = np.append(lat_p,latn)
    
        if nlon == 0:
            nlon = np.nan
        nstat = np.append(nstat,nlon)
        
    loc = (lon_p*100+lat_p)
    
    print(np.shape(usc_i))
    print(np.shape(hrr_i))
    print(np.shape(cpc_i))
    print(np.shape(lon_p))
    print(np.shape(lat_p))
    print(np.shape(loc))
    print(np.shape(net_i))
    
    d = {'insitu': usc_i,
            'hrrr': hrr_i,
            'cpc': cpc_i,
            'lon': lon_p,
            'lat': lat_p,
            'uloc': loc,
            'net': net_i}
    df = pandas.DataFrame(data = d)
    
    print('DataFrame Values')
    print(np.nanmax(df['hrrr'].values))
    print(np.nanmin(df['hrrr'].values))
    print(np.nanmax(df['cpc'].values))
    print(np.nanmin(df['cpc'].values))
    
    # Screen out stations with larger relative error variances
    savename2 = '3Y-15_0.08'
    pickpath = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/python/code/Map_Plots/stat_lists/hera/'
    pickpath = '/Volumes/PassBack/CIRA/GSL/python/code/Map_Plots/stat_lists/hera/'
    pickfile = 'Stat_List_Pass_Auto_'+savename2+'.p'
    
    [sta_g,val_g,lat_g,lon_g,len_g] = pickle.load(open(pickpath+pickfile,"rb"))
    
    uloc_g = np.zeros(len(sta_g))
    for i in np.arange(0,len(sta_g)):
        uloc_g[i] = lon_g[i]*100 + lat_g[i]
    
    df2 = df[df['uloc'].isin(uloc_g)]
    df = copy.deepcopy(df2)
    
    print('DataFrame Values')
    print(np.nanmax(df['hrrr'].values))
    print(np.nanmin(df['hrrr'].values))
    print(np.nanmax(df['cpc'].values))
    print(np.nanmin(df['cpc'].values))
    
    # Calculate mean values for all remaining stations
    df_s = df.groupby(['uloc']).mean()
    df_su = df_s[df_s['net'] == 1]
    df_sc = df_s[df_s['net'] == 2]
    
    df_std = df.groupby(['uloc']).std()
    df_su2 = df_std[df_std['net'] == 1]
    df_sc2 = df_std[df_std['net'] == 2]
    
    print(np.shape(df_su))
    print(np.shape(df_sc))
    
    
hrr_ii = df_s['hrrr'].values
cpc_ii = df_s['cpc'].values
usc_ii = df_s['insitu'].values
lon_pp = df_s['lon'].values
lat_pp = df_s['lat'].values

hrr_iii = df_std['hrrr'].values
cpc_iii = df_std['cpc'].values
usc_iii = df_std['insitu'].values

# Calculate mean ISM station values for all remaining stations
mean_val = np.nanmean([hrr_ii,usc_ii,cpc_ii],axis=0)        

# Estimate percentile bin locations based on mean values
npcts = 20
pcts = np.arange(0,101,npcts)
x_arr = np.zeros(len(pcts))
for i in np.arange(0,len(pcts)):
    x_arr[i] = np.nanpercentile(mean_val,pcts[i])

# Save indicies of the different quintiles in the data
qids = OrderedDict()
for j in np.arange(0,len(x_arr)-1):
       qids[j] = np.where((mean_val >= x_arr[j]) & (mean_val < x_arr[j+1]))[0]
       print(len(qids[j]))



# Read shapefile data
reader = Reader(shp_path+'ne_50m_admin_1_states_provinces')
states_provinces = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.5)
reader = Reader(shp_path+'ne_50m_coastline')
coastline = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.5)
reader = Reader(shp_path+'ne_50m_land')
land = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.5)
reader = Reader(shp_path+'ne_50m_admin_0_countries')
country = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.5)
reader = Reader(shp_path+'ne_50m_lakes')
lakes = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=0.5)


fs = 11.5; plt.rcParams.update({'font.size': fs}) # Fontsize
ax = OrderedDict()
#fig = plt.figure(figsize=[7.5,28])
fig = plt.figure(figsize=[11,5])
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, wspace=0.15, hspace=0.02,
                         left = 0.05, right = 0.97, top = 0.97, bottom = 0.03 )
gx = 0; gy = 0;

title_l = ['(a) HRRR ISM Mean (mm)','(b) CPC ISM Mean (mm)','(c) HRRR ISM Std. Dev. (mm)','(d) CPC ISM Std. Dev. (mm)']

# Symbols for the different quintiles
syms = ['v','s','o','D','^']

clvls = OrderedDict()
clvls[0] = np.arange(60,601,40)
clvls[1] = np.arange(0,130,10)

cmaps = OrderedDict()
#cmaps[0] = plt.cm.Greens
cmaps[0] = plt.cm.gist_rainbow
cmaps[1] = plt.cm.gist_rainbow
cmap_sm = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red','coral','gold','skyblue','blue'])
cmaps[0] = cmap_sm
cmaps[1] = cmap_sm

# Individual Data Plots
i = 0
for j in np.arange(0,2):
    for k in np.arange(0,2):
        if j == 0:
            varp = avg
        elif j == 1:
            varp = std
        ax[i] = plt.subplot(spec[j,k], projection=ccrs.PlateCarree())
        ax[i].add_feature(states_provinces)
        ax[i].add_feature(coastline)
        ax[i].add_feature(land)
        ax[i].add_feature(country)
        ax[i].add_feature(lakes)
    
        print(np.max(varp[k]))
        print(np.min(varp[k]))
        print(np.max(clvls[j]))
        print(np.min(clvls[j]))
        
        print('NAN CALCs')
        print(np.nanmax(varp[k]))
        print(np.nanmin(varp[k]))
        print(np.nanmax(clvls[j]))
        print(np.nanmin(clvls[j]))


        a = ax[i].contourf(lons, lats, varp[k], levels=clvls[j], cmap=cmaps[j],vmin=np.nanmin(clvls[j]),vmax=np.nanmax(clvls[j]),extend='both', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(a,ax=ax[i])
        if j == 0:
            cbar.ax.set_ylabel('ISM Mean (mm)')
        if j == 1:
            cbar.ax.set_ylabel('ISM Std. Dev. (mm)')
            

#        ax[i].scatter(df_su['lon'].values,df_su['lat'].values,c='k',s=3,marker='o')
#        ax[i].scatter(df_sc['lon'].values,df_sc['lat'].values,c='k',s=3,marker='x')
#        ax[i].scatter(df_su['lon'].values,df_su['lat'].values,c=df_su['insitu'].values,s=4,marker='o',linewidths=0.1,edgecolors='w',cmap=cmaps[j],vmin=np.min(clvls[j]),vmax=np.max(clvls[j]))
#        ax[i].scatter(df_sc['lon'].values,df_sc['lat'].values,c=df_sc['insitu'].values,s=4,marker='o',linewidths=0.1,edgecolors='w',cmap=cmaps[j],vmin=np.min(clvls[j]),vmax=np.max(clvls[j]))

        ax[i].add_feature(states_provinces)
        ax[i].add_feature(coastline)
        ax[i].add_feature(land)
        ax[i].add_feature(country)
        ax[i].add_feature(lakes)

        for ii in np.arange(0,5):

            if j == 0:
                ax[i].scatter(df_s['lon'].values[qids[ii]],df_s['lat'].values[qids[ii]],c=df_s['insitu'].values[qids[ii]],s=40,marker=syms[ii],linewidths=0.7,edgecolors='k',cmap=cmaps[j],vmin=np.min(clvls[j]),vmax=np.max(clvls[j]))
            elif j == 1:
                ax[i].scatter(df_s['lon'].values[qids[ii]],df_s['lat'].values[qids[ii]],c=df_std['insitu'].values[qids[ii]],s=40,marker=syms[ii],linewidths=0.7,edgecolors='k',cmap=cmaps[j],vmin=np.min(clvls[j]),vmax=np.max(clvls[j]))


        ax[i].set_xlim([-128,-65])
        ax[i].set_ylim([24,50])
        ax[i].set_title(title_l[i])
        gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabels_top = False
        gl.ylabels_right = False
        
        i = i+1

plt.tight_layout()
#plt.savefig(savepath+'Figure2.png',dpi=600)
plt.savefig(savepath+'Figure2_300dpi.pdf',dpi=300)
#plt.savefig(savepath+'Figure2.eps',dpi=600)
