#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 23:28:30 2021

@author: petermarinescu
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from datetime import datetime
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import pandas
from scipy import stats
from SMA_fx import read_in_data

# Custom, color-blind friendly colorbar
cmap_sm = matplotlib.colors.LinearSegmentedColormap.from_list("", ['saddlebrown','sandybrown','gold','lawngreen','green'])
cmap_sm = matplotlib.colors.LinearSegmentedColormap.from_list("", ['blue','skyblue','gold','coral','red'])
cmap_sm_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red','coral','gold','skyblue','blue'])
cmap_sm_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red','coral','white','skyblue','blue'])
cmap_sm_rmad = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white','skyblue','blue'])

savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/WAF/RevFigSubmit/'
openpaths = ['/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_lqc/',
            '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_sqc/']

# Input Parameters in Script
varn = 'TempR' # 'Mean','Std. Dev.','TempR'
varn2 = '' # 'MAD' or '' 
pv = 0.01 # 99% statistical significant
start_date = datetime(2018,7,12) # Begin Date
end_date = datetime(2020,12,3) # End Date
ism_txt = ''
savename = 'NOSEL_Hv3_median'
pair = [5,0] # USCRN, HRRR
if varn == 'TempR':
    cmap_sm_r = plt.cm.YlOrRd

# Read in data   
# insitu ism, hrrr ism, cpc ism, insitu level data, hrr level data, latitude, longitude, unique location identifier
[usc_i,hrr_i,cpc_i,usc_p,hrr_p,lon_p,lat_p,loc] = read_in_data(start_date,end_date,pair,ism_txt,openpaths,'ism')

# Place data into a pandas dataframe
d = {'insitu': usc_i,
        'hrrr': hrr_i,
        'cpc': cpc_i,
        'lon': lon_p,
        'lat': lat_p,
        'uloc': loc}
df = pandas.DataFrame(data = d)

# Calculate differences for individual days
df['hmc'] = df['hrrr']-df['cpc']
df['hmi'] = df['hrrr']-df['insitu']
df['cmi'] = df['cpc']-df['insitu']

# Calculate mean for each station location
df_stat = df.groupby(['uloc']).mean()
print(len(df_stat))

# Screen out stations with larger relative error variances using separate file
savename2 = '3Y-15_0.08'
pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/'
pickfile = 'Stat_List_Pass_Auto_'+savename2+'.p'
[sta_g,val_g,lat_g,lon_g,len_g] = pickle.load(open(pickpath+pickfile,"rb"))

# Loop through stations that pass error variance tests
uloc_g = np.zeros(len(sta_g))
for i in np.arange(0,len(sta_g)):
    # Calculate station unique identifier for stations that pass error variance testings
    uloc_g[i] = lon_g[i]*100 + lat_g[i]

# Only include good stations into new dataframe 
df2 = df[df['uloc'].isin(uloc_g)]


# For noqc tests
statlist = df2.groupby(['uloc']).mean() # List of stations based on unique identifier
statlist_all = df.groupby(['uloc']).mean() # List of stations based on unique identifier


#################
###### PLOT TIME
################


# Set font sizes
plt.rcParams.update({'font.size': 14})
tfs = 16 # title fontsize
ss = 30;
fig = plt.figure(figsize=[7,3.5])

# Make map projection axes
gs = GridSpec(1, 1, figure=fig, left = 0.08, right = 0.97, top = 0.97, bottom = 0.03 )
ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

states110 = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='110m',
            facecolor='none',
            edgecolor='k')



# Map Plot Specifications
ax.set_extent([-128,-65,23,48])
ax.add_feature(cfeature.LAND,facecolor='whitesmoke',zorder=0)
ax.add_feature(cfeature.LAKES,edgecolor='dodgerblue',facecolor='white',zorder=0)
ax.add_feature(cfeature.BORDERS,zorder=1)
ax.add_feature(states110,zorder=1)
ax.coastlines(resolution='50m', color='black', linewidth=1,zorder=1)
ax.add_feature(cfeature.RIVERS,edgecolor='dodgerblue',color='dodgerblue',zorder=1)    
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-120,-110,-100,-90,-80,-70])
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()

ax.scatter(statlist_all['lon'],statlist_all['lat'],marker='x',s=ss,edgecolor='r',c='r')
ax.scatter(statlist['lon'],statlist['lat'],marker='x',s=ss,edgecolor='k',c='k')

plt.tight_layout()
plt.subplots_adjust(left=0.05)
plt.savefig(savepath+'Figure1.png')
plt.savefig(savepath+'Figure1.pdf')