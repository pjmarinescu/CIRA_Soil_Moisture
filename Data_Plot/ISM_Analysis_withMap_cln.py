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
varn = 'Std. Dev.' # 'Mean','Std. Dev.','TempR','TempR2
varn2 = '' # 'MAD' or '' # Calculate mean absolute differences
pv = 0.001 # 99% statistical significant
start_date = datetime(2018,7,12) # Begin Date
end_date = datetime(2020,12,3) # End Date
ism_txt = ''
savename = 'NOSEL_Hv3_median'
pair = [5,0] # USCRN, HRRR
if varn == 'TempR' or varn == 'TempR2':
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
pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Code/git/Station_Lists/'
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
#df2 = df
station_list = np.unique(df2['uloc']) # List of stations based on unique identifier


if varn == 'Mean' or varn == 'TempR' or varn == 'TempR2':
    df_stat1 = df2.groupby(['uloc']).mean()
    df_stat2 = df2.groupby(['uloc']).mean()
elif varn == 'Std. Dev.':
    df_stat1 = df2.groupby(['uloc']).mean()
    df_stat2 = df2.groupby(['uloc']).std()

# Calculate mean values for all remaining stations
df_stat2['TempRivh'] = np.zeros(len(df_stat2))
df_stat2['TempRivc'] = np.zeros(len(df_stat2))
df_stat2['TempRhvc'] = np.zeros(len(df_stat2))

# Grab each station:
temp_r_samp_size = np.zeros(len(station_list))
cnti = 0
for i in np.arange(0,len(station_list)):
    # get data from each station
    cur_stat_data = df2[df2['uloc']==station_list[i]]
               
    # Exclude stations that only have 45 days of overlapping data
    if  len(cur_stat_data) < 45:
        df_stat2['TempRivc'].loc[station_list[i]] = np.nan
        df_stat2['TempRivh'].loc[station_list[i]] = np.nan
        df_stat2['TempRivc'].loc[station_list[i]] = np.nan
        continue
                
    df_stat2['TempRhvc'].loc[station_list[i]] = np.corrcoef(cur_stat_data['hrrr'],cur_stat_data['cpc'])[0,1]
    df_stat2['TempRivh'].loc[station_list[i]] = np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1]
    df_stat2['TempRivc'].loc[station_list[i]] = np.corrcoef(cur_stat_data['insitu'],cur_stat_data['cpc'])[0,1]

    sumcorrstr = str(np.round(df_stat2['TempRhvc'].loc[station_list[i]]+df_stat2['TempRivc'].loc[station_list[i]]+df_stat2['TempRivh'].loc[station_list[i]],2))
    corrstr = str(np.round(df_stat2['TempRhvc'].loc[station_list[i]],2))+str(np.round(df_stat2['TempRivh'].loc[station_list[i]],2))+str(np.round(df_stat2['TempRivc'].loc[station_list[i]],2))
    #fig,ax = plt.subplots(1,1)
    #plt.plot(cur_stat_data['hrrr'],label='HRRR hvc'+str(np.round(df_stat2['TempRhvc'].loc[station_list[i]],2)))
    #plt.plot(cur_stat_data['insitu'],label='insitu ivh'+str(np.round(df_stat2['TempRivh'].loc[station_list[i]],2)))
    #plt.plot(cur_stat_data['cpc'],label='cpc ivc'+str(np.round(df_stat2['TempRivc'].loc[station_list[i]],2)))
    #plt.grid()
    #plt.legend()
    #plt.title(str(station_list[i])+','+str(np.max(cur_stat_data['lat']))+','+str(np.max(cur_stat_data['lon'])))
    #plt.tight_layout()    
    #plt.savefig('/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/WAF/Time_Series/Time_Series_Station_'+sumcorrstr+'_'+str(station_list[i])+'.png')
    #plt.close()

    print(len(cur_stat_data['hrrr']),cur_stat_data['hrrr'].count())
    temp_r_samp_size[i] = cur_stat_data['hrrr'].count()


# Current pandas dataframe data into indidual numpy arrays
hrr_ii = df_stat1['hrrr'].values
cpc_ii = df_stat1['cpc'].values
usc_ii = df_stat1['insitu'].values
lon_pp = df_stat1['lon'].values
lat_pp = df_stat1['lat'].values

# Calculate mean ISM station values for all remaining stations
mean_val = np.nanmean([hrr_ii,usc_ii,cpc_ii],axis=0)        

# Convert statistic data into numpy arrays
hrr_iii = df_stat2['hrrr'].values
cpc_iii = df_stat2['cpc'].values
usc_iii = df_stat2['insitu'].values
ivh_iii = df_stat2['TempRivh'].values
ivc_iii = df_stat2['TempRivc'].values
hvc_iii = df_stat2['TempRhvc'].values

if varn == 'TempR2':
    ivh_iii = np.power(df_stat2['TempRivh'].values,2.0)
    ivc_iii = np.power(df_stat2['TempRivc'].values,2.0)
    hvc_iii = np.power(df_stat2['TempRhvc'].values,2.0)

x_arr = np.arange(50,751,100)

# Estimate percentile bin locations based on mean values
npcts = 20
pcts = np.arange(0,101,npcts)
x_arr = np.zeros(len(pcts))
for i in np.arange(0,len(pcts)):
    x_arr[i] = np.nanpercentile(mean_val,pcts[i])

# Calculte Difference results based
avg_arr = np.zeros((3,len(x_arr)))
med_arr = np.zeros((3,len(x_arr)))
std_arr = np.zeros((3,len(x_arr)))
cnt_arr = np.zeros((3,len(x_arr)))
tstat = np.zeros((3,len(x_arr)))
pval = np.zeros((3,len(x_arr)))
# Loop through the 5 quintiles
for j in np.arange(0,len(x_arr)-1):
    # Loop through the three different comparisons
    for i in np.arange(0,3):
        
        if varn == 'TempR' or varn == 'TempR2':
            if i == 0: 
                v3_iip = ivh_iii[mean_val >= x_arr[j]]
            elif i == 2:
                v3_iip = hvc_iii[mean_val >= x_arr[j]]
            elif i == 1:
                v3_iip = ivc_iii[mean_val >= x_arr[j]]
            mva_iip = mean_val[mean_val >= x_arr[j]]
            v3_iip = v3_iip[mva_iip < x_arr[j+1]]
        
            avg_arr[i,j] = np.nanmean(v3_iip)
            med_arr[i,j] = np.nanmedian(v3_iip)
            cnt_arr[i,j] = len(v3_iip)
            pval[i,j] = 1.0
        else:        
            if i == 0: 
                v1_iip = hrr_iii[mean_val >= x_arr[j]]
                v2_iip = usc_iii[mean_val >= x_arr[j]]
            elif i == 2:
                v1_iip = hrr_iii[mean_val >= x_arr[j]]
                v2_iip = cpc_iii[mean_val >= x_arr[j]]
            elif i == 1:
                v1_iip = cpc_iii[mean_val >= x_arr[j]]
                v2_iip = usc_iii[mean_val >= x_arr[j]]
     
            mva_iip = mean_val[mean_val >= x_arr[j]]
            v1_iip = v1_iip[mva_iip < x_arr[j+1]]
            v2_iip = v2_iip[mva_iip < x_arr[j+1]]
    
            # Calculate student's t-test
            [tstat[i,j],pval[i,j]] = stats.ttest_rel(v1_iip,v2_iip,nan_policy='raise')       
        
            # Calculate mean, median, standard deviation and length of difference array
            if varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
                avg_arr[i,j] = np.nanmean(np.abs(v1_iip-v2_iip))
                med_arr[i,j] = np.nanmedian(np.abs(v1_iip-v2_iip))
            else:
                avg_arr[i,j] = np.nanmean(v1_iip-v2_iip)
                med_arr[i,j] = np.nanmedian(v1_iip-v2_iip)
            std_arr[i,j] = np.nanstd(v1_iip-v2_iip)
            cnt_arr[i,j] = len(v1_iip-v2_iip)


# Save indicies of the different quintiles in the data
qids = OrderedDict()
for j in np.arange(0,len(x_arr)-1):
       qids[j] = np.where((mean_val >= x_arr[j]) & (mean_val < x_arr[j+1]))[0]
       print(len(qids[j]))


#################
###### PLOT TIME
################


# Set font sizes
plt.rcParams.update({'font.size': 14})
tfs = 16 # title fontsize

# Different Axes for Different Statistics
if (varn2 == 'MAD') and (varn == 'Mean'):
    ytix = [0,125,250,375,500]
    ylims = [0,500]
    vvals = [0,300]
elif varn == 'Mean':
    ytix = [-500,-250,0,250,500]
    ylims = [-500,500]
    vvals = [-250,250]
elif (varn2 == 'MAD') and (varn == 'Std. Dev.'):
    ytix = [0,50,100,150]
    ylims = [0,150]
    vvals = [0,100]
elif varn == 'Std. Dev.':
    ytix = [-150,-75,0,75]
    ylims = [-150,100]
    vvals = [-75,75]
elif varn == 'TempR2':
    ytix = np.arange(0,1.01,0.25)
    ylims = [0,1]
    vvals = [.25,.75]
elif varn == 'TempR':
    ytix = np.arange(-0.25,1.01,0.25)
    ylims = [-0.25,1]
    vvals = [0,.8]



xlims = [75,625]
xtix = [150,250,350,450,550]
    
c1 = 'darkviolet'
lw1 = 2
ls1 = ':'

c2 = 'black'
lw2 = 3
ls2 = ':'
ss = 40;

# no statistical significant if using mean absolute deviations
if varn2 == 'MAD':
    c1 = 'black'

# Symbols for the different quintiles
syms = ['v','s','o','D','^']


# 3 row by 2 col figure
fig = plt.figure(figsize=[14,15])
gs = GridSpec(3, 2, figure=fig)

# Make map projection axes
axm = OrderedDict()
axm[0] = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
axm[1] = fig.add_subplot(gs[1,0], projection=ccrs.PlateCarree())
axm[2] = fig.add_subplot(gs[2,0], projection=ccrs.PlateCarree())

states110 = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='110m',
            facecolor='none',
            edgecolor='k')


# Make other subpanel axes
ax = OrderedDict()
ax[0] = fig.add_subplot(gs[0,1])
ax[1] = fig.add_subplot(gs[1,1])
ax[2] = fig.add_subplot(gs[2,1])

if varn == 'TempR' or varn == 'TempR2':
    axmtit = ['(a) Model (HRRR) vs. In Situ',
             '(c) Model (CPC) vs. In Situ',
             '(e) Model (HRRR) vs. Model (CPC)']
    
    
    axtit = ['(b) Model (HRRR) vs. In Situ',
             '(d) Model (CPC) vs. In Situ',
             '(f) Model (HRRR) vs. Model (CPC)']
else: 
    axmtit = ['(a) Model (HRRR) - In Situ',
              '(c) Model (CPC) - In Situ',
              '(e) Model (HRRR) - Model (CPC)']


    axtit = ['(b) Model (HRRR) - In Situ',
             '(d) Model (CPC) - In Situ',
             '(f) Model (HRRR) - Model (CPC)']

pdata = OrderedDict()
pdata[0] = [hrr_iii,usc_iii]
pdata[1] = [cpc_iii,usc_iii]
pdata[2] = [hrr_iii,cpc_iii]

tdata = OrderedDict()
tdata[0] = ivh_iii
tdata[1] = ivc_iii
tdata[2] = hvc_iii
#### PLOTS for HRRR versus INSITU
for p in np.arange(0,3):

    if varn == 'TempR' or varn == 'TempR2':
        avg = np.nanmean(tdata[p])
        med = np.nanmedian(tdata[p])
        pval_n = 1.0
    elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
        avg = np.nanmean(np.abs(pdata[p][0]-pdata[p][1]))
        med = np.nanmedian(np.abs(pdata[p][0]-pdata[p][1]))
        [tstat_n,pval_n] = stats.ttest_rel(pdata[p][0],pdata[p][1],nan_policy='raise')       
    else:
        avg = np.nanmean(pdata[p][0]-pdata[p][1])
        med = np.nanmedian(pdata[p][0]-pdata[p][1])
        [tstat_n,pval_n] = stats.ttest_rel(pdata[p][0],pdata[p][1],nan_policy='raise')       

    # Make Map Plots
    #am = axm[0].scatter(lon_pp,lat_pp,c=(hrr_ii-usc_ii)[subs],s=20,edgecolor='k',transform = ccrs.PlateCarree(),vmin=-300,vmax=300,cmap=cmap_sm_r,zorder=2)
    for i in np.arange(0,5):
        if varn == 'TempR' or varn == 'TempR2':
            am = axm[p].scatter(lon_pp[qids[i]],lat_pp[qids[i]],c=tdata[p][qids[i]],marker=syms[i],s=ss,edgecolor='k',transform = ccrs.PlateCarree(),vmin=vvals[0],vmax=vvals[1],cmap=cmap_sm_r,zorder=2)
        elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
            am = axm[p].scatter(lon_pp[qids[i]],lat_pp[qids[i]],c=np.abs((pdata[p][0]-pdata[p][1])[qids[i]]),marker=syms[i],s=ss,edgecolor='k',transform = ccrs.PlateCarree(),vmin=vvals[0],vmax=vvals[1],cmap=cmap_sm_rmad,zorder=2)
        else:
            am = axm[p].scatter(lon_pp[qids[i]],lat_pp[qids[i]],c=(pdata[p][0]-pdata[p][1])[qids[i]],marker=syms[i],s=ss,edgecolor='k',transform = ccrs.PlateCarree(),vmin=vvals[0],vmax=vvals[1],cmap=cmap_sm_r,zorder=2)
    bm = plt.colorbar(am,ax=axm[p])
    if varn == 'TempR':
        bm.ax.set_ylabel('1.6m ISM \n Temporal Correlation (r)')
    elif varn == 'TempR2':
        bm.ax.set_ylabel('1.6m ISM \n Temporal Correlation (r$^{2}$)')
    elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
        bm.ax.set_ylabel('Absolute Difference (mm) \n in 1.6m ISM '+varn)
    else:
        bm.ax.set_ylabel('Difference (mm) in \n 1.6m ISM '+varn)
    axm[p].set_title(axmtit[p],fontsize=tfs)

    # Make Quintile Plots
    #a = ax[0].scatter(mean_val[subs],(hrr_ii-usc_ii)[subs],s=ss,c=lon_pp[subs],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
    for i in np.arange(0,5):
        if varn == 'TempR' or varn == 'TempR2':
            a = ax[p].scatter(mean_val[qids[i]],tdata[p][qids[i]],marker=syms[i],s=ss,c=lon_pp[qids[i]],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
        elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
            a = ax[p].scatter(mean_val[qids[i]],np.abs((pdata[p][0]-pdata[p][1])[qids[i]]),marker=syms[i],s=ss,c=lon_pp[qids[i]],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
        else:            
            a = ax[p].scatter(mean_val[qids[i]],(pdata[p][0]-pdata[p][1])[qids[i]],marker=syms[i],s=ss,c=lon_pp[qids[i]],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
    b = plt.colorbar(a,ax=ax[p])
    b.ax.set_ylabel('Longitude')

    # Add entire dataset mean,median lines
    if pval_n <= pv:
        a = ax[p].plot(xlims,[avg, avg],c=c1,lw=lw1,ls='-')
        a = ax[p].plot(xlims,[med, med],c=c1,lw=lw1,ls=':')
    else:
        a = ax[p].plot(xlims,[avg, avg],c=c2,lw=lw1,ls='-')
        a = ax[p].plot(xlims,[med, med],c=c2,lw=lw1,ls=':')

    # Quintile Plot Specifications
    if varn == 'TempR2':
        ax[p].set_ylabel('1.6m ISM \n Temporal Correlation (r$^{2}$)')
    elif varn == 'TempR':
        ax[p].set_ylabel('1.6m ISM \n Temporal Correlation (r)')
    elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
        ax[p].set_ylabel('Absolute Difference (mm) \n in 1.6m ISM '+varn)
    else:
        ax[p].set_ylabel('Difference (mm) in \n 1.6m ISM '+varn)
    ax[p].set_yticks(ytix)
    ax[p].set_xticks(xtix)
    ax[p].set_title(axtit[p],fontsize=tfs)
    ax[p].set_xlim(xlims)
    ax[p].set_ylim(ylims)
    ax[p].grid()
    ax[p].set_xlabel('1.6m ISM Mean (mm)')

    # Add quintile lines
    for j in np.arange(0,len(x_arr)-1):
        if pval[p,j] <= pv:
            a = ax[p].plot([x_arr[j],x_arr[j+1]],[avg_arr[p,j],avg_arr[p,j]],ls='-',color=c1,lw=lw2)
            a = ax[p].plot([x_arr[j],x_arr[j+1]],[med_arr[p,j],med_arr[p,j]],ls=ls2,color=c1,lw=lw2)
        else:
            a = ax[p].plot([x_arr[j],x_arr[j+1]],[avg_arr[p,j],avg_arr[p,j]],ls='-',color=c2,lw=lw2)
            a = ax[p].plot([x_arr[j],x_arr[j+1]],[med_arr[p,j],med_arr[p,j]],ls=ls2,color=c2,lw=lw2)

    # Map Plot Specifications
    axm[p].set_extent([-128,-65,23,48])
    axm[p].add_feature(cfeature.LAND,facecolor='whitesmoke',zorder=0)
    axm[p].add_feature(cfeature.LAKES,edgecolor='k',facecolor='white',zorder=0)
    axm[p].add_feature(cfeature.BORDERS,zorder=1)
    axm[p].add_feature(states110,zorder=1)
    axm[p].coastlines(resolution='50m', color='black', linewidth=1,zorder=1)
    axm[p].add_feature(cfeature.RIVERS,zorder=1)    
    gl = axm[p].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-120,-110,-100,-90,-80,-70])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

plt.tight_layout()
plt.subplots_adjust(left=0.05)
plt.savefig(savepath+'ISM_'+varn+'_'+str(npcts)+'_SIG'+str(1-pv)+savename+savename2+varn2+'.png')
plt.savefig(savepath+'ISM_'+varn+'_'+str(npcts)+'_SIG'+str(1-pv)+savename+savename2+varn2+'.pdf')
