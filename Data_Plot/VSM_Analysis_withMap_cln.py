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
import scipy
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
pv = 0.001 # 99.9% statistical significant
varn = 'Std. Dev.' # 'Mean','Std. Dev.','TempR','TempR2'
varn2 = 'MAD' # 'MAD' or '' # Calculate mean absolute differences
start_date = datetime(2018,7,12) # Begin Date
end_date = datetime(2020,12,3) # End Date
ism_txt = ''
savename = 'NOSEL_Hv3_median'
pair = [5,0] # USCRN, HRRR
if varn == 'TempR' or varn == 'TempR2': 
    cmap_sm_r = plt.cm.YlOrRd

pairs = [[5,0],
        [5,5],
        [10,10],
        [100,100]]# USCRN, HRRR

# Create dictionary items for the indivdual levels
usc_p = OrderedDict(); hrr_p = OrderedDict(); lon_p= OrderedDict(); lat_p = OrderedDict()
for p in np.arange(0,len(pairs)):
    usc_p[p] = []; hrr_p[p] = []; lat_p[p] = []; lon_p[p] = [] 


for p in np.arange(0,len(pairs)):
    # Read in data   
    #insitu ism, hrrr ism, cpc ism, insitu level data, hrr level data, latitude, longitude, unique location identifier
    [usc_i,hrr_i,cpc_i,usc_p[p],hrr_p[p],lon_p[p],lat_p[p],loc] = read_in_data(start_date,end_date,pairs[p],ism_txt,openpaths,'vsm')


# Create recalculated dataframes for each pair that includes station mean data
hrr_ii = OrderedDict()
usc_ii = OrderedDict()
lat_ii = OrderedDict()
lon_ii = OrderedDict()
hrr_iii = OrderedDict()
usc_iii = OrderedDict()
lat_iii = OrderedDict()
lon_iii = OrderedDict()
ivh_iii = OrderedDict()
dfm = OrderedDict()
for p in np.arange(0,len(pairs)):
    loc = (lon_p[p]*100+lat_p[p])

    d = {'insitu': usc_p[p],
            'hrrr': hrr_p[p],
            'lon': lon_p[p],
            'lat': lat_p[p],
            'uloc': loc}
    
    df = pandas.DataFrame(data = d)
    
    # Screen out stations with larger relative error variances
    savename2 = '3Y-15_0.08'
    pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Code/git/Station_Lists/'
    pickfile = 'Stat_List_Pass_Auto_'+savename2+'.p'
    
    [sta_g,val_g,lat_g,lon_g,len_g] = pickle.load(open(pickpath+pickfile,"rb"))
    
    uloc_g = np.zeros(len(sta_g))
    for i in np.arange(0,len(sta_g)):
        uloc_g[i] = lon_g[i]*100 + lat_g[i]
    df2 = df[df['uloc'].isin(uloc_g)]
    # Grab each station:
        
    # for noqc tests
    #df2 = df

    station_list = np.unique(df2['uloc'])

    if varn == 'Mean' or varn == 'TempR' or varn == 'TempR2':
        df_stat1 = df2.groupby(['uloc']).mean()
        df_stat2 = df2.groupby(['uloc']).mean()
    elif varn == 'Std. Dev.':
        df_stat1 = df2.groupby(['uloc']).mean()
        df_stat2 = df2.groupby(['uloc']).std()
    
    # Create array for temporal correlation
    df_stat2['TempRivh'] = np.zeros(len(df_stat2))
    
    temp_r_samp_size = np.zeros((len(pairs),len(station_list)))
    cnti = 0
    for i in np.arange(0,len(station_list)):

        # get data from each station
        cur_stat_data = df2[df2['uloc']==station_list[i]]
        corr_str = str(np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1])

#        fig, ax = plt.subplots(1,1,figsize=[8,5])
#        ax.plot(cur_stat_data['date'],cur_stat_data['insitu'],label='insitu')
#        ax.plot(cur_stat_data['date'],cur_stat_data['hrrr'],label='hrrr')
#        ax.set_xlabel('Time')
#        ax.set_ylabel('Soil Moisture')
#        ax.legend()
#        ax.set_title(str(station_list[i])+' '+str(p)+': '+corr_str)               
#        plt.savefig('/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/ISM_CMP_Plots_Final/Temp_R_TimeSeries/Time_Series_'+str(station_list[i])+'_'+str(p)+'.png')
#        plt.close(fig)
              
        temp_r_samp_size[p,i] = len(cur_stat_data)
        if len(cur_stat_data) < 45:
            df_stat2['TempRivh'].loc[station_list[i]] = np.nan
                
        df_stat2['TempRivh'].loc[station_list[i]] = np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1]
                
        ivh_iii[p] = df_stat2['TempRivh'].values    
        if varn == 'TempR2':
            ivh_iii[p] = np.power(df_stat2['TempRivh'].values,2.0)
    
    
    hrr_ii[p] = df_stat1['hrrr'].values
    usc_ii[p] = df_stat1['insitu'].values
    lon_ii[p] = df_stat1['lon'].values
    lat_ii[p] = df_stat1['lat'].values
    
    hrr_iii[p] = df_stat2['hrrr'].values
    usc_iii[p] = df_stat2['insitu'].values
    lon_iii[p] = df_stat2['lon'].values
    lat_iii[p] = df_stat2['lat'].values


# Calculate mean VSM values for all remaining stations for each depth
mean_val = OrderedDict()
for p in np.arange(0,len(pairs)):
    mean_val[p] = np.nanmean([hrr_ii[p],usc_ii[p]],axis=0)        


# Estimate percentile bin locations based on mean values
npcts = 20
pcts = np.arange(0,101,npcts)
x_arr = np.zeros((4,len(pcts)))
for p in np.arange(0,len(pairs)):
    for i in np.arange(0,len(pcts)):
        x_arr[p,i] = np.nanpercentile(mean_val[p],pcts[i])    


# Calculte Difference results based
avg_arr = np.zeros((len(pairs),len(pcts)))
med_arr = np.zeros((len(pairs),len(pcts)))
std_arr = np.zeros((len(pairs),len(pcts)))
cnt_arr = np.zeros((len(pairs),len(pcts)))
tstat = np.zeros((len(pairs),len(pcts)))
pval = np.zeros((len(pairs),len(pcts)))
for j in np.arange(0,len(pcts)-1):
    for p in np.arange(0,len(pairs)):

        if varn == 'TempR' or varn == 'TempR2':        
            v3_ip = ivh_iii[p][mean_val[p] >= x_arr[p,j]]
            mva_ip = mean_val[p][mean_val[p] >= x_arr[p,j]]
            v3_ip = v3_ip[mva_ip < x_arr[p,j+1]]
        
            avg_arr[p,j] = np.nanmean(v3_ip)
            med_arr[p,j] = np.nanmedian(v3_ip)
            cnt_arr[p,j] = len(v3_ip)            
            [tstat[p,j],pval[p,j]] = [np.nan,np.nan]       
            # Calculate mean, median, standard deviation and length of difference array
        elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
            v1_ip = hrr_iii[p][mean_val[p] >= x_arr[p,j]]
            v2_ip = usc_iii[p][mean_val[p] >= x_arr[p,j]]
            mva_ip = mean_val[p][mean_val[p] >= x_arr[p,j]]

            v1_ip = v1_ip[mva_ip < x_arr[p,j+1]]
            v2_ip = v2_ip[mva_ip < x_arr[p,j+1]]

            avg_arr[p,j] = np.nanmean(np.abs(v1_ip-v2_ip))
            med_arr[p,j] = np.nanmedian(np.abs(v1_ip-v2_ip))
            std_arr[p,j] = np.nanstd(np.abs(v1_ip-v2_ip))
            cnt_arr[p,j] = len(v1_ip-v2_ip)
        else:
            v1_ip = hrr_iii[p][mean_val[p] >= x_arr[p,j]]
            v2_ip = usc_iii[p][mean_val[p] >= x_arr[p,j]]
            mva_ip = mean_val[p][mean_val[p] >= x_arr[p,j]]
     
            v1_ip = v1_ip[mva_ip < x_arr[p,j+1]]
            v2_ip = v2_ip[mva_ip < x_arr[p,j+1]]
        
            avg_arr[p,j] = np.nanmean(v1_ip-v2_ip)
            med_arr[p,j] = np.nanmedian(v1_ip-v2_ip)
            std_arr[p,j] = np.nanstd(v1_ip-v2_ip)
            cnt_arr[p,j] = len(v1_ip-v2_ip)
    
            print(len(v1_ip))
            print(len(v2_ip))
            [tstat[p,j],pval[p,j]] = scipy.stats.ttest_rel(v1_ip,v2_ip,nan_policy='raise')       


# Save indicies of the different quintiles in the data
qids = OrderedDict()
for j in np.arange(0,len(x_arr[0])-1):
    for p in np.arange(0,len(pairs)):
       qids[p,j] = np.where((mean_val[p] >= x_arr[p,j]) & (mean_val[p] < x_arr[p,j+1]))[0]
       print(len(qids[p,j]))


#################
###### PLOT TIME
################


# Set font sizes
plt.rcParams.update({'font.size': 10})
tfs = 12 # title fontsize

if (varn2 == 'MAD') and (varn == 'Mean'):
    ytix = [0,0.1,0.2,0.3]
    ylims = [0,0.3]
    vvals = [0,0.2]
elif varn == 'Mean':
    ytix = [-0.45,-0.3,-0.15,0,0.15,0.3]
    ylims = [-0.45,0.3]
    vvals = [-0.2,0.2]
elif (varn2 == 'MAD') and (varn == 'Std. Dev.'):
    ytix = [0,0.025,0.05,0.075,0.1]
    ylims = [0,0.11]
    vvals = [0,0.075]
elif varn == 'Std. Dev.':
    ytix = [-0.15,-0.1,-0.05,0,0.05,0.1]
    ylims = [-0.15,0.1]
    vvals = [-0.1,0.1]
elif varn == 'TempR':
    ytix = np.arange(-0.25,1.01,0.25)
    ylims = [-0.25,1]
    vvals = [0.0,0.9]
elif varn == 'TempR2':
    ytix = np.arange(0,1.01,0.25)
    ylims = [0,1]
    vvals = [0.25,0.75]
    

xlims = [0,0.45]
xtix = [0.1,0.2,0.3,0.4]
    
c1 = 'darkviolet'
lw1 = 2
ls1 = ':'

c2 = 'black'
lw2 = 3
ls2 = ':'
ss = 25;

# no statistical significant if using mean absolute deviations
if varn2 == 'MAD':
    c1 = 'black'

# Symbols for the different quintiles
syms = ['v','s','o','D','^']


# 3 row by 2 col figure
fig = plt.figure(figsize=[10,18])
gs = GridSpec(4, 2, figure=fig)

# Make map projection axes
axm = OrderedDict()
axm[0] = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
axm[1] = fig.add_subplot(gs[1,0], projection=ccrs.PlateCarree())
axm[2] = fig.add_subplot(gs[2,0], projection=ccrs.PlateCarree())
axm[3] = fig.add_subplot(gs[3,0], projection=ccrs.PlateCarree())

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
ax[3] = fig.add_subplot(gs[3,1])

if varn == 'TempR' or varn == 'TempR2':
    axmtit = ['(a) 0 cm: HRRR vs. 5 cm: In Situ',
             '(c) 5 cm: HRRR vs. In Situ',
             '(e) 10 cm: HRRR vs. In Situ', 
             '(g) 100 cm: HRRR vs. In Situ']
    
    axtit = ['(b) 0 cm: HRRR vs. 5 cm: In Situ',
             '(d) 5 cm: HRRR vs. In Situ',
             '(f) 10 cm: HRRR vs. In Situ', 
             '(h) 100 cm: HRRR vs. In Situ']

else: 
    axmtit = ['(a) 0 cm: HRRR - 5 cm: In Situ',
             '(c) 5 cm: HRRR - In Situ',
             '(e) 10 cm: HRRR - In Situ', 
             '(g) 100 cm: HRRR - In Situ']
    
    
    axtit = ['(b) 0 cm: HRRR - 5 cm: In Situ',
             '(d) 5 cm: HRRR - In Situ',
             '(f) 10 cm: HRRR - In Situ', 
             '(h) 100 cm: HRRR - In Situ']



tstat_n = OrderedDict()
pval_n = OrderedDict()
#### PLOTS for HRRR versus INSITU
for p in np.arange(0,len(pairs)):

    if varn == 'TempR' or varn == 'TempR2':
        avg = np.nanmean(ivh_iii[p]) 
        med = np.nanmedian(ivh_iii[p]) 
        pval_n[p] = 1.0
    elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
        avg = np.nanmean(np.abs(hrr_iii[p]-usc_iii[p]))
        med = np.nanmedian(np.abs(hrr_iii[p]-usc_iii[p]))
        [tstat_n[p],pval_n[p]] = scipy.stats.ttest_rel(hrr_iii[p],usc_iii[p],nan_policy='raise')       
    else:       
        avg = np.nanmean(hrr_iii[p]-usc_iii[p])   
        med = np.nanmedian(hrr_iii[p]-usc_iii[p])   
        [tstat_n[p],pval_n[p]] = scipy.stats.ttest_rel(hrr_iii[p],usc_iii[p],nan_policy='raise')       

    # Make Map Plots
    #am = axm[0].scatter(lon_pp,lat_pp,c=(hrr_ii-usc_ii)[subs],s=20,edgecolor='k',transform = ccrs.PlateCarree(),vmin=-300,vmax=300,cmap=cmap_sm_r,zorder=2)
    for i in np.arange(0,5):
        if varn == 'TempR' or varn == 'TempR2':
            am = axm[p].scatter(lon_ii[p][qids[p,i]],lat_ii[p][qids[p,i]],c=ivh_iii[p][qids[p,i]],marker=syms[i],s=ss,edgecolor='k',transform = ccrs.PlateCarree(),vmin=vvals[0],vmax=vvals[1],cmap=cmap_sm_r,zorder=2)
        elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
            am = axm[p].scatter(lon_ii[p][qids[p,i]],lat_ii[p][qids[p,i]],c=np.abs(hrr_iii[p]-usc_iii[p])[qids[p,i]],marker=syms[i],s=ss,edgecolor='k',transform = ccrs.PlateCarree(),vmin=vvals[0],vmax=vvals[1],cmap=cmap_sm_rmad,zorder=2)
        else:
            am = axm[p].scatter(lon_ii[p][qids[p,i]],lat_ii[p][qids[p,i]],c=(hrr_iii[p]-usc_iii[p])[qids[p,i]],marker=syms[i],s=ss,edgecolor='k',transform = ccrs.PlateCarree(),vmin=vvals[0],vmax=vvals[1],cmap=cmap_sm_r,zorder=2)
    bm = plt.colorbar(am,ax=axm[p])
    bm.ax.set_ylabel('Difference in \n VSM '+varn)
    if varn == 'TempR':
        bm.ax.set_ylabel('VSM Temporal \n Correlation (r)')
    elif varn == 'TempR2':
        bm.ax.set_ylabel('VSM Temporal \n Correlation (r$^{2}$)')
    elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
        bm.ax.set_ylabel('Absolute Difference \n in VSM '+varn)
    else:        
        bm.ax.set_ylabel('Difference in \n VSM '+varn)
    axm[p].set_title(axmtit[p],fontsize=tfs)

    # Make Quintile Plots
    #a = ax[0].scatter(mean_val[subs],(hrr_ii-usc_ii)[subs],s=ss,c=lon_pp[subs],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
    for i in np.arange(0,5):
        if varn == 'TempR' or varn == 'TempR2':
            a = ax[p].scatter(mean_val[p][qids[p,i]],ivh_iii[p][qids[p,i]],marker=syms[i],s=ss,c=lon_ii[p][qids[p,i]],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
        elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
            a = ax[p].scatter(mean_val[p][qids[p,i]],np.abs(hrr_iii[p]-usc_iii[p])[qids[p,i]],marker=syms[i],s=ss,c=lon_ii[p][qids[p,i]],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
        else:            
            a = ax[p].scatter(mean_val[p][qids[p,i]],(hrr_iii[p]-usc_iii[p])[qids[p,i]],marker=syms[i],s=ss,c=lon_ii[p][qids[p,i]],vmin=-120,vmax=-70,cmap=plt.cm.viridis)
    b = plt.colorbar(a,ax=ax[p])
    b.ax.set_ylabel('Longitude')

    # Add entire dataset mean,median lines
    if pval_n[p] <= pv:
        a = ax[p].plot(xlims,[avg, avg],c=c1,lw=lw1,ls='-')
        a = ax[p].plot(xlims,[med, med],c=c1,lw=lw1,ls=':')
    else:
        a = ax[p].plot(xlims,[avg, avg],c=c2,lw=lw1,ls='-')
        a = ax[p].plot(xlims,[med, med],c=c2,lw=lw1,ls=':')

    # Quintile Plot Specifications
    if varn == 'TempR':
        ax[p].set_ylabel('VSM Temporal \n Correlation (r)')
    elif varn == 'TempR2':
        ax[p].set_ylabel('VSM Temporal \n Correlation (r$^{2}$)')
    elif varn2 == 'MAD' and ((varn == 'Mean') or (varn == 'Std. Dev.')):
        ax[p].set_ylabel('Absolute Difference \n in VSM '+varn)
    else:        
        ax[p].set_ylabel('Difference in \n VSM '+varn)
    ax[p].set_yticks(ytix)
    ax[p].set_xticks(xtix)
    ax[p].set_title(axtit[p],fontsize=tfs)
    ax[p].set_xlim(xlims)
    ax[p].set_ylim(ylims)
    ax[p].grid()
    ax[p].set_xlabel('Mean VSM')


    # Add quintile lines
    for j in np.arange(0,len(pcts)-1):
        if pval[p,j] <= pv:
            a = ax[p].plot([x_arr[p,j],x_arr[p,j+1]],[avg_arr[p,j],avg_arr[p,j]],ls='-',color=c1,lw=lw2)
            a = ax[p].plot([x_arr[p,j],x_arr[p,j+1]],[med_arr[p,j],med_arr[p,j]],ls=':',color=c1,lw=lw2)
        else:
            a = ax[p].plot([x_arr[p,j],x_arr[p,j+1]],[avg_arr[p,j],avg_arr[p,j]],ls='-',color=c2,lw=lw2)
            a = ax[p].plot([x_arr[p,j],x_arr[p,j+1]],[med_arr[p,j],med_arr[p,j]],ls=':',color=c2,lw=lw2)


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
plt.savefig(savepath+'VSM_'+varn+'_'+str(npcts)+'_SIG'+str(1-pv)+savename+savename2+varn2+'.png')
plt.savefig(savepath+'VSM_'+varn+'_'+str(npcts)+'_SIG'+str(1-pv)+savename+savename2+varn2+'.pdf')