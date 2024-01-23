#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Code to compare various error thresholds various based on different testing parameters

@author: petermarinescu
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import pickle

savenames = ['180_15_fullE_anom2_1521',
             '180_30_fullE_anom2_1521',
             '180_60_fullE_anom2_1521',
             '180_15_fullE_anom2',
             '180_30_fullE_anom2',
             '180_60_fullE_anom2']

snames = ['7Y-15','7Y-30','7Y-60','3Y-15','3Y-30','3Y-60']
openpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/'

stat = OrderedDict()
rerr = OrderedDict()
slen = OrderedDict()
slat = OrderedDict()
slon = OrderedDict()

for s in np.arange(0,len(savenames)):
    savename = savenames[s]
    filenames = ['USCRN_'+savename+'.p',
             'SCAN_'+savename+'.p']
    
    l_stat = OrderedDict()
    l_rat_err = OrderedDict()
    l_len = OrderedDict()
    l_lat = OrderedDict()
    l_lon = OrderedDict()
    for i in np.arange(0,len(filenames)):
        [l_stat[i], l_rat_err[i], l_len[i], l_lat[i], l_lon[i]] = pickle.load(open(openpath+filenames[i], "rb" ))
    
    l_rat_err[0][l_rat_err[0] <= 0] = np.nan
    l_rat_err[1][l_rat_err[1] <= 0] = np.nan


    stat[snames[s]] = []
    rerr[snames[s]] = []
    slen[snames[s]] = []
    slon[snames[s]] = []
    slat[snames[s]] = []
    
    for i in np.arange(0,2):
        for j in np.arange(0,len(l_stat[i])):
            stat[snames[s]] = np.append(stat[snames[s]],l_stat[i][j])
            rerr[snames[s]] = np.append(rerr[snames[s]],l_rat_err[i][j])
            slen[snames[s]] = np.append(slen[snames[s]],l_len[i][j])
            slon[snames[s]] = np.append(slon[snames[s]],l_lon[i][j])
            slat[snames[s]] = np.append(slat[snames[s]],l_lat[i][j])
    
    
    
for sin in [0,1,3]:
    for thr in [0.03,0.05,0.08,0.1,0.12]:
        sta_g = []; val_g = []; lat_g = []; lon_g = []; len_g = []
        nstat = len(rerr[snames[sin]])
        for i in np.arange(0,nstat):
            if rerr[snames[sin]][i] <= thr:
                sta_g = np.append(sta_g,stat[snames[sin]][i])
                val_g = np.append(val_g,rerr[snames[sin]][i])
                lat_g = np.append(lat_g,slat[snames[sin]][i])
                lon_g = np.append(lon_g,slon[snames[sin]][i])
                len_g = np.append(len_g,slen[snames[sin]][i])
        
        fileout = 'Stat_List_Pass_Auto_'+snames[sin]+'_'+str(thr)+'.p'
        pickle.dump([sta_g,val_g,lat_g,lon_g,len_g],open(openpath+fileout,'wb'))

fig,ax = plt.subplots(1,2,figsize=[10,10],sharey=True)
cs = ['b','r','g']
lbls = ['15day','30day','60day']

for i in np.arange(0,3):
    bins = np.array([0.1,1,3,5,10,20,50,100])
    hist,bins2 = np.histogram(np.log10(rerr[snames[i]]*100),bins=np.log10(bins))
    hist = np.insert(hist,0,0)
    phist = hist/np.sum(hist)
#    phist = hist
    ax[0].step(np.log10(bins),np.cumsum(phist),lw=3,c=cs[i],label=lbls[i]+' ('+str(int(np.nansum(hist)))+')')
    ax[0].set_xticks(np.log10(bins))
    ax[0].set_xticklabels(bins)
ax[0].legend()

for i in np.arange(3,6):
    bins = np.array([0.1,1,3,5,10,20,50,100])
    hist,bins2 = np.histogram(np.log10(rerr[snames[i]]*100),bins=np.log10(bins))
    hist = np.insert(hist,0,0)
    phist = hist/np.sum(hist)
#    phist = hist
    ax[1].step(np.log10(bins),np.cumsum(phist),lw=3,c=cs[i-3],label=lbls[i-3]+' ('+str(int(np.nansum(hist)))+')')
    ax[1].set_xticks(np.log10(bins))
    ax[1].set_xticklabels(bins)
ax[1].legend()


plt.rcParams.update({'font.size': 13})

fig,ax = plt.subplots(1,1,figsize=[9,6],sharey=True)
cs = ['b','r','g']
lbls = ['15day','30day','60day']

for i in np.arange(0,3):
    bins = np.array([0.1,1,3,5,10,20,50,100])
    hist,bins2 = np.histogram(np.log10(rerr[snames[i]]*100),bins=np.log10(bins))
    hist = np.insert(hist,0,0)
    phist = hist/np.sum(hist)
#    phist = hist
    ax.step(np.log10(bins),np.cumsum(phist),lw=3,c=cs[i],label='7YR: '+lbls[i]+' ('+str(int(np.nansum(hist)))+')')
    ax.set_xticks(np.log10(bins))
    ax.set_xticklabels(bins)
ax.legend()

for i in np.arange(3,6):
    bins = np.array([0.1,1,3,5,10,20,50,100])
    hist,bins2 = np.histogram(np.log10(rerr[snames[i]]*100),bins=np.log10(bins))
    hist = np.insert(hist,0,0)
    phist = hist/np.sum(hist)
#    phist = hist
    ax.step(np.log10(bins),np.cumsum(phist),ls=':',lw=3,c=cs[i-3],label='3YR: '+lbls[i-3]+' ('+str(int(np.nansum(hist)))+')')
    ax.set_xticks(np.log10(bins))
    ax.set_xticklabels(bins)
ax.legend(fontsize=10)
ax.set_xlabel('Ratio of Error Variance to ISM Variance')
ax.set_ylabel('Cumulative Relative Frequency')
ax.grid()


fig,ax = plt.subplots(1,1,figsize=[9,6],sharey=True)
cs = ['b','r','g']
lbls = ['15day','30day','60day']

for i in np.arange(0,3):
    bins = np.array([0,10,20,30,40,50,60,70,80,90,100])/100
    hist,bins2 = np.histogram(np.sqrt(rerr[snames[i]]),bins=bins)
    hist = np.insert(hist,0,0)
    phist = hist/np.sum(hist)
#    phist = hist
    ax.step(bins,np.cumsum(phist),lw=3,c=cs[i],label='7YR: '+lbls[i]+' ('+str(int(np.nansum(hist)))+')')
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
ax.legend()

for i in np.arange(3,6):
    bins = np.array([0,10,20,30,40,50,60,70,80,90,100])/100
    hist,bins2 = np.histogram(np.sqrt(rerr[snames[i]]),bins=bins)
    hist = np.insert(hist,0,0)
    phist = hist/np.sum(hist)
#    phist = hist
    ax.step(bins,np.cumsum(phist),ls=':',lw=3,c=cs[i-3],label='3YR: '+lbls[i-3]+' ('+str(int(np.nansum(hist)))+')')
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
ax.legend(fontsize=10)
ax.set_xlabel('Ratio of Error SD to ISM SD')
ax.set_ylabel('Cumulative Relative Frequency')
ax.grid()
