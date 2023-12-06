#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 23:28:30 2021

@author: petermarinescu
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import copy
import os.path
import scipy
import pandas

savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/WAF/RevFigSubmit/'


openpaths = ['/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_lqc/',
            '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_sqc/']

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n*1)

#start_date = datetime(2018,1,1)
#end_date = datetime(2021,1,1)
start_date = datetime(2018,7,12)
end_date = datetime(2020,12,3)
ism_txt = ''
savename = 'NOSEL'
varn = 'TempR' # 'STD or AVG or TempR or TempR2
plt.rcParams.update({'font.size': 13})

pair = [5,0] # USCRN, HRRR
usc_i = []
hrr_i = []
cpc_i = []
lon_i = []
lat_i = []
nstat = []

usc_d = OrderedDict(); hrr_d = OrderedDict(); lon_d= OrderedDict(); lat_d = OrderedDict(); cpc_d = OrderedDict(); nstat_d = OrderedDict()
for d in np.arange(0,12):    
    usc_d[d] = []; hrr_d[d] = []; cpc_d[d] = []; lon_d[d] = []; lat_d[d] = []; nstat_d[d] = []


for t in daterange(start_date,end_date):

    a_date = t
    
    # Convert date to desired datestrs
    YYYY = a_date.strftime("%Y")
    YY = a_date.strftime("%y")
    YY = a_date.strftime("%y")
    MM = a_date.strftime("%m")
    DD = a_date.strftime("%d")
    DOY = a_date.strftime("%j")

#    if MM != '11':
#        continue

    datestr = MM+DD

    # if datestr in datelist:
    #     print('Skip: ',datestr)
    #     continue


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

        nlon = nlon+len(lonn)
    
        usc_i = np.append(usc_i,ism_usc)
        hrr_i = np.append(hrr_i,hrrr_int[ism_txt])
        cpc_i = np.append(cpc_i,cpc_int)   
        lon_i = np.append(lon_i,lonn)
        lat_i = np.append(lat_i,latn)

        for d in np.arange(0,12):
            if int(MM) == int(d+1):
                usc_d[d] = np.append(usc_d[d],ism_usc)
                hrr_d[d] = np.append(hrr_d[d],hrrr_int[ism_txt])
                cpc_d[d] = np.append(cpc_d[d],cpc_int)
                lon_d[d] = np.append(lon_d[d],lonn)
                lat_d[d] = np.append(lat_d[d],latn)
                nstat_d[d] = np.append(nstat_d[d],nlon)    
    
    if nlon == 0:
        nlon = np.nan
    nstat = np.append(nstat,nlon)

hrr_ii = OrderedDict()
usc_ii = OrderedDict()
cpc_ii = OrderedDict()
lat_ii = OrderedDict()
lon_ii = OrderedDict()
hrr_iii = OrderedDict()
usc_iii = OrderedDict()
cpc_iii = OrderedDict()
lat_iii = OrderedDict()
lon_iii = OrderedDict()
temprivh_ii = OrderedDict()
temprivc_ii = OrderedDict()
temprhvc_ii = OrderedDict()
for d in np.arange(0,12):
    loc = (lon_d[d]*100+lat_d[d])

    dm = {'insitu': usc_d[d],
            'hrrr': hrr_d[d],
            'cpc': cpc_d[d],
            'lon': lon_d[d],
            'lat': lat_d[d],
            'uloc': loc}
    
    df = pandas.DataFrame(data = dm)


    savename2 = '3Y-15_0.08'
    pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/'
    pickfile = 'Stat_List_Pass_Auto_'+savename2+'.p'
    
    [sta_g,val_g,lat_g,lon_g,len_g] = pickle.load(open(pickpath+pickfile,"rb"))
    
    uloc_g = np.zeros(len(sta_g))
    for i in np.arange(0,len(sta_g)):
        uloc_g[i] = lon_g[i]*100 + lat_g[i]
    
    df2 = df[df['uloc'].isin(uloc_g)]
    df = copy.deepcopy(df2)

    ndf = df.groupby('uloc').nunique()

    if varn == 'AVG' or varn == 'TempR' or varn == 'TempR2':
        df_stat1 = df.groupby(['uloc']).mean()
        df_stat1 = df_stat1[ndf['insitu']>10]

        df_stat2 = df.groupby(['uloc']).mean()
        df_stat2 = df_stat2[ndf['insitu']>10]

    elif varn == 'STD':
        df_stat1 = df.groupby(['uloc']).mean()
        df_stat1 = df_stat1[ndf['insitu']>10]

        df_stat2 = df.groupby(['uloc']).std()
        df_stat2 = df_stat2[ndf['insitu']>10]

    
    hrr_ii[d] = df_stat1['hrrr'].values
    cpc_ii[d] = df_stat1['cpc'].values
    usc_ii[d] = df_stat1['insitu'].values
    lon_ii[d] = df_stat1['lon'].values
    lat_ii[d] = df_stat1['lat'].values

    hrr_iii[d] = df_stat2['hrrr'].values
    cpc_iii[d] = df_stat2['cpc'].values
    usc_iii[d] = df_stat2['insitu'].values
    lon_iii[d] = df_stat2['lon'].values
    lat_iii[d] = df_stat2['lat'].values

    # Include Temporal Correlations
    df_stat2['TempRivh'] = np.zeros(len(df_stat2))
    df_stat2['TempRivc'] = np.zeros(len(df_stat2))
    df_stat2['TempRhvc'] = np.zeros(len(df_stat2))
    station_list = np.unique(df['uloc'])
    #tempR = pd.DataFrame(columns=['stat','lat','lon','season','year','Rivh','Rivc','Rhvc'],index=np.arange(0,len(station_list)*10))
    cnti = 0
    for i in np.arange(0,len(station_list)):
        # get data from each station
        cur_stat_data = df[df['uloc']==station_list[i]]
        # Calculate correlation
        if varn == 'TempR2':
            df_stat2['TempRivh'].loc[station_list[i]] = np.power(np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1],2.0)
            df_stat2['TempRivc'].loc[station_list[i]] = np.power(np.corrcoef(cur_stat_data['insitu'],cur_stat_data['cpc'])[0,1],2.0)
            df_stat2['TempRhvc'].loc[station_list[i]] = np.power(np.corrcoef(cur_stat_data['cpc'],cur_stat_data['hrrr'])[0,1],2.0)
            
        else:
            df_stat2['TempRivh'].loc[station_list[i]] = np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1]
            df_stat2['TempRivc'].loc[station_list[i]] = np.corrcoef(cur_stat_data['insitu'],cur_stat_data['cpc'])[0,1]
            df_stat2['TempRhvc'].loc[station_list[i]] = np.corrcoef(cur_stat_data['cpc'],cur_stat_data['hrrr'])[0,1]
    
    temprivc_ii[d] = df_stat2['TempRivc'].values
    temprivh_ii[d] = df_stat2['TempRivh'].values
    temprhvc_ii[d] = df_stat2['TempRhvc'].values



mean_val = OrderedDict()
for d in np.arange(0,12):
    mean_val[d,0] = np.nanmean([hrr_ii[d],usc_ii[d],cpc_ii[d]],axis=0)        

x_arr = np.arange(50,751,100)

npcts = 20
pcts = np.arange(0,101,npcts)
x_arr = np.zeros((12,len(pcts)))
for d in np.arange(0,12):
    for i in np.arange(0,len(pcts)):
        x_arr[d,i] = np.nanpercentile(mean_val[d,0],pcts[i])

avg_arr = np.zeros((12,3,len(pcts)-1))
med_arr = np.zeros((12,3,len(pcts)-1))
pavg_arr = np.zeros((12,3,len(pcts)-1))
std_arr = np.zeros((12,3,len(pcts)-1))
cnt_arr = np.zeros((12,3,len(pcts)-1))
v1_arr = np.zeros((12,3,len(pcts)-1))
v2_arr = np.zeros((12,3,len(pcts)-1))
tstat = np.zeros((12,3,len(pcts)-1))
pval = np.zeros((12,3,len(pcts)-1))
for d in np.arange(0,12):
    for j in np.arange(0,len(pcts)-1):
        for i in np.arange(0,3):

            if varn == 'AVG':
                if i == 0: 
                    v1_iip = hrr_ii[d][mean_val[d,0] >= x_arr[d,j]]
                    v2_iip = usc_ii[d][mean_val[d,0] >= x_arr[d,j]]
                elif i == 2:
                    v1_iip = hrr_ii[d][mean_val[d,0] >= x_arr[d,j]]
                    v2_iip = cpc_ii[d][mean_val[d,0] >= x_arr[d,j]]
                elif i == 1:
                    v1_iip = cpc_ii[d][mean_val[d,0] >= x_arr[d,j]]
                    v2_iip = usc_ii[d][mean_val[d,0] >= x_arr[d,j]]
     
            if varn == 'STD':
                if i == 0: 
                    v1_iip = hrr_iii[d][mean_val[d,0] >= x_arr[d,j]]
                    v2_iip = usc_iii[d][mean_val[d,0] >= x_arr[d,j]]
                elif i == 2:
                    v1_iip = hrr_iii[d][mean_val[d,0] >= x_arr[d,j]]
                    v2_iip = cpc_iii[d][mean_val[d,0] >= x_arr[d,j]]
                elif i == 1:
                    v1_iip = cpc_iii[d][mean_val[d,0] >= x_arr[d,j]]
                    v2_iip = usc_iii[d][mean_val[d,0] >= x_arr[d,j]]



            if (varn == 'AVG') or (varn == 'STD'):
                mva_iip = mean_val[d,0][mean_val[d,0] >= x_arr[d,j]]
                v1_iip = v1_iip[mva_iip < x_arr[d,j+1]]
                v2_iip = v2_iip[mva_iip < x_arr[d,j+1]]

        
                pavg_arr[d,i,j] = np.nanmean((v1_iip-v2_iip)/v2_iip*100)
    #            avg_arr[d,i,j] = (np.nanmean(v1_iip)-np.nanmean(v2_iip))/np.nanmean(v2_iip)*100
                avg_arr[d,i,j] = np.nanmean((v1_iip-v2_iip))
                med_arr[d,i,j] = np.nanmedian((v1_iip-v2_iip))
                std_arr[d,i,j] = np.nanstd(v1_iip-v2_iip)
                cnt_arr[d,i,j] = len(v1_iip-v2_iip)
                v1_arr[d,i,j] = np.nanmean(v1_iip)
                v2_arr[d,i,j] = np.nanmean(v2_iip)
#                [tstat[d,i,j],pval[d,i,j]] = scipy.stats.ttest_rel(v1_iip,v2_iip,nan_policy='raise')       
    

            if varn == 'TempR' or varn == 'TempR2':
                if i == 0: 
                    v3_iip = temprivh_ii[d][mean_val[d,0] >= x_arr[d,j]]
                elif i == 2:
                    v3_iip = temprhvc_ii[d][mean_val[d,0] >= x_arr[d,j]]
                elif i == 1:
                    v3_iip = temprivc_ii[d][mean_val[d,0] >= x_arr[d,j]]
     
                mva_iip = mean_val[d,0][mean_val[d,0] >= x_arr[d,j]]
                v3_iip = v3_iip[mva_iip < x_arr[d,j+1]]
            
                avg_arr[d,i,j] = np.nanmean(v3_iip)
                med_arr[d,i,j] = np.nanmedian(v3_iip)
                cnt_arr[d,i,j] = len(v3_iip)

                [tstat[d,i,j],pval[d,i,j]] = [999,999] # No statistical significant testing

    
    


cs = ['red','coral','goldenrod','skyblue','blue']
mons = np.arange(1,13)
msx = OrderedDict()
msx[0] = 'v'
msx[2] = 'o'
msx[4] = '^'
ms = 8
lw = 2 # Line Width
lfs = 10 # Legend Font Size
if varn == 'TempR' or varn == 'TempR2':
    tits = ['(a) 1.6 m ISM: HRRR vs. In Situ ','(b) 1.6 m ISM: CPC vs. In Situ','(c) 1.6 m ISM: HRRR vs. CPC']
    xlims = [0.5,12.5]

    if varn == 'TempR':
        ylims = [-0.2,1.0]
        ylbl = 'ISM Temporal Correlation (r)'
    elif varn == 'TempR2':
        ylims = [0.0,0.9]
        ylbl = 'ISM Temporal Correlation (r$^{2}$)'
        
    xtix = np.arange(1,13)
    fig,ax = plt.subplots(3,1,figsize=[5,13])
        
    lss = ['dashed','dashdot','dotted']
    
    cnt1 = 0; 
    for k in np.arange(0,3):
        for j in np.arange(0,5,2):
            if j == 0:
               ax[cnt1].plot(mons,avg_arr[:,k,j],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{0'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
               ax[cnt1].plot(mons,med_arr[:,k,j],c=cs[j],ls=':',lw=lw,marker=msx[j],ms=ms,label='_no_legend_')
            else:
               ax[cnt1].plot(mons,avg_arr[:,k,j],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
               ax[cnt1].plot(mons,med_arr[:,k,j],c=cs[j],ls=':',lw=lw,marker=msx[j],ms=ms,label='_no_legend_')
    
        ax[cnt1].plot([0,13],[0,0],'-k')
        ax[cnt1].set_xlim(xlims)
        ax[cnt1].set_ylim(ylims)
        ax[cnt1].set_xticks(xtix)
        ax[cnt1].set_title(tits[cnt1])
        ax[cnt1].legend(ncol=3,loc='lower left',fontsize=lfs+1)
        ax[cnt1].grid()
        if cnt1 == 2:
            ax[cnt1].set_xlabel('Month')
    
    #    if k == 0 or k == 2:
     #       ax[cnt1].set_ylabel('Model - In-Situ \n Mean 1.6m ISM Difference')
    
        ax[cnt1].set_ylabel(ylbl)
    
        if cnt1 < 2:
            ax[cnt1].set_xticklabels('')
    
    
        cnt1 = cnt1 + 1
    plt.tight_layout()
    plt.savefig(savepath+'MonthSeries_iISM_cLon'+savename+savename2+varn+'.png')
    plt.savefig(savepath+'MonthSeries_iISM_cLon'+savename+savename2+varn+'.pdf')
    
else:
    
    xlims = [0.5,12.5]
    xtix = np.arange(1,13)
    fig,ax = plt.subplots(3,2,figsize=[9,11])

    if varn == 'AVG':
        
        tits = ['(a) Monthly ISM Mean for $L_{00-20}$','(b) Monthly ISM Mean for $L_{40-60}$','(c) Monthly ISM Mean for $L_{80-100}$',
            '(d) HRRR - In Situ Difference','(e) CPC - In Situ Difference','(f) HRRR - CPC Difference']
        ylims0 = [50,300]  
        ylims1 = [200,500]  
        ylims2 = [300,800]  
        ylims = [-300,200]  
        ylbl = 'ISM Mean (mm)'
    
    elif varn == 'STD':
    
        tits = ['(a) $L_{00-20}$ Monthly ISM Std. Dev.','(b) $L_{40-60}$ Monthly ISM Std. Dev.','(c) $L_{80-100}$ Monthly ISM Std. Dev.',
            '(d) HRRR - In Situ Difference','(e) CPC - In Situ Difference','(f) HRRR - CPC Difference']
        ylims0 = [0,80]  
        ylims1 = [0,80]  
        ylims2 = [0,80]  
        ylims = [-60,20]  
        ylbl = 'ISM Std. Dev. (mm)'
            
    #ylims = [-50,210]
    
    lsx = OrderedDict()
    lsx[0] = ':'
    lsx[1] = '--'
    lsx[2] = '-'
    
    lcs = OrderedDict()
    lcs[0] = 'lightsalmon'
    lcs[1] = 'tomato'
    lcs[2] = 'firebrick'
    
    lss = ['dashed','dashdot','dotted']
    k = 0
    cnt1 = 0; cnt2 = 0; 
    j = 0
    #ax[cnt1,cnt2].plot(mons,v1_arr[:,0,j],ls=lsx[0],lw=lw,c=cs[j],marker=msx[0],ms=ms,label='HRRR ISM')
    #ax[cnt1,cnt2].plot(mons,v1_arr[:,1,j],ls=lsx[1],lw=lw,c=cs[j],marker=msx[0],ms=ms,label='CPC ISM')
    #ax[cnt1,cnt2].plot(mons,v2_arr[:,1,j],ls=lsx[2],lw=lw,c=cs[j],marker=msx[0],ms=ms,label='In Situ ISM')
    ax[cnt1,cnt2].plot(mons,v1_arr[:,0,j],ls=lsx[0],lw=lw,c=lcs[0],marker=msx[0],ms=ms,label='HRRR')
    ax[cnt1,cnt2].plot(mons,v1_arr[:,1,j],ls=lsx[1],lw=lw,c=lcs[1],marker=msx[0],ms=ms,label='CPC')
    ax[cnt1,cnt2].plot(mons,v2_arr[:,1,j],ls=lsx[2],lw=lw,c=lcs[2],marker=msx[0],ms=ms,label='In Situ')
    ax[cnt1,cnt2].plot([0,13],[0,0],'-k')
    ax[cnt1,cnt2].legend(ncol=3,fontsize=lfs)
    ax[cnt1,cnt2].set_xlim(xlims)
    ax[cnt1,cnt2].set_ylim(ylims0)
    ax[cnt1,cnt2].set_xticks(xtix)
    ax[cnt1,cnt2].set_title(tits[k])
    ax[cnt1,cnt2].set_ylabel(ylbl)
    ax[cnt1,cnt2].set_xlabel('Month')
    ax[cnt1,cnt2].grid()
    
    
    lcs = OrderedDict()
    lcs[0] = 'tan'
    lcs[1] = 'orange'
    lcs[2] = 'darkgoldenrod'
    
    cnt1 = 1; cnt2 = 0; 
    k = 1; j = 2
    #ax[cnt1,cnt2].plot(mons,v1_arr[:,0,j],ls=lsx[0],lw=lw,c=cs[j],marker=msx[2],ms=ms,label='HRRR ISM')
    #ax[cnt1,cnt2].plot(mons,v1_arr[:,1,j],ls=lsx[1],lw=lw,c=cs[j],marker=msx[2],ms=ms,label='CPC ISM')
    #ax[cnt1,cnt2].plot(mons,v2_arr[:,1,j],ls=lsx[2],lw=lw,c=cs[j],marker=msx[2],ms=ms,label='In Situ ISM')
    ax[cnt1,cnt2].plot(mons,v1_arr[:,0,j],ls=lsx[0],lw=lw,c=lcs[0],marker=msx[2],ms=ms,label='HRRR')
    ax[cnt1,cnt2].plot(mons,v1_arr[:,1,j],ls=lsx[1],lw=lw,c=lcs[1],marker=msx[2],ms=ms,label='CPC')
    ax[cnt1,cnt2].plot(mons,v2_arr[:,1,j],ls=lsx[2],lw=lw,c=lcs[2],marker=msx[2],ms=ms,label='In Situ')
    ax[cnt1,cnt2].plot([0,13],[0,0],'-k')
    ax[cnt1,cnt2].legend(ncol=3,loc='lower left',fontsize=lfs)
    ax[cnt1,cnt2].set_xlim(xlims)
    ax[cnt1,cnt2].set_ylim(ylims1)
    ax[cnt1,cnt2].set_xticks(xtix)
    ax[cnt1,cnt2].set_title(tits[k])
    ax[cnt1,cnt2].set_ylabel(ylbl)
    ax[cnt1,cnt2].set_xlabel('Month')
    ax[cnt1,cnt2].grid()
    
    
    lcs = OrderedDict()
    lcs[0] = 'dodgerblue'
    lcs[1] = 'royalblue'
    lcs[2] = 'darkblue'
    
    
    cnt1 = 2; cnt2 = 0; 
    k = 2; j = 4
    #ax[cnt1,cnt2].plot(mons,v1_arr[:,0,j],ls=lsx[0],lw=lw,c=cs[j],marker=msx[4],ms=ms,label='HRRR ISM')
    #ax[cnt1,cnt2].plot(mons,v1_arr[:,1,j],ls=lsx[1],lw=lw,c=cs[j],marker=msx[4],ms=ms,label='CPC ISM')
    #ax[cnt1,cnt2].plot(mons,v2_arr[:,1,j],ls=lsx[2],lw=lw,c=cs[j],marker=msx[4],ms=ms,label='In Situ ISM')
    ax[cnt1,cnt2].plot(mons,v1_arr[:,0,j],ls=lsx[0],lw=lw,c=lcs[0],marker=msx[4],ms=ms,label='HRRR')
    ax[cnt1,cnt2].plot(mons,v1_arr[:,1,j],ls=lsx[1],lw=lw,c=lcs[1],marker=msx[4],ms=ms,label='CPC')
    ax[cnt1,cnt2].plot(mons,v2_arr[:,1,j],ls=lsx[2],lw=lw,c=lcs[2],marker=msx[4],ms=ms,label='In Situ')
    ax[cnt1,cnt2].plot([0,13],[0,0],'-k')
    ax[cnt1,cnt2].legend(ncol=3,loc='lower left',fontsize=lfs)
    ax[cnt1,cnt2].set_xlim(xlims)
    ax[cnt1,cnt2].set_ylim(ylims2)
    ax[cnt1,cnt2].set_xticks(xtix)
    ax[cnt1,cnt2].set_title(tits[k])
    ax[cnt1,cnt2].set_ylabel(ylbl)
    ax[cnt1,cnt2].set_xlabel('Month')
    ax[cnt1,cnt2].grid()
    
    
    cnt1 = 0; cnt2 = 1; 
    for k in np.arange(0,3):
        for j in np.arange(0,5,2):
    #        if j == 0 or j == 4:
    #            lw = 4
    #            lw = 2
    #        else:
    #            lw = 2
    #        ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],marker='o',label='VSM: '+str(np.round(x_arr[j],1))+'-'+str(np.round(x_arr[j+1],1)))
    #        ax[cnt1,cnt2].plot(mons,avg_arr[:,k,j],c=cs[j],lw=lw,marker='x',label='ISM: '+str(np.round(pcts[j],0))+'%-'+str(np.round(pcts[j+1],0))+'%')
             lw = 2
    
             if j == 0:
                ax[cnt1,cnt2].plot(mons,avg_arr[:,k,j],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{0'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
                ax[cnt1,cnt2].plot(mons,med_arr[:,k,j],c=cs[j],lw=lw,ls=':',marker=msx[j],ms=ms,label='_no_legend_')
             else:
                ax[cnt1,cnt2].plot(mons,avg_arr[:,k,j],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
                ax[cnt1,cnt2].plot(mons,med_arr[:,k,j],c=cs[j],lw=lw,ls=':',marker=msx[j],ms=ms,label='_no_legend_')
                   
            
        ax[cnt1,cnt2].plot([0,13],[0,0],'-k')
        ax[cnt1,cnt2].legend(ncol=3,fontsize=lfs)
        ax[cnt1,cnt2].set_xlim(xlims)
        ax[cnt1,cnt2].set_ylim(ylims)
        ax[cnt1,cnt2].set_xticks(xtix)
        ax[cnt1,cnt2].set_title(tits[k+3])
        ax[cnt1,cnt2].grid()
        ax[cnt1,cnt2].set_ylabel('Diff. in '+ylbl)
        ax[cnt1,cnt2].set_xlabel('Month')
    
    #    if k == 2 or k == 3:
    #        ax[cnt1,cnt2].set_xlabel('Month')
    
    #    if k == 0 or k == 2:
    #        ax[cnt1,cnt2].set_ylabel('% Change')
    
    #    if k == 0 or k == 1:
    #        ax[cnt1,cnt2].set_xticklabels('')
    
        cnt1 = cnt1 + 1
    
    plt.tight_layout()
    plt.savefig(savepath+'MonthSeries_iISM_DiffxMeancLon'+savename+savename2+varn+'.png')
    plt.savefig(savepath+'MonthSeries_iISM_DiffxMeancLon'+savename+savename2+varn+'.pdf')
    
    
    
    
    
    
    
