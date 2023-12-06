#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 23:28:30 2021

@author: petermarinescu
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import pickle
import copy
import os.path
import pandas

openpaths = ['/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_lqc/',
            '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_sqc/']
            
savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/WAF/RevFigSubmit/'
  

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n*1)

start_date = datetime(2018,7,12)
end_date = datetime(2020,12,3)
varn = 'TempR' # 'AVG' or 'STD' or 'TempR' or 'TempR2'

#start_date = datetime(2018,1,1)
#end_date = datetime(2021,1,1)
ism_txt = ''
savename = 'NOSEL_Hv3'
savename = '2018-2020'

#start_date = datetime(2018,12,1)
#end_date = datetime(2019,12,1)
#start_date = datetime(2018,1,1)
#end_date = datetime(2021,1,1)
#ism_txt = ''
#savename = 'NOSEL_Hv3'
#savename = '2019'

#start_date = datetime(2019,12,1)
#end_date = datetime(2020,12,1)
#start_date = datetime(2018,1,1)
#end_date = datetime(2021,1,1)
#ism_txt = ''
#savename = 'NOSEL_Hv3'
#savename = '2020'


plt.rcParams.update({'font.size': 13})
fs = 10

pairs = [[5,0],
        [5,5],
        [10,10],
        [100,100]]# USCRN, HRRR

usc_p = OrderedDict(); hrr_p = OrderedDict(); lon_p= OrderedDict(); lat_p = OrderedDict()
usc_d = OrderedDict(); hrr_d = OrderedDict(); lon_d= OrderedDict(); lat_d = OrderedDict()
for p in np.arange(0,len(pairs)):
    usc_p[p] = []; hrr_p[p] = []; lat_p[p] = []; lon_p[p] = [] 
    for d in np.arange(0,12):    
        usc_d[p,d] = []; hrr_d[p,d] = []; lon_d[p,d] = []; lat_d[p,d] = []


for p in np.arange(0,len(pairs)):
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
             
                latn = lat['v'+str(pairs[p][0])]
                lonn = lon['v'+str(pairs[p][0])]                
                lat_o = copy.deepcopy(latn)
                #Eliminate data from Selma, AL station (often very high VSM values (>0.6) at 100 cm depth)   
                usc[str(pairs[p][0])] = usc[str(pairs[p][0])][(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                vsm[pairs[p][1]] = vsm[pairs[p][1]][(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
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
                latn = lat['v'+str(pairs[p][0])]
                lonn = lon['v'+str(pairs[p][0])]
                                
    
            uscn = copy.deepcopy(usc[str(pairs[p][0])])
            vsmn = copy.deepcopy(vsm[pairs[p][1]])
            
            vsmn = vsmn[uscn>0]
            lonn = lonn[uscn>0]
            latn = latn[uscn>0]
            uscn = uscn[uscn>0]
    
            usc_p[p] = np.append(usc_p[p],uscn)
            hrr_p[p] = np.append(hrr_p[p],vsmn)
            lon_p[p] = np.append(lon_p[p],lonn)
            lat_p[p] = np.append(lat_p[p],latn)
    
            for d in np.arange(0,12):
                if int(MM) == int(d+1):
                    usc_d[p,d] = np.append(usc_d[p,d],uscn)
                    hrr_d[p,d] = np.append(hrr_d[p,d],vsmn)
                    lon_d[p,d] = np.append(lon_d[p,d],lonn)
                    lat_d[p,d] = np.append(lat_d[p,d],latn)


# Create recalculated dataframes for each pair that includes station mean data
hrr_ii = OrderedDict()
usc_ii = OrderedDict()
lat_ii = OrderedDict()
lon_ii = OrderedDict()
tempr_ii = OrderedDict()
hrr_iii = OrderedDict()
usc_iii = OrderedDict()
lat_iii = OrderedDict()
lon_iii = OrderedDict()
dfm = OrderedDict()
for p in np.arange(0,len(pairs)):
    for d in np.arange(0,12):
        loc = (lon_d[p,d]*100+lat_d[p,d])
    
        dm = {'insitu': usc_d[p,d],
                'hrrr': hrr_d[p,d],
                'lon': lon_d[p,d],
                'lat': lat_d[p,d],
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
        df_stat1 = df.groupby(['uloc']).mean()
        hrr_ii[p,d] = df_stat1['hrrr'].values
        usc_ii[p,d] = df_stat1['insitu'].values
        lon_ii[p,d] = df_stat1['lon'].values
        lat_ii[p,d] = df_stat1['lat'].values

        if varn == 'AVG' or varn == 'TempR' or varn == 'TempR2':        
            df_stat2 = df.groupby(['uloc']).mean()
        elif varn == 'STD':
            df_stat2 = df.groupby(['uloc']).std()
        hrr_iii[p,d] = df_stat2['hrrr'].values
        usc_iii[p,d] = df_stat2['insitu'].values
        lon_iii[p,d] = df_stat2['lon'].values
        lat_iii[p,d] = df_stat2['lat'].values
        
        # Include Temporal Correlations
        df_stat2['TempR'] = np.zeros(len(df_stat2))
        station_list = np.unique(df['uloc'])
        #tempR = pd.DataFrame(columns=['stat','lat','lon','season','year','Rivh','Rivc','Rhvc'],index=np.arange(0,len(station_list)*10))
        cnti = 0
        for i in np.arange(0,len(station_list)):
            # get data from each station
            cur_stat_data = df[df['uloc']==station_list[i]]
            # Calculate correlation
            if varn == 'TempR2':
                df_stat2['TempR'].loc[station_list[i]] = np.power(np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1],2.0)
            else:
                df_stat2['TempR'].loc[station_list[i]] = np.corrcoef(cur_stat_data['insitu'],cur_stat_data['hrrr'])[0,1]
        
        tempr_ii[p,d] = df_stat2['TempR'].values



# Calculate mean value for each level and each month
mean_val = OrderedDict()
for p in np.arange(0,len(pairs)):
    for d in np.arange(0,12):
        mean_val[p,d] = np.nanmean([hrr_ii[p,d],usc_ii[p,d]],axis=0)        


pcts = [5,25,50,75,95]
pcts = [0,20,40,60,80,100,0,100]
x_arr = np.zeros((len(pairs),len(pcts),12))
for p in np.arange(0,len(pairs)):
    for d in np.arange(0,12):
        for i in np.arange(0,len(pcts)):
            x_arr[p,i,d] = np.nanpercentile(mean_val[p,d],pcts[i])    

v1_arr = np.zeros((len(pairs),len(pcts)-1,12))
v2_arr = np.zeros((len(pairs),len(pcts)-1,12))
avg_arr = np.zeros((len(pairs),len(pcts)-1,12))
med_arr = np.zeros((len(pairs),len(pcts)-1,12))
std_arr = np.zeros((len(pairs),len(pcts)-1,12))
cnt_arr = np.zeros((len(pairs),len(pcts)-1,12))
for j in np.arange(0,len(pcts)-1):
    for p in np.arange(0,len(pairs)):
        for d in np.arange(0,12):
            
            if varn == 'TempR' or varn == 'TempR2':                
                v3_ip = tempr_ii[p,d][mean_val[p,d] >= x_arr[p,j,d]]
                mva_ip = mean_val[p,d][mean_val[p,d] >= x_arr[p,j,d]]
         
                v3_ip = v3_ip[mva_ip < x_arr[p,j+1,d]]
                
                avg_arr[p,j,d] = np.nanmean(v3_ip)
                med_arr[p,j,d] = np.nanmedian(v3_ip)
                cnt_arr[p,j,d] = len(v3_ip)

            else:
                v1_ip = hrr_iii[p,d][mean_val[p,d] >= x_arr[p,j,d]]
                v2_ip = usc_iii[p,d][mean_val[p,d] >= x_arr[p,j,d]]
    
                mva_ip = mean_val[p,d][mean_val[p,d] >= x_arr[p,j,d]]     
                v1_ip = v1_ip[mva_ip < x_arr[p,j+1,d]]
                v2_ip = v2_ip[mva_ip < x_arr[p,j+1,d]]
                
                print(len(v1_ip),len(v2_ip))
                avg_arr[p,j,d] = np.nanmean(v1_ip-v2_ip)
                med_arr[p,j,d] = np.nanmedian(v1_ip-v2_ip)
                std_arr[p,j,d] = np.nanstd(v1_ip-v2_ip)
                cnt_arr[p,j,d] = len(v1_ip-v2_ip)
                v1_arr[p,j,d] = np.nanmean(v1_ip)
                v2_arr[p,j,d] = np.nanmean(v2_ip)



tits = ['(a) 0 cm (HRRR) - 5 cm (In Situ)','(b) 5 cm: HRRR - In Situ','(c) 10 cm: HRRR - In Situ', '(d) 100 cm: HRRR - In Situ']

cs = ['red','coral','goldenrod','skyblue','blue']
mons = np.arange(1,13)
xlims = [0.5,12.5]
xtix = np.arange(1,13)
msx = OrderedDict()
msx[0] = 'v'
msx[2] = 'o'
msx[4] = '^'
ms = 8
lw = 2 # Line Width
lfs = 10 # Legend Font Size


if varn == 'TempR' or varn == 'TempR2':

    tits = ['(a) 0 cm (HRRR) vs. 5 cm (In Situ)','(b) 5 cm: HRRR vs. In Situ','(c) 10 cm: HRRR vs. In Situ', '(d) 100 cm: HRRR vs. In Situ']

    if varn == 'TempR':
        ylims = [-0.3,0.8]
        ylbl = 'VSM Temporal Correlation (r)'
    elif varn == 'TempR2':
        ylims = [0,0.7]
        ylbl = 'VSM Temporal Correlation (r$^{2}$)'

    fig,ax = plt.subplots(2,2,figsize=[9,7])
    
    cnt1 = 0; cnt2 = 0; 
    for k in np.arange(0,4):
        for j in np.arange(0,5,2):   
            if j == 0:
               ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{0'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
               ax[cnt1,cnt2].plot(mons,med_arr[k,j,:],c=cs[j],lw=lw,ls=':',marker=msx[j],ms=ms,label='_no_legend_')
            else:
               ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
               ax[cnt1,cnt2].plot(mons,med_arr[k,j,:],c=cs[j],lw=lw,ls=':',marker=msx[j],ms=ms,label='_no_legend_')
    
    
        ax[cnt1,cnt2].plot([0,13],[0,0],'-k')
        ax[cnt1,cnt2].legend(ncol=3,loc='lower left',fontsize=lfs)
        ax[cnt1,cnt2].set_xlim(xlims)
        ax[cnt1,cnt2].set_ylim(ylims)
        ax[cnt1,cnt2].set_xticks(xtix)
        ax[cnt1,cnt2].set_title(tits[k])
        ax[cnt1,cnt2].grid()
        if k == 2 or k == 3:
            ax[cnt1,cnt2].set_xlabel('Month')
    
        if k == 0 or k == 2:
            ax[cnt1,cnt2].set_ylabel(ylbl)
    
        if k == 0 or k == 1:
            ax[cnt1,cnt2].set_xticklabels('')
    
        cnt2 = cnt2 + 1
        if cnt2 == 2:
            cnt1 = cnt1 + 1
            cnt2 = 0
    plt.tight_layout()
    plt.savefig(savepath+'MonthSeries_iVSM_'+savename+'_'+varn+'.png')
    plt.savefig(savepath+'MonthSeries_iVSM_'+savename+'_'+varn+'.pdf')    


else:

    fig,ax = plt.subplots(2,2,figsize=[9,7])

    if varn == 'AVG':
        ylims = [-0.3,0.1]
        ylbl = 'VSM Mean Difference'

    elif varn == 'STD':
        ylims = [-0.05,0.04]
        ylbl = 'VSM Std. Dev. Difference'
    
    cnt1 = 0; cnt2 = 0; 
    for k in np.arange(0,4):
        for j in np.arange(0,5,2):
    #        ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],marker='o',label='VSM: '+str(np.round(x_arr[j],1))+'-'+str(np.round(x_arr[j+1],1)))
    #        ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],marker='o',label='VSM: '+str(np.round(pcts[j],0))+'%-'+str(np.round(pcts[j+1],0))+'%')
    
            if j == 0:
               ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{0'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
               ax[cnt1,cnt2].plot(mons,med_arr[k,j,:],c=cs[j],lw=lw,ls=':',marker=msx[j],ms=ms,label='_no_legend_')
            else:
               ax[cnt1,cnt2].plot(mons,avg_arr[k,j,:],c=cs[j],lw=lw,marker=msx[j],ms=ms,label='$L_{'+str(np.round(pcts[j],0))+'-'+str(np.round(pcts[j+1],0))+'}$')
               ax[cnt1,cnt2].plot(mons,med_arr[k,j,:],c=cs[j],lw=lw,ls=':',marker=msx[j],ms=ms,label='_no_legend_')
    
    
        ax[cnt1,cnt2].plot([0,13],[0,0],'-k')
        ax[cnt1,cnt2].legend(ncol=3,fontsize=lfs)
        ax[cnt1,cnt2].set_xlim(xlims)
        ax[cnt1,cnt2].set_ylim(ylims)
        ax[cnt1,cnt2].set_xticks(xtix)
        ax[cnt1,cnt2].set_title(tits[k])
        ax[cnt1,cnt2].grid()
        if k == 2 or k == 3:
            ax[cnt1,cnt2].set_xlabel('Month')
    
        if k == 0 or k == 2:
            ax[cnt1,cnt2].set_ylabel(ylbl)
    
        if k == 0 or k == 1:
            ax[cnt1,cnt2].set_xticklabels('')
    
        cnt2 = cnt2 + 1
        if cnt2 == 2:
            cnt1 = cnt1 + 1
            cnt2 = 0
    plt.tight_layout()
    plt.savefig(savepath+'MonthSeries_iVSM_DiffxMeancLon'+savename+'_'+varn+'.png')
    plt.savefig(savepath+'MonthSeries_iVSM_DiffxMeancLon'+savename+'_'+varn+'.pdf')
