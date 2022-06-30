#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: petermarinescu
# Code for plotting ISM and VSM Standard Deviations on the same plot

"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import copy
import os.path
from sklearn import linear_model
regr = linear_model.LinearRegression()
import scipy
import pandas

# Function to loop over a data range by a certain number of days
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n*1)

# Define function to get VSM values from individual date files
def get_vsm_std(start_date,end_date,pv,ism_txt,openpaths,savename2):

    # Specificy soil levels of pairs (Insitu, HRRR)
    pairs = [[5,0],
            [5,5],
            [10,10],
            [100,100]]
    
    usc_p = OrderedDict(); hrr_p = OrderedDict(); lon_p= OrderedDict(); lat_p = OrderedDict()
    for p in np.arange(0,len(pairs)):
        usc_p[p] = []; hrr_p[p] = []; lat_p[p] = []; lon_p[p] = [] 
    
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
            datestr = MM+DD

            # Loop through each of two datasets (USCRN and SCAN)        
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
                                    
        
                # Get in situ and hrrr data from each of the pairs of depths for current day
                uscn = copy.deepcopy(usc[str(pairs[p][0])])
                vsmn = copy.deepcopy(vsm[pairs[p][1]])
                
                # Eliminate bad / missing data (negative VSMs)
                vsmn = vsmn[uscn>0]
                lonn = lonn[uscn>0]
                latn = latn[uscn>0]
                uscn = uscn[uscn>0]

                # Append current day data to all day data
                usc_p[p] = np.append(usc_p[p],uscn)
                hrr_p[p] = np.append(hrr_p[p],vsmn)
                lon_p[p] = np.append(lon_p[p],lonn)
                lat_p[p] = np.append(lat_p[p],latn)
        
    
    # Create recalculated dataframes for each pair that includes station mean and standard deviation (2) data
    hrr_ii = OrderedDict()
    usc_ii = OrderedDict()
    lat_ii = OrderedDict()
    lon_ii = OrderedDict()
    hrr2_ii = OrderedDict()
    usc2_ii = OrderedDict()
    lat2_ii = OrderedDict()
    lon2_ii = OrderedDict()
    dfm = OrderedDict()
    # Loop through each set of pairs
    for p in np.arange(0,len(pairs)):
        loc = (lon_p[p]*100+lat_p[p])
    
        d = {'insitu': usc_p[p],
                'hrrr': hrr_p[p],
                'lon': lon_p[p],
                'lat': lat_p[p],
                'uloc': loc}
        
        df = pandas.DataFrame(data = d)
        
        # Screen out stations with larger relative error variances (calculated from separate script)
        #savename2 = '3Y-15_0.08'
        pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/'
        pickfile = 'Stat_List_Pass_Auto_'+savename2+'.p'
        
        [sta_g,val_g,lat_g,lon_g,len_g] = pickle.load(open(pickpath+pickfile,"rb"))
        
        # Create unique identifies for each station based on lon / lat value of each station
        uloc_g = np.zeros(len(sta_g))
        for i in np.arange(0,len(sta_g)):
            uloc_g[i] = lon_g[i]*100 + lat_g[i]
        
        # Only include stations that pass error variance (those present in pickle file)
        df2 = df[df['uloc'].isin(uloc_g)]
        df = copy.deepcopy(df2)

        # Calculate mean for each station
        df_stat = df.groupby(['uloc']).mean()
        
        # Save mean value for each set of pairs (p)
        hrr_ii[p] = df_stat['hrrr'].values
        usc_ii[p] = df_stat['insitu'].values
        lon_ii[p] = df_stat['lon'].values
        lat_ii[p] = df_stat['lat'].values
    
        # Calculate standard deviation for each station
        df_stat2 = df.groupby(['uloc']).std()
        
        # Save std value for each set of pairs (p)
        hrr2_ii[p] = df_stat2['hrrr'].values
        usc2_ii[p] = df_stat2['insitu'].values
        lon2_ii[p] = df_stat2['lon'].values
        lat2_ii[p] = df_stat2['lat'].values

    # Calculate mean VSM values from mean hrrr and mean in situ data for each station
    mean_val = OrderedDict()
    for p in np.arange(0,len(pairs)):
        mean_val[p] = np.nanmean([hrr_ii[p],usc_ii[p]],axis=0)        
    
    # Calculate quintile values based on mean_val (data mean of station means)
    npcts = 20
    pcts = np.arange(0,101,npcts)
    x_arr = np.zeros((4,len(pcts)))
    for p in np.arange(0,len(pairs)):
        for i in np.arange(0,len(pcts)):
            x_arr[p,i] = np.nanpercentile(mean_val[p],pcts[i])    
    
    avg_arr = np.zeros((len(pairs),len(pcts)))
    pdif_arr = np.zeros((len(pairs),len(pcts)))
    std_arr = np.zeros((len(pairs),len(pcts)))
    cnt_arr = np.zeros((len(pairs),len(pcts)))
    tstat = np.zeros((len(pairs),len(pcts)))
    pval = np.zeros((len(pairs),len(pcts)))
    # Loop through 5 quintiles
    for j in np.arange(0,len(pcts)-1):
        # Loop through 4 different soil level comparisons
        for p in np.arange(0,len(pairs)):
    
            # split up std values (xxx2) based on mean soil moisture values
            v1_ip = hrr2_ii[p][mean_val[p] >= x_arr[p,j]]
            v2_ip = usc2_ii[p][mean_val[p] >= x_arr[p,j]]
            mva_ip = mean_val[p][mean_val[p] >= x_arr[p,j]]
     
            v1_ip = v1_ip[mva_ip < x_arr[p,j+1]]
            v2_ip = v2_ip[mva_ip < x_arr[p,j+1]]
        
            avg_arr[p,j] = np.nanmean(v1_ip-v2_ip)
            std_arr[p,j] = np.nanstd(v1_ip-v2_ip)
            cnt_arr[p,j] = len(v1_ip-v2_ip)
            pdif_arr[p,j] = np.nanmean((v1_ip-v2_ip)/v2_ip*100)
    
            #Run paired t-test on station mean data
            [tstat[p,j],pval[p,j]] = scipy.stats.ttest_rel(v1_ip,v2_ip,nan_policy='raise')       
    
    # Similar function to that for the means, but note that the output variables lon2, hrr2, and usc2 are output instead (std values)
    return pairs, lon_ii, hrr2_ii, usc2_ii, mean_val, x_arr, avg_arr, tstat, pval, pcts, pdif_arr
    
# Define function to get ISM values from individual date files
def get_ism_std(start_date,end_date,pv,ism_txt,openpaths,savename2):

    # Random pair of levels for each dataset -- dummy variable and does not impact calculations
    pair = [5,0] # USCRN, HRRR
    
    usc_i = []; hrr_i = []; cpc_i = []
    usc_p = []; hrr_p = []; lon_p = []; lat_p = []; nstat = []
    
    # Pull in data from files for each date
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
    
        nlon = 0
        # Loop through both USCRN and SCAN datasets
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
        
            # Append each days data to arrays
            usc_i = np.append(usc_i,ism_usc)
            hrr_i = np.append(hrr_i,hrrr_int[ism_txt])
            cpc_i = np.append(cpc_i,cpc_int)
        
            usc_p = np.append(usc_p,usc[str(pair[0])])
            hrr_p = np.append(hrr_p,vsm[pair[1]])
            lon_p = np.append(lon_p,lonn)
            lat_p = np.append(lat_p,latn)
    
        if nlon == 0:
            nlon = np.nan
        nstat = np.append(nstat,nlon)
        
    # define unique identified for each station based on its lat/lon position
    loc = (lon_p*100+lat_p)
    
    d = {'insitu': usc_i,
            'hrrr': hrr_i,
            'cpc': cpc_i,
            'lon': lon_p,
            'lat': lat_p,
            'uloc': loc}
    df = pandas.DataFrame(data = d)
       
    # Screen out stations with larger relative error variances
    #savename2 = '3Y-15_0.08'
    pickpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/'
    pickfile = 'Stat_List_Pass_Auto_'+savename2+'.p'
    
    [sta_g,val_g,lat_g,lon_g,len_g] = pickle.load(open(pickpath+pickfile,"rb"))
    
    uloc_g = np.zeros(len(sta_g))
    for i in np.arange(0,len(sta_g)):
        uloc_g[i] = lon_g[i]*100 + lat_g[i]
    
    df2 = df[df['uloc'].isin(uloc_g)]
    
    # Calculate mean values for all remaining stations
    df_stat1 = df2.groupby(['uloc']).mean()
    df_stat2 = df2.groupby(['uloc']).std()
    
    
    hrr_ii = df_stat1['hrrr'].values
    cpc_ii = df_stat1['cpc'].values
    usc_ii = df_stat1['insitu'].values
    lon_pp = df_stat1['lon'].values
    lat_pp = df_stat1['lat'].values

    hrr_iii = df_stat2['hrrr'].values
    cpc_iii = df_stat2['cpc'].values
    usc_iii = df_stat2['insitu'].values
    
    # Calculate mean values for all remaining stations
    mean_val = OrderedDict()
    mean_val[0] = np.nanmean([hrr_ii,usc_ii,cpc_ii],axis=0)        
    mean_val[1] = np.nanmean([hrr_ii,cpc_ii,usc_ii],axis=0)        
    mean_val[2] = np.nanmean([cpc_ii,usc_ii,hrr_ii],axis=0)        
    
    # Estimate percentile bin locations based on mean values
    npcts = 20
    pcts = np.arange(0,101,npcts)
    x_arr = np.zeros(len(pcts))
    for i in np.arange(0,len(pcts)):
        x_arr[i] = np.nanpercentile(mean_val[0],pcts[i])
    
    # Calculte Difference results based
    avg_arr = np.zeros((3,len(x_arr)))
    pavg_arr = np.zeros((3,len(x_arr)))
    std_arr = np.zeros((3,len(x_arr)))
    cnt_arr = np.zeros((3,len(x_arr)))
    tstat = np.zeros((3,len(x_arr)))
    pval = np.zeros((3,len(x_arr)))
    for j in np.arange(0,len(x_arr)-1):
        for i in np.arange(0,3):
            if i == 0: 
                v1_iip = hrr_iii[mean_val[i] >= x_arr[j]]
                v2_iip = usc_iii[mean_val[i] >= x_arr[j]]
            elif i == 2:
                v1_iip = hrr_iii[mean_val[i] >= x_arr[j]]
                v2_iip = cpc_iii[mean_val[i] >= x_arr[j]]
            elif i == 1:
                v1_iip = cpc_iii[mean_val[i] >= x_arr[j]]
                v2_iip = usc_iii[mean_val[i] >= x_arr[j]]
     
            mva_iip = mean_val[i][mean_val[i] >= x_arr[j]]
            v1_iip = v1_iip[mva_iip < x_arr[j+1]]
            v2_iip = v2_iip[mva_iip < x_arr[j+1]]
        
            # Run paired t-test on quintile samples
            [tstat[i,j],pval[i,j]] = scipy.stats.ttest_rel(v1_iip,v2_iip,nan_policy='raise')       
                
            avg_arr[i,j] = np.nanmean(v1_iip-v2_iip)
            pavg_arr[i,j] = np.nanmean((v1_iip-v2_iip)/v2_iip)
            std_arr[i,j] = np.nanstd(v1_iip-v2_iip)
            cnt_arr[i,j] = len(v1_iip-v2_iip)
    
    return lon_pp, hrr_iii, usc_iii, cpc_iii, mean_val, x_arr, avg_arr, tstat, pval

# Specify location of input data files and paths to save files
openpaths = ['/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_lqc/',
            '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat_fin_sqc/']
savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/ISM_CMP_Plots_Final/'
pv = 0.001 # 99.9% statistical significant
start_date = datetime(2018,7,12)
end_date = datetime(2020,12,3)
ism_txt = ''
savename2 = '3Y-15_0.08' #Specify the name of quality control screening

# For Plotting
plt.rcParams.update({'font.size': 10})
savename = 'NOSEL_Hv3'
savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/ISM_CMP_Plots_Final/'

# Call ISM Mean Calculations
[lon_pp, hrr_ii, usc_ii, cpc_ii, mean_val, x_arr, avg_arr, tstat, pval] = get_ism_std(start_date,end_date,pv,ism_txt,openpaths,savename2)


# Plotting Specifications
ytix = [-150,-75,0,75]
ylims = [-150,100]
xlims = [50,650]
xtix = [50,150,250,350,450,550,650]
tits = ['(a) ISM: HRRR - In Situ','(b) ISM: CPC - In Situ','(c) ISM: HRRR - CPC']
subs = np.arange(0,len(lon_pp))
c1 = 'gray'
lw1 = 3
ls1 = ':'
c2 = 'red'
c2 = 'red'
lw2 = 3
ls2 = ':'
ss = 7;
# Integrated Soil Moisture Comparison Scatter
fig,ax = plt.subplots(4,2,figsize=[9,16])

avg = np.nanmean(hrr_ii-usc_ii)
std = np.nanstd(hrr_ii-usc_ii)
[tstat_n,pval_n] = scipy.stats.ttest_rel(hrr_ii,usc_ii,nan_policy='raise')       

a = ax[0,0].scatter(mean_val[0][subs],(hrr_ii-usc_ii)[subs],s=ss,c=lon_pp[subs],vmin=-120,vmax=-70)
#b = plt.colorbar(a,ax=ax[0,0])
#b.ax.set_ylabel('Longitude')
a = ax[0,0].plot([0,800],[0,0],'-k')
if pval_n <= pv:
    a = ax[0,0].plot(xlims,[avg, avg],c=c1,lw=lw1,ls='-')
else:
    a = ax[0,0].plot(xlims,[avg, avg],c=c1,lw=lw1,ls=ls1)#a = ax[0].plot(xlims,[std, std],c='darkviolet')

avg = np.nanmean(cpc_ii-usc_ii)
std = np.nanstd(cpc_ii-usc_ii)
[tstat_n,pval_n] = scipy.stats.ttest_rel(usc_ii,cpc_ii,nan_policy='raise')       
a = ax[1,0].scatter(mean_val[2][subs],(cpc_ii-usc_ii)[subs],s=ss,c=lon_pp[subs],vmin=-120,vmax=-70)
#b = plt.colorbar(a,ax=ax[1,0])
#b.ax.set_ylabel('Longitude')
a = ax[1,0].plot([0,800],[0,0],'-k')
if pval_n <= pv:
    a = ax[1,0].plot(xlims,[avg, avg],c=c1,lw=lw1,ls='-')
else:
    a = ax[1,0].plot(xlims,[avg, avg],c=c1,lw=lw1,ls=ls1)
#a = ax[2].plot(xlims,[std, std],c='darkviolet')

avg = np.nanmean(hrr_ii-cpc_ii)
std = np.nanstd(hrr_ii-cpc_ii)
[tstat_n,pval_n] = scipy.stats.ttest_rel(hrr_ii,cpc_ii,nan_policy='raise')       
a1 = ax[2,0].scatter(mean_val[1][subs],(hrr_ii-cpc_ii)[subs],s=ss,c=lon_pp[subs],vmin=-120,vmax=-70)
#b=plt.colorbar(a,ax=ax[2,0])
#b.ax.set_ylabel('Longitude')
a = ax[2,0].plot([0,800],[0,0],'-k')
if pval_n <= pv:
    a = ax[2,0].plot(xlims,[avg, avg],c=c1,lw=lw1,ls='-')
else:
    a = ax[2,0].plot(xlims,[avg, avg],c=c1,lw=lw1,ls=ls1)#a = ax[1].plot(xlims,[std, std],c='darkviolet')
ax[2,0].set_xlabel('1.6m ISM Mean Value (mm)')

for i in np.arange(0,3):
    ax[i,0].set_ylabel('1.6m ISM \n STD Difference (mm)')
    ax[i,0].set_yticks(ytix)
    ax[i,0].set_xticks(xtix)
    ax[i,0].set_title(tits[i])
    ax[i,0].set_xlim(xlims)
    ax[i,0].set_ylim(ylims)
    ax[i,0].grid()

    for j in np.arange(0,len(x_arr)-1):
        if pval[i,j] <= pv:
            a = ax[i,0].plot([x_arr[j],x_arr[j+1]],[avg_arr[i,j],avg_arr[i,j]],ls='-',color=c2,lw=lw2)
        else:
            a = ax[i,0].plot([x_arr[j],x_arr[j+1]],[avg_arr[i,j],avg_arr[i,j]],ls=ls2,color=c2,lw=lw2)


# Add Colorbar in empty axis
fig.delaxes(ax[3,0])
cax = fig.add_axes([0.11, 0.2, 0.375, 0.01])
b=fig.colorbar(a1,cax=cax, orientation='horizontal')
b.ax.set_xlabel('Longitude')

# Call VSM Mean Calculations
[pairs, lon_ii, hrr_ii, usc_ii, mean_val, x_arr, avg_arr, tstat, pval, pcts, pdif_arr] = get_vsm_std(start_date,end_date,pv,ism_txt,openpaths,savename2)


ytix = [-0.15,-0.1,-0.05,0,0.05,0.1]
xtix = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
ylims = [-0.15,0.1]
xlims = [0.02,0.43]
tits = ['(d) 0-5 cm VSM: HRRR - In Situ','(e) 5 cm VSM: HRRR - In Situ','(f) 10 cm VSM: HRRR - In Situ', '(g) 100 cm VSM: HRRR - In Situ']
#c1 = 'gray'
#lw1 = 3
#ls1 = ':'
#c2 = 'red'
#lw2 = 3
#ls2 = ':'

tstat_n = OrderedDict()
pval_n = OrderedDict()
for p in np.arange(0,len(pairs)):

    subs = np.arange(0,len(lon_ii[p]),1)
    avg = np.nanmean(hrr_ii[p]-usc_ii[p])   
    pdif = np.nanmean((hrr_ii[p]-usc_ii[p])/usc_ii[p]*100)
    print(pdif)
    [tstat_n[p],pval_n[p]] = scipy.stats.ttest_rel(hrr_ii[p],usc_ii[p],nan_policy='raise')       

    a = ax[p,1].scatter(mean_val[p][subs],(hrr_ii[p]-usc_ii[p])[subs],s=7,c=lon_ii[p][subs],vmin=-120,vmax=-70)
    #b = plt.colorbar(a,ax=ax[p,1])
#    b.ax.set_ylabel('Longitude')
    if pval_n[p] <= pv:
        a = ax[p,1].plot(xlims,[avg, avg],c=c1,lw=lw1,ls='-')
    else:
        a = ax[p,1].plot(xlims,[avg, avg],c=c1,lw=lw1,ls=ls1)

    a = ax[p,1].plot(xlims,[0, 0],c='k',lw=1)

    ax[p,1].set_ylabel('VSM STD Difference')
    ax[p,1].set_yticks(ytix)
    ax[p,1].set_xticks(xtix)
    ax[p,1].set_title(tits[p])
    ax[p,1].set_xlim(xlims)
    ax[p,1].set_ylim(ylims)
    ax[p,1].grid()

    for j in np.arange(0,len(pcts)-1):
        if pval[p,j] <= pv:
            a = ax[p,1].plot([x_arr[p,j],x_arr[p,j+1]],[avg_arr[p,j],avg_arr[p,j]],ls='-',color=c2,lw=lw2)
        else:
            a = ax[p,1].plot([x_arr[p,j],x_arr[p,j+1]],[avg_arr[p,j],avg_arr[p,j]],ls=ls2,color=c2,lw=lw2)

ax[p,1].set_xlabel('Mean VSM')
plt.tight_layout()
fig.subplots_adjust(wspace=0.32,hspace=0.4)
plt.savefig(savepath+'Scatter_COMBiv_DiffxSTDcLon_SCAN_USCRN_PCT20_SIG'+str(1-pv)+savename+savename2+'.png')
plt.savefig(savepath+'Scatter_COMBiv_DiffxSTDcLon_SCAN_USCRN_PCT20_SIG'+str(1-pv)+savename+savename2+'.pdf')