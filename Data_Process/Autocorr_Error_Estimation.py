#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Code to calculate the Autocorrelation based ratio of error variance to soil moisture variance, as described in manuscript

@author: petermarinescu
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import copy
import sys
from sklearn import linear_model
regr = linear_model.LinearRegression()
import scipy
import pandas as pd
sys.path.append("/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Code/Hera/") 
from Read_USCRN_2 import read_USCRN
sys.path.append("/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Code/Hera/") 
from Read_SCAN_2 import read_SCAN

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

# Define path where in situ data
openpath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/Hera_data/save/ISM_scat/'
datanet = 'SCAN' # 'SCAN or 'USCRN'

# Time Range for in situ data analysis
start_date = datetime(2018,1,1)
end_date = datetime(2022,1,1)
savepath = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/'
anom = 1 # 1 = anomalies are used based on monthly mean calculations, 0 = raw data are used
savename = '180_15_fullE_anom2_3Y_b1'

# Load station id identifiers
if datanet == 'USCRN':
    stats = pickle.load(open(savepath+'USCRN_stationids.p', "rb" ))
elif datanet == 'SCAN':
    lfile = '/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/GIT/NOAA_Soil_Moisture/Data_Process/nwcc_inventory.csv'
    ldata = pd.read_csv(lfile)
    stats = ldata['station id'].values

# Scale datasets so units are the same -- 
# note this was included before the read fucntions were corrected to account for this
# note different read functions are called in this script
if datanet == 'USCRN':
    scale = 1
elif datanet == 'SCAN':
    scale = 100


# Define variables
l_stat = np.zeros(len(stats))
l_rat_err = np.zeros(len(stats))
l_len = np.zeros(len(stats))
l_lat = np.zeros(len(stats))
l_lon = np.zeros(len(stats))

# Loop through stations
for s in np.arange(0,len(stats)):
#for s in [66]:
    cnt = 0
    # Loop through individual days in analysis
    for t in daterange(start_date,end_date):
        #print(t)
        
        stat_cur = stats[s]
        
        a_date = t
        # Convert date to desired datestrs
        YYYY = a_date.strftime("%Y")
        YY = a_date.strftime("%y")
        YY = a_date.strftime("%y")
        MM = a_date.strftime("%m")
        DD = a_date.strftime("%d")
        DOY = a_date.strftime("%j")
        
        #Because functions pull entire years worth of data, only need to do once during a year (on the first day here)
        if DOY != '001':
            continue
    
        # Read data from SCAN / USCRN files
        if datanet == 'USCRN':
            data_i = read_USCRN(YYYY+MM+DD,stat_cur)
            # Only include CONUS
            data_i = data_i[data_i['LATITUDE']<55]
            data_i = data_i[data_i['LATITUDE']>23]
            data_i = data_i[data_i['ism_1p6']>0]
            
            #Eliminate Coastal Stations That cannot be Resolved in HRRR
            data_i = data_i[data_i['WBANNO']!=53152] # Santa Barbara, CA
            data_i = data_i[data_i['WBANNO']!=63856] # Brunswick, GA
            data_i = data_i[data_i['WBANNO']!=93245] # Bodega, CA

        elif datanet == 'SCAN':
            data_i = read_SCAN(YYYY+MM+DD,stat_cur)
            if len(np.shape(data_i)) < 2:
                continue
            else:
                data_i = data_i[data_i['LATITUDE']<55]
                data_i = data_i[data_i['LATITUDE']>23]
                data_i = data_i[data_i['ism_1p6']>0]
            

        if len(np.shape(data_i)) < 2:
            continue
        
        # Combine data from individual days for each station
        if cnt == 0:
            data_comb = copy.deepcopy(data_i)
        else:
            data_comb = pd.concat([data_comb,data_i])
    
        cnt = cnt + 1

    if len(np.shape(data_i)) < 2:
        continue

       
    # Go back through dataset and fill in missing data with nan, so continuous time series
    data_comb['DOY'] = np.ones(len(data_comb))*-99
    cntdd = 0
    for tt in daterange(start_date,end_date):
        YYYY = tt.strftime("%Y")
        MM = tt.strftime("%m")
        DD = tt.strftime("%d")
        DOY = tt.strftime("%j")
                
        # If no data for a certain date, add a nan
        if len(data_comb[data_comb.eq(int(YYYY+MM+DD)).any(1)]) == 0:
            # Create nan arrays for specific number of columsn for each dataset
            if datanet == 'USCRN':
                nan_arr = np.empty((1,30))
                nan_arr[0,:] = np.nan
                nan_arr[0,1] = int(YYYY+MM+DD)       
                nan_arr[0,29] = int(DOY)       
            elif datanet == 'SCAN':
                nan_arr = np.empty((1,16))
                nan_arr[0,:] = np.nan
                nan_arr[0,14] = int(YYYY+MM+DD)       
                nan_arr[0,15] = int(DOY)       
            df_nan = pd.DataFrame(nan_arr,columns=data_comb.columns)
            data_comb = pd.concat([data_comb,df_nan])
        else:
            data_comb.iloc[cntdd,data_comb.columns.get_loc('DOY')] = int(DOY)
            cntdd = cntdd + 1
            #continue

    # Ensure data is chronological
    data_comb = data_comb.sort_values(by=['LST_DATE'])
    data_comb = data_comb.reset_index()

    if anom == 1:
        # Calculate Monthly Mean for Anomaly Calculation
        m_arr = np.zeros(len(data_comb))    
        for m in np.arange(0,len(data_comb)):
            m_arr[m] = float(str(data_comb['LST_DATE'].values[m])[4:6])
        data_comb['MM'] = m_arr   
        data_mon = data_comb.groupby('MM').mean()    
    
    
    # Use 1.6m soil moisture
    varn = 'ism_1p6'
    # Elimate data over 0.6 as unphysical
    data_comb[varn][data_comb['SOIL_MOISTURE_5_DAILY']>0.6*scale] = np.nan
    data_comb[varn][data_comb['SOIL_MOISTURE_10_DAILY']>0.6*scale] = np.nan
    data_comb[varn][data_comb['SOIL_MOISTURE_20_DAILY']>0.6*scale] = np.nan
    data_comb[varn][data_comb['SOIL_MOISTURE_50_DAILY']>0.6*scale] = np.nan
    data_comb[varn][data_comb['SOIL_MOISTURE_100_DAILY']>0.6*scale] = np.nan
    # Can change ism_ip6 here to the specific variable of interest
    temp_arr = copy.deepcopy(data_comb[varn])


    # Calculate and substract out, monthly anomalies for each day
    # Monthly anomalies are used, since it is a realitively short time frame and daily anomalies are too noisy
    # Interpolate Monthly anomalies to each day of the year for the anomaly calculation
    if anom == 1:
        varns = ['ism_1p6']
        dfg = data_comb.groupby('MM')
        dfg_keys = list(dfg.groups.keys())
        mon_avg = data_comb.groupby('MM').mean()
        mon_cnt = data_comb.groupby('MM').count()
        
        for v in np.arange(0,len(varns)):
            varn = varns[v]
            mon_mid_vals = mon_avg[varn].values
            
            #pad monthly mean values with 2 months of the periodic data
            mon_mid_vals2 = np.insert(mon_mid_vals,len(mon_mid_vals),mon_mid_vals[0])
            mon_mid_vals2 = np.insert(mon_mid_vals2,0,mon_mid_vals[len(mon_mid_vals)-1])
            #                    Jul, Aug  ,Sept,Octob,Nov,Decem,Janu,Fe,Marc,Apr,May  ,Jun,July ,Augus,Sep,Octob,Nov,Decem,Janua,Feb,March,Apr,May  ,Jun
            doy_mid = np.array([-168,-137.5,-107,-76.5,-46,-15.5,15.5,45,74.5,105,135.5,166,196.5,227.5,258,288.5,319,349.5,380.5,411,441.5,472,502.5,533])
            
            cur_doy_mid = []
            for mm in np.arange(0,len(mon_avg)):
                cur_doy_mid = np.append(cur_doy_mid,doy_mid[int(dfg_keys[mm])+5])
            cur_doy_mid = np.insert(cur_doy_mid,len(cur_doy_mid),doy_mid[int(dfg_keys[0]+17)])    
            cur_doy_mid = np.insert(cur_doy_mid,0,doy_mid[5+int(dfg_keys[len(dfg_keys)-1])-12])            
                
            # Interpolate monthly mean ISM values to each day of year for a more continuous subtraction
            dmi = np.zeros(366)
            for ii in np.arange(0,366):
               dmi[ii] = np.interp(ii,cur_doy_mid,mon_mid_vals2)
            
            doy_data = data_comb['DOY'].values
            # Loop through data, and substract mean doy value for that day
            # i.e. Removing the seasonal cycle
            for i in np.arange(0,len(temp_arr)):
                doy_sub = dmi[int(doy_data[i])-1]
                temp_arr[i] = temp_arr[i] - doy_sub
  
    # Get Lat and lon data and save them for each station (s)
    clat = np.nanmax(data_comb['LATITUDE'])
    clon = np.nanmin(data_comb['LONGITUDE'])
    l_lat[s] = clat
    l_lon[s] = clon

    # Eliminate potentially bad data (isms > 800m, VSM > 0.6 and negative VSMs)
    temp_arr[temp_arr>800] = np.nan
    for varnt in ['SOIL_MOISTURE_5_DAILY','SOIL_MOISTURE_10_DAILY','SOIL_MOISTURE_20_DAILY','SOIL_MOISTURE_50_DAILY','SOIL_MOISTURE_100_DAILY']:
        temp_arr[data_comb[varnt] > 0.6*scale] = np.nan
        temp_arr[data_comb[varnt] < 0] = np.nan

    # if there is one day of bad data, separated by days with good data, linearly interpolate to create a more contintuous time series
    for iii in np.arange(1,len(temp_arr)-1):
        if np.isnan(temp_arr[iii]):
            temp_arr[iii] = (temp_arr[iii-1]+temp_arr[iii+1])/2
            
    # Calculate autocorrelation up to ncor days lag 
    ncor = 15
    narr2 = np.shape(temp_arr)[0]
    cor_arr = np.zeros(ncor-1)
    xval = np.arange(1,ncor)
    for i in xval:
        c_arr1 = temp_arr.values[0:narr2-i]
        c_arr2 = temp_arr.values[i:narr2]
        cf_arr1 = []; cf_arr2 = []
        for c in np.arange(0,len(c_arr1)):
            # if there is a nan in either of the original or lagged series, ignore those lagged comparisons
            if np.isnan(c_arr1[c]) or np.isnan(c_arr2[c]):
                c;
            else:
                cf_arr1 = np.append(cf_arr1,c_arr1[c])
                cf_arr2 = np.append(cf_arr2,c_arr2[c])

        lanl = len(cf_arr1) # Calculate length of analysis series
        cor_arr[i-1] = np.corrcoef(cf_arr1,cf_arr2)[0,1]   # calculate lagged autocorrelation
      
    # Fit best fit line using scipy.stats linear regression        
    linreg = scipy.stats.linregress(xval, np.log(cor_arr))
    m = linreg.slope
    b = linreg.intercept

    # If in unable to fit a line through the data, give a -99 error value
    if np.isnan(m):
        l_stat[s] = stat_cur
        l_rat_err[s] = -99
        l_len[s] = lanl
    # Else use the extrapolated y-intercept to estimate the ratio of errors
    # Based on Robock et al., 1995 and other studies as described in manuscript
    else:
        # Calc displacement from 0 y intercept
        a = 1 - np.exp(b)
        a_pos = np.abs(a)
        rat_err = a_pos / (1-a_pos)
    
        # Create a figure with the best fit line for the data
        # Fit best fit line           
        plt.figure(figsize=[6,6])
        plt.plot(xval,np.log(cor_arr))
        plt.xlabel('lag')
        plt.ylabel('ln(r)')
        plt.title(str(stat_cur)+'_'+str(clat)+str(clon)+': y = '+str(np.round(m,1))+'x + '+str(np.round(b,3))+' | '+str(np.round(a,3))+' | '+str(np.round(rat_err,3)))
        plt.plot(xval,m*xval + b)
        plt.grid()
        plt.savefig('/Users/petermarinescu/Research/CIRA/Incu_SoilMoisture/AutoCorr_Plots/LT'+datanet+'ISM_'+savename+'_'+str(stat_cur)+'.png')
        plt.close()
        
        # Save station number, ratio of errors, and length of time series
        l_stat[s] = stat_cur
        l_rat_err[s] = rat_err
        l_len[s] = lanl

# Save Station and associated error,  dataseries length, lat, lon,
filename='LT'+datanet+'_'+savename+'.p'
pickle.dump([l_stat, l_rat_err, l_len, l_lat, l_lon],open(savepath+filename, "wb" ))