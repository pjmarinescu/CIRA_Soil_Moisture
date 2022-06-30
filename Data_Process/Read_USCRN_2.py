#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:38:35 2021

@author: petermarinescu
"""

import numpy as np
import pandas as pd
import glob 
import matplotlib.pyplot as plt

def read_USCRN(datein,stat_id):

    print(stat_id)
    # Pathname to USCRN data    
    pathname = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/uscrn/USCRN2/'+datein[0:4]+'/'
    pathname = '/Users/petermarinescu/Desktop/USCRN2/'+datein[0:4]+'/'
    files = glob.glob(pathname+'*')
    
    cnt = 0
    for f in np.arange(0,len(files)):
        if cnt == 0:
            data = pd.read_fwf(files[f],header=None)
            data.columns = ['WBANNO', 'LST_DATE', 'CRX_VN', 'LONGITUDE', 'LATITUDE', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_MEAN', 'T_DAILY_AVG', 'P_DAILY_CALC', 'SOLARAD_DAILY', 'SUR_TEMP_DAILY_TYPE', 'SUR_TEMP_DAILY_MAX', 'SUR_TEMP_DAILY_MIN', 'SUR_TEMP_DAILY_AVG', 'RH_DAILY_MAX', 'RH_DAILY_MIN', 'RH_DAILY_AVG', 'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY', 'SOIL_TEMP_5_DAILY', 'SOIL_TEMP_10_DAILY', 'SOIL_TEMP_20_DAILY', 'SOIL_TEMP_50_DAILY', 'SOIL_TEMP_100_DAILY'] 
           # data = data[data['LST_DATE']==int(datein)]
            data = data[data['WBANNO']==int(stat_id)]
            if len(data) > 0:
                cnt = cnt + 1
        else:
            tmp_data = pd.read_fwf(files[f],header=None)
            tmp_data.columns = ['WBANNO', 'LST_DATE', 'CRX_VN', 'LONGITUDE', 'LATITUDE', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_MEAN', 'T_DAILY_AVG', 'P_DAILY_CALC', 'SOLARAD_DAILY', 'SUR_TEMP_DAILY_TYPE', 'SUR_TEMP_DAILY_MAX', 'SUR_TEMP_DAILY_MIN', 'SUR_TEMP_DAILY_AVG', 'RH_DAILY_MAX', 'RH_DAILY_MIN', 'RH_DAILY_AVG', 'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY', 'SOIL_TEMP_5_DAILY', 'SOIL_TEMP_10_DAILY', 'SOIL_TEMP_20_DAILY', 'SOIL_TEMP_50_DAILY', 'SOIL_TEMP_100_DAILY'] 
           # tmp_data = tmp_data[tmp_data['LST_DATE']==int(datein)]
            tmp_data = tmp_data[tmp_data['WBANNO']==int(stat_id)]
            if len(tmp_data) > 0:
                cnt = cnt + 1
                frames = [data, tmp_data]
                data = pd.concat(frames)
                
    z_lvls = [0.05,0.1,0.2,0.5,1.0] 
    z_i = np.arange(0,1.6001,0.01)         
    ism_uscrn = np.zeros(len(data))
    for i in np.arange(0,len(data)):
        m_lvls = [data.iloc[[i]]['SOIL_MOISTURE_5_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_10_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_20_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_50_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_100_DAILY'].values[0]]
    
        if any(m < -1 for m in m_lvls):
            ism_uscrn[i] = np.nan
        else:
            m_i = np.interp((z_i[1:]+z_i[:-1])/2,z_lvls,m_lvls,left=m_lvls[0],right=m_lvls[-1])
            ism_uscrn[i] = np.nansum(np.diff(z_i)*m_i)*1000
    
    data['ism_1p6'] = ism_uscrn
    
    return data
    
    #fig,ax = plt.subplots(3,2,figsize=[10,7])
    #a = ax[0,0].scatter(data['LONGITUDE'],data['LATITUDE'],s=5,c=data['ism'])
    #plt.colorbar()
    
