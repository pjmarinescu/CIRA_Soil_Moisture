#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Function to read NOAA/NCEI USCRN Soil Moisture Data from downloaded USCRN files

@author: petermarinescu
"""

import numpy as np
import pandas as pd
import glob 

# datein format should be string, YYYYMMDD
def read_USCRN(datein):

    # Pathname to USCRN data    
    pathname = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/uscrn/USCRN2/'+datein[0:4]+'/'
    files = glob.glob(pathname+'*')
    
    # Loop through all files for a given data (i.e., different stations)
    cnt = 0
    for f in np.arange(0,len(files)):
        if cnt == 0:
            # Read data into a dataframe
            data = pd.read_fwf(files[f],header=None)
            # Create the same column names
            data.columns = ['WBANNO', 'LST_DATE', 'CRX_VN', 'LONGITUDE', 'LATITUDE', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_MEAN', 'T_DAILY_AVG', 'P_DAILY_CALC', 'SOLARAD_DAILY', 'SUR_TEMP_DAILY_TYPE', 'SUR_TEMP_DAILY_MAX', 'SUR_TEMP_DAILY_MIN', 'SUR_TEMP_DAILY_AVG', 'RH_DAILY_MAX', 'RH_DAILY_MIN', 'RH_DAILY_AVG', 'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY', 'SOIL_TEMP_5_DAILY', 'SOIL_TEMP_10_DAILY', 'SOIL_TEMP_20_DAILY', 'SOIL_TEMP_50_DAILY', 'SOIL_TEMP_100_DAILY'] 
            # Only allow data from current date
            data = data[data['LST_DATE']==int(datein)]
            if len(data) > 0:
                cnt = cnt + 1
        else:
            tmp_data = pd.read_fwf(files[f],header=None)
            tmp_data.columns = ['WBANNO', 'LST_DATE', 'CRX_VN', 'LONGITUDE', 'LATITUDE', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_MEAN', 'T_DAILY_AVG', 'P_DAILY_CALC', 'SOLARAD_DAILY', 'SUR_TEMP_DAILY_TYPE', 'SUR_TEMP_DAILY_MAX', 'SUR_TEMP_DAILY_MIN', 'SUR_TEMP_DAILY_AVG', 'RH_DAILY_MAX', 'RH_DAILY_MIN', 'RH_DAILY_AVG', 'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY', 'SOIL_TEMP_5_DAILY', 'SOIL_TEMP_10_DAILY', 'SOIL_TEMP_20_DAILY', 'SOIL_TEMP_50_DAILY', 'SOIL_TEMP_100_DAILY'] 
            tmp_data = tmp_data[tmp_data['LST_DATE']==int(datein)]
 
            # Concatnate all the different station data for a given date 
            if len(tmp_data) > 0:
                cnt = cnt + 1
                frames = [data, tmp_data]
                data = pd.concat(frames)
                
                
    z_lvls = [0.05,0.1,0.2,0.5,1.0] 
    z_i = np.arange(0,1.6001,0.01) # interpolate levels to finer grid (unnecessary for linear interpolation that is being applied)         
    ism_uscrn = np.zeros(len(data))
    for i in np.arange(0,len(data)):
        # Grab the 5 specified levels from the data files
        m_lvls = [data.iloc[[i]]['SOIL_MOISTURE_5_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_10_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_20_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_50_DAILY'].values[0],
                  data.iloc[[i]]['SOIL_MOISTURE_100_DAILY'].values[0]]

        # If any level is missing, do not calculate the ISM    
        if any(m < -1 for m in m_lvls):
            ism_uscrn[i] = np.nan
        else:
            # Calculate 1.6 m ISM    
            m_i = np.interp((z_i[1:]+z_i[:-1])/2,z_lvls,m_lvls,left=m_lvls[0],right=m_lvls[-1])
            ism_uscrn[i] = np.nansum(np.diff(z_i)*m_i)*1000
    
    # add 1.6m ISM to the dataframe
    data['ism_1p6'] = ism_uscrn
   
    # Data is a pandas dataframe with the data from the USCRN text files for the specific date
    # Note there is more data in the text files that are not read in here -- I only read in data up to the soil moisture data
    # Note I also calculated integrated soil moisture for 1.6 m depths and place that into the dataframe  
    return data
    
    
