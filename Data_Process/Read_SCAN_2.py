#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:30:58 2022

@author: petermarinescu
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


# datein format should be string, YYYYMMDD
def read_SCAN(datein,stat_id):

    #stat_id = 2062
    #datein='20180101'
    # Pathname to SCAN data    
    pathname = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/scan/'+datein[0:4]+'/'
    pathname = '/Users/petermarinescu/Desktop/SCAN/'+datein[0:4]+'/'
    files = sorted(glob.glob(pathname+str(stat_id)+'*.csv'))
    datestr = datein[0:4]+'-'+datein[4:6]+'-'+datein[6:8]
    lfile = '/Users/petermarinescu/Desktop/SCAN/nwcc_inventory.csv'
#    lfile = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/scan/nwcc_inventory.csv' 
    
    cnt = 0
    for f in np.arange(0,len(files)):
    
        if cnt == 0:
            data = pd.read_csv(files[f],header=1)
            if np.shape(data)[0] == 0:
                continue
    
            col_out = []
            col_in = data.columns
            for ii in np.arange(0,len(col_in)):
                col_out = np.append(col_out,col_in[ii][0:17])            
            data.columns = col_out
    
            if any(data.columns.str.contains('-2')) and any(data.columns.str.contains('-4')) and any(data.columns.str.contains('-8')) and any(data.columns.str.contains('-20')) and any(data.columns.str.contains('-40')):
                print('')
            else:
                print('Not all levels present at '+files[f])
                #print(data.columns)
                continue
    
#            cols1 = ['Site Id', 'Date', 'Time', 'TOBS.I-1 (degC) ',
#           'TMAX.D-1 (degC) ', 'TMIN.D-1 (degC) ', 'TAVG.D-1 (degC) ',
#           'PRCP.D-1 (in) ', 'SMS.I-1:-2 (pct) ',
#           'SMS.I-1:-4 (pct) ', 'SMS.I-1:-8 (pct) ',
#           'SMS.I-1:-20 (pct)', 'SMS.I-1:-40 (pct)','RHUM.I-1 (pct) ']

            cols1 = ['Site Id', 'Date', 'Time', 'TOBS.I-1 (degC) ',
           'TMAX.D-1 (degC) ', 'TMIN.D-1 (degC) ', 'TAVG.D-1 (degC) ',
           'SMS.I-1:-2 (pct) ',
           'SMS.I-1:-4 (pct) ', 'SMS.I-1:-8 (pct) ',
           'SMS.I-1:-20 (pct)', 'SMS.I-1:-40 (pct)']
    
            # Only choose certain columns from individual files
            #print(data.columns)
            data = data[cols1]
    
#            data.columns = ['WBANNO', 'LST_DATE', 'Time', 'T_DAILY', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_AVG', 'P_DAILY_CALC',  'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY', 'RH_DAILY_AVG' ]
            data.columns = ['WBANNO', 'LST_DATE', 'Time', 'T_DAILY', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_AVG', 'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY']
 #           data = data[data['LST_DATE']==datestr]
            if len(data) > 0:
                cnt = cnt + 1
        else:
            tmp_data = pd.read_csv(files[f],header=1)
    
            if len(tmp_data) == 0:
                continue
            
            col_out = []
            col_in = tmp_data.columns
            for ii in np.arange(0,len(col_in)):
                col_out = np.append(col_out,col_in[ii][0:17])            
            tmp_data.columns = col_out
                
            if any(tmp_data.columns.str.contains('-2')) and any(tmp_data.columns.str.contains('-4')) and any(tmp_data.columns.str.contains('-8')) and any(tmp_data.columns.str.contains('-20')) and any(tmp_data.columns.str.contains('-40')):
                print('')
            else:
                print('Not all levels present at '+files[f])
    #            print(tmp_data.columns)
                continue
            
            if tmp_data['Site Id'][0] in [2027,2039]: # Ordinal for 100cm level is 5 (worse quality)
                continue
    
#            cols1 = ['Site Id', 'Date', 'Time', 'TOBS.I-1 (degC) ','TMAX.D-1 (degC) ', 'TMIN.D-1 (degC) ', 'TAVG.D-1 (degC) ',
#           'PRCP.D-1 (in) ', 'SMS.I-1:-2 (pct) ','SMS.I-1:-4 (pct) ','SMS.I-1:-8 (pct) ',
#           'SMS.I-1:-20 (pct)', 'SMS.I-1:-40 (pct)','RHUM.I-1 (pct) ']
    
            # Only choose certain columns from individual files
            tmp_data = tmp_data[cols1]
    
#            tmp_data.columns = ['WBANNO', 'LST_DATE', 'Time', 'T_DAILY', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_AVG', 'P_DAILY_CALC',  'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY', 'RH_DAILY_AVG' ]
            tmp_data.columns = ['WBANNO', 'LST_DATE', 'Time', 'T_DAILY', 'T_DAILY_MAX', 'T_DAILY_MIN', 'T_DAILY_AVG', 'SOIL_MOISTURE_5_DAILY', 'SOIL_MOISTURE_10_DAILY', 'SOIL_MOISTURE_20_DAILY', 'SOIL_MOISTURE_50_DAILY', 'SOIL_MOISTURE_100_DAILY']
    
#            tmp_data = tmp_data[tmp_data['LST_DATE']==datestr]
            if len(tmp_data) > 0:
                cnt = cnt + 1
                frames = [data, tmp_data]
                data = pd.concat(frames)
    
    if len(files) == 0 or np.shape(data)[0] == 0 or cnt == 0:
        data = np.nan
 
    else:
        #print(data.columns)
        ldata = pd.read_csv(lfile)
        lat = np.zeros(np.shape(data)[0])
        lon = np.zeros(np.shape(data)[0])
        for i in np.arange(0,np.shape(data)[0]):
            cdata = ldata.loc[ldata['station id'] == data['WBANNO'].values[i]]
            
            lat[i] = cdata[' lat'].values[0]
            lon[i] = cdata['lon'].values[0]
        
        data['LONGITUDE'] = lon
        data['LATITUDE'] = lat
        
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
                ism_uscrn[i] = np.nansum(np.diff(z_i)*m_i)*1000/100
        
        data['ism_1p6'] = ism_uscrn
        
        datenum = np.zeros(len(data))
        for t in np.arange(0,len(data)):
    #        print('new code')
            testdate = data['LST_DATE'][t]
    #        print(testdate)
            numdate = str(testdate[0:4])+str(testdate[5:7])+str(testdate[8:10])
            datenum[t] = np.int(numdate)
    
    #    print(data['LST_DATE'][t])
        data = data.drop(columns=['LST_DATE'])
        data['LST_DATE'] = datenum
    #    print(data['LST_DATE'][t])
    
    # Data is a pandas dataframe with the data from the USCRN text files for the specific date
    # Note there is more data in the text files that are not read in here -- I only read in data up to the soil moisture data
    # Note I also calculated integrated soil moisture for 1.6 m depths and place that into the dataframe  
    return data

#test = read_SCAN('20180101',1232)
    
    
    
