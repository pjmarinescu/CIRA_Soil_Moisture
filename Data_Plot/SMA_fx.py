#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:02:53 2023

@author: petermarinescu
"""

# Functions for soil moisture analysis

# Import libraries needed for these functions
import numpy as np
from datetime import datetime, timedelta
import pickle
import copy
import os

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n*1)

def read_in_data(start_date,end_date,pair,ism_txt,openpaths,switch):

    # Define variables
    usc_i = []    
    hrr_i = []
    cpc_i = []
    usc_p = []
    hrr_p = []
    lon_p = []
    lat_p = []
    nstat = []


    # Loop through individual dates during time period
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
        # Loop through two observational datasates
        for o in np.arange(0,len(openpaths)):
            
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
             
                if switch == 'ism':
                    latn = lat['ism']
                    lonn = lon['ism']
                    lat_o = copy.deepcopy(latn)
                    #Eliminate data from Selma, AL station (often very high VSM values (>0.6) at 100 cm depth)   
                    ism_usc = ism_usc[(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                    cpc_int = cpc_int[(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                    hrrr_int[ism_txt] = hrrr_int[ism_txt][(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                elif switch == 'vsm':
                    latn = lat['v'+str(pair[0])]
                    lonn = lon['v'+str(pair[0])]                
                    lat_o = copy.deepcopy(latn)
                    usc[str(pair[0])] = usc[str(pair[0])][(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                    vsm[pair[1]] = vsm[pair[1]][(latn<32.45) | (latn > 32.47) | (lonn<-87.25) | (lonn>-87.23)]
                  
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
                if switch == 'ism':
                    latn = lat['ism']
                    lonn = lon['ism']
                elif switch == 'vsm':
                    latn = lat['v'+str(pair[0])]
                    lonn = lon['v'+str(pair[0])]                
    
            nlon = nlon+len(lonn)
        
            usc_i = np.append(usc_i,ism_usc)
            hrr_i = np.append(hrr_i,hrrr_int[ism_txt])
            cpc_i = np.append(cpc_i,cpc_int)

            if switch == 'vsm':
                uscn = copy.deepcopy(usc[str(pair[0])])
                vsmn = copy.deepcopy(vsm[pair[1]])
            
                vsmn = vsmn[uscn>0]
                lonn = lonn[uscn>0]
                latn = latn[uscn>0]
                uscn = uscn[uscn>0]

                usc_p = np.append(usc_p,uscn)
                hrr_p = np.append(hrr_p,vsmn)
                lon_p = np.append(lon_p,lonn)
                lat_p = np.append(lat_p,latn)
            
            else:
                usc_p = np.append(usc_p,usc[str(pair[0])])
                hrr_p = np.append(hrr_p,vsm[pair[1]])
                lon_p = np.append(lon_p,lonn)
                lat_p = np.append(lat_p,latn)

        if nlon == 0:
            nlon = np.nan
        nstat = np.append(nstat,nlon)
        
    loc = (lon_p*100+lat_p)

    return (usc_i,hrr_i,cpc_i,usc_p,hrr_p,lon_p,lat_p,loc)
