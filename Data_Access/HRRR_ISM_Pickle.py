#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:21:01 2021

@author: petermarinescu
"""

import pygrib 
import numpy as np
from datetime import datetime, timedelta
import pickle
import os

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = datetime(2020,7,1)
end_date = datetime(2021,1,1)
inpath = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/hrrr/'

# Specify HRRR grid size
nslvls = 9; # Number of soil levels   
nx = 1799; # Number of x points
ny = 1059; # Number of y points

# Read in lat/lon grid,  landmask, and maximum soil moisture content data for HRRR grid
[lons,lats,lm] = pickle.load(open( inpath+'HRRR_lm.p', 'rb' ))
[maxsmc_t,maxsmc_b] = pickle.load(open( inpath+'maxsmc.p','rb' ))

# Define vertical levels in HRRR and for integration
z_lvls = np.array([0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0])
z_hafs = (z_lvls[1:]+z_lvls[:-1])/2
z_hafs = np.insert(z_hafs,0,0)
#z_i = [0,0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4,1.6]

# Loop over days
for t in daterange(start_date,end_date):

    an_date = t
    # Convert date to desired datestrs
    YYYY = an_date.strftime("%Y")
    YY = an_date.strftime("%y")
    YY = an_date.strftime("%y")
    MM = an_date.strftime("%m")
    DD = an_date.strftime("%d")
    DOY = an_date.strftime("%j")
    
    # HRRR File
    file=inpath+'SM_'+YYYY+MM+DD+'1800.grb'
    
    fs = os.path.getsize(file)
    if fs == 0:
        continue
       
    # Read in VSM data from HRRR Grib Files
    grbs = pygrib.open(file)
    grb = grbs.select(name='Volumetric soil moisture content')     
    m_lvls = np.zeros((ny,nx,nslvls))
    for i in np.arange(0,nslvls):   
        m_lvls[:,:,i] = grb[i].values

    # soil moistures values greater than 0.485 are not physical (communication with Tanya Smirnova)
    m_lvls[m_lvls > 0.485] = np.nan     
        
    # Define ISM variable for HRRR
    ism_fin = np.zeros((ny,nx))
    
    startTime = datetime.now()
    print(startTime)

    
    cnt = 0
    # Loop over x and y points over the HRRR domain
    for ii in np.arange(0,ny):
          for jj in np.arange(0,nx):
              # If water region, continue
              if lm[ii,jj] == 0:
                  cnt = cnt + 1
                  continue
              else:
                  #print(cnt)

                  # Integrate soil moisture vertically in HRRR
                  for k in np.arange(0,nslvls-2):
                      cur_sm_val = np.min([maxsmc_t[ii,jj],m_lvls[ii,jj,k]])
                      ism_fin[ii,jj] = ism_fin[ii,jj] + (cur_sm_val * (z_hafs[k+1]-z_hafs[k]) * 1000)

    	          # Add 30 cm from 1.3m to 1.6 m
                  k = 7
                  cur_sm_val = np.min([maxsmc_t[ii,jj],m_lvls[ii,jj,k]])
                  ism_fin[ii,jj] = ism_fin[ii,jj] + (cur_sm_val * (1.6-z_hafs[k]) * 1000)
                  cnt = cnt + 1

    endloop2 = datetime.now()
    print(endloop2)    
   
    # Saved HRRR ISM and VSM data to files specified below
    pickpath = '/scratch1/RDARCH/rda-ghpcs/Peter.Marinescu/data/hrrr/pickle_fin/'
    filename = 'HRRR_ISM_'+YYYY+MM+DD+'_lm.p'
    #pickle.dump([ism_pjm,ism_ts1,m_lvls,z_lvls,nx,ny,lons,lats,nslvls,an_date], open( pickpath+filename, "wb" ))
    pickle.dump([ism_fin,m_lvls,z_lvls,nx,ny,lons,lats,nslvls,an_date], open( pickpath+filename, "wb" ))


