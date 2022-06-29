#!/bin/bash

# Bash script to grab HRRRv3 model soil moisture data from NOAA HPSS

# Define start and end dates for gra
d=2021-11-01T18:00:00Z
d_end=2021110418

dstr=$(date -d "$d" +%Y%m%d%H)
echo $dstr

# Loop over each time
while [ "$dstr" != "$d_end" ]; do 

  # Create datestr variables
  dstr=$(date -d "$d" +%Y%m%d%H)
  yyyy=$(date -d "$d" +%Y)
  dd=$(date -d "$d" +%d)
  mm=$(date -d "$d" +%m)
  hh=$(date -d "$d" +%H)
  #echo $d
  echo $dstr

  # Create filename of HRRRv3 data
  filename='/ESRL/BMC/fdr/Permanent/'$yyyy'/'$mm'/'$dd'/grib/hrrr_wrfprs/7/0/83/0_1905141_30/'$yyyy$mm$dd$hh'00.zip'
  echo $filename

  # Grab the file
  hsi get $filename

  # Unzip the file
  unzip -j $yyyy$mm$dd$hh'00.zip' '*000000'
  rm *.zip

  file='*000000'

  fileout='SM_'$yyyy$mm$dd$hh'00.grb'
  echo $fileout

  # Grab the soil moisture variables from the HRRR files and resave new files locally
  wgrib2 $file -s | egrep '(:SOILW)' | wgrib2 -i $file -grib $fileout
 
  rm $file

  echo $d
  # Add 6 hours to previous time
  d=$(date -d "$d + 24 hour")
  echo $d
done
