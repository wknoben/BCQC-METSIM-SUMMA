############################################################
### Python concatenates windspeed data from nldas		 ###
### and attempts to create time series files for snoTEl  ###
### that will be used as an input for SUMMA				 ###	
############################################################

# load all the necessary modules and packages
import pandas as pd
import numpy as np
import re as re
import datetime
from datetime import date, timedelta
import glob
import scipy
import os
from scipy.io import netcdf
import netCDF4 as nc4
import time
import csv
import os

#######################################
# Open several different lists
#######################################
os.chdir('/home/chart/3DOWNLOAD_ERA5_WIND/LISTS') 					# change directory into folder List Data
with open ("LIST_Lat.csv") as myfile:								# Declare SNOTEL file names and corresponding HRU # to find
	lat_list = myfile.read().split('\n')							# Split up original .CSV into rows

with open ("LIST_Lon.csv") as myfile:								# Declare SNOTEL file names and corresponding HRU # to find
	lon_list = myfile.read().split('\n')							# Split up original .CSV into rows

with open ("LIST_ERA5.csv") as myfile:								# Declare SNOTEL file names and corresponding HRU # to find
	era5_list = myfile.read().split('\n')							# Split up original .CSV into rows
		
with open ("LIST_METSIM.csv") as myfile:							# Declare SNOTEL file names and corresponding HRU # to find
	metsim_list = myfile.read().split('\n')							# Split up original .CSV into rows
			
metsimlength=608 #len(lat_list)											# Calculatie length of these data files
for o in range(metsimlength):										# Iterate through files	
	time_data=[]													# create time data variable
	wind_data=[]													# create wind data variable

	########################################
	# read in ERA5 data and open netCDF	file
	########################################
	era5length=len(era5_list)	# 	tt=1							# Calculate number of NLDAS data files (number of rows)
	for z in range(era5length):										# Iterate through files		
		os.chdir('/project/6008034/Model_Output/ClimateForcingData/ERA5_NA') # change directory into folder with ERA5 data 	
		filename=era5_list[z]										# Assign a file name variable to a FILE name from the nldas name list file
		mcid = nc4.Dataset(filename, "r", format="NETCDF4")			# Open the NLDAS data file for reading	
			
		#######################################
		# Format windspeed @ time stamp 
		#######################################		
		raw_data = mcid.variables['time']							# Assign a raw_data variable to the time variable from the NLDAS data
		time_data = np.append(time_data,raw_data)					# Append the raw_data to the entire time time-series
		raw_data = mcid.variables['windspd']						# Assign a raw_data variable to the windspd variable from the NLDAS data		
		wind_data = np.append(wind_data,raw_data[:,lat_list[o],lon_list[o]])	# Append the raw_data to the entire windspd time-series
		mcid.close()												# close NLDAS Datafile

	print(len(wind_data))

	#######################################
	# netCDF creating a new file
	#######################################	
	os.chdir('/home/chart/3DOWNLOAD_ERA5_WIND/OUTPUT')	 			# Change directory into the folder containing NLDAS data
	outfile=metsim_list[o]										    # Create name for an output file from and the existing snoTEL file
	outfile+='.wind.nc'												# Concatenate a wind variable identifier to the original file name
	ncid = nc4.Dataset(outfile, "w", format="NETCDF4")				# Create a file for writing NLDAS data using this new name
	# Declare dimensions for the new output file				
	dimid_T = ncid.createDimension('time',15360)
	dimid_hru = ncid.createDimension('hru',1)
	# Declare attributes for the new variable 				
	wind_varid = ncid.createVariable('windspd','f',('time','hru'))	# Create variable used to assign the Windspd time-series to
	wind_varid.long_name      = 'wind speed at the measurement height' # Create long_name attribute for the windspd variable
	wind_varid.units          = 'm s-1'								# Create units attribute for the windspd variable
	wind_varid.FillValue      = '-999'								# Create a FillValue attribute for the windspd variable
	# Assign variable  value
	wind_varid[:]=wind_data											# Assign NLDAS data to file
	# Declare attributes for the new variable 				
	time_varid = ncid.createVariable('time','d',('time','hru'))		# Create variable used to assign the time time-series to
	time_varid.long_name      = 'Observation time'					# Create long_name attribute for the time variable
	time_varid.units          = 'days since 1900-01-01 00:00:00'	# Create units attribute for the time variable
	time_varid.FillValue      = '-999'								# Create a FillValue attribute for the time variable
	# Assign variable a dummy value
	time_varid[:]=time_data											# Assign NLDAS data to file

	ncid.close()													# Close WINDSPD Datafile	
	
