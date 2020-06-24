############################################################
### Python MetSim State file creator					 ###
### this script opens up snotel files finds the desired  ###
### time period writes netCDF FILES for state inputs     ### 
### for a meteorological simulator						 ###	
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
import netCDF4 as nc4
import time
import csv

#######################################
# read in the names of files to format
# read in the names of .nc output
#######################################
with open ("LIST_STATION.csv") as myfile:						# Open a file that contains a list of snoTEl station names
	datafile = myfile.read().split('\n')						  # Split up original .CSV into rows

with open ("LIST_FORCING.csv") as myfile:						# Open a file that contains a list of forcing flie names
	outputfile = myfile.read().split('\n')						# Split up original .CSV into rows

#######################################
# define some time variables
# Used for searching snoTEL data
# must be 1 day longer than the config
#######################################
telstart='20170401'												  # Decalre a variable that == a datetime time string at the start of the metSIM simulation
telend='20180930'												    # Decalre a variable that == a datetime time string at the end of the metSIM simulation

#######################################
# The date range used to declare this variable has to be 1 day
# longer than the range used in the CONFIG file, but it is not the actual 
# range of the data. 
########################################
startdate = date(1900,1,1)									# Declare a startdate variable, which will be used to calculate a time vector and display info
date1 = date(2017,4,1)											# Declare a date = to the telstart variable
date2 = date(2018,9,30)											# Declare a date = to the telend variable
time1 = (date1-startdate).days							# Convert this date1 variable into days
time2 = (date2-startdate).days							# Convert this date2 variable into days
td=time2-time1													    # Calculate the time difference between these two numbers of days
int_time1 = int(time1)											# Convert the (# of days) time1 variable into and integer value
int_time2 = int(time2)											# Convert the (# of days) time2 variable into and integer value

time3 = np.arange(int_time1, int_time2)							# Arrange a time-series starting and ending at the two integer values with a timestep of a day (default)

#######################################
# IMOPORT .csv files and find the 
# desired time-series within the file
#######################################
start=90
finish=637
data_length=len(datafile)												    # Calculate the length of the snoTEL station list file
for z in range(data_length):												# Iterate through the different snoTEL stations
	os.chdir('/home/cjh458/2CREATE_METSIM_INPUTS/dataCSV_bcqc')	# Change directory into a folder containing the snoTEL data
	data = pd.read_csv(datafile[z], sep=',')					# Use pandas read_csv function to declare a variable = to a .csv file with a ; delimiter
	length=len(data)											            # Calculate the length of the this vaiable
	lle = pd.read_csv('LLE.csv', sep=',')			        # Use pandas read_csv function to declare a variable = to a .csv file with a ; delimiter

	#######################################
	# Extract data from csv files and if
	# necessary, conduct some basic QC and
	# and data conversions
	#######################################
	length_series=finish-start									      # Calculate the number of rows between the start and finish strings with the .csv -- This maybe different than the date time-series bc of gaps in data

	sid = lle['sid'].values[z]								    # Extract the snoTEL station Id number
	lon = lle['lon'].values[z]							    # Extract the geographical coordinates - latitude
	lat = lle['lat'].values[z]							    # Extract the geographical coordinates - longitude
	elev = lle['elev'].values[z]								      # Extract the geographical elevation
	Tmin = data['temperature_min'].values[start:finish]
	Tmax = data['temperature_max'].values[start:finish]
	Prec = data['precipitation'].values[start:finish]
	for i in range(len(Prec)):
		Tmin[i] = Tmin[i]
		Tmax[i] = Tmax[i]
		Prec[i] = Prec[i]
	
	length_T=len(Tmin)
	for i in range(length_T):
		if Tmin[i]>Tmax[i]:
			Tmax[i]=Tmin[i]+0.5	
	#######################################
	# netCDF creation
	#######################################
	os.chdir('/home/cjh458/2CREATE_METSIM_INPUTS/')

	outfile=outputfile[z]

	ncid = nc4.Dataset(outfile, "w", format="NETCDF4")

	dimid_lon = ncid.createDimension('lon',1)
	dimid_lat = ncid.createDimension('lat',1)
	dimid_T = ncid.createDimension('time',td)

	####################################### Variables: Latitude & Longitude
	Tmax_varid = ncid.createVariable('Tmax','f',('time','lat','lon',))
	lat_varid = ncid.createVariable('lat','d',('lat',))

	# Attributes
	lat_varid.standard_name  = 'latitude'
	lon_varid.standard_name  = 'longitude'
	lat_varid.long_name      = 'latitude'
	lon_varid.long_name      = 'longitude'
	lat_varid.units          = 'degrees_north'
	lon_varid.units          = 'degrees_east'
	lat_varid.axis           = 'Y'
	lon_varid.axis           = 'X'

	# Write data
	lat_varid[:] = lat
	lon_varid[:] = lon

	####################################### Variable: Time
	### The date range used to declare this variable has to be 1 day
	### longer than the range used in the CONFIG file, but it is not the actual
	### range of the data. For MetSim to work this range will have to be from a time period
	### originally declared in the config file starting in 1950/1/1
	########################################
	time_varid = ncid.createVariable('time','d',('time',))

	# Attributes
	time_varid.standard_name  = 'time'
	time_varid.longname       = 'Time axis'
	time_varid.units          = 'days since 1900-01-01 00:00:00'
	time_varid.calendar       = 'standard'

	# Write data
	time_varid[:] = time3

	####################################### Variable: Precipitation
	Prec_varid = ncid.createVariable('Prec','f',('time','lat','lon',))
# 	Prec_varid = ncid.createVariable('Prec','f',('time',))

	# Attributes
	Prec_varid.units            = 'mm'
	Prec_varid.FillValue        = '1.00e+20'
	Prec_varid.missingvalue     = '1.00e+20'
	Prec_varid.long_name        = 'precipitation'

	# Write data
	Prec_varid[:] = Prec

	####################################### Variable: Temp maximum
	Tmax_varid = ncid.createVariable('Tmax','f',('time','lat','lon',))
# 	Tmax_varid = ncid.createVariable('Tmax','f',('time',))

	# Attributes
	Tmax_varid.units            = 'C'
	Tmax_varid.FillValue        = '1.00e+20'
	Tmax_varid.missingvalue    = '1.00e+20'
	Tmax_varid.long_name        = 'Daily maximum temperature'

	# Write data
	Tmax_varid[:] = Tmax

	###################################### Variable: Temp minimum
	Tmin_varid = ncid.createVariable('Tmin','f',('time','lat','lon',))
# 	Tmin_varid = ncid.createVariable('Tmin','f',('time',))

	# Attributes
	Tmin_varid.units            = 'C'
	Tmin_varid.FillValue        = '1.00e+20'
	Tmin_varid.missingvalue    = '1.00e+20'
	Tmin_varid.long_name        = 'Daily minimum temperature'

	# Write data
	Tmin_varid[:] = Tmin

	####################################### Header
	ncid.License     = 'The file was created by C.Hart, https://github.com/ChristianHart2019'
	ncid.history     = 'Created ' + time.ctime(time.time())

	#####################
	ncid.close()
