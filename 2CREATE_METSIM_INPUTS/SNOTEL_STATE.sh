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
# iterate through time periods
#######################################
tf=1
for o in range(tf):
	date1 = date(2017,1,1)
	delta = timedelta(90)
	offset = str(date1 - delta)

	#######################################
	# read in the names of files to format
	# read in the names of .nc output
	#######################################
	with open ("LIST_STATION.csv") as myfile:
		datafile = myfile.read().split('\n')
	
	with open ("LIST_STATE.csv") as myfile:
		outputfile = myfile.read().split('\n')

	#######################################
	# define some time variables
	# Used for searching snoTEL data
	#######################################
	telstart='20170101'											# Decalre a variable that == a datetime time string at the start of the metSIM simulation
	telend='20170331'

	#######################################
	# IMOPORT .csv files and find time .
	#######################################
	start=0
	finish=90
	data_length=len(datafile)

	for z in range(data_length):	# Change directory into a folder containing the snoTEL data
		os.chdir('/home/cjh458/2CREATE_METSIM_INPUTS/dataCSV_bcqc')	# Change directory into a folder containing the snoTEL data	
		data = pd.read_csv(datafile[z], sep=',')					# Use pandas read_csv function to declare a variable = to a .csv file with a ; delimiter
		length=len(data)											            # Calculate the length of the this vaiable
		lle = pd.read_csv('LLE.csv', sep=',')			        # Use pandas read_csv function to declare a variable = to a .csv file with a ; delimiter

		#######################################
		# extract data from .csv
		#######################################
		length_series=finish-start									      # Calculate the number of rows between the start and finish strings with the .csv -- This maybe different than the date time-series bc of gaps in data

		sid = lle['sid'].values[z]								          # Extract the snoTEL station Id number
		lon = lle['lon'].values[z]							            # Extract the geographical coordinates - latitude
		lat = lle['lat'].values[z]							            # Extract the geographical coordinates - longitude
		elev = lle['elev'].values[z]								        # Extract the geographical elevation
		t_min = data['temperature_min'].values[start:finish]
		t_max = data['temperature_max'].values[start:finish]
		Prec = data['precipitation'].values[start:finish]
		for i in range(len(Prec)):
			t_min[i] = t_min[i]
			t_max[i] = t_max[i]
			Prec[i] = Prec[i]

		length_T=len(t_min)
		for i in range(length_T):
			if t_min[i]>t_max[i]:
				t_max[i]=t_min[i]+0.5

		#######################################
		# netCDF creation
		#######################################
		os.chdir('/home/cjh458/2CREATE_METSIM_INPUTS/')	# Change directory into a folder containing the snoTEL data
		outfile=outputfile[z]
		
		ncid = nc4.Dataset(outfile, "w", format="NETCDF4")

		dimid_T = ncid.createDimension('time',90)
		dimid_lat = ncid.createDimension('lat',1)
		dimid_lon = ncid.createDimension('lon',1)

		####################################### Variable: Time
		time_varid = ncid.createVariable('time','i8',('time',))
		length_time=90
		time2 = [i for i in range(length_time)]

		startdate = date(1900,1,1)										# Declare a startdate variable, which will be used to calculate a time vector and display info
		date1 = date(2017,1,1)											# Declare a date = to the telstart variable
		date2 = date(2017,4,1)											# Declare a date = to the telend variable
		time1 = (date1-startdate).days									# Convert this date1 variable into days
		time2 = (date2-startdate).days									# Convert this date2 variable into days
		td=time2-time1													# Calculate the time difference between these two numbers of days
		int_time1 = int(time1)											# Convert the (# of days) time1 variable into and integer value
		int_time2 = int(time2)											# Convert the (# of days) time2 variable into and integer value

		time3 = np.arange(int_time1, int_time2)							# Arrange a time-series starting and ending at the two integer values with a timestep of a day (default)
		# Attributes
		time_varid.units         = 'days since 1900-01-01 00:00:00'
		time_varid.calendar      = 'proleptic_gregorian'

		# Write data
		time_varid[:] = time3

		####################################### Variables: Latitude & Longitude
		lon_varid = ncid.createVariable('lon','d',('lon',))
		lat_varid = ncid.createVariable('lat','d',('lat',))

		# Attributes
		lat_varid._fillvalue     = 'nan'
		lon_varid._fillvalue     = 'nan'
		lat_varid.standard_name  = 'latitude'
		lon_varid.standard_name  = 'longitude'
		lat_varid.units          = 'degrees_north'
		lon_varid.units          = 'degrees_east'
		lat_varid.axis           = 'Y'
		lon_varid.axis           = 'X'

		# Write data
		lat_varid[:] = lat
		lon_varid[:] = lon

		####################################### Variable: Precipitation
		prec_varid = ncid.createVariable('prec','d',('time','lat','lon',))

		# Attributes
		prec_varid.names         = '_fillvalue'
		prec_varid.value        = 'nan'

		# Write data
		prec_varid[:] = Prec

		###################################### Variable: Temp minimum
		t_min_varid = ncid.createVariable('t_min','d',('time','lat','lon',))

		# Attributes
		t_min_varid.names         = '_fillvalue'
		t_min_varid.value        = 'nan'

		# Write data
		t_min_varid[:] = t_min

		####################################### Variable: Temp maximum
		t_max_varid = ncid.createVariable('t_max','d',('time','lat','lon',))

		# Attributes
		t_max_varid.names         = '_fillvalue'
		t_max_varid.value        = 'nan'

		# Write data
		t_max_varid[:] = t_max

		####################################### Header 
		ncid.License     = 'The file was created by C.Hart, https://github.com/ChristianHart2019'
		ncid.history     = 'Created ' + time.ctime(time.time())

		#####################
		ncid.close()
