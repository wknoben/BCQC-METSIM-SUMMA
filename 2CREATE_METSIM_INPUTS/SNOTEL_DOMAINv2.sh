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
import glob
import scipy
import os
import netCDF4 as nc4
import time
import csv

#######################################
# iterate through months
#######################################
tf=1			#number of time periods to simulate 
for o in range(tf):

	#######################################
	# read in the names of files to format
	# read in the names of .nc output
	#######################################
	with open ("LIST_STATION.csv") as myfile:
		datafile = myfile.read().split('\n')
	
	with open ("LIST_DOMAIN.csv") as myfile:
		outputfile = myfile.read().split('\n')
	
	#######################################
	# define some time variables
	# Used for searching snoTEL data
	# Used 90 days before simulation period
	#######################################
	telstart='20170401'												# Decalre a variable that == a datetime time string at the start of the metSIM simulation
	telend='20181231'	
 
	#######################################
	# IMOPORT .csv files and find time .
	#######################################
	data_length=len(datafile)
# 	print(data_length)
# 	tt=1
	for z in range(data_length):
		os.chdir('/home/cjh458/2CREATE_METSIM_INPUTS/dataCSV_bcqc')	# Change directory into a folder containing the snoTEL data
# 		os.chdir('/Users/cjh458/Desktop/1CREATE_METSIM_INPUTS_2/ghcndData')	
		data = pd.read_csv(datafile[z], sep=',')
		lle = pd.read_csv('LLE.csv', sep=',')			# Use pandas read_csv function to declare a variable = to a .csv file with a ; delimiter

		start=90
		finish=729
		
		length=len(data)
		for i in range(length): 
			val = data['date'].values[i]
			coms = (val==telstart)				
			comf = (val==telend)				
			if coms:							#find start of time-series
				start=i
			if comf:							#find end of time-series
				finish=i

		#######################################
		# extract data from .csv
		#######################################
		length_series=finish-start

		sid = lle['sid'].values[z]								          # Extract the snoTEL station Id number
		lon = lle['lon'].values[z]							            # Extract the geographical coordinates - latitude
		lat = lle['lat'].values[z]							            # Extract the geographical coordinates - longitude
		elev = lle['elev'].values[z]								        # Extract the geographical elevation
		t_min = data['temperature_min'].values[start:finish]# Extract the min temp
		t_max = data['temperature_max'].values[start:finish]# Extract the max temp
		Prec = data['precipitation'].values[start:finish]   # Extract the precipitation

		length_T=len(t_min)         # Calculate the length of the temperature time-series
		for i in range(length_T):   # Iterate through the lenght of this number
			if t_min[i]>t_max[i]:     # If the minimum is larger than the maximum
				t_max[i]=t_min[i]+0.5   # Than adjust the maximum temperature


		#######################################
		# netCDF creation
		#######################################
		os.chdir('/home/cjh458/2CREATE_METSIM_INPUTS/')	# Change directory into a folder containing the snoTEL data
		outfile=outputfile[z]
# 		outfile+='.'
# 		outfile+=str(telstart)
# 		outfile+='.nc'
		
		ncid = nc4.Dataset(outfile, "w", format="NETCDF4")

		dimid_lon = ncid.createDimension('lon',1)
		dimid_lat = ncid.createDimension('lat',1)
		dimid_m = ncid.createDimension('month',12)

		####################################### Variable: Month
		month_varid = ncid.createVariable('month','i4',('month',))
		length_month=12
		month = [i for i in range(length_month)]

		# Attributes
		# Write data
		month_varid[:] = month

		####################################### Variable: Mask
		mask_varid = ncid.createVariable('mask','i4',('lat','lon',))

		# Attributes
		mask_varid.long_name     = 'domain mask'
		mask_varid.comment       = '0 indicates cell is not active'

		# Write data
		mask_varid[:] = 1

		####################################### Variable: Frac
		frac_varid = ncid.createVariable('frac','d',('lat','lon',))

		# Attributes
		frac_varid.units         = '1'
		frac_varid.long_name     = 'fraction of grid cell that is active'
		frac_varid.FillValue     = 'NaN'

		# Write data
		frac_varid[:] = 1

		####################################### Variable: Elev
		elev_varid = ncid.createVariable('elev','d',('lat','lon',))

		# Attributes
		elev_varid.units         = 'm'
		elev_varid.long_name     = 'gridcell_elevation'
		elev_varid.FillValue    = 'NaN'

		# Write data
		elev_varid[:] = elev
	
		####################################### Variable: Area
		area_varid = ncid.createVariable('area','d',('lat','lon',))

		# Attributes
		area_varid.units         = 'm2'
		area_varid.long_name     = 'area of grid cell'
		area_varid.standardname  = 'area'
		area_varid.FillValue    = 'NaN'

		# Write data
		area_varid[:] = 1

		####################################### Variables: Latitude & Longitude
		lon_varid = ncid.createVariable('lon','d',('lon',))
		lat_varid = ncid.createVariable('lat','d',('lat',))
	
		# Attributes
		lat_varid.names          = '_FillValue'
		lon_varid.names          = '_FillValue'
		lat_varid.value          = 'NaN'
		lon_varid.value          = 'NaN'
	
		# Write data
		lat_varid[:] = lat
		lon_varid[:] = lon

		####################################### Header 
		ncid.License     = 'The file was created by C.Hart, https://github.com/ChristianHart2019'
		ncid.history     = 'Created ' + time.ctime(time.time())

		#####################
		ncid.close()
