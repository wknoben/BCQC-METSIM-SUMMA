############################################################
### This script formats output variables from MetSim     ###
###   and NLDAS netCDF files for use with SUMMA			     ###
############################################################

# load all the necessary modules and packages
import pandas as pd
import numpy as np
import re as re
import datetime
from datetime import date, timedelta
from datetime import datetime
import glob
import scipy
import os
from scipy.io import netcdf
import netCDF4 as nc4
import time
import csv
import os

#######################################
# read in the names of metsim output files
#######################################
os.chdir('/Users/cjh458/Desktop/GHCND-metSIM-SUMMA-workflow-master/4CREATE_SUMMA_FILES/LISTS')						# Change directory into the folder containing lists of names

with open ("LIST_METSIM.csv") as myfile:				# Open list containing names of Metsim-snoTEL simulations
	datafile = myfile.read().split('\n')					# Split up original .CSV into rows

with open ("LIST_WIND.csv") as myfile:				  # Open list containing names of ERA5 wind data
	windfile = myfile.read().split('\n')					# Split up original .CSV into rows

with open ("LIST_STATION.csv") as myfile:				# Open list containing names of Metsim-snoTEL simulations
	hrufile =  myfile.read().split('\n')					# Split up original .CSV into rows

data_length=len(datafile)	# 							    	# Calculate length of MetSim data (number of rows)

#######################################
# netCDF opening and creation
#######################################
outfile='snotel_forcing.nc'
os.chdir('/Users/cjh458/Desktop/BCQC-metSIM-SUMMA/4CREATE_SUMMA_FILES/METSIM_RESULTS')						# Change directory into the folder containing lists of names
mcid = nc4.Dataset(datafile[1], "r", format="NETCDF4")		# Open the MetSim data file for reading
os.chdir('/Users/cjh458/Desktop/BCQC-metSIM-SUMMA/4CREATE_SUMMA_FILES/OUTPUT')						# Change directory into the folder containing lists of names
ncid = nc4.Dataset(outfile, "w", format="NETCDF4")			# Open a new file for writing Metsim and ERA5 data

time_len=(len(mcid.variables['time']))    # calculate how many time-steps are in the dataset

dimid_T = ncid.createDimension('time', time_len) # Declare time dimension that is the same length of the data file
dimid_hru = ncid.createDimension('hru', data_length)					# Declare hydrological response unit dimension of 1

#######################################
# Change names and attrib. of radiation
#######################################
os.chdir('/Users/cjh458/Desktop/BCQC-metSIM-SUMMA/4CREATE_SUMMA_FILES/METSIM_RESULTS')						# Change directory into the folder containing lists of names
# Create longwave variable in the new datafile
LWRadAtm_varid = ncid.createVariable('LWRadAtm','f',('time','hru'))

# Declare variable attributes
LWRadAtm_varid.units          = 'W m-2'
LWRadAtm_varid.long_name      = 'downward longwave radiation at the upper boundary'
LWRadAtm_varid.FillValue      = '-999.f'

# Write data to the variable
for z in range(data_length):											# Iterate through snotel sites
	mcid = nc4.Dataset(datafile[z], "r", format="NETCDF4")	# Open the MetSim data file for reading
	raw_data = mcid.variables['longwave']					# Open the longwave variable
	data = np.copy(raw_data)								# Make a copies of these variables
	LWRadAtm_varid[0:time_len,z] = data[0:time_len]			# Write date to LWRadAtm

# Create shortwave variable in the new datafile
SWRadAtm_varid = ncid.createVariable('SWRadAtm','f',('time','hru'))

# Declare variable attributes
SWRadAtm_varid.units          = 'W m-2'
SWRadAtm_varid.long_name      = 'downward shortwave radiation at the upper boundary'
SWRadAtm_varid.FillValue      = '-999.f'

# Write data to the variable
for z in range(data_length):											# Iterate through snotel sites
	mcid = nc4.Dataset(datafile[z], "r", format="NETCDF4")	# Open the MetSim data file for reading
	raw_data = mcid.variables['shortwave']					# Open the longwave variable
	data = np.copy(raw_data)								# Make a copies of these variables
	SWRadAtm_varid[0:time_len,z] = data[0:time_len]			# Write data to SWRadAtm

#######################################
# Format the airpres
#######################################
pres_varid = ncid.createVariable('airpres','f',('time','hru'))	# Create airpressure variable in the new datafile

# Declare variable attributes
pres_varid.units          = 'Pa'
pres_varid.long_name      = 'air pressure at the measurement height'
pres_varid.FillValue      = '-999.f'

# Write data to the variable
for z in range(data_length):											        # Iterate through snotel sites
	mcid = nc4.Dataset(datafile[z], "r", format="NETCDF4")	# Open the MetSim data file for reading
	raw_data = mcid.variables['air_pressure']				        # Open the air_pressure variable
	data = np.copy(raw_data)								                # Make a copy of this variable
	presdata=(data*1000)									                  # Convert kPa to Pa
	pres_varid[0:time_len,z] = presdata[0:time_len]			    # Update the original air pressure to the converted one

#######################################
# Format the airtemp
#######################################
temp_varid = ncid.createVariable('airtemp','f',('time','hru'))	# Create airtemp variable in the new datafile

# Declare variable attributes
temp_varid.units          = 'K'
temp_varid.long_name      = 'air temperature at the measurement height'
temp_varid.FillValue      = '-999.f'

# Write data to the variable
for z in range(data_length):											        # Iterate through snotel sites
	mcid = nc4.Dataset(datafile[z], "r", format="NETCDF4")	# Open the MetSim data file for reading
	raw_data = mcid.variables['temp']						            # Open the air_pressure variable
	data = np.copy(raw_data)								                # Make a copy of this variable
	tempdata = (data-237.15)								                # Convert C to K
	temp_varid[0:time_len,z] = tempdata[0:time_len]			    # Update the original air temperature to the converted one

#######################################
# Create data_step varaiable
#######################################
datastep_varid = ncid.createVariable('data_step','d')		# Create data_step variable in the new datafile

# Declare variable attributes
datastep_varid.long_name      = 'data step length in seconds'
datastep_varid.units          = 'seconds'

# Write data to the variable
datastep_varid[0] = 3600									# carry over the original data_step to the new one

#######################################
# Format Precipitation variable
#######################################
prec_varid = ncid.createVariable('pptrate','d',('time','hru'))	# Create pptrate variable in the new datafile

# Declare variable attributes
prec_varid.FillValue      = '-999.'
prec_varid.long_name      = 'Precipitation rate'
prec_varid.units          = 'kg m-2 s-1'

# Write data to the variable
for z in range(data_length):											# Iterate through snotel sites
	mcid = nc4.Dataset(datafile[z], "r", format="NETCDF4")	# Open the MetSim data file for reading
	raw_data = mcid.variables['prec']						# Open the prec variable
	data = np.copy(raw_data)								# Make a copy of this variable
	prec_data=(data/3600)									# Convert mm/timestep to kg m-2 s-1
	prec_varid[0:time_len,z] = prec_data[0:time_len]		# Update the original precipitation data and add to the converted dataset

#######################################
# Format Spechum variable
#######################################
spechum_varid = ncid.createVariable('spechum','f',('time','hru'))	# Create specific humidity variable in the new datafile

# Declare variable attributes
spechum_varid.units          = 'g g-1'
spechum_varid.long_name      = 'specific humidity at the measurement height'
spechum_varid.FillValue      = '-999.f'

# Write data to the variable
for z in range(data_length):											        # Iterate through snotel sites
	mcid = nc4.Dataset(datafile[z], "r", format="NETCDF4")	# Open the MetSim data file for reading
	raw_data = mcid.variables['vapor_pressure']				      # Open the vapor pressure variable
	raw_data1 = mcid.variables['air_pressure']				      # Open the air pressure variable
	data = np.copy(raw_data)								                # Make a copies of these variables
	data1 = np.copy(raw_data1)								              # Make a copies of these variables
	spechum_data=(0.622*data)/(data1-data*(1-0.622))		    # Convert air and vapor pressure to specific humidity
	spechum_varid[0:time_len,z] = spechum_data[0:time_len]	# Update the original specific humidity data and add to the converted dataset

#######################################
# Format the time stamp
#######################################
time_varid = ncid.createVariable('time','d',('time'))	# Create time variable in the new datafile

# Declare variable attributes
time_varid.units          = 'days since 1990-01-01 00:00:00'    # Declare time formatting
time_varid.long_name      = 'Observation time'                  # Add time attribute
time_varid.calendar       = 'standard'                          # Add another time attribute

# Write data to the variable
startdate = date(1990,1,1)     # Declare a start date the is used in all SUMMA simulations
date1 = date(2017,4,1)				# Declare a variable == to the time of the start of the SUMMA simulation
date2 = date(2019,1,1)				# Declare a variable == to the time of the end of the SUMMA simulation
time1 = (date1-startdate).days		# Calculate the time difference in days between the start of simulation and startdate
time2 = (date2-startdate).days		# Calculate the time difference between the end of simulation and startdate
int_time1 = int(time1)*24			# Multiply this difference by 24 and add the left over hours in the day to convert to total hours
int_time2 = int(time2)*24			# Multiply this difference by 24 and add the left over hours in the day to convert to total hou
time3 = np.arange(int_time1, int_time2)/24				# Create a time-series from the start and end date integers
time_varid[0:time_len] = time3[0:time_len] 				# Update the original time stamp to the converted one

#######################################
# attain and format the windspeed
#######################################
os.chdir('/Users/cjh458/Desktop/BCQC-metSIM-SUMMA/4CREATE_SUMMA_FILES/ERA5_WIND')						# Change directory into the folder containing lists of names
wind_varid = ncid.createVariable('windspd','d',('time','hru'))	# Create time variable in the new datafile

# Declare attributes for the new variable
wind_varid.units          = 'm s-1'
wind_varid.long_name      = 'wind speed at the measurement height'
wind_varid.FillValue      = '-999.f'

# Assign variable  value
for z in range(data_length):											      # Iterate through snotel sites
	mcid = nc4.Dataset(windfile[z], "r", format="NETCDF4")	# Open the wind data file for reading
	raw_data = mcid.variables['windspd']				          # Open the vapor pressure variable
	wind_data = np.copy(raw_data)								          # Make a copies of these variables
	wind_varid[0:time_len,z] = wind_data[0:time_len]               # Assign ERA5 data to file

#######################################
# Create hruId varaiable
#######################################
hruId_varid = ncid.createVariable('hruId','i',('hru'))		# Create hruId variable in the new datafile

# Declare variable attributes
hruId_varid.long_name      = 'hru id'
hruId_varid.units          = '-'

# Write data to the variable
for z in range(data_length):				# Iterate through snotel sites
	hruId_varid[z]=z+1 #hrufile[z]         # Assign hruId corresponding to the station used

mcid.close()												# close MetSim Datafile
ncid.close()												# close Output Datafile
