############################################################
### This script creates an initial conditions  files for ###
### SUMMA that is specific to snoTEL site locations	 	 ###	
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
import os
from datetime import date, timedelta
from scipy.io import netcdf
from pathlib import Path

# Define the path with Path() because that avoids issues with how different OS' handle '/' and '\'
list_path = Path('/Users/cjh458/Desktop/GHCND-metSIM-SUMMA-workflow/4CREATE_SUMMA_FILES/LISTS/')		# A directory containing list control variables
data_path = Path('/Users/cjh458/Desktop/GHCND-metSIM-SUMMA-workflow/4CREATE_SUMMA_FILES/LISTS/')		# A path that has an existing initial conditions file
output_path = Path('/Users/cjh458/Desktop/GHCND-metSIM-SUMMA-workflow/4CREATE_SUMMA_FILES/OUTPUT/')	# A path where the output file will be written

# read in a .csv containing snoTEL information and parse into variables
os.chdir(list_path)															# Change directory into the folder containing list data

datafile='LIST_LLES.csv'												# Open a file that contains the snoTEl information to be written into the attributes file
data = pd.read_csv(datafile, sep=',')										# Read this file using pandas read_csv function to separate variables
latitude = data['lat'].values[:]											# Extract an Latitude variable from this file
longitude = data['lon'].values[:]											# Extract an Longitude variable from this file
elevation = data['elev'].values[:]											# Extract an Elevation variable from this file
snotelid = data['site_id'].values[:]											# Extract an snotel id variable from this file
hruid = data['hru_id'].values[:]												# Extract an hru id variable from this file

with open ("LIST_METSIM.csv") as myfile:									# Open list containing names of Metsim-snoTEL simulations
	datafile = myfile.read().split('\n')									# Split up original .CSV into rows

data_length=len(datafile)														# Calculate length of forcing data files (number of rows)
os.chdir(data_path)															# Change directory into the folder that will contain the attribute files

#######################################
# netCDF reading and creation
#######################################

outfile='snotel_trialParams.nc'												# Create name for an output file
infile='trialParams.nc'														# Create name for a data input file

os.chdir(data_path)															# Change directory into the folder that will contain the attribute files
icid = nc4.Dataset(infile, "r", format="NETCDF4")							# Open a file for reading original attributes data
os.chdir(output_path)														# Change directory into the folder that will contain the attribute files
ncid = nc4.Dataset(outfile, "w", format="NETCDF4")							# Open a new file for writing altered attributes data
# Declare dimensions for the new output file				
dimid_hru = ncid.createDimension('hru',data_length)							# Declare hydrological response unit dimension of 631
dimid_gru = ncid.createDimension('gru',data_length)							# Declare geographical response unit dimension of 631

#######################################
# Declare an hruId value = declare
#######################################
							
hruId_varid = ncid.createVariable('hruId','i4',('hru')) 					# Create hruId variable in the new datafile
# Declare attributes for the new variable 												
hruId_varid.long_name      = 'Ids for hydrological response units'
hruId_varid.units          = '-'
hruId_varid.longname       = 'hru id'
# Assign variable  value
for z in range(data_length):													# loop through the snotel sites
	hruId_varid[z]= z+1 													# Assign variable a value from the snoTEL information .csv


#######################################
# Declare an gruId value = declare
#######################################
							
gruId_varid = ncid.createVariable('gruId','i4',('gru')) 					# Create gruId variable in the new datafile
# Declare attributes for the new variable 												
gruId_varid.long_name      = 'Ids for geographical response units'
gruId_varid.units          = '-'
gruId_varid.longname       = 'gru id'
# Assign variable  value
for z in range(data_length):													# loop through the snotel sites
	gruId_varid[z]= z+1 


#######################################
# Declare an critSoilTranspire value = declare
#######################################
							
critSoilTranspire_varid = ncid.createVariable('critSoilTranspire','d',('gru')) 	# Create critSoilTranspire variable in the new datafile
# Assign variable  value
for z in range(data_length):													# loop through the snotel sites
	critSoilTranspire_varid[z]= 0.175


#######################################
# Declare an theta_res value = declare
#######################################
							
theta_res_varid = ncid.createVariable('theta_res','d',('gru')) 				# Create theta_res variable in the new datafile
# Assign variable  value
for z in range(data_length):													# loop through the snotel sites
	theta_res_varid[z]= 0.139


#######################################
# Declare an theta_sat value = declare
#######################################
							
theta_sat_varid = ncid.createVariable('theta_sat','d',('gru')) 				# Create theta_sat variable in the new datafile
# Assign variable  value
for z in range(data_length):													# loop through the snotel sites
	theta_sat_varid[z]= 0.55


#######################################
# Declare an critical soil wilting
#######################################
							
critSoilWilting_varid = ncid.createVariable('critSoilWilting','d',('gru')) 	# Create critSoilWilting variable in the new datafile
# Assign variable  value
for z in range(data_length):													# loop through the snotel sites
	critSoilWilting_varid[z]= 0.145		#0.075

########### write soil properties from table #############

soil_dens_intr_varid = ncid.createVariable('soil_dens_intr','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	soil_dens_intr_varid[z]= 2700.0000
	
thCond_soil_varid = ncid.createVariable('thCond_soil','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	thCond_soil_varid[z]= 5.5000
	
frac_sand_varid = ncid.createVariable('frac_sand','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	frac_sand_varid[z]= 0.1600
	
frac_silt_varid = ncid.createVariable('frac_silt','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	frac_silt_varid[z]= 0.2800
	
frac_clay_varid = ncid.createVariable('frac_clay','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	frac_clay_varid[z]= 0.5600
	
fieldCapacity_varid = ncid.createVariable('fieldCapacity','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	fieldCapacity_varid[z]= 0.2000
	
wettingFrontSuction_varid = ncid.createVariable('wettingFrontSuction','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	wettingFrontSuction_varid[z]= 0.3000
	
theta_mp_varid = ncid.createVariable('theta_mp','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	theta_mp_varid[z]= 0.4010
	
vGn_alpha_varid = ncid.createVariable('vGn_alpha','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	vGn_alpha_varid[z]= -0.8400
	
vGn_n_varid = ncid.createVariable('vGn_n','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	vGn_n_varid[z]= 1.3000
	
mpExp_varid = ncid.createVariable('mpExp','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	mpExp_varid[z]= 5.0000
	
k_soil_varid = ncid.createVariable('k_soil','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	k_soil_varid[z]= 0.0000075
	
k_macropore_varid = ncid.createVariable('k_macropore','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	k_macropore_varid[z]= 0.0030
	
kAnisotropic_varid = ncid.createVariable('kAnisotropic','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	kAnisotropic_varid[z]= 1.0000
	
zScale_TOPMODEL_varid = ncid.createVariable('zScale_TOPMODEL','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	zScale_TOPMODEL_varid[z]= 2.5000

compactedDepth_varid = ncid.createVariable('compactedDepth','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	compactedDepth_varid[z]= 1.0000
	
aquiferBaseflowRate_varid = ncid.createVariable('aquiferBaseflowRate','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	aquiferBaseflowRate_varid[z]= 0.1000
	
aquiferScaleFactor_varid = ncid.createVariable('aquiferScaleFactor','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	aquiferScaleFactor_varid[z]= 0.3500

aquiferBaseflowExp_varid = ncid.createVariable('aquiferBaseflowExp','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	aquiferBaseflowExp_varid[z]= 2.0000

qSurfScale_varid = ncid.createVariable('qSurfScale','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	qSurfScale_varid[z]= 50.0000

specificYield_varid = ncid.createVariable('specificYield','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	specificYield_varid[z]= 0.2000

specificStorage_varid = ncid.createVariable('specificStorage','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	specificStorage_varid[z]= 0.000006

f_impede_varid = ncid.createVariable('f_impede','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	f_impede_varid[z]= 2.0000
	
soilIceScale_varid = ncid.createVariable('soilIceScale','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	soilIceScale_varid[z]= 0.1300
	
soilIceCV_varid = ncid.createVariable('soilIceCV','d',('gru')) 				# Create theta_sat variable in the new datafile
for z in range(data_length):													# loop through the snotel sites
	soilIceCV_varid[z]= 0.4500

# soil_dens_intr            |    2700.0000
# thCond_soil               |       5.5000 
# frac_sand                 |       0.1600
# frac_silt                 |       0.2800
# frac_clay                 |       0.5600
# fieldCapacity             |       0.2000
# wettingFrontSuction       |       0.3000
# theta_mp                  |       0.4010
# theta_sat                 |       0.5500
# theta_res                 |       0.1390
# vGn_alpha                 |      -0.8400
# vGn_n                     |       1.3000
# mpExp                     |       5.0000
# k_soil                    |      7.5d-06
# k_macropore               |      1.0d-03
# kAnisotropic              |       1.0000
# zScale_TOPMODEL           |       2.5000
# compactedDepth            |       1.0000
# aquiferBaseflowRate       |       0.1000
# aquiferScaleFactor        |       0.3500
# aquiferBaseflowExp        |       2.0000
# qSurfScale                |      50.0000
# specificYield             |       0.2000
# specificStorage           |       1.d-06
# f_impede                  |       2.0000
# soilIceScale              |       0.1300
# soilIceCV                 |       0.4500

ncid.close()																# close New Datafile
icid.close()																# close "coldState" initial conditions file	




