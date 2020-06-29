{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf600
{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
# Download wind data from ERA5\
\
ERA5 is the most recent climate reanalysis dataset produced by ECMWF (European Centre for Medium-Range Weather Forcasts). This data set contains several land surface, ocean, and atmospheric variables at a sub-daily frequency. This dataset was produced using 4D-Var data assimilation in ECMWF's integrated forecast system and was quality controlled by Wouter Knoben using ERA5_QualityControl_Level0.py. The dataset is currently stored on the Graham.computecanada high-performance computing cluster. First, a Coordinates.csv was produced by finding the closest match of the locations of SUMMA hydrological response units to those of the ERA5 reanalysis product using the CalculateCoordinates.m script. This coordinates file is then used by DOWNLOAD_ERA5_WIND.sh to download the time periods and locations of interest from the ERA5 database. Results are then stored in the OUTPUT folder.}