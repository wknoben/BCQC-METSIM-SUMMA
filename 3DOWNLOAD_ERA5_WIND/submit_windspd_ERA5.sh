#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --account=def-kshook

module load python/3.7.4
module load scipy-stack/2019a
module load netcdf
# module laod hdf5

python3 DOWNLOAD_ERA5_WIND.sh
