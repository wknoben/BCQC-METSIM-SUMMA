#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=10G
#SBATCH --time=03:15:00
#SBATCH --account=def-kshook
#SBATCH --job-name=run_24620 
mkdir -p /home/chart/summa-develop-graham/settings/SUMMA_0331/OUTPUT
#SBATCH --nodes=5 
SUMMA_EXE=/home/chart/summa-develop-graham/bin/summa.exe 
echo "Starting run at: `date`" 
#${SUMMA_EXE} -r never -m ~/summa_scripts/4_settings/NLDAS_default/fileManager.txt 
${SUMMA_EXE} -g 713 1 -r never -m /home/chart/summa-develop-graham/settings/SUMMA_0331/01_SnowEval_201704_201812/fileManager_plato.txt
echo "Program finished with exit code $? at: `date`" 
