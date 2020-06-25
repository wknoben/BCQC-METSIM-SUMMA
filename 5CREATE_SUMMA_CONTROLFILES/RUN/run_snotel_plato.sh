#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=10G 
#SBATCH --time=1:00:00 
#SBATCH --job-name=run_24620 
mkdir -p /home/cjh458/summa/summa-22-10-2019/settings/SETTINGS-snoTEL_03-31/OUTPUT
SUMMA_EXE=/home/cjh458/summa/summa-22-10-2019/build/bin/summa.exe 
echo "Starting run at: `date`" 
#${SUMMA_EXE} -r never -m ~/summa_scripts/4_settings/NLDAS_default/fileManager.txt 
${SUMMA_EXE} -g 713 1 -r never -m /home/cjh458/summa/summa-22-10-2019/settings/SETTINGS-snoTEL_03-31/01_SnowEval_201704_201812/fileManager_plato.txt
echo "Program finished with exit code $? at: `date`" 