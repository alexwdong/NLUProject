#!/bin/bash

#SBATCH --job-name=hdrnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load anaconda3/2020.07
export PYTHONPATH="./"

# Execute the script
python ~/NLUProject/scripts/download_data.py