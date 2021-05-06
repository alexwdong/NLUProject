#!/bin/bash

#SBATCH --job-name=hdrnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load anaconda3/2020.07

source ~/.bashrc
conda activate NLUProject

cd ~

export PYTHONPATH="./"

# Execute the script
py3=/scratch/awd275/miniconda3/envs/NLUProject/bin/python3
pyscript=~/NLUProject/download_data.py

$py3 $pyscript 




