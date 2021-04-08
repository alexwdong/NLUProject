#!/bin/bash
#SBATCH --job-name=hdrnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=awd275@nyu.edu
#SBATCH --output=slurm_%j.out


# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate NLUProject

cd /home/awd275/NLUProject/

export PYTHONPATH="./"

# Execute the script
py3=/scratch/awd275/miniconda3/envs/NLUProject/bin/python3
pyscript=/home/awd275/NLUProject/scripts/make_text_splits_dict.py

$py3 $pyscript 




