#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=8:00:00

module purge
. ~/.bashrc

overlay_ext3="/scratch/$USER/overlay-25GB-500K.ext3"
conda_env="/ext3/miniconda3/envs/dsga1012"

dataset_name=$1

singularity \
    exec --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/centos-7.8.2003.sif \
    /bin/bash -c ". /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh; \
                  conda activate $conda_env
                  python ./download_data.py --dataset_name=$dataset_name"