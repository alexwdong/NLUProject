#!/bin/bash
#SBATCH --job-name=make_cutoff_indices
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

module purge
. ~/.bashrc

overlay_ext3="/scratch/$USER/overlay-25GB-500K.ext3"
conda_env="/ext3/miniconda3/envs/dsga1012"

dataset=$1
model=$2
threshold=$3

singularity \
    exec --nv --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/centos-7.8.2003.sif \
    /bin/bash -c ". /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh; \
                  conda activate $conda_env
                  python ./make_cutoff_indices.py -d $dataset -m $model -t $threshold"