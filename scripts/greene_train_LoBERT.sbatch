#!/bin/bash
#SBATCH --job-name=train_LoBERT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=9:00:00

module purge
. ~/.bashrc

overlay_ext3="/scratch/$USER/overlay-25GB-500K.ext3"
conda_env="/ext3/miniconda3/envs/dsga1012"

epochs=$1
input_file=$2

singularity \
    exec --nv --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/centos-7.8.2003.sif \
    /bin/bash -c ". /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh; \
                  conda activate $conda_env
                  python ./train_LoBERT.py -e $epochs -i $input_file"
