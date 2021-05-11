#!/bin/bash
#SBATCH --job-name=baseline_encoded_segments
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=9:00:00

module purge
. ~/.bashrc

overlay_ext3="/scratch/$USER/overlay-25GB-500K.ext3"
conda_env="/ext3/miniconda3/envs/dsga1012"

dataset=$1
model=$2
sequence_length=$3
shift_length=$4

singularity \
    exec --nv --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/centos-7.8.2003.sif \
    /bin/bash -c ". /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh; \
                  conda activate $conda_env
                  python ./make_baseline_bert_encoded_segments.py -d $dataset -m $model -l $sequence_length -s $shift_length"
