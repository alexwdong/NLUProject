#! /bin/bash

threshold=.95
mode=newsgroup
data_dir='/home/jolteon/NLUProject/data/20news/'
processed_dir='/home/jolteon/NLUProject/data/20news/processed/'
python3='/mnt/e/miniconda3/envs/NLUProject/python.exe'
conda activate NLUProject
echo $python3
echo $data_dir
$python3 ./make_text_splits_dict.py -m $mode -d $data_dir -p $processed_dir
$python3 ./make_cutoff_indices.py -t $threshold -m $mode -d $data_dir -p $processed_dir
$python3 ./make_bert_encoded_segments.py -t $threshold -m $mode -d $data_dir -p $processed_dir