#! /bin/bash

mode=newsgroup
data_dir='/home/jolteon/NLUProject/data/20news/'
processed_dir='/home/jolteon/NLUProject/data/20news/processed/'
python3='/mnt/e/miniconda3/envs/NLUProject/python.exe'
length=200
shift=50

echo $python3
echo $data_dir

$python3 ./make_baseline_bert_encoded_segments.py -m $mode -d $data_dir -p $processed_dir -l $length -s $shift

