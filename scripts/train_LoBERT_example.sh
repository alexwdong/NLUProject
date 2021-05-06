#! /bin/bash

epochs=50
input_file='/home/jolteon/NLUProject/data/20news/processed/train/bert_encoded_segments_list_overlap_200_50.pkl'
model_save_path='/home/jolteon/NLUProject/data/20news/processed/train/bert_encoded_segments_list_overlap_200_50.pkl'
output_path='/home/jolteon/NLUProject/models/standard_overlap_200_50_results.pkl'

python3='/mnt/e/miniconda3/envs/NLUProject/python.exe'

conda activate NLUProject
echo $python3
$python3 ./train_LoBERT.py -e $epochs -i $input_file -m $model_save_path -o $output_path