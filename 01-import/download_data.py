from datasets import load_dataset,load_from_disk

# BEFORE RUNNING: Make a symbolic link between ~/NLU_data and /scratch/[user]/NLU_data
# example: ln -s /scratch/ay1626/NLU_data ~/NLU_data
# also, mkdir ~/NLU_data/raw
dataset = load_dataset('trivia_qa', 'rc',cache_dir = '~/NLU_data/raw')
dataset.save_to_disk('~/NLU_data/raw/trivia_qa')
