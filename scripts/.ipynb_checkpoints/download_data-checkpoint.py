from datasets import load_dataset,load_from_disk

# Make a symbolic link between ~/NLU_data and /scratch/[user]/NLU_data
# example: ln -s /scratch/ay1626/NLU_data ~/NLU_data
dataset = load_dataset('trivia_qa', 'rc',cache_dir = '~/NLU_data')
dataset.save_to_disk('.')
