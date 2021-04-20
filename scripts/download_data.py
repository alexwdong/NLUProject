from datasets import load_dataset,load_from_disk

dataset = load_dataset('trivia_qa', 'rc',cache_dir = '/scratch/awd275/NLU_data')
dataset.save_to_disk('.')
