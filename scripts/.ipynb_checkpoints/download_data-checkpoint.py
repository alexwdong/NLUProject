from datasets import load_dataset, load_from_disk

exec(open("../header.py").read())



try:
    os.mkdir(RAW_DIR('trivia_qa'))
except FileExistsError:
    print(f"{RAW_DIR('trivia_qa')} already exists.")

dataset = load_dataset('trivia_qa', 'rc',cache_dir = RAW_DIR())
dataset.save_to_disk(RAW_DIR('trivia_qa'))

try:
    os.mkdir(RAW_DIR('newsgroup'))
except FileExistsError:
    print(f"{RAW_DIR('newsgroup')} already exists.")
    
dataset2 = load_dataset('newsgroup', 'rc',cache_dir = RAW_DIR())
dataset2.save_to_disk(RAW_DIR('newsgroup'))
