from datasets import load_dataset, load_from_disk

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloads dataset specified')
    parser.add_argument('-n', '--dataset_name', help='the dataset that is to be downloaded', required=True)

    args = vars(parser.parse_args())
    dataset_name = args['dataset_name']

    # BEFORE RUNNING: Make a symbolic link between ~/NLU_data and /scratch/[user]/NLU_data
    # example: ln -s /scratch/ay1626/NLU_data ~/NLU_data
    # also, mkdir ~/NLU_data/raw

    BASE_PATH = '../data/raw/'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    dataset = load_dataset(dataset_name, 'rc', cache_dir=BASE_PATH)
    dataset.save_to_disk(BASE_PATH + dataset_name) 