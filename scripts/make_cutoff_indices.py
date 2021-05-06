import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import BertTokenizer,BertModel

from torch.utils.data import Dataset

from datasets import load_from_disk,load_dataset

import pickle
import argparse

def get_sentence_list(tokenizer,context):
    sentence_list = sent_tokenize(context)
    #print('len sentence_list: ',len(sentence_list))
    encoded_sentence_list = [(tokenizer.encode(sentence)) for sentence in sentence_list]
    #print('len encoded_sentence_list: ',len(encoded_sentence_list))
    tokens_per_sentence_list = [len(sentence)for sentence in encoded_sentence_list]
    #print('tokens_per_sentence_list: ', tokens_per_sentence_list)
    return encoded_sentence_list, tokens_per_sentence_list

def apply_threshold(prob_seq,tokens_per_sentence_list,threshold):
    '''
    If prob_seq is empty, we will return and empty list.
    '''
    # Initialize
    cutoff_indices = []
    running_length = tokens_per_sentence_list[0]
    # 
    for ii,prob in enumerate(prob_seq):
        if prob <= threshold:
            cutoff_indices.append(ii)
            running_length = tokens_per_sentence_list[ii+1]
            
        elif running_length + tokens_per_sentence_list[ii+1] > 512:
            cutoff_indices.append(ii)
            running_length = tokens_per_sentence_list[ii+1]
            
        else:
            running_length += tokens_per_sentence_list[ii+1]
        
    return cutoff_indices

def get_cutoff_indices(text, threshold, nsp_model,tokenizer, device):
    
    prob_seq, sentence_list = get_probabilities_on_text_w_NSP(nsp_model, text, tokenizer,device)
    tokens_per_sentence_list = get_tokens_per_sentence_list(tokenizer, sentence_list)
    cutoff_indices = apply_threshold(prob_seq, tokens_per_sentence_list, threshold=.5)
    
    return cutoff_indices


# ArgParse
parser = argparse.ArgumentParser(description='Takes "qid_struct" pickle file, and creates label_to_cutoff_indices')

parser.add_argument('-t','--threshold',help='Probability Threshold where the split occurs if NSP falls below the threshold',required = True)
args = vars(parser.parse_args())

threshold = float(args['threshold'])
data_dir = args['data_dir']
processed_dir = args['processed_dir']

splits = ['train','test']

# Start Script
if __name__ == "__main__":
    # Use full sized bert model tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for split in splits: #Loop over train, test
        #Load all newsgroups into dataset_list
        dataset_list = []
        for config in newsgroup_configs:
            subset_path = RAW_DIR(f'20news/{split}/{config}')
            dataset_list.append((config,load_from_disk(subset_path)))
        
        # Create label_to_cutoff_indices_dict
        label_to_cutoff_indices_dict = {}
        for label, sub_dataset in dataset_list:
            #Load the probability of split from qid_struct
            with open(SEGMENT_DIR(f'20news/{split}/{label}_qid_struct.pkl'), 'rb') as handle:
                #qid struct is just index ii for newsgroup dataset
                #qid struct is question id for wikihop dataset
                qid_struct = pickle.load(handle)

            idx_to_cutoff_indices = {}
            for ii, entry in enumerate(sub_dataset):
                context = entry['text']
                sentence_list, tokens_per_sentence_list = get_sentence_list(tokenizer,context)
                prob_seq = qid_struct[ii]
                cutoff_indices = apply_threshold(prob_seq, tokens_per_sentence_list, threshold=threshold)
                idx_to_cutoff_indices[ii] = cutoff_indices
            label_to_cutoff_indices_dict[label] = idx_to_cutoff_indices
        
        output_file = SEGMENT_DIR(f'20news/{split}/label_to_cutoff_indices_{threshold}.pkl')
        with open(output_file, 'wb') as handle:
            pickle.dump(label_to_cutoff_indices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



