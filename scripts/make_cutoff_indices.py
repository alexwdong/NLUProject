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


# Start Script
if __name__ == "__main__":
    # Use full sized bert model tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    newsgroup_configs = ['bydate_alt.atheism',
                         'bydate_comp.graphics',
                         'bydate_comp.os.ms-windows.misc',
                         'bydate_comp.sys.ibm.pc.hardware',
                         'bydate_comp.sys.mac.hardware',
                         'bydate_comp.windows.x',
                         'bydate_misc.forsale',
                         'bydate_rec.autos',
                         'bydate_rec.motorcycles',
                         'bydate_rec.sport.baseball',
                         'bydate_rec.sport.hockey',
                         'bydate_sci.crypt',
                         'bydate_sci.electronics',
                         'bydate_sci.med',
                         'bydate_sci.space',
                         'bydate_soc.religion.christian',
                         'bydate_talk.politics.guns',
                         'bydate_talk.politics.mideast',
                         'bydate_talk.politics.misc',
                         'bydate_talk.religion.misc']

    splits = ['train','test']

    data_dir = r'\\wsl$\Ubuntu-20.04\home\jolteon\NLUProject\data\20news\\'
    processed_dir = data_dir + 'processed\\'
    print('using data_dir: ', data_dir)
    threshold=1.0 
    print('making splits using prob threshold of: ', threshold)
    for split in splits: #Loop over train, test
        #Load all newsgroups into dataset_list
        dataset_list = []
        for config in newsgroup_configs:
            subset_path = data_dir + split+ '\\'+ config
            dataset_list.append((config,load_from_disk(subset_path)))
        
        # Create label_to_cutoff_indices_dict
        label_to_cutoff_indices_dict = {}
        file_name = 'label_to_cutoff_indices_'
        for label, sub_dataset in dataset_list:
            #Load the probability of split from qid_struct
            with open(processed_dir + split + '\\' + label+"_qid_struct.pkl", 'rb') as handle:
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
        
        with open(processed_dir + split + '\\' + file_name + str(threshold) +'.pkl', 'wb') as handle:
            pickle.dump(label_to_cutoff_indices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



