from datasets import load_from_disk,load_dataset
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertModel

import argparse
import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def create_segments_list(cutoff_indices, sentence_list,tokenizer):
    '''
        Input:
            cutoff_indices: a list of cutoff indices. each index should be in the range of 0 to n-1, where n=len(sentence_list)
            sentence_list: a list of sentences from sent_tokenize
            tokenizer: the tokenizer for the model.
        Returns:
            segments_list: a list of 3-tuples of type BatchEncoding. This 3-tuple is the output of encode_plus
    '''
    segments_list = []
    #If cutoff indices is an empty list, means we don't split at all. then all the sentences get joined into one segment
    if len(cutoff_indices) == 0: 
        segment = "".join(sentence_list)
        encoded_segment = tokenizer.encode_plus(segment, add_special_tokens=True, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        segments_list.append(encoded_segment)
        return segments_list
    #Make first n-1 splits
    start_idx = 0
    segments_list = []
    for split_idx in cutoff_indices: 
        grouped_sentences_list = sentence_list[start_idx:split_idx+1] 
        segment = "".join(grouped_sentences_list)
        encoded_segment = tokenizer.encode_plus(segment, add_special_tokens=True, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        segments_list.append(encoded_segment)
        start_idx = split_idx+1
    # make last split
    grouped_sentences_list = sentence_list[start_idx:] 
    segment = "".join(grouped_sentences_list)
    encoded_segment = tokenizer.encode_plus(segment, add_special_tokens=True, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    segments_list.append(encoded_segment)
    #Return 
    return segments_list

class SegmentDataset(Dataset):
    def __init__(self, dataset_list,configs, label_to_cutoff_indices_dict,tokenizer):
        self.label_to_label_idx_dict = {}
        for ii,label in enumerate(configs):
            self.label_to_label_idx_dict[label]=ii
        
        self.data = []
        for label, sub_dataset in dataset_list:
            print('applying splits for label: ',label)
            cutoff_indices_dict = label_to_cutoff_indices_dict[label]
            for ii, entry in enumerate(sub_dataset):
                context = entry['text']
                sentence_list = sent_tokenize(context)
                cutoff_indices = cutoff_indices_dict[ii]
                segments_list = create_segments_list(cutoff_indices, sentence_list,tokenizer)
                data_entry = (self.label_to_label_idx_dict[label], segments_list)
                self.data.append(data_entry)
        
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self,idx):
        return(self.data[idx])
    
class OnTheFlyDataset(Dataset):
    def __init__(self, encode_plus_out_list):
        self.encode_plus_out_list = encode_plus_out_list
        
    def __len__(self):
        return len(self.encode_plus_out_list)
 
    def __getitem__(self,idx):
        return(self.encode_plus_out_list[idx])

    
def squeeze_tensors(batch):
    '''
        batch has four dimensions (b_size,useless,useless, 512 (representing padded tokens))
        We want to squeeze the second and third dimensions
    '''
    batch['input_ids'] = batch['input_ids'].squeeze(axis=1).squeeze(axis=1)
    batch['token_type_ids'] = batch['token_type_ids'].squeeze(axis=1).squeeze(axis=1)
    batch['attention_mask'] = batch['attention_mask'].squeeze(axis=1).squeeze(axis=1)
    return batch


# ArgParse
parser = argparse.ArgumentParser(description='Takes "label_to_cutoff_indices" pickle file, and creates BERT encoded segments')

parser.add_argument('-t','--threshold',help='threshold. This isnt technically required, because the threshold is already used in the previous script (make_cutoff_indices), but this helps for loading the correct file.', required=True)
parser.add_argument('-d', '--dataset', help='what dataset are we using (currently only newsgroup is accepted)', default='newsgroup')
parser.add_argument('-m', '--model', help='A string, the model id of a pretrained model hosted inside a model repo on huggingface.co.', required=True)
args = vars(parser.parse_args())

threshold = float(args['threshold'])
dataset = args['dataset']
model = args['model']
raw_dir = '../data/raw/' + dataset + '/'
segmentations_dir = '../data/segmentations/' + dataset + '/' + model + '/'
embeddings_dir = '../data/embeddings/' + dataset + '/' + model + '/'

if dataset == '20news':
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
    splits = ['train', 'test']
    for split in splits:
        if not os.path.exists(embeddings_dir + split):
            os.makedirs(embeddings_dir + split)

# Start Script
if __name__ == "__main__":

    # Start Script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    
    tokenizer = BertTokenizer.from_pretrained(model)
    bert_model= BertModel.from_pretrained(model)
    bert_model.eval()
    bert_model.to(device)
    for split in splits:
        dataset_list = []
        #Create (train, val or test) Dataset list 
        for config in newsgroup_configs:
            subset_path = raw_dir + split + '/'+ config
            dataset_list.append((config, load_from_disk(subset_path)))
        
        # Load the label_to_cutoff_indices pkl file, which contains the sentence splits for each long document.
        label_to_cutoff_indices_file = segmentations_dir + split + '/label_to_cutoff_indices_' + str(threshold) + '.pkl'
        with open(label_to_cutoff_indices_file, 'rb') as handle:
            label_to_cutoff_indices_dict = pickle.load(handle)


        #Create a Segment Dataset which contains tuples of (label - int, list of segments - list of 3-tuple which is output from tokenizer.encode_plus))
        split_set = SegmentDataset(dataset_list, newsgroup_configs, label_to_cutoff_indices_dict, tokenizer)
        split_loader = DataLoader(split_set, batch_size=1, shuffle=False, pin_memory=True)
        
        #Initialize bert_encoded_segments_list, this will contain the output that we want to dump
        bert_encoded_segments_list = []
        with torch.no_grad():
            for idx, batch in enumerate(split_loader):
                label =  batch[0]
                encoded_segments = batch[1]
                onthefly_dataset = OnTheFlyDataset(encoded_segments)
                onthefly_loader = DataLoader(onthefly_dataset, batch_size=128, shuffle=False, pin_memory=True)
                batch_encoded_seg_list = []
                for ii, small_batch in enumerate(onthefly_loader):
                    small_batch = squeeze_tensors(small_batch)
                    batch_input_ids = small_batch['input_ids'].to(device)
                    batch_token_type_ids = small_batch['token_type_ids'].to(device)
                    batch_attention_mask = small_batch['attention_mask'].to(device)
                    out = bert_model(batch_input_ids, batch_token_type_ids, batch_attention_mask)
                    # out['last_hidden_state'] is bsize x seq_len x embedding_size. We want to take only the embedding
                    # which corresponds to the CLS token.
                    sub_bert_encoded_segments = out['last_hidden_state'][:,0,:] #take only the first
                    batch_encoded_seg_list.append(sub_bert_encoded_segments)
                bert_encoded_segments = torch.cat(batch_encoded_seg_list)
                bert_encoded_segments_list.append((label,bert_encoded_segments.cpu()))
        file_name = 'bert_encoded_segments_list_'
        with open(embeddings_dir + split +'/' + file_name + str(threshold) +'.pkl', 'wb') as handle:
            pickle.dump(bert_encoded_segments_list, handle, protocol=pickle.HIGHEST_PROTOCOL)