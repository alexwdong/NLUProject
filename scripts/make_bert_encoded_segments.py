import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import BertTokenizer,BertModel

from torch.utils.data import Dataset

from datasets import load_from_disk,load_dataset

import pickle

def create_segments_list(cutoff_indices, sentence_list,tokenizer):
    # prob_seq contains the indices on which to split on.
    # e.g, prob seq contains idx 3
    # then 0,1,2,3 are one segment, 4,5,... are another
    
    # e.g, prob seq contains idx 0
    # then 0 is one segment, 1,2,3,... are another
    segments_list = []
    #If cutoff indices is an empty list, means we don't split at all. then all the sentences get joined into one segment
    if len(cutoff_indices) == 0: 
        segment = "".join(sentence_list)
        encoded_segment = tokenizer.encode(segment,padding='max_length',max_length=512,truncation=True,return_tensors='pt')
        segments_list.append(encoded_segment)
        return segments_list
    #Make first n-1 splits
    start_idx = 0
    segments_list = []
    for split_idx in cutoff_indices: 
        grouped_sentences_list = sentence_list[start_idx:split_idx+1] 
        segment = "".join(grouped_sentences_list)
        encoded_segment = tokenizer.encode(segment,padding='max_length',max_length=512,truncation=True,return_tensors='pt')
        segments_list.append(encoded_segment)
        start_idx = split_idx+1
    # make last split
    grouped_sentences_list = sentence_list[start_idx:] 
    segment = "".join(grouped_sentences_list)
    encoded_segment = tokenizer.encode(segment,padding='max_length',max_length=512,truncation=True, return_tensors='pt')
    segments_list.append(encoded_segment)
    #Return 
    return segments_list

class NewsgroupDataset(Dataset):
    def __init__(self, dataset_list,configs, label_to_cutoff_indices_dict,tokenizer):
        print("In Newsgroup Dataset")
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
                segments_list = create_segments_list(cutoff_indices,sentence_list,tokenizer)
                data_entry = (self.label_to_label_idx_dict[label],segments_list)
                self.data.append(data_entry)
        
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self,idx):
        return(self.data[idx])
    
class OnTheFlyDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
        
    def __len__(self):
        return self.tensor.shape[0]
 
    def __getitem__(self,idx):
        return(self.tensor[idx])

# Start Script
if __name__ == "__main__":
    threshold = 1.0
    data_dir = r'\\wsl$\Ubuntu-20.04\home\jolteon\NLUProject\data\20news\\'
    processed_dir = data_dir + 'processed\\'
    
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
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    
    dataset_list = []
    splits = ['train','test']
    for split in splits:
        for config in newsgroup_configs:
            subset_path = data_dir + split + '\\'+ config
            dataset_list.append((config,load_from_disk(subset_path)))

        label_to_cutoff_indices_file = \
            r'\\wsl$\Ubuntu-20.04\home\jolteon\NLUProject\data\20news\processed\\' + \
            split + '\label_to_cutoff_indices_'+str(threshold)+'.pkl'
        with open(label_to_cutoff_indices_file, 'rb') as handle:
            label_to_cutoff_indices_dict = pickle.load(handle)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        split_set = NewsgroupDataset(dataset_list,newsgroup_configs,label_to_cutoff_indices_dict,tokenizer)

        bert_model= BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        bert_model.to(device)

        split_loader = DataLoader(split_set, batch_size=1, shuffle=False, pin_memory=True)
        
        bert_encoded_segments_list = []
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(train_loader)):
                label =  batch[0]
                segments = torch.cat(batch[1],axis=0)
                segments = segments.squeeze(axis=1)
                segments.to(device)
                onthefly_dataset = OnTheFlyDataset(segments)
                onthefly_loader =  DataLoader(onthefly_dataset, batch_size=4, shuffle=False, pin_memory=True)
                batch_encoded_seg_list = []
                for ii, small_batch in enumerate(onthefly_loader):
                    out = bert_model(input_ids=small_batch.to(device))
                    sub_bert_encoded_segments = out['pooler_output']
                    batch_encoded_seg_list.append(sub_bert_encoded_segments)
                bert_encoded_segments = torch.cat(batch_encoded_seg_list)
                bert_encoded_segments_list.append((label,bert_encoded_segments.cpu()))
            file_name = 'bert_encoded_segments_list_'
            with open(processed_dir+ split+'\\' + file_name + str(threshold) +'.pkl', 'wb') as handle:
                pickle.dump(bert_encoded_segments_list, handle, protocol=pickle.HIGHEST_PROTOCOL)