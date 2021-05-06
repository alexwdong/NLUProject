import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader

from transformers import BertTokenizer,BertModel

from torch.utils.data import Dataset

from datasets import load_from_disk,load_dataset

import pickle
import logging
import argparse

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

class OnTheFlyDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
        
    def __len__(self):
        return self.tensor.shape[0]
 
    def __getitem__(self,idx):
        return(self.tensor[idx])
    
def create_overlaps(tokens, sequence_length=200, shift_length=50):
    cls_token = tokenizer.encode(tokenizer.cls_token,add_special_tokens=False,return_tensors='pt')[0]
    start_index = 0
    first_time = True
    all_sub_seq = []
    tokens_left = len(tokens)
    # Start the While loop - here, we try to get spans of 200 tokens, with a shift of 50. 
    # E.g if the sequence is 300 tokens, we get [0,199][50,249],[100,299]

    while tokens_left > 0:
        sub_seq = tokens[start_index:start_index+sequence_length]
        #Update Tokens left
        if first_time is True:
            tokens_left -=sequence_length
            first_time=False
        else:
            tokens_left -=shift_length
        # add start_idx
        start_index+=shift_length
        # Add new sub_sequence to our list of sub_sequences
        sub_seq_w_cls =torch.cat([cls_token,sub_seq]).unsqueeze(0)
        if tokens_left <=0: #if this is the last run, make sure to pad the last sequence to be 201 tokens long:
            sub_seq_w_cls = tokenizer.encode(sub_seq_w_cls.tolist()[0],padding='max_length',max_length=sequence_length + 1,add_special_tokens=False,return_tensors='pt')
        all_sub_seq.append(sub_seq_w_cls)
        
    return all_sub_seq

# Argparse
parser = argparse.ArgumentParser(description='Make baseline bert encoded segments. Baseline is splits as described in Pappagari 2019. (e.g 200 tokens with a shift of 50.)')

parser.add_argument('-m', '--mode', help='what dataset are we using (currently only newsgroup is accepted)', default='newsgroup')

parser.add_argument('-d', '--data_dir', help='path_to_data_dir', required=True)
parser.add_argument('-p', '--processed_dir', help = 'path to processed_dir, which contains the label_to_cutoff_indices pickle file and also where the output of this script will be stored', required=True)
parser.add_argument('-l','--sequence_length',help='Sequence length. This is the number of tokens each segment contains.',required=True)
parser.add_argument('-s','--shift_length',help='Shift length. This is the number of tokens to shift between each segment. Two consecutive segments will overlap by sequence_length - shift_length tokens',required=True)

args = vars(parser.parse_args())

mode = args['mode']
data_dir = args['data_dir']
processed_dir = args['processed_dir']
sequence_length = int(args['sequence_length'])
shift_length = int(args['shift_length'])

if mode =='newsgroup':
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

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    
    #Initialize Tokenizer and Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)

    for split in splits:
        # Load each dataset
        dataset_list = []
        for config in newsgroup_configs:
            subset_path = data_dir + split + '/' + config
            dataset_list.append((config,load_from_disk(subset_path)))

        # create label_to_label_idx_dict
        label_to_label_idx_dict={}
        for ii, label in enumerate(newsgroup_configs):
            label_to_label_idx_dict[label] = ii

        bert_encoded_segments_list = []
        for label, sub_dataset in dataset_list: #Loop over all dataset
            for entry in sub_dataset: #Loop inside the dataset
                # get text and CLS token
                text = entry['text']
                tokens = tokenizer.encode(text,add_special_tokens=False,return_tensors='pt')[0]
                # Start the While loop - here, we try to get spans of 200 tokens, with a shift of 50. 
                # E.g if the sequence is 300 tokens, we get [0,199][50,249],[100,299]

                all_sub_seq = create_overlaps(tokens,sequence_length=sequence_length,shift_length=shift_length)

                #cat to make a tensor
                segments_tensor = torch.cat(all_sub_seq)
                # turn all_sub_seq into an OnTheFlyDataset    
                onthefly_dataset = OnTheFlyDataset(segments_tensor)
                onthefly_loader = DataLoader(onthefly_dataset, batch_size=64, shuffle=False, pin_memory=True)
                # At this point, onthefly_datset/loader contains the tokens for one single "long document", in one dataset
                with torch.no_grad():
                    batch_encoded_seg_list = []
                    for small_batch in onthefly_loader: #encode each segment in the long document
                        out = bert_model(input_ids=small_batch.to(device))
                        sub_bert_encoded_segments = out['last_hidden_state'][:,0,:]
                        batch_encoded_seg_list.append(sub_bert_encoded_segments)
                    bert_encoded_segments = torch.cat(batch_encoded_seg_list)
                    bert_encoded_segments_list.append((label_to_label_idx_dict[label],bert_encoded_segments.cpu()))

        file_name = 'bert_encoded_segments_list_overlap_' + str(seq_len) + '_' + str(shift_len)
        with open(processed_dir + split + '/' + file_name  + '.pkl', 'wb') as handle:
            pickle.dump(bert_encoded_segments_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
