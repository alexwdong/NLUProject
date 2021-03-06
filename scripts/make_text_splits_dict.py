from nltk.tokenize import sent_tokenize
from datasets import load_from_disk
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForNextSentencePrediction

import argparse
import json
import os
import pickle
import torch


class ContextDataset(Dataset):
    def __init__(self, sentence_pair_list):
        self.sentence_pair_list = sentence_pair_list

    def __len__(self):
        return len(self.sentence_pair_list)
 
    def __getitem__(self,idx):
        return(self.sentence_pair_list[idx])


def get_probabilities_on_text_w_NSP(nsp_model, text, tokenizer, device):
    '''
        Returns a sequence of probabilities which represent confidence that the next sentence is part of the same segment

        If text has n sentences, then prob_seq has n-1 probabilities. (If text has 1 sentence, prob_seq is [], the empty list.)
        The ii index of prob seq represents the NSP confidence of the ii and ii+1 sentences in text.
        Probabilities closer to 1 indicate confidence, Probabilities closer to 0 indicate no confidence.
    
    '''
    
    sentence_list = sent_tokenize(text)
    over_length_indices = []
    sentence_pair_list=[]
    indices_to_be_processed=[]
    
    #Create Sentence pair list
    if len(sentence_list)==1:
        return [],sentence_list # Return empty list for probs
    
    for ii in range(0,len(sentence_list)-1):
        sentence_1 = sentence_list[ii]
        sentence_2 = sentence_list[ii+1]

        #Encode temporarily, just to count
        encoded = tokenizer.encode_plus(sentence_1, text_pair=sentence_2, return_tensors='pt')
        if encoded['input_ids'].shape[1] > 512: # If two sentences are too long, just split them
            over_length_indices.append(ii)
        else:# add to list to be processed
            indices_to_be_processed.append(ii)
            sentence_pair_list.append([sentence_1,sentence_2])
    # Now, begin calculating probabilities
    with torch.no_grad():
        # Load into a dataset and dataloader. We do this for speed
        context_dataset = ContextDataset(sentence_pair_list)
        context_loader = DataLoader(context_dataset, batch_size=128, shuffle=False, pin_memory=True)
        probs_list = [] #Will be list of tensors
        for batch_idx,batch in enumerate(context_loader):
            if len(batch)==0:
                continue
            # I swear to god this is a legitimate pytorch bug, but we need to reorganize the batch from the dataloader. Whatever
            batch_fixed = [(s1,s2) for s1,s2 in zip(batch[0], batch[1])] 
            # Batch encode
            sentence_pairs = tokenizer.batch_encode_plus(batch_fixed, return_tensors='pt',padding=True)
            # Run through the model
            sentence_pairs.to(device)
            logits = nsp_model(**sentence_pairs)[0]
            # Get Probability of next sentence.
            probs = softmax(logits, dim=1)
            probs = probs[:,0]
            # Add to list of tensors
            probs_list.append(probs.cpu())
            
        #Cat the list of tensors to get a bsize x sequence_length tensors
        if len(probs_list)  == 0:
            all_probs = []
        else:
            all_probs = list(torch.cat(probs_list))
    # Now, we need to sort the probabilities. Some of probabilities are coming from over_length_indices, some of them are coming from indices_to_be_processed
    # We'll zip, then sort, then take the sorted probs.
    indices = over_length_indices + indices_to_be_processed
    one_probs = [1]*len(over_length_indices)
    probs = one_probs + all_probs
    probs = [x for _, x in sorted(zip(indices, probs))]
    # Return probabilities, and also return sentence list for use later as well
    return probs, sentence_list

# ArgParse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', help='what dataset are we using (currently only newsgroup is accepted)', default='20news')
parser.add_argument('-m', '--model', help='A string, the model id of a pretrained model hosted inside a model repo on huggingface.co.', required=True)

args = vars(parser.parse_args())
dataset = args['dataset']
model = args['model']

raw_data_dir = '../data/raw/' + dataset + '/'
segmentations_dir = '../data/segmentations/' + dataset + '/' + model + '/'

if not os.path.exists(segmentations_dir):
    os.makedirs(segmentations_dir)

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
        if not os.path.exists(segmentations_dir + split):
            os.makedirs(segmentations_dir + split)

# Start Script
if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_cuda_info(device)
      
    nsp_model = BertForNextSentencePrediction.from_pretrained(model)
    nsp_model.eval()
    nsp_model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model)

    if dataset == 'wikihop':
        #dataset = load_from_disk('/home/adong/School/NLUProject/data/trivia_qa_rc_tiny')
        #dataset = load_from_disk(r'\\wsl$\Ubuntu-20.04\home\jolteon\NLUProject\data\trivia_qa_rc_tiny')
        dataset = load_from_disk(raw_data_dir)
        qid_struct = {}
        for key in dataset.keys():
            sub_dataset = dataset[key]
            qid_struct = {}
            for ii, entry  in enumerate(sub_dataset):
                #if ii ==5:
                #    break
                print('started: ',str(ii))

                if len(entry['entity_pages']['wiki_context'])==0:
                    wiki_context_probs = None
                else:
                    wiki_context_probs = []
                    for context in entry['entity_pages']['wiki_context']:
                        prob_seq , _ = get_probabilities_on_text_w_NSP(nsp_model, context, tokenizer, device)
                        wiki_context_probs.append(prob_seq)

                if len(entry['search_results']['search_context']) == 0:
                     search_context_probs = None
                else:
                    search_context_probs = []
                    for context in entry['search_results']['search_context']:

                        prob_seq , _ = get_probabilities_on_text_w_NSP(nsp_model, context, tokenizer, device)
                        search_context_probs.append(prob_seq)

                qid_struct[entry['question_id']] = (wiki_context_probs,search_context_probs)
            file_name = key + '_qid_struct.pkl'

        for label, sub_dataset in dataset_list: #Loop over labels
            qid_struct = {}
            for ii, entry in enumerate(sub_dataset):# Loop over data entries with the same label
                context = entry['text']
                prob_seq , _ = get_probabilities_on_text_w_NSP(nsp_model, context, tokenizer, device)
                qid_struct[ii] = prob_seq
                                                 
            with open(SEGMENT_DIR(f'20news/{split}/{label}_qid_struct.pkl'), 'wb') as handle:
                pickle.dump(qid_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif dataset == '20news':
        for split in splits: # Loop over train test
            dataset_list = []
            for config in newsgroup_configs: #loop over labels
                subset_path = raw_data_dir + split + '/' + config
                dataset_list.append((config, load_from_disk(subset_path)))

            for label, sub_dataset in dataset_list: #Loop over labels
                qid_struct = {}
                for ii, entry in enumerate(sub_dataset): # Loop over data entries with the same label
                    context = entry['text']
                    prob_seq , _ = get_probabilities_on_text_w_NSP(nsp_model, context, tokenizer, device)
                    qid_struct[ii] = prob_seq
                file_name = label + '_qid_struct.pkl'
                with open(segmentations_dir + split + '/' + file_name, 'wb') as handle:
                    pickle.dump(qid_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)