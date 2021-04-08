from datasets import load_from_disk
import json
import torch
from nltk.tokenize import sent_tokenize
from torch.nn.functional import softmax
from transformers import BertTokenizer,BertForNextSentencePrediction
import pickle

def get_probabilities_on_text_w_NSP(nsp_model, text, tokenizer,device):
    '''
    Returns a sequence of probabilities which represent confidence that the next sentence is part of the same segment
    
    If text has n sentences, then prob_seq has n-1 probabilities. 
    The ii index of prob seq represents the NSP confidence of the ii and ii+1 sentences in text.
    Probabilities closer to 1 indicate confidence, Probabilities closer to 0 indicate no confidence.
     
    '''
    #Create sentence list
    sentence_list = sent_tokenize(text)
    prob_seq = []
    #Iterate over all sequential pairs
    for ii in range(0,len(sentence_list)-1):
        sentence_1 = sentence_list[ii]
        sentence_2 = sentence_list[ii+1]
        
        #Encode
        encoded = tokenizer.encode_plus(sentence_1, text_pair=sentence_2, return_tensors='pt')
        encoded.to(device=cuda)
        #print(encoded['input_ids'].shape[1])
        if encoded['input_ids'].shape[1] > 512: # If two sentences are too long, just split them
            prob_seq.append(0)
        else:
            #Not too long, pass through the model and get a probability
            with torch.no_grad():
                logits = nsp_model(**encoded)[0]

            probs = softmax(logits, dim=1)
            prob_seq.append(probs[0][0])
    #End for loop
    return prob_seq,sentence_list

def get_tokens_per_sentence_list(tokenizer,sentence_list):
    tokens_per_sentence_list = [len(tokenizer.encode(sentence)) for sentence in sentence_list]
    return tokens_per_sentence_list

def apply_threshold(prob_seq,tokens_per_sentence_list,threshold):
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
    
    prob_seq,sentence_list = get_probabilities_on_text_w_NSP(nsp_model, text, tokenizer,device)
    tokens_per_sentence_list = get_tokens_per_sentence_list(tokenizer, sentence_list)
    cutoff_indices = apply_threshold(prob_seq, tokens_per_sentence_list, threshold=.5)
    
    return cutoff_indices

# Start Script
if __name__ == "__main__":
    
    ##### CUDA CHECK
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    ###### LOAD DATASET
    #dataset = load_from_disk('/home/adong/School/NLUProject/data/trivia_qa_rc_tiny')
    #dataset = load_from_disk(r'\\wsl$\Ubuntu-20.04\home\jolteon\NLUProject\data\trivia_qa_rc')

    dataset = load_from_disk('/scratch/awd275/NLU_data/trivia_qa_rc')
    
    ###### INITIALIZE
    nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    nsp_model.eval()
    nsp_model.cuda()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    

    threshold = .5
    qid_struct = {}
    keys = dataset.keys()
    for key in dataset.keys():
        sub_dataset = dataset[key]
        
        for ii, entry  in enumerate(sub_dataset):
            print(entry['question_id'])
            print('num entity pages, num search context', len(entry['entity_pages']['wiki_context']),len(entry['search_results']['search_context']))

            wiki_context_splits = []
            for context in entry['entity_pages']['wiki_context']:
                cutoff_indices = get_cutoff_indices(context, threshold, nsp_model, tokenizer,device)
                wiki_context_splits.append(cutoff_indices)

            search_context_splits = []
            for context in entry['search_results']['search_context']:
                cutoff_indices = get_cutoff_indices(context, threshold, nsp_model, tokenizer,device)
                search_context_splits.append(cutoff_indices)

            qid_struct[entry['question_id']] = (wiki_context_splits,search_context_splits)
            if ii == 10:
                break
        file_name = key + '_qid_struct.pkl'
        with open('/scratch/awd275/NLU_data/' + file_name, 'wb') as handle:
            pickle.dump(qid_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)

