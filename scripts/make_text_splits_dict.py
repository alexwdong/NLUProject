exec(open("../header.py").read())

from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer,BertForNextSentencePrediction

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
splits = ['train','test']

# Start Script
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_cuda_info(device)
      
    nsp_model = BertForNextSentencePrediction.from_pretrained('prajjwal1/bert-small')
    nsp_model.eval()
    nsp_model.to(device)
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')

    for split in splits: # Loop over train test
        try:
            os.mkdir(SEGMENT_DIR(f'20news/{split}'))
            print("Created folder: " + SEGMENT_DIR(f'20news/{split}'))
        except FileExistsError:
            pass
        
        dataset_list = []
        for config in newsgroup_configs: #loop over labels
            subset_path = RAW_DIR(f'20news/{split}/{config}')
            dataset_list.append((config, load_from_disk(subset_path)))

        for label, sub_dataset in dataset_list: #Loop over labels
            qid_struct = {}
            for ii, entry in enumerate(sub_dataset):# Loop over data entries with the same label
                context = entry['text']
                prob_seq , _ = get_probabilities_on_text_w_NSP(nsp_model, context, tokenizer, device)
                qid_struct[ii] = prob_seq
                                                 
            with open(SEGMENT_DIR(f'20news/{split}/{label}_qid_struct.pkl'), 'wb') as handle:
                pickle.dump(qid_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        

