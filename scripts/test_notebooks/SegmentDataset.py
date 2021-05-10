from torch.utils.data import Dataset, DataLoader

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
                segments_list = create_segments_list(cutoff_indices,sentence_list,tokenizer)
                data_entry = (self.label_to_label_idx_dict[label],segments_list)
                self.data.append(data_entry)
        
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self,idx):
        return(self.data[idx])
    
