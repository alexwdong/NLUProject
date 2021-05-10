from torch.utils.data import Dataset, DataLoader

class OnTheFlyDataset(Dataset):
    def __init__(self, encode_plus_out_list):
        self.encode_plus_out_list = encode_plus_out_list
        
    def __len__(self):
        return len(self.encode_plus_out_list)
 
    def __getitem__(self,idx):
        return(self.encode_plus_out_list[idx])