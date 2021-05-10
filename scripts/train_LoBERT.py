from datasets import load_from_disk, load_dataset
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertModel

import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EncodedSegmentsDataset(Dataset):
    def __init__(self,data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
 
    def __getitem__(self,idx):
        return(self.data_list[idx])
    
class LSTMoverBERT(nn.Module):
    def __init__(self, model_save_dir):
        super().__init__()
        if 'large' in model_save_dir:
            self.LSTM = nn.LSTM(input_size=1024, hidden_size = 128, num_layers=1)
        else:
            self.LSTM = nn.LSTM(input_size=768, hidden_size = 128, num_layers=1)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(in_features=128,out_features=64)
        self.linear2 = nn.Linear(in_features=64,out_features=20)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x,verbose=False):
        
        #print('input x:', x.shape)
        LSTM_out,LSTM_states = self.LSTM(x)
        #print('LSTM out:', LSTM_out.shape)
        #print('LSTM states[0]:', LSTM_states[0].shape)
        #print('LSTM states[1]:', LSTM_states[1].shape)
        last_hidden_state = LSTM_states[0][0,:,:]
        #last_embedding = LSTM_out[:,-1,:]
        out = self.linear1(last_hidden_state)
        #print('linear out', out.shape) if verbose
        out = self.activation(out)
        #print('activation out', out.shape) if verbose
        out = self.linear2(out)
        out = self.softmax(out)
        return out

def pad_collate(batch):
    (labels_list, sequence_list) = zip(*batch)
    labels_tensor = torch.cat(labels_list)
    x_lens = [len(sequence) for sequence in sequence_list]
    sequence_list_padded = pack_sequence(sequence_list, enforce_sorted=False)

    return labels_tensor, sequence_list_padded
    
def train_loop(LoBERT_model, encoded_train_loader, optimizer, criterion):
    LoBERT_model.train()
    train_loss = 0
    train_correct = 0
    for idx, batch in enumerate(encoded_train_loader):
        optimizer.zero_grad()

        # Define and move to GPU
        label = batch[0]
        model_input = batch[1]
        label = label.to(device)
        model_input = model_input.to(device)
        # Forward Pass
        out = LoBERT_model(model_input)
        loss = criterion(out, label)
        #Record Metrics pt 1/2
        train_loss += loss.item()
        pred = torch.argmax(out, axis=1)
        train_correct += (pred == label).sum()

        #Backward pass
        loss.backward()
        optimizer.step()
    
    return train_loss, train_correct

def val_loop(LoBERT_model, encoded_val_loader, criterion):
    LoBERT_model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for idx, batch in enumerate(encoded_val_loader):
            
            # Define and move to GPU
            label = batch[0]
            model_input = batch[1]
            label = label.to(device)
            model_input = model_input.to(device)
            # Forward Pass
            out = LoBERT_model(model_input)
            loss = criterion(out,label)
            # Record metrics pt 1/2
            val_loss += loss.item()
            pred = torch.argmax(out, axis=1)
            val_correct +=(pred == label).sum()
    
    return val_loss, val_correct

    
parser = argparse.ArgumentParser(description='Trains the LoBERT Model and dumps results')

parser.add_argument('-e','--epochs', help='number of epochs to train', default=50)
parser.add_argument('-i','--input', help='bert_encoded_segments_list_file', required=True)

args = vars(parser.parse_args())
    
num_epochs = int(args['epochs'])
bert_encoded_input_file = args['input']
model_save_path = bert_encoded_input_file.replace('embeddings', 'models')
model_save_dir = '/'.join(model_save_path.split('/')[:-1])
results_path = bert_encoded_input_file.replace('embeddings', 'results')
results_dir = '/'.join(results_path.split('/')[:-1])

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
    
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if __name__ == "__main__":
    
    # Our experiment below
    # Baseline (Hierarchical Transformers Pappagari) below
    with open(bert_encoded_input_file, 'rb') as handle:
        bert_encoded_segments_list = pickle.load(handle)

    for document in bert_encoded_segments_list:
        print(document)
        break
        
    encoded_dataset = EncodedSegmentsDataset(bert_encoded_segments_list)
    val_prop =.1
    bsize = 128

    dataset_size = len(encoded_dataset)
    val_size = int(val_prop * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset =  torch.utils.data.random_split(encoded_dataset,[train_size,val_size])
    encoded_train_loader = DataLoader(train_dataset,batch_size=bsize, shuffle=True, pin_memory=True, collate_fn=pad_collate)
    encoded_val_loader = DataLoader(val_dataset,batch_size=bsize, shuffle=True, pin_memory=True, collate_fn=pad_collate)

    # Define LoBERT 
    LoBERT_model = LSTMoverBERT(model_save_dir)
    LoBERT_model.to(device)
    LoBERT_model.train()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(LoBERT_model.parameters(), lr=5e-4)

    # Train over epochs
    best_val_accuracy = 0
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        #START TRAIN
        
        train_loss, train_correct = train_loop(LoBERT_model, encoded_train_loader, optimizer, criterion)
        #Print and save
        train_accuracy = train_correct / train_size
        print('Epoch:', epoch, 'train_loss:', train_loss, 'accuracy: ', train_accuracy)
        train_loss_list.append(train_loss / train_size)
        train_accuracy_list.append(train_accuracy)

        # START VAL
        val_loss, val_correct = val_loop(LoBERT_model, encoded_val_loader, criterion)
        
        # Print and save
        val_accuracy = val_correct / val_size
        print('Epoch:', epoch, 'val_loss:', val_loss, 'accuracy: ', val_accuracy)
        val_loss_list.append(val_loss / val_size)
        val_accuracy_list.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(LoBERT_model.state_dict(), model_save_path)

        results = {'train_loss' : train_loss_list,
               'val_loss' : val_loss_list,
               'train_accuracy' : train_accuracy_list,
               'val_accuracy' : val_accuracy_list
        }
    with open(results_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)