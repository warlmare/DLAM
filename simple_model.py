! pip install transformers
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime#
from pprint import pprint
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import random
import sys
from nltk import ngrams
from string import ascii_lowercase, ascii_uppercase
from itertools import product
from itertools import islice

### AUXILIARY FUNCTIONS ###

def flatten(t):
    return [item for sublist in t for item in sublist]

#reads .csv files with hashes and generates a list
def read_csv_to_list(csv_file):
    '''reads .csv files with hashes and generates a list

    '''
    with open(csv_file, 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        flatlist_pre = list(readcsv)
        flatlist = flatten(flatlist_pre)

        
    return flatlist

def clean_ssdeep_hash(ssdeep_hash):
    '''takes a ssdeep hash and removes all unimportant information from a sseep hash
    like chunksize and the separation-character (:)

    ssdeep formatting: chunksize:chunk:double_chunk
    
    :param ssdeep_hash: string 
    '''

    # remove the first numbers until the :
    hash_without_chunksize = re.sub(r'^.*?:', ':', ssdeep_hash)

    # remove all :'s
    hash_without_sep_char = hash_without_chunksize.replace(":","")

    return hash_without_sep_char

# Collate function for padding, only needed when ssdeep, and mrshv2 hash are used
def simple_collate_fn(data):
  max_seq_len = max([len(tokens) for tokens, _ in data])
  padded_tokens = [torch.hstack([tokens, 
                                 torch.ones(max_seq_len-len(tokens))*PADDING_TOKEN]) 
                   for tokens, _ in data]
  batch = torch.vstack(padded_tokens)
  labels = torch.vstack([label for _, label in data])
  return batch.int(), labels.long()


def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def n_grammer(raw_hash):
    '''
    splits a string hash into 3-grams
    '''
    res_list = ["".join(x) for x in window(raw_hash, 3)]
    return res_list

# Chars in SSDEEP-Hashes
ALL_CHARS = '0abcdefghijklmnopqrstuvwxyz'
ALL_CHARS += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALL_CHARS += '123456789'
ALL_CHARS += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' + '\\'

# Chars in TLSH-Hashes
tlsh_chars =  'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789' 

# Alphabet for TLSH
tlsh_alphabet = [''.join(i) for i in product(tlsh_chars, repeat = 3)]


# Hashset

class HashDataset(Dataset):

    def __init__(self, hashes, labels, all_chars, hashing_algorithm="tlsh",  transform=False):
        """
        .
        """
        super(HashDataset, self).__init__()

        self._vocab_size = len(all_chars)

        # TLSH needs no padding which
        if hashing_algorithm == "tlsh":  
          self.char_to_int = dict((c, i) for i, c in enumerate(all_chars))
          self.int_to_char = dict((i, c) for i, c in enumerate(all_chars))
        else:
          self.char_to_int = dict((c, i) for i, c in enumerate(all_chars, 1))
          self.int_to_char = dict((i, c) for i, c in enumerate(all_chars, 1))

        self.hashes = hashes
        self.labels = labels

        self.transform = transform

        self.data_min = 0

        if hashing_algorithm == "tlsh":
          self.data_max = len(tlsh_alphabet)
        else:
          self.data_max = len(all_chars)

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # the n-grams are in nested lists 
        inputs = [self.char_to_int[char] for char in self.hashes[idx]]
          
        label = torch.LongTensor(self.labels[idx])

        if self.transform:
            inputs_scaled = inputs / self.data_max
            return torch.FloatTensor(inputs_scaled), label
        else:
            return torch.LongTensor(inputs), label

    def decode_hash(self, encoded_hash):
        return ''.join(self.int_to_char[_int] for _int in encoded_hash)

    @property
    def vocab_size(self):
        return self._vocab_size

### DATA PREPARATION AND LOADER CREATION

    
# set hashing algorithm here
hashing_algorithm = "tlsh"

# set dataset size here
dataset_size = 5000
dataset_split = int(dataset_size / 2)


training_data_normal = read_csv_to_list("dataset/normal_hashes_training_50000_pdf_tlsh.csv")
training_data_anom = read_csv_to_list("dataset/anomaly_hashes_training_50000_pdf_tlsh.csv")
training_data_list = random.sample(training_data_normal, dataset_split) + random.sample(training_data_anom, dataset_split)

# hashes are pre processed according to the literature
if hashing_algorithm == "ssdeep":
  training_data_list = list(map(clean_ssdeep_hash, training_data_list))
#elif hashing_algorithm == "mrshv2":
#  training_data_list = list(map(clean_mrshv2_hash, training_data_list)) 
elif hashing_algorithm == "tlsh":
   training_data_list = list(map(n_grammer, training_data_list))

n_training = len(training_data_list)


training_labels = [[0]] * (n_training // 2) + [[1]] * (n_training // 2)

train_dataset = HashDataset(
    hashes=training_data_list,
    labels=training_labels,
    all_chars=tlsh_alphabet if hashing_algorithm == "tlsh" else ALL_CHARS
)

max_hash_length = max([len(hash) for hash in train_dataset.hashes])
dataset_size = len(train_dataset)
vocabulary_size = train_dataset.vocab_size
input_size = vocabulary_size
# random padding token set to zero. 
PADDING_TOKEN = 0 


### SIMPLE MODEL ###

class SimpleModel(nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_size, vocab_size):

        super(SimpleModel, self).__init__()

        self.vocab_size = vocab_size

        self.char_embedding = nn.Embedding(vocab_size, vocab_size)

        self.net = nn.Sequential(
             nn.Linear(input_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU()
        )

        self.predictor = nn.Linear(vocab_size, 2) # TODO should probably be multi-layerd
        

    def forward(self, x):

        # embedding
        x_embed = self.char_embedding(x)

        # mean across token dimension
        x_embed = x_embed.mean(dim=1)

        return self.predictor(x_embed)




### TRAINING PARAMETERS 


device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_ct = torch.cuda.device_count()     
print(f'GPU devices: {gpu_ct}')

print_every = 50

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

max_steps =  1500 # 1230 3 epochs for 210000 training data and 1024 batch size
learning_rate = 1e-3 #1e-6 #1e-3 --> recommended
batch_size =   512 #1024 for mrshv2 bigger batchsize creates a memory issue
hidden_size = 128 #312 #128

model = SimpleModel(
    input_size=max_hash_length,
    hidden_size=hidden_size,
    vocab_size=vocabulary_size
)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TLSH hashes does not need any padding. Its same length. 
if hashing_algorithm == "ssdeep" or hashing_algorithm == "mrshv2":
  train_data_loader = DataLoader(train_dataset, batch_size, collate_fn=simple_collate_fn, shuffle=True, num_workers=2)
else:
  train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)

train_loader_generator = iter(train_data_loader)

### MAIN TRAINING LOOP

for step in range(max_steps):

    try:
        # Samples the batch
        batch_inputs, batch_labels = next(train_loader_generator)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        train_loader_generator = iter(train_data_loader)
        batch_inputs, batch_labels = next(train_loader_generator)

    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)

    
    prediction_logits = model(batch_inputs)
    
    loss = criterion(prediction_logits, batch_labels.view(-1))
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % print_every == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}], Step = {step}/{max_steps}, Loss = {loss}")
         
print('Done training.')
torch.save(model, "trained_simple_model.pth")
