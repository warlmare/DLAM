import math
from pickle import FALSE
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
from pprint import pprint
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import random 
import wandb
import sys
from nltk import ngrams
from string import ascii_lowercase, ascii_uppercase
from itertools import product
from itertools import islice

from main import TRAINING_DATASET_SIZE


# because mrshv2 produces very large hashes
csv.field_size_limit(sys.maxsize)


def read_csv_to_list(csv_file):
    '''reads .csv files with hashes and generates a list

    '''
    with open(csv_file, 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        flatlist_pre = list(readcsv)
        flatlist = flatten(flatlist_pre)
        
    return flatlist

def flatten(t):
    '''flattens a list of lists

    '''
    return [item for sublist in t for item in sublist]

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

def clean_mrshv2_hash(mrshv2_hash):

    #remove all unnecessary information until the last ":"

    hash_without_pre_set_information = re.search(r'(.*):(.*)',mrshv2_hash).group(2)
    hash_without_newline_char = hash_without_pre_set_information.strip('\n')

    return hash_without_newline_char

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
  res_list = ["".join(x) for x in window(raw_hash, 3)]
  return res_list

PADDING_TOKEN = 0

import numpy as np
def simple_collate_fn(data):
  m = max([len(tokens) for tokens, _ in data])
  padded_tokens = [torch.hstack([tokens, 
                                 torch.ones(m-len(tokens))*PADDING_TOKEN]) 
                   for tokens, _ in data]
  batch = torch.vstack(padded_tokens)
  labels = torch.vstack([label for _, label in data])
  return batch.int(), labels.long()


# encoding chars for hash -> embeddings
ALL_CHARS = '0abcdefghijklmnopqrstuvwxyz'
ALL_CHARS += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALL_CHARS += '123456789'
ALL_CHARS += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' + '\\'

# generate an alphabet for 3-grams, special for tlsh which only uses numbers and uppercase letters
tlsh_chars =  'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789' 
tlsh_alphabet = [''.join(i) for i in product(tlsh_chars, repeat = 3)]


class HashDataset(Dataset):

    def __init__(self, hashes, labels, all_chars, hashing_algorithm='tlsh', transform=False):
        """
        .
        """

        super(HashDataset, self).__init__()

        self._vocab_size = len(all_chars)

        #changed the enumerator to start at 1 s.t. 0 can be padding token 
        #if hashing_algorithm == "tlsh":  
        self.char_to_int = dict((c, i) for i, c in enumerate(all_chars))
        self.int_to_char = dict((i, c) for i, c in enumerate(all_chars))
        #else:
        #self.char_to_int = dict((c, i) for i, c in enumerate(all_chars, 1))
        #self.int_to_char = dict((i, c) for i, c in enumerate(all_chars, 1))


        self.hashes = hashes
        self.labels = labels

        self.transform = transform

        self.data_min = 0

#        if hashing_algorithm == "tlsh":
#          self.data_max = len(tlsh_alphabet)
#        else:
        self.data_max = len(all_chars)

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if hashing_algorithm == "tlsh":
          # the n-grams are in nested lists 
          # self.hashes[idx] = ['XC3', '235', '13C', ... ]
        inputs = [self.char_to_int[char] for char in self.hashes[idx]]
        #else:
        #  inputs = [self.char_to_int[char] for char in self.hashes[idx]]
        
        label = torch.LongTensor(self.labels[idx])

        # Does this transform really make sense?
        if self.transform:
            inputs_scaled = inputs / self.data_max
            return torch.FloatTensor(inputs_scaled), label
        else:
            return torch.LongTensor(inputs), label

    def decode_hash(self, encoded_hash):
#        if hashing_algorithm == "tlsh":
 #         return ''.join(self.int_to_ngram[_int] for _int in encoded_hash)
  #      else:
          return ''.join(self.int_to_char[_int] for _int in encoded_hash)

    @property
    def vocab_size(self):
        return self._vocab_size


from transformers import AutoModel, BertConfig, BertGenerationEncoder


class BERTModel(nn.Module):
    # Constructor
    def __init__(self, max_seq_len, hidden_size, vocab_size):

        super(BERTModel, self).__init__()

        # config for a tiny BERT, could be bigger
        config = BertConfig(hidden_size=hidden_size, 
                            intermediate_size=1200, #512
                            num_attention_heads=12, #2
                            num_hidden_layers=4,    #2
                            max_position_embeddings=max_seq_len,
                            vocab_size=vocab_size) 

        self.BERT = AutoModel.from_config(config)

        self.predictor = nn.Linear(hidden_size, 2)
        
    def forward(self, x, attention_mask=None):

        # pass through BERT 
        outputs_per_token = self.BERT(x, 
                                      attention_mask=attention_mask, 
                                      return_dict=True)["last_hidden_state"]

        # mean across token dimension 
        output = outputs_per_token.mean(dim=1) 
        
        return self.predictor(output)
  

# Model
model = BERTModel(
    max_seq_len=96,# max_hash_length, #64
    hidden_size=312, #hidden_size,     #312
    vocab_size=91 # vocabulary_size   #91
)


device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_ct = torch.cuda.device_count()     
print(f'GPU devices: {gpu_ct}')

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model = torch.load("trained_tiny_bert_model_nilsimsa.pth")
model.eval()

def hash_to_tensor(hash):
  input_hash_list = [hash]
  input_label = [[1]]
  single_instance_dataset = HashDataset(
    hashes=input_hash_list,
    labels=input_label,
    all_chars=ALL_CHARS
  )
  return single_instance_dataset

nilsimsa_hash = "befd117eb1136331e651ff29dac4e335b9cd747780b304cd32b7e260074207bf"

single_hash = hash_to_tensor(nilsimsa_hash)[0]


def model_single_prediction(single_embedding):
      custom_loader = DataLoader(single_embedding,1)
      input, label = iter(custom_loader)
      input = input.to(device)
      label = label.to(device)
      softmax_prediction= F.softmax(model(input), dim=1)
      argmax_prediction = torch.argmax(softmax_prediction, dim=1)
      if argmax_prediction == label: 
        return True
        #print("correctly identified")
      else:
        return False
        #print("incorrectly identified")



model_single_prediction(single_hash)
