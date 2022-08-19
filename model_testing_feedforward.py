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

def window(seq, n=2):
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
  res_list = ["".join(x) for x in window(raw_hash, 2)]
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

# Diese Funktion müsste meine inputs auf die Länge 96 padden. 
def ssdeep_collate_fn(data):
    max_hash_len = 96

    ssdeep_hashes, ssdeep_labels  = zip(*data)
    ssdeep_hashes = list(ssdeep_hashes)
    for i, _hash in enumerate(ssdeep_hashes):
        cur_hash_len = _hash.size(0)
        padding_size = max_hash_len - cur_hash_len
        ssdeep_hashes[i] = F.pad(_hash, (0, padding_size), "constant", 0)

    return torch.stack(ssdeep_hashes), torch.stack(ssdeep_labels)


# encoding chars for hash -> embeddings
ALL_CHARS = '0abcdefghijklmnopqrstuvwxyz'
ALL_CHARS += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALL_CHARS += '123456789'
ALL_CHARS += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' + '\\'

# generate an alphabet for 3-grams, special for tlsh which only uses numbers and uppercase letters
tlsh_chars =  'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789' 
tlsh_alphabet = [''.join(i) for i in product(tlsh_chars, repeat = 2)]

class HashDataset(Dataset):

    def __init__(self, hashes, labels, all_chars, hashing_algorithm, transform=False):#='TLSH', transform=False):
        """
        .
        """

        super(HashDataset, self).__init__()

        self._vocab_size = len(all_chars)

        #changed the enumerator to start at 1 s.t. 0 can be padding token 
        if hashing_algorithm == "TLSH":  
          self.char_to_int = dict((c, i) for i, c in enumerate(all_chars))
          self.int_to_char = dict((i, c) for i, c in enumerate(all_chars))
        else:
          self.char_to_int = dict((c, i) for i, c in enumerate(all_chars, 1))
          self.int_to_char = dict((i, c) for i, c in enumerate(all_chars, 1))


        self.hashes = hashes
        self.labels = labels

        self.transform = transform

        self.data_min = 0

        if hashing_algorithm == "TLSH":
          self.data_max = len(tlsh_alphabet)
        else:
          self.data_max = len(all_chars)

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, idx):#, hashing_algorithm='TLSH'):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if hashing_algorithm == "tlsh":
          # the n-grams are in nested lists 
          # self.hashes[idx] = ['XC3', '235', '13C', ... ]
        inputs = [self.char_to_int[char] for char in self.hashes[idx]]
        label = torch.FloatTensor(self.labels[idx])
        #else:
        #  inputs = [self.char_to_int[char] for char in self.hashes[idx]]
        
        #if hashing_algorithm == "TLSH":
        #  label = torch.FloatTensor(self.labels[idx]) #torch.LongTensor(self.labels[idx]) #
        #else:
        #  label = torch.LongTensor(self.labels[idx])

        # Does this transform really make sense?
        if self.transform:
            inputs_scaled = inputs / self.data_max
            return torch.FloatTensor(inputs_scaled), label
        else:
            return torch.LongTensor(inputs), label

    def decode_hash(self, encoded_hash):#, hashing_algorithm='TLSH'):
       if hashing_algorithm == "TLSH":
          return ''.join(self.int_to_ngram[_int] for _int in encoded_hash)
       else:
          return ''.join(self.int_to_char[_int] for _int in encoded_hash)

    @property
    def vocab_size(self):
        return self._vocab_size




class SimpleClassificationModel(nn.Module):
    def __init__(self, input_length, vocabulary_size, embedding_size=32, hidden_units_1=256, hidden_units_2=64, 
                 dropout_prob=0.125):
      
        super(SimpleClassificationModel, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        # Don't think we need this: nn.Flatten(start_dim=0, end_dim=1)
        # They probably just take a mean 

        # Because of flattening
        self.network_input_size = input_length * embedding_size

        self.net = nn.Sequential(
            # nn.BatchNorm1d(input_length),
            nn.BatchNorm1d(self.network_input_size),
            nn.Linear(self.network_input_size, hidden_units_1),
            nn.ReLU(), 
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(), 
            # nn.Linear(hidden_units_2, 2), # with CrossEntropyLoss
            nn.Linear(hidden_units_2, 1) # with CrossEntropyLossWithLogits
        )
        
        
    def forward(self, x):
        embed_x = self.embedding(x)

        # TODO: try with flattening, reshape = flattening while preserving batch size
        embed_x = embed_x.reshape(embed_x.size(0), self.network_input_size)
        
        # I think this is what they mean with "flatten", although thats
        # not what flatten usually means ???
        # mean_x = torch.mean(embed_x, dim=2)
        
        return self.net(embed_x)

def hash_to_embedding(fuzzy_hash_algorithm ,raw_hash, label):
    if fuzzy_hash_algorithm == "SSDEEP":
        hash = clean_ssdeep_hash(raw_hash)
    elif(fuzzy_hash_algorithm == "TLSH"):
        hash = n_grammer(raw_hash)
    else:
        hash = raw_hash
    input_hash_list = [hash]
    input_label = [[label]]
    single_instance_dataset = HashDataset(
        hashes=input_hash_list,
        labels=input_label,
        all_chars=tlsh_alphabet if fuzzy_hash_algorithm == "TLSH" else ALL_CHARS,
        hashing_algorithm = fuzzy_hash_algorithm)
    if fuzzy_hash_algorithm == "SSDEEP":
      return ssdeep_collate_fn(single_instance_dataset)
    else:
      return single_instance_dataset#[0]


def single_input_prediction(single_embedding, model, device):
      custom_loader = DataLoader(single_embedding,1)
      input, label = iter(custom_loader)
      input = input.to(device)
      label = label.to(device)
      target = F.sigmoid(model(input))
      round_prediction = torch.round(target)

      # extract the ints from the label-tensor and prediction-tensor
      ext_label = label.item()
      ext_pred = round_prediction.item()

      if ext_pred == 1 and ext_label == 1: 
        # tp for when the label is anomaly and the prediction is anomaly
        return "tp"
      elif(ext_pred == 1 and ext_label == 0):
        # fp for the label is "normal" and the predcition is anomaly
        return "fp"
      elif(ext_pred == 0 and ext_label == 0):
        # tn for the label is "normal" and the prediction is "normal"
        return "tn"
      elif(ext_pred == 0 and ext_label == 1):
        return "fn"


def evaluate_dataset_with_model(path_to_pretrained_model,
                                model_hash,
                                max_hash_length, 
                                hidden_size,
                                vocabulary_size, 
                                evaluation_dict,
                                results_dict):


    evaluation_hash_name = model_hash + "_MODEL_FEEDFORWARD"

    # Model
    model = SimpleClassificationModel(
    input_length=max_hash_length,
    vocabulary_size= vocabulary_size
    )

    #print(model.BERT.config)

    device = torch.device("cuda:0") #"cuda" if torch.cuda.is_available() else "cpu"
    gpu_ct = torch.cuda.device_count()     
    print(f'GPU devices: {gpu_ct}')

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #model = nn.DataParallel(model)

    model = torch.load(path_to_pretrained_model)
    model.eval()

    ctr = 0
    tp_ctr = 0
    fp_ctr = 0
    tn_ctr = 0
    fn_ctr = 0

    tp_lst = []
    fp_lst = []
    tn_lst = []
    fn_lst = []

    for testfile in evaluation_dict:
        if testfile != "anomaly":
            ctr += 1

            

            # extract the raw hash
            raw_hash = evaluation_dict[testfile][model_hash]

            if evaluation_dict[testfile]["label"]  == "anomaly":                
                label = 1
            else:
                label = 0 

            hash_embedding = hash_to_embedding(model_hash,raw_hash,label)

            if model_hash == "SSDEEP":
                model_prediction = single_input_prediction(hash_embedding,model, device)#
            else:   
                model_prediction = single_input_prediction(hash_embedding[0],model, device)
            

            if model_prediction == "tp": 
                tp_ctr += 1
                tp_lst.append(testfile)
            elif model_prediction == "fp":
                fp_ctr += 1
                fp_lst.append(testfile)
            elif model_prediction == "tn":
                tn_ctr += 1
                tn_lst.append(testfile)
            elif model_prediction == "fn":
                fn_ctr += 1
                fn_lst.append(testfile) 

    
    # calculations
    accuracy = (tp_ctr + tn_ctr) / (tp_ctr + tn_ctr + fp_ctr + fn_ctr)
    precision = tp_ctr / (tp_ctr + fp_ctr )
    recall = tp_ctr / (tp_ctr + fn_ctr)

    # fill the results dict that will be turned into evaluation.yml
    results_dict[evaluation_hash_name] = {}
    results_dict[evaluation_hash_name]["accuracy"] = accuracy
    results_dict[evaluation_hash_name]["precision"] = precision
    results_dict[evaluation_hash_name]["recall"] = recall
    results_dict[evaluation_hash_name]["tp_files"] = tp_lst
    results_dict[evaluation_hash_name]["fp_files"] = fp_lst
    results_dict[evaluation_hash_name]["tn_files"] = tn_lst
    results_dict[evaluation_hash_name]["fn_files"] = fn_lst


    return results_dict
