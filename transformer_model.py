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

#######################
#  LOGGING IN TO W&B  #
#######################

# Log in to W&B account, use 
wandb.login()

#######################
# AUXILIARY FUNCTIONS #
#######################

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

################
# HASH DATASET #
################

# encoding chars for hash -> embeddings
ALL_CHARS = '0abcdefghijklmnopqrstuvwxyz'
ALL_CHARS += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALL_CHARS += '123456789'
ALL_CHARS += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' + '\\'

# generate an alphabet for 3-grams, special for tlsh which only uses numbers and uppercase letters
tlsh_chars =  'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789' 
tlsh_alphabet = [''.join(i) for i in product(tlsh_chars, repeat = 2)]


class HashDataset(Dataset):

    def __init__(self, hashes, labels, all_chars, hashing_algorithm='tlsh', transform=False):
        """
        .
        """

        super(HashDataset, self).__init__()

        self._vocab_size = len(all_chars)

        #changed the enumerator to start at 1 s.t. 0 can be padding token 
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
       if hashing_algorithm == "tlsh":
          return ''.join(self.int_to_ngram[_int] for _int in encoded_hash)
       else:
          return ''.join(self.int_to_char[_int] for _int in encoded_hash)

    @property
    def vocab_size(self):
        return self._vocab_size


########################
#  DATASET PARAMETERS  #
########################

hashing_algorithm = "ssdeep"

anomalies_path = "evaluation_testcase_mixed/training_data_for_model/anomaly_hashes_49999_singlefragment_1-99_ssdeep_mixed.csv"#"evaluation_testcase_xlsx/training_data_for_model/anomaly_hashes_50000_singlefragment_1-99_tlsh_xlsx.csv"
normal_path = "evaluation_testcase_mixed/training_data_for_model/normal_hashes_49999_singlefragment_1-99_ssdeep_mixed.csv"#"evaluation_testcase_xlsx/training_data_for_model/normal_hashes_50000_singlefragment_1-99_tlsh_xlsx.csv"

# set training dataset size here
dataset_size = 99999

#####################################
# TRAINING & VALIDATION DATA LOADER #
#####################################

train_set_size = int((dataset_size * 0.85) / 2) 
val_set_size = int((dataset_size * 0.15) /2)


data_normal = read_csv_to_list(normal_path)
data_anom = read_csv_to_list(anomalies_path)


data_complete_list = data_normal + data_anom

# shuffle the data lists, sample returns a new shuffle list and the original one remains intact
data_normal = random.sample(data_normal, len(data_normal))
data_anom =  random.sample(data_anom, len(data_anom))

#take first half of the data for our validation data
val_data_normal = data_normal[0:val_set_size]
val_data_anom = data_anom[0:val_set_size]

#Take last half of the data for our training data
training_data_normal =  data_normal[-train_set_size:]
training_data_anom = data_anom[-train_set_size:]


training_data_list = training_data_normal + training_data_anom



# cleanup all ssdeep hashes (delete first two chars) and mrshv2 (delete newline char and unneccesary information)
if hashing_algorithm == "ssdeep":
  training_data_list = list(map(clean_ssdeep_hash, training_data_list))
  data_complete_list = list(map(clean_ssdeep_hash, data_complete_list ))
  print("clean")
elif hashing_algorithm == "mrshv2":
  training_data_list = list(map(clean_mrshv2_hash, training_data_list))
  data_complete_list = list(map(clean_mrshv2_hash, data_complete_list )) 
elif hashing_algorithm == "tlsh":
  training_data_list = list(map(n_grammer, training_data_list))

n_training = len(training_data_list)

# normal is [0] and  anomalie is [1].
training_labels = [[0]] * (n_training // 2) + [[1]] * (n_training // 2)

train_dataset = HashDataset(
    hashes=training_data_list,
    labels=training_labels,
    all_chars=tlsh_alphabet if hashing_algorithm == "tlsh" else ALL_CHARS
)

# VALIDATION DATA

val_data_list = val_data_normal + val_data_anom

# cleanup all ssdeep hashes (delete first two chars) and mrshv2 (delete newline char and unneccesary information)
if hashing_algorithm == "ssdeep":
  val_data_list = list(map(clean_ssdeep_hash, val_data_list))
elif hashing_algorithm == "mrshv2":
  val_data_list = list(map(clean_mrshv2_hash, val_data_list))
elif hashing_algorithm == "tlsh":
  val_data_list = list(map(n_grammer, val_data_list)) 

n_val = len(val_data_list)

# normal is [0] and  anomalie is [1].
val_labels = [[0]] * (n_val // 2) + [[1]] * (n_val // 2)

val_dataset = HashDataset(
    hashes=val_data_list,
    labels=val_labels,
    all_chars=tlsh_alphabet if hashing_algorithm == "tlsh" else ALL_CHARS
)



max_hash_length = max([len(hash) for hash in data_complete_list])#train_dataset.hashes])
#dataset_size = len(train_dataset)
vocabulary_size = train_dataset.vocab_size
input_size = vocabulary_size

#training_sample_lengths =[len(t) for t in training_data_list]
#max_training_sample_length = max(training_sample_lengths)


print("training dataset: ",len(training_data_list)," validation dataset: ", len(val_data_list))
##########################
#     TEST DATA LOADER   #
##########################

#These files represent spefically created test files that the models accuracy is tested on 

# test data is not validation data! 

# test_anomalies_path = "dataset/anomaly_hashes_50000_singlefragment_1-99_tlsh_pdf.csv"
# test_normal_path = "dataset/normal_hashes_50000_singlefragment_1-99_tlsh_pdf.csv"

# #set validation dataset size here
# test_dataset_size = 3000

# test_dataset_split = int(test_dataset_size / 2)

# test_data_normal = read_csv_to_list(test_normal_path)
# test_data_anom = read_csv_to_list(test_anomalies_path)

# test_data_list = random.sample(test_data_normal, test_dataset_split) + random.sample(test_data_anom, test_dataset_split)

# # cleanup all ssdeep hashes (delete first two chars) and mrshv2 (delete newline char and unneccesary information)
# if hashing_algorithm == "ssdeep":
#   test_data_list = list(map(clean_ssdeep_hash, test_data_list))
# elif hashing_algorithm == "mrshv2":
#   test_data_list = list(map(clean_mrshv2_hash, test_data_list)) 

# n_test = len(test_data_list)

# test_labels = [[0]] * (n_test // 2) + [[1]] * (n_test // 2)

# test_dataset = HashDataset(
#     hashes=test_data_list,
#     labels=test_labels,
#     all_chars=tlsh_alphabet if hashing_algorithm == "tlsh" else ALL_CHARS
# )

# print("test dataset: ", len(test_data_list))




#######################
# PADDING & COLLATING #
#######################


# Padding and colate function for padding with ssdeep
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

#######################
# TRAINING PARAMETERS #
#######################

print_every = 50

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0")# "cuda" if torch.cuda.is_available() else "cpu"
#gpu_ct = torch.cuda.device_count()     
#print(f'GPU devices: {gpu_ct}')

max_steps =  1170 # 12 epochs #585 # 6 epochs 1500 # 1230 3 epochs for 210000 training data and 1024 batch size
learning_rate = 1e-3 #1e-6 #1e-3 --> recommended
batch_size =   1024 #1024 for mrshv2 bigger batchsize creates a memory issue
hidden_size = 312 #312 #128

##############
# BERT MODEL #
##############

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

########################
# MODEL INITIALIZATION #
########################        

# Model
model = BERTModel(
    max_seq_len=max_hash_length,
    hidden_size=hidden_size,
    vocab_size=vocabulary_size
)
print(max_hash_length)
print(model.BERT.config)

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)

model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# padding is only used for ssdeep & mrshv2 hashing all other algorithms  produce fixed length hashes 
if hashing_algorithm == "ssdeep" or hashing_algorithm == "mrshv2":
  train_data_loader = DataLoader(train_dataset, batch_size, collate_fn=simple_collate_fn, shuffle=True, num_workers=2)
  #test_data_loader = DataLoader(test_dataset, batch_size, collate_fn=simple_collate_fn, shuffle=True, num_workers=2)
  val_data_loader = DataLoader(val_dataset, batch_size, collate_fn=simple_collate_fn, shuffle=True, num_workers=2)
else:
  train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
  #test_data_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=2)
  val_data_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=2)

train_loader_generator = iter(train_data_loader)
#test_loader_generator = iter(test_data_loader)
val_loader_generator = iter(val_data_loader)

#print(train_dataset[1][0])
######################
# MAIN TRAINING LOOP #
###################### 

total_training_loss = []
total_validation_loss = []
total_val_acc = []
iters = []
overall_accuracy = 0 
overall_loss =0
ctr = 0

#weights an biases initialization
# wandb.init(
#   # project where this run will be logged
#  project="transformer_meets_fuzzy_hashes", 
#   # Track hyperparameters and run metadata
#  config={
#  "learning_rate": learning_rate,
#  "architecture": "BERT",
#  "dataset_size": dataset_size,
#  "max_steps": max_steps,
#  "batch_size": batch_size
#  })

# # Copy config 
# config = wandb.config


# main training loop
for step in range(max_steps):
    ctr += 1
    try:
        # Samples the batch
        batch_inputs, batch_labels = next(train_loader_generator)
        batch_attention_mask = batch_inputs.bool().int()
        val_batch_inputs, val_batch_labels = next(val_loader_generator)

    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        train_loader_generator = iter(train_data_loader)
        batch_inputs, batch_labels = next(train_loader_generator)
        batch_attention_mask = batch_inputs.bool().int()

        val_loader_generator = iter(val_data_loader)
        val_batch_inputs, val_batch_labels = next(val_loader_generator)

    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)
    batch_attention_mask = batch_attention_mask.to(device)

    prediction_logits = model(batch_inputs, attention_mask=batch_attention_mask)

    # LOSS
    loss = criterion(prediction_logits, batch_labels.view(-1))
    overall_loss += loss
    avg_loss = (overall_loss / ctr)
    
    optimizer.zero_grad()

    loss.backward()

    # update the optimizers
    optimizer.step()

    # VALIDATION LOSS 
    model.eval()

    val_batch_inputs = val_batch_inputs.to(device)
    val_batch_labels = val_batch_labels.to(device)
    val_prediction_logits = model(val_batch_inputs)
    val_loss = criterion(val_prediction_logits, val_batch_labels.view(-1))

    # VALIDATION ACCURACY
    
    # flatten the labels of the batch
    val_batch_labels = torch.flatten(val_batch_labels)
    
    # get the predictions for the batch
    softmax_predictions = F.softmax(val_prediction_logits, dim=0)
    argmax_predictions = torch.argmax(softmax_predictions, dim=1)

    # checking the number of equal values in labels and predictions
    correct_predictions = (val_batch_labels == argmax_predictions).sum().cpu().numpy()
    val_acc = (correct_predictions / val_prediction_logits.shape[0]) 
    overall_accuracy += val_acc
    avg_accuracy = ( overall_accuracy / ctr)

    # LOGGING
    train_loss = loss.cpu().detach().numpy()
    val_loss = val_loss.cpu().detach().numpy()

    total_training_loss.append(train_loss)
    total_validation_loss.append(val_loss)
    total_val_acc.append(val_acc)
    iters.append(step)

    # log metrics to weights and biases
    metrics = {"train/train_loss": train_loss,
               "train/step": step} 
    #wandb.log(metrics)

    if step % print_every == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}], Step= {step}/{max_steps}, Loss= {loss:.3f}, Val Loss={val_loss:.3f}, Val Acc={val_acc:.3f}, Avg Loss={avg_loss:.3f}, Avg Acc={avg_accuracy:.3f}")

#wandb.finish()

print('Done training.')
torch.save(model, "evaluation_testcase_mixed/transformer_model_{}.pth".format(hashing_algorithm))

#plotting
plt.plot(iters , total_training_loss, label = "training loss")
plt.plot(iters,  total_validation_loss, label = "validation loss")
plt.plot(iters, total_val_acc, label = "validation accuracy")
plt.legend(loc="upper right")
plt.title("training loss rate (batch_size={}, lr={}, hash={})".format(batch_size,learning_rate,hashing_algorithm))
plt.show()



print("max_hash_len:  {}  hidden_size: {} vocabulary_size: {} ".format(max_hash_length, hidden_size, vocabulary_size))


########################
# TEST DATA EVALUATION #
########################

# model = torch.load("evaluation_testcase_xlsx/trained_tiny_bert_model_{}.pth".format(hashing_algorithm))
# model.eval()



# def test(loader_generator, data_loader, dataset_size):

#   avg_accuracy = 0 
#   ctr = 0
  

#   for step in range(dataset_size):
#       ctr += 1

#       try:
#           # Samples the batch
#           batch_inputs, batch_labels = next(loader_generator)
#       except StopIteration:
#           # restart the generator if the previous generator is exhausted.
#           loader_generator = iter(data_loader)
#           batch_inputs, batch_labels = next(loader_generator)

#       batch_inputs = batch_inputs.to(device)
#       batch_labels = batch_labels.to(device)

#       prediction_logits = model(batch_inputs)


#       # flatten the labels of the batch
#       batch_labels = torch.flatten(batch_labels)
      
#       # get the predictions for the batch
#       softmax_predictions = F.softmax(prediction_logits, dim=0)
#       argmax_predictions = torch.argmax(softmax_predictions, dim=1)


#       # checking the number of equal values in labels and predictions
#       correct_predictions = (batch_labels == argmax_predictions).sum().cpu().numpy()

#       label_tensor_size = batch_labels.size()
#       pred_tensor_size = argmax_predictions.size()


#       if label_tensor_size == pred_tensor_size:
#         total = pred_tensor_size
#         accuracy_per_batch = (100 * correct_predictions / total)
#         avg_accuracy += accuracy_per_batch 
#         overall_accuracy = ( avg_accuracy / ctr)

#       if step % print_every == 0:
#         print(f"Step = {step}/{dataset_size}, Accuracy per batch = {accuracy_per_batch} avg Accuracy = {overall_accuracy}")



# test(train_loader_generator, train_data_loader, 300)





# #parse the result of a model evaluation model(input)
# def tensor_float_parser(tensor_str):
#   tensor_str = str(tensor_str)
#   try:
#     pat = r'.*?\[(.*)].*'            
#     match = re.search(pat, tensor_str)
#     target_str = match.group(1)  
#     arr = np.array([float(i) for i in re.findall(r"[-+]?\d*\.\d+|\d+", target_str)])
#   except:# when the tensor_str is a label with [1. , 0.]
#     arr = np.array([int(s) for s in tensor_str.split() if s.isdigit()])
#   return arr



# # normal is [0] and anomalie is [1].
# # raw prediction logit syntax: [normal, anomaly]
# def check_prediction_for_single_input(single_instance):
#   '''takes an input and a label and checks wether the 
#   model predicts correctly 
#   '''
#   custom_loader = DataLoader(single_instance,1)
#   input, label = iter(custom_loader)
#   input = input.to(device)
#   prob = F.softmax(model(input), dim=1)

#   # turns the softmax prediction into a readable result
#   parsed_prediction= tensor_float_parser(prob)
#   print(parsed_prediction)
#   if parsed_prediction[0] < parsed_prediction[1]:
#     prediction_string = "normal"
#   elif parsed_prediction[0] > parsed_prediction[1]:
#     prediction_string = "anomalous"
#   # case that the prediction is incoconclusive
#   elif parsed_prediction[0] == parsed_prediction[1]:
#     result_dict = {"correct " : False,
#                    "original label" : tensor_float_parser(label)[0],
#                    "prediction" : "indecisive"} 
#     pprint(result_dict, width=1)
#     return 

#   # parses the label 
#   arr_lab = tensor_float_parser(label)
#   if arr_lab[0] == 0: 
#     label_str = "normal"
#   elif arr_lab[0] == 1:
#     label_str = "anomalous"
  
#   # check if prediction is correct
#   if label_str == prediction_string:
#     prediction_correctness = "correct"
#   else:
#     prediction_correctness = "false"
#   result_dict = {"correct " : prediction_correctness,
#                  "original label" : label_str,
#                  "prediction" : prediction_string} 
  
#   pprint(result_dict, width=1)




# def test_prediction_bool(single_instance):
#   '''takes an input and a label and checks wether the 
#   model predicts correctly 
#   '''
#   custom_loader = DataLoader(single_instance,1)
#   input, label = iter(custom_loader)
#   input = input.to(device)
#   prob = F.softmax(model(input), dim=1)

#   # turns the softmax prediction into a readable result
#   parsed_prediction= tensor_float_parser(prob)
#   if parsed_prediction[0] < parsed_prediction[1]:
#     prediction_string = "normal"
#   elif parsed_prediction[0] > parsed_prediction[1]:
#     prediction_string = "anomalous"
#   elif parsed_prediction[0] == parsed_prediction[1]:
#     return False

#   # parses the label 
#   arr_lab = tensor_float_parser(label)
#   if arr_lab[0] == 0: 
#     label_str = "normal"
#   elif arr_lab[0] == 1:
#     label_str = "anomalous"
  
#   # check if prediction is correct
#   if label_str == prediction_string:
#     return True
#   else:
#     return False
