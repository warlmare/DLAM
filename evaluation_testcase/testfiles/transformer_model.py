import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from transformers import AutoModel, BertConfig

# encoding chars for hash -> embeddings
ALL_CHARS = '0abcdefghijklmnopqrstuvwxyz'
ALL_CHARS += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALL_CHARS += '123456789'
ALL_CHARS += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' + '\\'

#reads .csv files with hashes and generates a list
def read_csv_to_list(csv_file):
    with open(csv_file, 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        flatlist = (list(readcsv))

    return flatlist

def flatten(t):
    return [item for sublist in t for item in sublist]

class HashDataset(Dataset):

    def __init__(self, hashes, labels, all_chars, transform=False):
        """
        .
        """
        super(HashDataset, self).__init__()

        self._vocab_size = len(all_chars)

        self.char_to_int = dict((c, i) for i, c in enumerate(all_chars))
        self.int_to_char = dict((i, c) for i, c in enumerate(all_chars))

        self.hashes = hashes
        self.labels = labels

        self.transform = transform

        self.data_min = 0
        self.data_max = len(all_chars)

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = [self.char_to_int[char] for char in self.hashes[idx]]
        label = torch.FloatTensor(self.labels[idx])

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


# Creating a Dataloader and a generator
batch_size = 16

training_data_normal = read_csv_to_list("dataset/normal_hashes_training_15k_pdf_tlsh.csv")
training_data_anom = read_csv_to_list("dataset/anomaly_hashes_training_15k_pdf_tlsh.csv")
training_data_list_unflat = training_data_normal + training_data_anom
training_data_list = flatten(training_data_list_unflat)

print(training_data_list[0])

n_training = len(training_data_list)

# anomalie is [0, 1] and normal is [1, 0].
# training_labels = [[1, 0] for i in range(n_training // 2)] + [[0, 1] for i in range(n_training // 2)]
training_labels = [[0]] * (n_training // 2) + [[1]] * (n_training // 2)

train_dataset = HashDataset(
    hashes=training_data_list,
    labels=training_labels,
    all_chars=ALL_CHARS
)

max_hash_length = max([len(hash) for hash in train_dataset.hashes])
dataset_size = len(train_dataset)
vocabulary_size = train_dataset.vocab_size
input_size = vocabulary_size

print("Debug print: first tensor of train_dataset: ",train_dataset.__getitem__(1))

train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
train_loader_generator = iter(train_data_loader)



# training parameter

print_every = 50

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0")

max_steps = 50000
learning_rate = 0.01
batch_size = 16 #256 512
hidden_size = 128


# Transformer Model
class BERTModel(nn.Module):
    # Constructor
    def __init__(self, max_seq_len, hidden_size, vocab_size):
        super(BERTModel, self).__init__()

        # config for a tiny BERT, feel free to make bigger
        config = BertConfig(hidden_size=hidden_size,
                            intermediate_size=512,  # needs adjusting
                            num_attention_heads=2,  # needs adjusting
                            num_hidden_layers=2,  # needs adjusting
                            max_position_embeddings=max_seq_len,
                            vocab_size=vocab_size)

        self.BERT = AutoModel.from_config(config)

        self.predictor = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # pass through BERT
        outputs_per_token = self.BERT(x, return_dict=True)["last_hidden_state"]

        # mean across token dimension
        output = outputs_per_token.mean(dim=1)

        return self.predictor(output)

# Model
model = BERTModel(
    max_seq_len=max_hash_length,
    hidden_size=hidden_size,
    vocab_size=vocabulary_size
)
print(model.BERT.config)
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
train_loader_generator = iter(train_data_loader)

# main training loop
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
torch.save(model, "trained_tiny_bert_model.pth")
