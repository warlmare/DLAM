#!pip install sgt
#!pip install py-tlsh

from datetime import datetime
import string
import random
import re
import tlsh
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SimpleClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, vocabulary_size):
        """
        .
        """
        super(SimpleClassificationModel, self).__init__()

        self.char_embedding = nn.Embedding(vocabulary_size, vocabulary_size)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.predictor = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.char_embedding(x)
        x = self.net(x)

        # Perform Global average pooling for tensor resizing
        x = x.mean(1)

        return self.predictor(x)

n_training = 10_000
n_testing = 2_000

S = 100
anomaly_len = 30
anomaly = "_" * anomaly_len

def get_normal_hash():
    ran_a = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(S))
    ran_tlsh_a = tlsh.hash(str.encode(ran_a))
    return ran_tlsh_a
 
def get_anom_hash():
    ran_b = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(S))
    pos = random.randint(anomaly_len,S - anomaly_len)
    new_str = ran_b[:pos] + anomaly + ran_b[pos+anomaly_len:]
    ran_tlsh_b = tlsh.hash(str.encode(new_str))
    return ran_tlsh_b

# Note: You need balanced training data.
training_data_normal = [get_normal_hash() for i in range(n_training // 2)]
training_data_anom = [get_anom_hash() for i in range(n_training // 2)]
training_data_list = training_data_normal + training_data_anom

# Assuming that anomalie is [0, 1] and normal is [1, 0].
training_labels = [[1, 0] for i in range(n_training // 2)] + [[0, 1] for i in range(n_training // 2)]

# Also for testing having balanced data makes interpretation easier.
test_data_normal = [get_normal_hash() for i in range(n_testing // 2)]
test_data_anom = [get_anom_hash() for i in range(n_testing // 2)]
test_data_list = test_data_normal + test_data_anom

# Assuming that anomalie is [0, 1] and normal is [1, 0].
test_labels = [[1, 0] for i in range(n_testing // 2)] + [[0, 1] for i in range(n_testing // 2)]

class HashDataset(Dataset):
    """Freedoms fancy hash dataset with anomalies"""

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

print_every = 50

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0")

max_steps = 50000
learning_rate = 0.01
batch_size = 512
hidden_size = 128

ALL_CHARS = '0abcdefghijklmnopqrstuvwxyz'
ALL_CHARS +='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALL_CHARS += '123456789'
ALL_CHARS += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' +'\\'

train_dataset = HashDataset(
    hashes=training_data_list,
    labels=training_labels,
    all_chars=ALL_CHARS
)
dataset_size = len(train_dataset)
vocabulary_size = train_dataset.vocab_size
input_size = vocabulary_size

train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
train_loader_generator = iter(train_data_loader)


# Initialize the model that we are going to use
model = SimpleClassificationModel(
    input_size=input_size,
    hidden_size=hidden_size,
    vocabulary_size=vocabulary_size
)
print(model)

model.to(device)

# Setup the loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    loss = criterion(prediction_logits, batch_labels)
    #accuracy would be nice

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % print_every == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}], Step = {step}/{max_steps}, Loss = {loss}")
         
print('Done training.')
torch.save(model, "trained_model.pth")