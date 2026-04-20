# THEME: GPT BASE MODEL AT CHARACTER LEVEL
# PART 0: DATA PREPROCESSING AND HYPERPARAMETERS
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Data loading
path_data_str = os.path.join('data', 'shakespeare.txt')
with open(path_data_str, 'r', encoding='utf-8') as f:
    text = f.read()

# Hyperparameters
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"
print(f"device using: {device}")
batch_size = 8
context_length = 32
emb_dim = 32
N_blocks = 4
n_heads = 4
head_dim = emb_dim//n_heads
p_drop = 0.02
lr = 1e-3
max_iters = 5000
show_lossi = 200
interval_lossi = 100
print_lossi = 500

# Data preprocessing, batch function
tokens = sorted(set(text))
vocab_size = len(tokens)
itos = {i:s for i,s in enumerate(tokens)}
stoi = {s:i for i,s in enumerate(tokens)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text)) # memory on gpu is expensive, data stay on cpu
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[ix:ix+context_length] for ix in ixs])
    y = torch.stack([data[ix+1:ix+1+context_length] for ix in ixs])
    x, y = x.to(device), y.to(device)
    return x,y

# Check loss at evaluation time
@torch.no_grad()
def loss_estimate():
    model.eval()
    splits = ['train', 'val']
    out = {}
    for split in splits:
        loss_m = torch.zeros(interval_lossi)
        for k in range(interval_lossi):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            loss_m[k] = loss
        loss_m = loss_m.mean()
        out[split] = loss_m.item()
    model.train()
    return out