# THEME: INITIALIZATION, BATCHNORM1D, ACTIVATIONS AND GRADIENTS STATISTICS
# PART 0: PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import os

# Open and prepare the file
path_data_str = os.path.join('data', 'names.txt')
with open(path_data_str, 'r') as f:
    names = f.read().splitlines()
random.seed(42)
random.shuffle(names)

# Letters and their indices
vocab = sorted(set(''.join(names)))
vocab.insert(0, '.')
sz_voc = len(vocab)
itos = {i:s for i,s in enumerate(vocab)}
stoi = {s:i for i,s in itos.items()}

# PART 1: HYPERPARAMETERS AND DATA PREPARATION
# Hyperparameters
block_size = 3
n_emb = 10
n_hidden = 100
n_iters = 10000
batch_size = 32
lr = 0.1
lr_decay = 0.01

# Data preparation: input and labels, train/validation/test splits
def build_split(names):
    X, Y = [], []
    for name in names:
        context = [0] * block_size
        for ch in name+'.':
            X.append(context)
            ix = stoi[ch]
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

n1 = int(0.8*len(names))
n2 = int(0.9*len(names))

Xtr,Ytr = build_split(names[:n1])
num_tr = Ytr.shape[0]
Xval,Yval = build_split(names[n1:n2])
Xte,Yte = build_split(names[n2:])
print(f'total names: {len(names)}')
print(f'bigram training examples: {num_tr}')
print(f'bigram validation examples: {Yval.shape[0]}')
print(f'bigram test examples: {Yte.shape[0]}')