# THEME: MANUAL BACKPROPOGATION OF TENSOR-LEVEL GRADIENTS
# PART 0: HYPERPARAMETERS AND DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.preprocess_names import get_splits_names

# Hyperparameters
block_size = 3
n_emb = 10
n_hidden = 100
n_iters = 10000
batch_size = 32
lr = 0.1
lr_decay = 0.01
n = batch_size

# Preprocess names.txt to feed into the model
Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr = get_splits_names(block_size=block_size)
print(f"Xtr.shape: {Xtr.shape}")

# PART 1: INIT
# Get 1 batch
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((sz_voc, n_emb), generator = g)
W1 = torch.randn((n_emb*block_size, n_hidden), generator = g)*(5/3)/(n_emb*block_size)**0.5
b1 = torch.randn(n_hidden, generator = g)*0.1
W2 = torch.randn((n_hidden, sz_voc), generator = g)*0.1
b2 = torch.randn(sz_voc, generator = g)*0.1
gamma = torch.randn((1,n_hidden), generator = g)*0.1+1
beta = torch.randn((1,n_hidden), generator = g)*0.1
parameters = [C,W1,b1,W2,b2,gamma,beta]

for p in parameters:
    p.requires_grad = True
print(f'num parameters: {sum(p.nelement() for p in parameters)}')

batch = torch.randint(0, num_tr, (batch_size,), generator = g)
Xb, Yb = Xtr[batch], Ytr[batch]