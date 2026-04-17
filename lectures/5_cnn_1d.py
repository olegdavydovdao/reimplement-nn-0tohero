# THEME: WAVENET AS CNN 1 DIMENSION
# PART 0: HYPERPARAMETERS AND DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.preprocess_names import get_splits_names

# Hyperparameters
torch.manual_seed(42)
block_size = 8
n_emb = 10
n_hidden = 32
n_iters = 1000
batch_size = 32
lr = 0.1
lr_decay = 0.01
n_consec = 2

# Preprocess names.txt to feed into the model
Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr = get_splits_names(block_size=block_size)
print(f"Xtr.shape: {Xtr.shape}")