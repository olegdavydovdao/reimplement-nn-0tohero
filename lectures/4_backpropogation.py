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

# PART 1: INIT AND FORWARD PASS
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

# Extended forward pass (with atomic ops)
emb = C[Xb] # 32,3,10 = 27,10[32,3]
embcat = emb.view(emb.shape[0], -1) # 32,30
hprebn = embcat @ W1 + b1 # 32,64

bnmean = 1/n*hprebn.sum(0, keepdim=True) # 1,64
bndiff = hprebn - bnmean
bndiff2 = bndiff**2 # 32,64
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # 1,64
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = gamma * bnraw + beta # 32,64; 1,64

h = torch.tanh(hpreact) # 32,64
logits = h @ W2 + b2 # 32,27

logit_maxes = logits.max(1, keepdim=True).values # 32,1
norm_logits = logits - logit_maxes
counts = norm_logits.exp() # 32, 27
counts_sum = counts.sum(1, keepdim=True) # 32,1
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log() # 32,27
loss = -logprobs[range(n), Yb].mean()

for p in parameters:
  p.grad = None
for inter in [logprobs, probs,counts_sum_inv, counts_sum, counts,
          norm_logits, logit_maxes, logits, h, hpreact, 
          bnraw, bnvar_inv, bnvar, bndiff2, bndiff,
          bnmean, hprebn, embcat, emb]:
    inter.retain_grad()
loss.backward()