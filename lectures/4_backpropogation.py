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

# PART 2: MANUAL BACKWARD PASS AND COMPARE WITH .BACKWARD() OF PYTORCH
# Compare my backprop with loss.backward()
def compare(string, d_manual, tensor):
    exactly = torch.all(d_manual == tensor.grad).item()
    approximate = torch.allclose(d_manual, tensor.grad)
    difference = (d_manual-tensor.grad).abs().max().item()
    print(f'{string:15s} | exactly:{str(exactly):5s} | approximate:{str(approximate):5s} | max_difference:{difference}')

# Manual backprop with atomic ops grad
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] -= 1/n
dprobs = dlogprobs * 1/probs
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim = True)
dcounts = dprobs * counts_sum_inv
dcounts_sum = dcounts_sum_inv * -counts_sum**-2
dcounts += dcounts_sum * torch.ones_like(counts)
dnorm_logits = dcounts * norm_logits.exp()
dlogit_maxes = (dnorm_logits * -1).sum(1, keepdim = True)
dlogits = dnorm_logits.clone()
dlogits[torch.arange(n), logits.max(1).indices] += dlogit_maxes.squeeze(1)
db2 = dlogits.sum(0)
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
dhpreact = dh * (1-h**2)
dgamma = (dhpreact * bnraw).sum(0, keepdim=True)
dbeta = dhpreact.sum(0, keepdim=True)
dbnraw = dhpreact * gamma
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
dbndiff = dbnraw * bnvar_inv
dbnvar = dbnvar_inv * -0.5*(bnvar + 1e-5)**-1.5*1.0
dbndiff2 = dbnvar * (1/(n-1)) * torch.ones_like(bndiff2)
dbndiff += dbndiff2 * (2.0 * bndiff)
dhprebn = dbndiff.clone()
dbnmean = (-dbndiff.clone()).sum(0, keepdim=True)
dhprebn += dbnmean * (1/n) * torch.ones_like(hprebn)
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for i in range(demb.shape[0]):
    for j in range(demb.shape[1]):
        ix = Xb[i,j]
        dC[ix] += demb[i,j]

# Compare all results
compare('logprobs', dlogprobs, logprobs)
compare('probs', dprobs, probs)
compare('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
compare('counts_sum', dcounts_sum, counts_sum)
compare('counts', dcounts, counts)
compare('norm_logits', dnorm_logits, norm_logits)
compare('logit_maxes', dlogit_maxes, logit_maxes)
compare('logits', dlogits, logits)
compare('h', dh, h)
compare('W2', dW2, W2)
compare('b2', db2, b2)
compare('hpreact', dhpreact, hpreact)
compare('gamma', dgamma, gamma)
compare('beta', dbeta, beta)
compare('bnraw', dbnraw, bnraw)
compare('bnvar_inv', dbnvar_inv, bnvar_inv)
compare('bnvar', dbnvar, bnvar)
compare('bndiff2', dbndiff2, bndiff2)
compare('bndiff', dbndiff, bndiff)
compare('bnmean', dbnmean, bnmean)
compare('hprebn', dhprebn, hprebn)
compare('embcat', dembcat, embcat)
compare('W1', dW1, W1)
compare('b1', db1, b1)
compare('emb', demb, emb)
compare('C', dC, C)

# PART 3: DERIVE GRAD THROUGH CROSS ENTROPY LOSS AND BATCHNORM
# Fast derivative of cross entropy loss
dlogits_fast = F.softmax(logits, 1)
dlogits_fast[range(n), Yb] -= 1
dlogits_fast /= n

# Fast derivative of batchnorm
dhprebn_fast = gamma*bnvar_inv*(dhpreact - bnraw/(n-1)*(dhpreact*bnraw).sum(0) - dhpreact.sum(0)/n)

compare('logits', dlogits_fast, logits)
compare('hprebn', dhprebn_fast, hprebn)

# activations derivative of logits
plt.figure(figsize=(5,5))
plt.imshow(dlogits.detach(), 'gray')
plt.show()

# PART 4: TRAIN THE MODEL WITH MANUAL BACKPROP
# Init parameters and logs
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((sz_voc, n_emb), generator = g)
W1 = torch.randn((n_emb*block_size, n_hidden), generator = g)*(5/3)/(n_emb*block_size)**0.5 # kaiming
b1 = torch.randn(n_hidden, generator = g)*0.1
W2 = torch.randn((n_hidden, sz_voc), generator = g)*0.1
b2 = torch.randn(sz_voc, generator = g)*0.1
gamma = torch.randn((1,n_hidden), generator = g)*0.1+1
beta = torch.randn((1,n_hidden), generator = g)*0.1
parameters = [C,W1,b1,W2,b2,gamma,beta]
lossi = []

for p in parameters:
    p.requires_grad = True
print(f'num parameters: {sum(p.nelement() for p in parameters)}')

# Train the net with torch.no_grad()
with torch.no_grad():
    for i in range(n_iters):
        batch = torch.randint(0, num_tr, (batch_size,), generator = g)
        Xb, Yb = Xtr[batch], Ytr[batch]
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        bnmean = hprebn.mean(0,keepdim = True)
        bnvar = hprebn.var(0,keepdim = True) # with Bessels
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw = (hprebn-bnmean)*bnvar_inv
        hpreact = gamma * bnraw + beta
        h = torch.tanh(hpreact) # 32,64
        logits = h @ W2 + b2
        
        loss = F.cross_entropy(logits, Yb)
        if (i+1)%(n_iters/10) == 0 or i == 0 or i == n_iters-1:
            print(f'{i:6} | loss: {loss.item():.4f}')
        lossi.append(loss.log10().item())
    
        # for p in parameters:
        #     p.grad = None
        # for inter_tr in [logits, h, hpreact, bnraw, bnvar_inv, 
        #                  bnvar, bnmean, hprebn, embcat, emb]:
        #     inter_tr.retain_grad()
        # loss.backward()
        
        # Manual backprop with fast differentiation formulas
        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        db2 = dlogits.sum(0)
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        dhpreact = dh * (1-h**2)
        dgamma = (dhpreact * bnraw).sum(0, keepdim=True)
        dbeta = dhpreact.sum(0, keepdim=True)
        dhprebn = gamma*bnvar_inv*(dhpreact - bnraw/(n-1)*(dhpreact*bnraw).sum(0) - dhpreact.sum(0)/n)
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(demb.shape[0]):
            for j in range(demb.shape[1]):
                ix = Xb[k,j]
                dC[ix] += demb[k,j]
    
        d_grads = [dC,dW1,db1,dW2,db2,dgamma,dbeta]
        
        lri = lr if i < int(0.75*n_iters) else lr_decay
        for p, d_grad in zip(parameters, d_grads):
            # p.data -= lri * p.grad
            p.data -= lri * d_grad
    # (p.grad-d_grad).abs().max().item()