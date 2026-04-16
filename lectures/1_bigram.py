# THEME: BIGRAM CHARACTER LEVEL LANGUAGE MODEL WITH 1 LINEAR LAYER
# PART 0: DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
from utils.preprocess_names import get_splits_names

# In original lecture, splits and shuffle names come later
# I don't want to repeat slightly different code spelling but with the same logic
Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr = get_splits_names(block_size=1)
xs = Xtr.squeeze(1)
ys = Ytr

# PART 1: INIT, TRAIN AND SAMPLE
# Parameters init and generator
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator = g, requires_grad = True)
# Vanilla Gradient Descent
# Train the net: forward, backward, update, evaluate loss, cross entropy loss with regularization
for k in range (100):
    xenc = F.one_hot(xs, num_classes = sz_voc).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim = True)
    loss = -probs[torch.arange(num_tr), ys].log().mean() + 0.01*(W**2).mean()
    if k%100==0 or (k+1) % 100 == 0:
        print(f"{k} | loss = {loss.item()}")
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

# Sample new names
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    ix = 0
    new_name = []
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes = sz_voc).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim = True)
        ix = torch.multinomial(probs, 1, generator = g)
        new_name.append(itos[ix.item()])
        if ix == 0:
            break
    print(''.join(new_name))
