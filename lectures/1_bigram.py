# THEME: BIGRAM CHARACTER LEVEL LANGUAGE MODEL WITH 1 LINEAR LAYER
# PART 0: DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import os

# Open and prepare the file
path_data_str = os.path.join('data', 'names.txt')
with open(path_data_str, 'r') as f:
    names = f.read().splitlines()

# Letters and their indices
vocab = sorted(set(''.join(names)))
vocab.insert(0, '.')
sz_voc = len(vocab)
itos = {i:s for i,s in enumerate(vocab)}
stoi = {s:i for i,s in itos.items()}

# Inputs and labels
xs, ys = [], []
for name in names:
    chs = ['.'] + list(name) + ['.']
    for (ch1, ch2) in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        xs.append(ix1); ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'number of data points: {num}')

# PART 1: TRAIN AND SAMPLE
# Parameters and generator
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator = g, requires_grad = True)
# Train the net: forward, backward, update, evaluate loss
for k in range (100):
    xenc = F.one_hot(xs, num_classes = sz_voc).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim = True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
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
