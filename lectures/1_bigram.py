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
