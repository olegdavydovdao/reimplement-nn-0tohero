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

# PART 0: PYTORCHIFYING CODE AND INITIALIZATION
# Build modules of nn
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator = g)/fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x@self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.training = True
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0)
            xvar = x.var(0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * xmean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum * xvar
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean)/(xvar+self.eps)**0.5
        self.out = self.gamma * xhat + self.beta
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]
        
class Tanh:
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
# Init model's layers as modules in stack
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((sz_voc, n_emb), generator = g)
layers = [
    Linear(n_emb*block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, sz_voc, bias=False), BatchNorm1d(sz_voc),
]
# Parameters and logs
parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True
print(f'num parameters: {sum(p.nelement() for p in parameters)}')
ud = []

# Adjust initialization
with torch.no_grad():
    # layers[-1].weight *= 0.1
    layers[-1].gamma *= 0.1

    for layer in layers[:-1]:
        if isinstance(layer,Linear):
            layer.weight *= 5/3