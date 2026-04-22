# THEME: WAVENET AS CNN 1 DIMENSION
# PART 0: HYPERPARAMETERS AND DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.preprocess_names import get_splits_names
from utils.savefig import save_figf

# Hyperparameters
torch.manual_seed(42)
block_size = 8
n_emb = 10
n_hidden = 32
n_iters = 10000
batch_size = 32
lr = 0.1
lr_decay = 0.01
n_consec = 2

# Preprocess names.txt to feed into the model
Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr = get_splits_names(block_size=block_size)

# PART 1: PYTORCHIFYING CODE
# Build modules of nn
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out))/fan_in**0.5
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
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim)
            xvar = x.var(dim)
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

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    def parameters(self):
        return [self.weight]
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1]==1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
# PART 2: INIT AND TRAIN
# Layers as modules in stack, parameters and logs
model = Sequential([ Embedding(sz_voc, n_emb),
    FlattenConsecutive(n_consec), Linear(n_emb*n_consec, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(n_consec), Linear(n_hidden*n_consec, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(n_consec), Linear(n_hidden*n_consec, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, sz_voc)
])
parameters = model.parameters()
for p in parameters:
    p.requires_grad = True
print(f'num parameters: {sum(p.nelement() for p in parameters)}')

with torch.no_grad():
    model.layers[-1].weight *= 0.1

lossi = []

# Train the net
for i in range(n_iters):
    batch = torch.randint(0, num_tr, (batch_size,))
    logits = model(Xtr[batch])
    loss = F.cross_entropy(logits, Ytr[batch])
    if (i+1)%(n_iters/10) == 0 or i == 0 or i == n_iters-1:
        print(f'{i:6} | loss: {loss.item():.4f}')
    lossi.append(loss.log10().item())

    for p in parameters:
        p.grad = None
    loss.backward()

    lri = lr if i < int(0.75*n_iters) else lr_decay
    for p in parameters:
        p.data -= lri * p.grad
    # break

# PART 3: RESULTS
# Loss graph without noise
plt.plot(torch.tensor(lossi).view(-1, 100).mean(1))
dir_sublogs = '5_cnn_1d_logs'
save_figf(dir_sublogs, 'log10_loss_graph.png')
# model.eval() as evaluation mode of model
for layer in model.layers:
    layer.training = False

# Loss of different splits
@torch.no_grad()
def loss_split(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xval, Yval),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

loss_split('train')
loss_split('val')

# Sample new names
for _ in range(5):
    context = [0] * block_size
    new_name = []
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, 1).item()
        context = context[1:] + [ix]
        new_name.append(itos[ix])
        if ix == 0:
            break
    print(''.join(new_name))

# Test loss once at the end
loss_split('test')