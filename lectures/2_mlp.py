# THEME: N-GRAM MLP CHARACTER LEVEL LANGUAGE MODEL
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
n_hid = 200
n_iters = 100000
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

# PART 2: MODEL INIT, TRAINING, AND SAMPLING
# Model init: parameters and logs
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((sz_voc, n_emb), generator = g)
W1 = torch.randn((n_emb*block_size, n_hid), generator = g)
b1 = torch.randn(n_hid, generator = g)
W2 = torch.randn((n_hid, sz_voc), generator = g)
b2 = torch.randn(sz_voc, generator = g)
parameters = [C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad = True
print(f'num parameters: {sum(p.nelement() for p in parameters)}')

lre = torch.linspace(-3, 0, n_iters)
lrs = 10**lre
lossi = []
stepi = []
lrei = []
lri = []

# Train the net with mini-batches
for i in range(n_iters):
    batch = torch.randint(0, num_tr, (batch_size,), generator = g)
    emb = C[Xtr[batch]]
    h = torch.tanh(emb.view(batch_size,W1.shape[0]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[batch])
    if (i+1)%(n_iters/10) == 0 or i == 0:
        print(f'{i:6} | loss: {loss.item():.4f}')

    for p in parameters:
        p.grad = None
    loss.backward()

    # lr = lrs[i]
    lr = lr if i < int(0.7*n_iters) else lr_decay
    for p in parameters:
        p.data -= lr * p.grad

    # Logs
    # lrei.append(lre[i])
    # lri.append(lr)
    lossi.append(loss.log().item())
    # stepi.append(i)

# Sample new names
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    context = [0] * block_size
    new_name = []
    while True:
        emb = C[context]
        h = torch.tanh(emb.view(1,W1.shape[0]) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, 1)
        ix = torch.multinomial(probs, 1, generator = g).item()
        context = context[1:] + [ix]
        new_name.append(itos[ix])
        if ix == 0:
            break
    print(''.join(new_name))

# PART 3: RESULTS: LOSSES, GRAPHS, EMBEDDING VISUALIZATION
# Graphs
# plt.plot(lrei, lossi)
# plt.plot(lri, lossi)
# plt.plot(stepi, lossi)
plt.plot(lossi)
plt.show()

# Loss train
emb = C[Xtr]
h = torch.tanh(emb.view(-1,W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
loss_tr = F.cross_entropy(logits, Ytr)
print(f'train loss: {loss_tr:.4f}')

# Loss validation
emb = C[Xval]
h = torch.tanh(emb.view(-1,W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
loss_v = F.cross_entropy(logits, Yval)
print(f'validation loss: {loss_v:.4f}')

# Loss test
emb = C[Xte]
h = torch.tanh(emb.view(-1,W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
loss_test = F.cross_entropy(logits, Yte)
print(f'test loss: {loss_test:.4f}')

# (x,y) of embeddings in 2d space (only for 2 first dim of vectors in embeddings)
C4 = C.detach()
plt.plot(figsize=(10,10))
plt.scatter(C4[:,0], C4[:,1], s=200)
for i in range(C4.shape[0]):
    plt.text(C4[i,0], C4[i,1], itos[i], ha='center', va='center', color='white')
plt.grid()
plt.show()
