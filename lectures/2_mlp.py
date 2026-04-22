# THEME: N-GRAM MLP CHARACTER LEVEL LANGUAGE MODEL
# PART 0: HYPERPARAMETERS AND DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.preprocess_names import get_splits_names
from utils.savefig import save_figf

# Hyperparameters
block_size = 3
n_emb = 10
n_hidden = 200
n_iters = 10000
batch_size = 32
lr = 0.1
lr_decay = 0.01

# Preprocess names.txt to feed into the model
Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr = get_splits_names(block_size=block_size)

# PART 1: MODEL INIT, TRAINING
# Model init: parameters and logs
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((sz_voc, n_emb), generator = g)
W1 = torch.randn((n_emb*block_size, n_hidden), generator = g)
b1 = torch.randn(n_hidden, generator = g)
W2 = torch.randn((n_hidden, sz_voc), generator = g)
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

# PART 3: RESULTS
# Different graphs, lr value choice
# plt.plot(lrei, lossi)
# plt.plot(lri, lossi)
plt.plot(lossi)
dir_sublogs = '2_mlp_logs'
save_figf(dir_sublogs, 'log_loss_graph.png')

# Different losses
@torch.no_grad()
def loss_split(split):
    Xs,Ys = {
        'train': (Xtr, Ytr),
        'val': (Xval, Yval),
        'test': (Xte, Yte),
    }[split]
    emb = C[Xs]
    h = torch.tanh(emb.view(-1,W1.shape[0]) @ W1 + b1)
    logits = h @ W2 + b2
    loss_tr = F.cross_entropy(logits, Ys)
    print(f'{split} loss: {loss_tr:.4f}')
loss_split('train')
loss_split('val')

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

# Embedding visualization. (x,y) of embeddings in 2d space (only for 2 first dim of vectors in embeddings)
C4 = C.detach()
plt.plot(figsize=(10,10))
plt.scatter(C4[:,0], C4[:,1], s=200)
for i in range(C4.shape[0]):
    plt.text(C4[i,0], C4[i,1], itos[i], ha='center', va='center', color='white')
plt.grid()
save_figf(dir_sublogs, 'emb_visual_2d.png')

# Test loss once at the end
loss_split('test')