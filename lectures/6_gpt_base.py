# THEME: GPT BASE MODEL AT CHARACTER LEVEL
# PART 0: DATA PREPROCESSING AND HYPERPARAMETERS
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Data loading
path_data_str = os.path.join('data', 'shakespeare.txt')
with open(path_data_str, 'r', encoding='utf-8') as f:
    text = f.read()

# Hyperparameters
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"
print(f"device using: {device}")
batch_size = 8
context_length = 32
emb_dim = 32
N_blocks = 4
n_heads = 4
head_dim = emb_dim//n_heads
p_drop = 0.02
lr = 1e-3
max_iters = 5000
show_lossi = 200
interval_lossi = 100
print_lossi = 500

# Data preprocessing, batch function
tokens = sorted(set(text))
vocab_size = len(tokens)
itos = {i:s for i,s in enumerate(tokens)}
stoi = {s:i for i,s in enumerate(tokens)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text)) # memory on gpu is expensive, data stay on cpu
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[ix:ix+context_length] for ix in ixs])
    y = torch.stack([data[ix+1:ix+1+context_length] for ix in ixs])
    x, y = x.to(device), y.to(device)
    return x,y

# Check loss at evaluation time
@torch.no_grad()
def loss_estimate():
    model.eval()
    splits = ['train', 'val']
    out = {}
    for split in splits:
        loss_m = torch.zeros(interval_lossi)
        for k in range(interval_lossi):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            loss_m[k] = loss
        loss_m = loss_m.mean()
        out[split] = loss_m.item()
    model.train()
    return out

# Single head self-attention
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(emb_dim, head_dim,bias=False)
        self.key = nn.Linear(emb_dim, head_dim,bias=False)
        self.value = nn.Linear(emb_dim, head_dim,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length,context_length)))
        self.dropout = nn.Dropout(p_drop)
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x) # B, T, H
        k = self.key(x)
        v = self.value(x) # new T - new token
        aff = q @ k.transpose(-2, -1) / head_dim**0.5 # new T - token who appreciated previous tokens
        aff = aff.masked_fill(self.tril[:T, :T]==0, float('-inf')) # B, T, T
        aff = F.softmax(aff, -1) # aff is affinity
        aff = self.dropout(aff)
        out = aff @ v # new T - token who agregate previous tokens
        return out # B, T, H
        # for N_block == 2: new T - token who agregate previous tokens who themselves saw history of previous - hierarchical approach
        # i.e. level up abstraction level through sequintial Transformer blocks

# Multiple heads of self-attention processing parallel and same input
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
    
# MLP block too think about whats happens in attention block
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_seq = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim), # == self.proj
        )
    def forward(self, x):
        return self.ff_seq(x)
    
# Single transformer block
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.m_sa = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(emb_dim)
        self.fforward = FeedForward()
        self.dropout = nn.Dropout(p_drop)
    def forward(self, x):
        x = x + self.dropout(self.m_sa(self.ln1(x)))
        x = x + self.dropout(self.fforward(self.ln2(x)))
        return x

# Stack multiple class and layers bulding blocks into one architecture GPT
class Gpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(context_length, emb_dim)
        self.blocks = nn.Sequential(*[Block() for _ in range (N_blocks)])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, xb, yb=None):
        tok_emb = self.tok_emb_table(xb)
        B,T,C = tok_emb.shape # every row - next tokens with C information
        pos_emb = self.pos_emb_table(torch.arange(T,device=device)) # T,C
        x = tok_emb + pos_emb # B,T,C
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x)) # B,T,vocab_size


        # get loss
        if yb is None:
            loss = None
        else:
            logits = logits.view(B*T, vocab_size)
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb)
        return logits, loss

    def generate(self, x, max_new):
        for _ in range(max_new):
            x_context = x[:,-context_length:]
            logits, loss = self(x_context)
            f_logit = logits[:,-1,:] # B,C | take last who saw all running_context in last abstarction level
            probs = F.softmax(f_logit,1)
            ix = torch.multinomial(probs, 1) # B,1
            x = torch.cat((x, ix), dim=-1)
        return x # tensor of int

# Initialization GPT
model = Gpt()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_train_graph = []
loss_val_graph = []
step_val_graph = []

# Train GPT
for i in range(max_iters):
    # get intermidiate loss
    if i % show_lossi==0 or i == max_iters-1 or i%print_lossi==0:
        lossi = loss_estimate()
        loss_val_graph.append(lossi["val"])
        step_val_graph.append(i)
        if i%print_lossi==0:
            print(f'{i:5d} | train loss:{lossi['train']:.4f} | val loss:{lossi['val']:.4f}')

    # train
    xb,yb = get_batch('train')
    optimizer.zero_grad()
    logits, loss = model(xb,yb)
    loss_train_graph.append(loss.detach())
    loss.backward()
    optimizer.step()