# THEME: GPT2 BASE MODEL
# PART 0: DATA PREPROCESSING
# Import libriries
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import tiktoken
import time
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import inspect
import matplotlib.pyplot as plt

# Congfiguration and hyperparameters for model
@dataclass
class Config:
    # Initialization tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # Model hyperparams
    # 124M_new         |# 20M_old
    emb_dim: int = 768 #256
    vocab_size: int = 50304
    context_size: int = 1024 #128
    batches_size: int = 8 # 4
    total_batch_tokens: int = 1024*8 # 1024
    N_tran_blocks: int = 12 #4
    prob_dropout: float = 0.02
    num_heads: int = 12 #4
    head_dim: int = emb_dim//num_heads
    expand_mlp_dim: int = 4
    сoef_train_val_split: float = 0.95

    # Optimization, evaluation, generation hyperparams
    learning_rate: float = 6e-4 #1e-3
    min_lr: float = learning_rate * 0.1
    betas: tuple = (0.9, 0.95)
    num_loop_val: int = None
    weight_decay2d: int = 0.1
    batch_gen: int = 2
    max_gen_tokens: int = 100
    topk_gen_variants: int = 50

    # Steps hyperparameters
    epochs: int = 1
    steps_for_epoch_train = None # calculate later in code
    steps_for_epoch_val = None
    train_steps = None # 313
    warmup_iters: int = None
    cosine_iters_end: int = None
    val_gen_step: int = 4
    checkpoint_interval: int = 5000

    # DDP logic, processes and devices
    ddp_bool = int(os.environ.get('RANK', -1)) != -1
    if ddp_bool:
        assert torch.cuda.is_available(), 'need cuda for ddp'
        init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        unique_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        unique_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device: str = 'cuda'
            gpu_t4_bool = True
            compile_bool = True # torch.compile break rng reproducibility
        else:
            device: str = 'cpu'
            gpu_t4_bool = False
            compile_bool = False
    master_process = unique_rank==0

    # gradient accumulation hyperparameter
    assert total_batch_tokens % (batches_size*context_size*world_size) == 0, 'total_batch_tokens % != 0'
    grad_accum_steps = total_batch_tokens//(batches_size*context_size*world_size)

# Pre-process data and get batch
class DataLoader:
    def __init__(self, config, split):
        assert split in {'train', 'val'}
        # Data loading
        path_data_str = os.path.join('data', 'shakespeare.txt')
        with open(path_data_str, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = config.tokenizer.encode(text)
        limit = int(len(tokens)*config.сoef_train_val_split)
        tokens = tokens[:limit] if split=='train' else tokens[limit:]
        self.tokens = torch.tensor(tokens)
        self.B = config.batches_size
        self.T = config.context_size
        self.device = config.device
        self.reset()
        if split == 'train':
            config.steps_for_epoch_train = len(tokens)//config.total_batch_tokens
        else:
            config.steps_for_epoch_val = len(tokens)//config.total_batch_tokens
        steps_for_epoch = config.steps_for_epoch_train if split=='train' else config.steps_for_epoch_val
        if config.master_process:
            print(f'{split:5s} | len_tok={len(tokens)} | epoch:{steps_for_epoch} steps')
    def reset(self):
        self.current_position = config.unique_rank*self.B*self.T
    def next_batch(self):
        buffer = self.tokens[self.current_position:self.current_position+self.B*self.T+1]
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        self.current_position += self.B*self.T*config.world_size
        if self.current_position+config.world_size*self.B*self.T+1 > len(self.tokens):
            self.reset()
        x, y = x.to(self.device), y.to(self.device)
        return x,y

# Self-attention with multiple head
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim==config.head_dim*config.num_heads, 'wrong heads and its dim'
        self.qkv = nn.Linear(config.emb_dim, 3*config.emb_dim)
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.proj.FLAG = 1

    def forward(self,x):
        B,T,C = x.shape
        qkv = self.qkv(x)
        q,k,v = qkv.split(config.emb_dim, 2)
        q,k,v = [m.view(B, T, config.num_heads, config.head_dim).transpose(1,2) for m in [q,k,v]]
        x = F.scaled_dot_product_attention(q,k,v, dropout_p=config.prob_dropout, is_causal=True) # B, n, T, H | 2,4,256,64
        x = x.transpose(1,2).contiguous().view(B, T, C)
        x = self.proj(x)
        return x

# Feed-forward module
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.emb_dim, config.expand_mlp_dim*config.emb_dim)
        # self.actfunc = nn.GELU(approximate = 'tanh')
        self.actfunc = nn.SiLU()
        self.proj = nn.Linear(config.expand_mlp_dim*config.emb_dim, config.emb_dim)
        self.proj.FLAG = 1
    def forward(self, x):
        x = self.linear(x)
        x = self.actfunc(x)
        x = self.proj(x)
        return x