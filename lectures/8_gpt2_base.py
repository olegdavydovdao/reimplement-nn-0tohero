# THEME: GPT2 BASE MODEL
# PART 0: DATA PREPROCESSING AND HYPERPARAMETERS
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
    emb_dim: int = 256#768 #256
    vocab_size: int = 50304
    context_size: int = 128#1024 #128
    batches_size: int = 4#8 # 4
    total_batch_tokens: int = 1024#1024*8 # 1024
    N_tran_blocks: int = 4#12 #4
    prob_dropout: float = 0.02
    num_heads: int = 4#12 #4
    head_dim: int = emb_dim//num_heads
    expand_mlp_dim: int = 4
    сoef_train_val_split: float = 0.95

    # Optimization, evaluation, generation hyperparams
    learning_rate: float = 1e-3#6e-4 #1e-3
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

# PART 1: GPT LOGIC
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

# Decoder transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.mh_attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.prob_dropout)

    def forward(self,x):
        x = x + self.dropout(self.mh_attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x

# GPT2 model, manual ititialization weights, optimizer, lr scheduler
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            tok_emb_table = nn.Embedding(self.config.vocab_size, self.config.emb_dim),
            pos_emb_table = nn.Embedding(self.config.context_size, self.config.emb_dim),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.N_tran_blocks)]),
            ln_f = nn.LayerNorm(self.config.emb_dim)
        ))
        self.lm_head = nn.Linear(self.config.emb_dim, self.config.vocab_size, bias=False)
        self.transformer.tok_emb_table.weight = self.lm_head.weight
        self.apply(self.init_weights_)

    def init_weights_(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'FLAG'):
                std *= (2*config.N_tran_blocks)**-0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self,x,y=None):
        assert x.size(1)<=config.context_size, 't > context_size'
        emb_tok = self.transformer.tok_emb_table(x)
        pos_buf = torch.arange(x.size(1), device=self.config.device)
        pos_tok = self.transformer.pos_emb_table(pos_buf)
        x = emb_tok + pos_tok
        for h in self.transformer.h:
            x = h(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            logits = logits.view(config.batches_size*config.context_size, config.vocab_size)
            y = y.view(config.batches_size*config.context_size)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def get_optimizer_lrshed(self, weight_decay2d, lr, betas):
        # Oprimizer and its groups
        all_params = {n:t for n,t in self.named_parameters() if t.requires_grad}
        d2_params = [t for n,t in all_params.items() if t.ndim>=2]
        d1_params = [t for n,t in all_params.items() if t.ndim<=1]
        optim_groups = [
            {'params': d2_params, 'weight_decay': weight_decay2d},
            {'params': d1_params, 'weight_decay': 0.0}
        ] # optimizer.param_groups
        fused_bool = None
        if ('cuda' in config.device) and ('fused' in inspect.signature(torch.optim.AdamW).parameters):
            fused_bool = True
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=fused_bool)
        if config.master_process:
            print(f"d2_params: {sum(p.numel() for p in d2_params):,}")
            print(f"d1_params: {sum(p.numel() for p in d1_params):,}")
            print(f"fused AdamW:{fused_bool}")

        # Lr scheduler
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_iters)
        decay_sched = CosineAnnealingLR(optimizer, T_max=config.cosine_iters_end-config.warmup_iters, eta_min=config.min_lr)
        min_sched = ConstantLR(optimizer, factor = 0.1, total_iters=config.train_steps-config.cosine_iters_end)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, decay_sched, min_sched],
            milestones=[config.warmup_iters, config.cosine_iters_end]
        )
        return optimizer, scheduler

# PART 2: INITIALIZATION
# Reproducibility, precision, init config
torch.manual_seed(40)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium') # 'high'=tf32; 'medium'=bf16 | only to internal matmul
config = Config()

# Create directory and file to log history of model updates
if config.master_process:
    log_dir = 'log_dir_gpt2'
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, 'log_file.txt')
    with open(file_path, 'w') as f:
        pass
    print(file_path, type(file_path))
    print(f'total_batch_tokens: {config.total_batch_tokens}')
    print(f'grad_accum_steps: {config.grad_accum_steps}')
    print(f"ddp_use: {config.ddp_bool} | master_process_device: {config.device}")

# Inizialization model
train_loader = DataLoader(config, 'train')
val_loader = DataLoader(config, 'val')
config.train_steps = config.steps_for_epoch_train*config.epochs
config.warmup_iters = int(config.train_steps * 0.1)
config.cosine_iters_end = int(config.train_steps * 0.8)
config.num_loop_val = config.steps_for_epoch_val
if config.master_process:
    print(f'{config.train_steps=}')
    print(f'{config.warmup_iters=} | {config.cosine_iters_end=} | {config.num_loop_val=}')
model = GPT2(config)
model.to(config.device)
if config.compile_bool:
    model = torch.compile(model) # raw_model = model._orig_mod
if config.ddp_bool:
    model = DDP(model, device_ids=[config.local_rank])
raw_model = model.module if config.ddp_bool else model
optimizer, scheduler = raw_model.get_optimizer_lrshed(weight_decay2d=config.weight_decay2d, lr=config.learning_rate, betas=config.betas)
if torch.cuda.is_available():
    scaler = torch.amp.GradScaler('cuda', enabled = config.gpu_t4_bool)
else:
    scaler = torch.amp.GradScaler('cpu', enabled = False) # its just skip and scaler.update is optimizer.update
loss_train_graph = []
loss_val_graph = []
step_val_graph = []

# PART 3: TRAIN, LOSS EVAL, SAMPLE
# Train gpt2, evaluate validation loss, generate new tokens
for step in range(config.train_steps):
    # break # for debugging
    t0 = time.time()
    # Validation evaluate
    if step % config.val_gen_step == 0 or step+1==config.train_steps:
        model.eval()
        with torch.no_grad():
            val_loss = torch.zeros(config.num_loop_val)
            for i in range(config.num_loop_val):
                # val_loader.reset()
                x, y = val_loader.next_batch()
                logits, loss = model(x,y)
                val_loss[i] = loss.detach()
            val_loss = val_loss.mean()
            if config.ddp_bool:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            # rng for generate new toks
            rng = torch.Generator(device=config.device)
            rng.manual_seed(2+config.unique_rank)
            if config.master_process:
                print(f'---- step:{step} | val_loss: {val_loss:.4f} ----')
                loss_val_graph.append(val_loss.item())
                step_val_graph.append(step)
                with open(file_path, 'a') as f:
                    f.write(f'---- step:{step} | val_loss: {val_loss:.4f} ----\n')
                if step>0 and (step % config.checkpoint_interval == 0 or step+1==config.train_steps):
                    checkpoint_path = os.path.join(log_dir, f'model_{step:4d}.pt')
                    checkpoint={
                        'model.state_dict': raw_model.state_dict(),
                        'model.config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss.item(),
                        'optimizer.state_dict': optimizer.state_dict(),
                        'sheduler_state': scheduler.state_dict(),
                        'rng_state_cpu': torch.get_rng_state(), # for torch.manual_seed(40)
                        'rng_state_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,# for torch.cuda.manual_seed(40)
                        'custom_rng_state': rng.get_state(),
                        'scaler_state_dict': scaler.state_dict()
                    }
                    torch.save(checkpoint, checkpoint_path)

        # Generate new tokens: in no_grad and model.eval
            if step+1==config.train_steps or step ==0:
                gen = config.tokenizer.encode("How do you like that, Elon Musk?")
                gen = torch.tensor(gen).unsqueeze(0).repeat(config.batch_gen,1)
                gen = gen.to(config.device)
                for _ in range(config.max_gen_tokens-len(gen)):
                    logits, loss = model(gen)
                    logits = logits[:,-1,:]
                    probs = F.softmax(logits, dim=-1)
                    topk_values, topk_indices = torch.topk(probs, config.topk_gen_variants, dim=-1)
                    new_gen = torch.multinomial(topk_values, num_samples=1, generator=rng)
                    new_gen = torch.gather(topk_indices, -1, new_gen)
                    gen = torch.cat((gen, new_gen), dim=-1)
                gen = gen.tolist()
                for k in range(len(gen)):
                    gen_str = config.tokenizer.decode(gen[k])
                    # without master_process
                    print(f"rank:{config.unique_rank} | sample_k:{k} | {gen_str}")
                    with open(file_path, 'a') as f:
                        f.write(f"rank:{config.unique_rank} | sample_k:{k} | {gen_str}\n")
    # Training
        model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    for micro_step in range (config.grad_accum_steps):
        x, y = train_loader.next_batch()
        if config.ddp_bool:
            model.require_backward_grad_sync = (micro_step+1==config.grad_accum_steps) # model with ddp wrapper
        if 'cuda' in config.device:
            data_type = torch.float16 if config.gpu_t4_bool else torch.bfloat16
            with torch.autocast(device_type='cuda', dtype=data_type):
                logits, loss = model(x,y)
        else: # for cpu
            logits, loss = model(x,y)
        loss = loss/config.grad_accum_steps
        loss_total += loss.detach()
        scaler.scale(loss).backward() # instead loss.backward()
    scaler.unscale_(optimizer) # fp16 -> fp32 | grad_unscale = grad/scaler
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer) # instead optimizer.step()
    scaler.update()
    if config.ddp_bool:
        dist.all_reduce(loss_total, op = dist.ReduceOp.AVG)
    current_lr = scheduler.get_last_lr()
    # optim_cur_lr = optimizer.param_groups[0]['lr'] # other way check lr
    scheduler.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_per_step = config.batches_size*config.context_size*config.world_size*config.grad_accum_steps
    assert config.total_batch_tokens==tokens_per_step, 'wrong tokens_per_step'
    if config.master_process:
        print(f"step:{step:4d} | loss_total:{loss_total.item():.4f} | current_lr:{current_lr[0]:.2e} | norm:{norm:.2f} | dt:{dt:.2f} | tok/sec: {tokens_per_step/dt:.2f}")
        loss_train_graph.append(loss_total.item())
        with open(file_path, 'a') as f:
            f.write(f"step:{step:4d} | loss_total:{loss_total.item():.4f} | current_lr:{current_lr[0]:.2e} | norm:{norm:.2f} | dt:{dt:.2f} | tok/sec: {tokens_per_step/dt:.2f}\n")
    # break
if config.ddp_bool:
    dist.destroy_process_group()

# PART 4: Loss graph
legends = []
plt.figure(figsize=(6,4))
plt.plot(loss_train_graph)
legends.append('my gpt(124M) train loss')
plt.plot(step_val_graph, loss_val_graph)
legends.append('my gpt(124M) val loss')
plt.axhline(y=3.2924, color='red', linestyle=':')
legends.append('OpenAI GPT-2(124M) checkpoint val loss')
plt.legend(legends)
plt.title('loss graph')
plt.xlabel('steps')
plt.ylabel('loss');