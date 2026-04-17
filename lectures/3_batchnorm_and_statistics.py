# THEME: INITIALIZATION, BATCHNORM1D, ACTIVATIONS AND GRADIENTS STATISTICS
# PART 0: HYPERPARAMETERS AND DATA PREPARATION
# Import libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.preprocess_names import get_splits_names

# Hyperparameters
block_size = 3
n_emb = 10
n_hidden = 100
n_iters = 1000
batch_size = 32
lr = 0.1
lr_decay = 0.01

# Preprocess names.txt to feed into the model
Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr = get_splits_names(block_size=block_size)

# PART 1: PYTORCHIFYING CODE
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
    
# PART 2: INIT PYTORCH-LIKE CODE
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

# PART 3: TRAINING
for i in range(n_iters):
    batch = torch.randint(0, num_tr, (batch_size,), generator = g)
    emb = C[Xtr[batch]]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ytr[batch])
    if (i+1)%(n_iters/10) == 0 or i == 0 or i == n_iters-1:
        print(f'{i:6} | loss: {loss.item():.4f}')

    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = lr if i < int(0.7*n_iters) else lr_decay
    for p in parameters:
        p.data -= lr * p.grad

    with torch.no_grad():
        ud.append([(lr*p.grad.std()/p.data.std()).log10().item() for p in parameters])

    # if i == 1000:
    #     print(loss)
    #     break

# PART 4: NN STATISTICS AND THEIR GRAPHS
plt.ion()
def graph_statistics_activations(choice):
    assert choice in ['activations', 'activations_gradients']
    if choice == 'activations':
        print('Activations of layer.out')
    else:
        print('Gradients of layer.out.grad')
    plt.figure(figsize=(20,4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            if choice == 'activations':
                t = layer.out
                print(f'Layer {i} | ({layer.__class__.__name__}, {tuple(t.data.shape)}) | mean:{t.data.mean():+.2f} | std:{t.data.std():.2f} | saturated:{(t.data.abs()>0.97).float().mean()*100:.2f}%')
            else:
                t = layer.out.grad
                print(f'Layer {i} | ({layer.__class__.__name__}, {tuple(t.data.shape)}) | mean:{t.data.mean():+.2e} | std:{t.data.std():.2e}')
            hy, hx = torch.histogram(t.data, density=True)
            plt.plot(hx[:-1], hy)
            legends.append(f'{layer.__class__.__name__} {i}, {tuple(t.data.shape)}')
    plt.legend(legends);
    plt.title('activation distribution') if choice == 'activations' else plt.title('activation gradients distribution')
    plt.show()
graph_statistics_activations('activations')
graph_statistics_activations('activations_gradients')

def graph_statistics_parameters(choice):
    assert choice in ['weight_grad', 'update_data_ratio']
    if choice == 'weight_grad':
        print('Gradients of parameters')
    else:
        print('Update/data ratio of parameters')
    plt.figure(figsize=(20,4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            t = p.grad
            if choice == 'weight_grad':
                print(f'{str(tuple(t.data.shape)):<10} | mean:{t.data.mean():+.2e} | std:{t.data.std():.2e}, grad/data: {t.std()/p.data.std():.2e}')
                hy, hx = torch.histogram(t.data, density=True)
                plt.plot(hx[:-1], hy)
            else:
                plt.plot([ud[j][i] for j in range(len(ud))])
                print(f'{str(tuple(t.data.shape)):<10} | mean:{t.data.mean():+.2e} | std:{t.data.std():.2e} | update/data: {lr*t.std()/p.data.std():.2e}')
                plt.plot([0,len(ud)], [-3,-3], 'k')
            legends.append(f'{i}, {tuple(t.data.shape)}')
    plt.legend(legends);
    plt.title('weight gradient distribution') if choice == 'weight_grad' else plt.title('update/data ratio')
    plt.show()
graph_statistics_parameters('weight_grad')
plt.ioff()
graph_statistics_parameters('update_data_ratio')

# PART 5: RESULTS
# Losses of different splits
@torch.no_grad()
def loss_split(split):
    Xs,Ys = {
        'train': (Xtr, Ytr),
        'val': (Xval, Yval),
        'test': (Xte, Yte),
    }[split]
    emb = C[Xs]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        if isinstance (layer, BatchNorm1d):
            layer.training = False
        x = layer(x)
    loss = F.cross_entropy(x, Ys)
    print(split,'loss:', loss.item())
loss_split('train')
loss_split('val')

# Sample new names
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    context = [0] * block_size
    new_name = []
    while True:
        emb = C[context]
        x = emb.view(1,-1)
        for layer in layers:
            if isinstance (layer, BatchNorm1d):
                layer.training = False
            x = layer(x)
        probs = F.softmax(x, 1)
        ix = torch.multinomial(probs, 1, generator = g).item()
        context = context[1:] + [ix]
        new_name.append(itos[ix])
        if ix == 0:
            break
    print(''.join(new_name))

# Test loss once at the end
loss_split('test')