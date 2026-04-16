# THEME: REUSABLE CODE OPEN FILE, LETTERS AND THEIR INDICES
# PART 0: PREPARATION
# Import libraries
import os
import random
import torch

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

def get_splits_names(block_size):
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
    return Xtr,Ytr,Xval,Yval,Xte,Yte,itos,stoi,sz_voc,num_tr

if __name__ == '__main__':
    get_splits_names(block_size=1)