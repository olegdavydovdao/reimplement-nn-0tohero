# THEME: TEXT TOKENIZATION
# PART 0: DATA PREPARATION
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import regex as re
import os

# Data loading
text_choise(split):
    assert split in ['tiny', 'big'], "incorrect argument"
    path_data_str = os.path.join('data', 'shakespeare.txt')
    with open(path_data_str, 'r', encoding='utf-8') as f:
        text = f.read()
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
text = text_choise('tiny')

# PART 1: TOKENIZER TRAIN
# Tokens separating with regex pattern
# Note: I use regex in train too instead original lecture inference only
pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}{1,3}+| ?[^\s\p{L}\p{N}]|\s+(?!\S)|\s+""")
list_text = pattern.findall(text)
tokens = []
for i in range(len(list_text)):
    tokens.append(list(list_text[i].encode('utf-8')))

# Stats of pairs tokens
def get_pair_frequency(tokens):
    stats = {}
    for chunk in tokens:
        for pair in zip(chunk, chunk[1:]):
            stats[pair] = stats.get(pair, 0) + 1
    return stats

# Update your tokens sequence
def merge(tokens, max_pair, new_token):
    new_tokens = []
    for chunk in tokens:
        i = 0
        new_chunk = []
        while i<len(chunk):
            if i+1<len(chunk) and chunk[i]==max_pair[0] and chunk[i+1]==max_pair[1]:
                new_chunk.append(new_token)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        new_tokens.append(new_chunk)
    return new_tokens

# Train function of tokinization with regex
default_toks = 256
vocab_size = 276
extra_merges = vocab_size-default_toks
merges_main = {}
for k in range(extra_merges):
    pa_fr = get_pair_frequency(tokens)
    max_pair = max(pa_fr, key=pa_fr.get)
    new_token = k + default_toks
    tokens = merge(tokens, max_pair, new_token)
    merges_main[max_pair] = new_token