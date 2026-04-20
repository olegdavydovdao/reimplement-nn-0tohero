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