# Reimplement Neural Networks: Zero to Hero
![Loss gpt 124m](logs/8_gpt2_logs/loss_graph_124M.png)   
Reimplement from scratch "Neural Networks: Zero to Hero" Andrej Karpathy's course.  
This course is an introduction to neural networks from the basics to modern architectures such as the GPT in code.  
Tech stack: Python, Pytorch.  
Links to the original course: [GitHub](https://github.com/karpathy/nn-zero-to-hero), [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), [Site](https://karpathy.ai/zero-to-hero.html).
### Course completion process
1. Watch youtube video lecture and write notebook code in parallel.
2. Close all hints and video and notebook code.
3. Reimplement all code from scratch all alone.

## Project structure
There are only 4 folders that matter in the project:
- **`data`** &mdash; 2 tiny datasets that are used in the repo.
- **`lectures`** &mdash; source code for all lectures.
- **`lectures/utils`** &mdash; reusable source code across lectures files.
- **`logs`** &mdash; folders that contains images of graphs that were created as a result of source code in the lectures folder.
### Source code
**`.py`** files in **lectures** and **lectures/utils** folders:
- **`preprocess_names`** — reusable prepare data for 1-5 lectures to feed into neural network.
- **`savefig`** — reusable save png image of graphs for all lectures.
- **`0_autograd`** — backpropogation autograd engine and train mlp at scalar level.
- **`1_bigram`** — bigram character level laguage model with 1 linear layer.
- **`2_mlp`** — n-gram character level mlp laguage model.
- **`3_batchnorm_and_statistics`** — statistics graphs of model and batch normalization layer.
- **`4_backpropogation`** — manual derive backpropagation of tensor-level gradients.
- **`5_cnn_1d`** — wavenet architecture as 1 dimensional cnn for text.
- **`6_gpt_base`** — transformer and gpt architecture pretraining stage without finetuning.
- **`7_tokenizer`** — bpe(byte pair encoding) algorithm for training and inference tokenizer.
- **`8_gpt2_base`** — OpenAI gpt2(124m parameters) architecture efficient ddp pretraining stage without finetuning. I train gpt2 on small dataset roughly 300K tokens 8 epochs. This can be improved by expanding the dataset (for example FineWeb-Edu 10B tokens dataset) and training time.

## Quick start
**Requirements**: [uv](https://docs.astral.sh/uv/), Git, Python 3.12+, a single Nvidia GPU for lecture 6,8 (tested on Nvidia T4 and Geforce GTX 1050Ti).  
```
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/olegdavydovdao/reimplement-nn-0tohero.git

# 3. Install dependencies in accordance pyproject.toml file
uv sync

# 4. Run code in lectures
uv run lectures/#choose_file.py

# 5. Run 8_gpt2_base with ddp (if you have 1 node with >= 2 GPUs). --nproc_per_node=n, where n is number of GPUs 
uv run torchrun --standalone --nproc_per_node=2 lectures/8_gpt2_base.py
```
## 8_gpt2_base.py sample
Context feed into model: `What makes a lord worthy?`  
Result:
```
What makes a lord worthy?

I am this? to us it to the noble heart from the mother as be we, I would the father.

Why-ADas you and you.
And the soul with a brother-ESre I'll I shall thee, we did she may it,

A:
And a good that, if byer in our love for not.
ROMELLFKE:
Now we think this had it do, then's love to my lord.
```
## License
MIT
