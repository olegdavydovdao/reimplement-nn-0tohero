# Reimplement Neural Networks: Zero to Hero
![Loss gpt2 124M](logs/8_gpt2_logs/loss_graph_124M.png)   
Reimplement from scratch "Neural Networks: Zero to Hero" Andrej Karpathy's course.  
This course is an introduction to neural networks from the basics to modern architectures such as the GPT in code.  
Tech stack: Python, Pytorch, DL math and principles.  
Links to the original course: [GitHub](https://github.com/karpathy/nn-zero-to-hero), [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), [Site](https://karpathy.ai/zero-to-hero.html).
### Course completion process
Repeat for all lectures:  
1. Watch 1 youtube video lecture and write notebook code in parallel.
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
- **`3_batchnorm_and_statistics`** — statistics graphs of mlp model and batch normalization layer.
- **`4_backpropogation`** — manual derive backpropagation of tensor-level gradients.
- **`5_cnn_1d`** — wavenet architecture as 1 dimensional cnn for text.
- **`6_gpt_base`** — minimum core for transformer and gpt architecture pretraining stage without finetuning.
- **`7_tokenizer`** — bpe(byte pair encoding) algorithm for training and inference tokenizer.
- **`8_gpt2_base`** — OpenAI gpt2(124M parameters) architecture efficient ddp pretraining stage without finetuning. I train gpt2 on small dataset roughly 1 epoch = 320K tokens (I run 8 epochs, this roughly 2.5M tokens).  
This can be improved:  
expanding the dataset (for example FineWeb-Edu 10B tokens dataset) and training time;  
adding Hellaswag evaluation;  
adding a fine-tuning stage to the model so that the model can interact as a Q&A manner;  
acceleration code by convert Python/Pytorch into C/CUDA.

## Quick start
**Requirements**: [uv](https://docs.astral.sh/uv/), Git, Python 3.12+, a single Nvidia GPU for lecture 6,8 (tested on Nvidia T4 and Geforce GTX 1050Ti).  
```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/olegdavydovdao/reimplement-nn-0tohero.git

# 3. Install dependencies in accordance pyproject.toml file
uv sync

# 4. Run code in lectures
uv run lectures/#choose_file.py

# 5. Optional: Run 8_gpt2_base.py with ddp. --nproc_per_node=n, where n is number of GPUs in 1 node
uv run torchrun --standalone --nproc_per_node=1 lectures/8_gpt2_base.py
```
## 8_gpt2_base.py sample
Context feed into model: `What makes a lord worthy?`  
Result:
```
What makes a lord worthy?
So'll shall it, you I all as you be a blood:
As that.
To
My lord and that thy king. Good!

You, and not be so's will come, and the king, we he; and all to the house to be

Which with thy love out of thy man,


Come we can so.
KINGENCE:
And as not is, but they not shall are in us is be, not have
```
## License
MIT
