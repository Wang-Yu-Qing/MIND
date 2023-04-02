# MIND
Pytorch implementation of paper: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf)

# Dataset
* [Amazon books ratings](https://nijianmo.github.io/amazon/index.html), in **Files -> "Small" subsets for experimentation -> choose Books ratings only in the table**.
* [Amazon books metadata](https://jmcauley.ucsd.edu/data/amazon_v2/index.html), int **Files -> Complete review data -> choose Books metadata in the table**.

# How to run
1. Create a folder called `data`. Download and unzip the data files into the `data` folder, such that the `data` folder contains two files: `meta_Books.json` and `ratings_Books.csv`.
2. Run with default config: `python main.py`
3. Then you can see the output like this:
```
preparing data
start training
Epoch 00 | Step 00000 | Loss 25.838057
Epoch 00 | Step 00200 | Loss 2.774709
Epoch 00 | Step 00400 | Loss 1.765235
Epoch 00 | Step 00600 | Loss 1.416502
...
Epoch 49 | Step 02000 | Loss 0.217909
Epoch 49 | Step 02200 | Loss 0.217899
Epoch 49 | Step 02400 | Loss 0.217884
start testing
100%|█████████████████████████████████████████████████| 1630/1630 [00:23<00:00, 67.99it/s]
recall@30: 0.06549379147251112, hitRate@30: 0.3280333304991072
```

# Something about sample building
Each rating action reflect user's interest upon that book, so each rating action is one postive sample, no matter what the actual rating value is. From my point of view, If the user read the book and gave a rate about it, he is interested in the book anyway. Low rate means he is disappointed with the content, rather than "not interested".

# TODOs
* l2 norm & test via faiss
* introduce meta data and:
  * item embedding distribution
  * take a look into interest caps