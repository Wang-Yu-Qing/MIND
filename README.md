# MIND (Under developing)
Implementation of paper: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf)

# Dataset
[Amazon books ratings](https://nijianmo.github.io/amazon/index.html), in **Files -> "Small" subsets for experimentation -> choose Books ratings only in the table**.

[Amazon books metadata](https://nijianmo.github.io/amazon/index.html), int **Files -> Complete review data -> choose Books metadata in the table**.

# How to run
1. Create a folder called `data`. Download and unzip the data files into the `data` folder, such that the `data` folder contains two files: `meta_Books.json` and `ratings_Books.csv`.
2. Run with default config: `python main.py`
3. The full data size is big. If you are eager to see some result or want to make some tests, use a small `data_frac` to decrease the amount of data used for sample building

# Something about sample building
1. Each rating action reflect user's interest upon that book, so each rating action is one postive sample, no matter what the actual rating value is. From my point of view, If the user read the book and gave a rate about it, he is interested in the book anyway. Low rate means he is disappointed with the content, rather than "not interested".
2. Split all the rating actions to 19:1 just as described in the paper.
3. From each rating action, use that user's rating actions before the current rating time as history behaviors.
4. For negative samples, keep the history behavior same as the positive sample, but randomly draw target books from the whole book pool.

# References
> [1] Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering R. He, J. McAuley WWW, 2016;
> 
> [2] Image-based recommendations on styles and substitutes J. McAuley, C. Targett, J. Shi, A. van den Hengel SIGIR, 2015;
> 
> [3] Justifying recommendations using distantly-labeled reviews and fined-grained aspects Jianmo Ni, Jiacheng Li, Julian McAuley Empirical Methods in Natural Language Processing (EMNLP), 2019;
