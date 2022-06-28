import json
import math
import pickle
import random
import argparse
import numpy as np
import pandas as pd


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_frac', type=float, default=0.05)
    argparser.add_argument('--min_user_freq', type=int, default=10)
    argparser.add_argument('--min_book_freq', type=int, default=10)
    argparser.add_argument('--max_user_freq', type=int, default=200)
    argparser.add_argument('--train_frac', type=int, default=0.95)
    argparser.add_argument('--his_len', type=int, default=100)
    argparser.add_argument('--n_neg', type=int, default=10)
    argparser.add_argument('--k', type=int, default=4)

    args = argparser.parse_args()

    return args


def get_valid_user_and_book_set(args):
    user_freq, book_freq = {}, {}
    with open('data/ratings_Books.csv', 'r') as f:
        for line in f.readlines():
            if random.random() < args.data_frac:
                # user, item, rating, timestamp
                splitted = line.split(",")
                try:
                    user_freq[splitted[0]] += 1
                except KeyError:
                    user_freq[splitted[0]] = 1
                try:
                    book_freq[splitted[1]] += 1
                except KeyError:
                    book_freq[splitted[1]] = 1

    valid_users = set([user_id for user_id, cnt in user_freq.items() if cnt >= args.min_user_freq and cnt <= args.max_user_freq])
    valid_books = set([book_id for book_id, cnt in book_freq.items() if cnt >= args.min_book_freq])

    return valid_users, valid_books
    

def parse_rating_data(valid_users, valid_books):
    user_rates = {}
    with open('data/ratings_Books.csv', 'r') as f:
        for line in f.readlines():
            line = line[:-1]
            splitted = line.split(",")
            user_id, book_id = splitted[0], splitted[1]
            if user_id in valid_users and book_id in valid_books:
                rate = [user_id, book_id, int(splitted[-1])]
                try:
                    user_rates[user_id].append(rate)
                except KeyError:
                    user_rates[user_id] = [rate]
    
    # convert user ratings to pd.DataFrame for efficient history quering
    for user_id, rates in user_rates.items():
        user_rates[user_id] = pd.DataFrame(rates, columns=['user_id', 'book_id', 'timestamp']).sort_values(by='timestamp', ignore_index=True)

    return user_rates


def read_book_meta(valid_books):
    book_cates = {}
    with open("data/meta_Books.json", "r") as f:
        for line in f.readlines():
            meta = json.loads(line[:-1])
            if meta['asin'] in valid_books:
                book_cates[meta['asin']] = meta['category']

    return book_cates


def encode_users(valid_users):
    encoder, decoder = {}, []
    encode_id = 0
    for user_id in valid_users:
        if user_id not in encoder:
            encoder[user_id] = encode_id
            decoder.append(user_id)
            encode_id += 1

    return encoder, decoder


def encode_books(valid_books):
    encoder, decoder = {'<pad>': 0}, ['<pad>']
    encode_id = 1
    for book_id in valid_books:
        if book_id not in encoder:
            encoder[book_id] = encode_id
            decoder.append(book_id)
            encode_id += 1
    
    return encoder, decoder


def encode_cates(book_cates):
    encoder, decoder = {'<pad>': 0}, ['<pad>']
    encode_id = 1
    for cate in book_cates:
        if cate not in encoder:
            encoder[cate] = encode_id
            decoder.append(cate)
            encode_id += 1
    
    return encoder, decoder


def split_train_and_test_rates(user_rates, train_frac):
    all_train_ratings, all_test_ratings = [], []
    for user_id, ratings in user_rates.items():
        train_rates = ratings.sample(frac=train_frac, replace=False)
        test_rates = ratings.drop(train_rates.index)

        for rate in train_rates.values: all_train_ratings.append(rate)
        for rate in test_rates.values: all_test_ratings.append(rate)
    
    return all_train_ratings, all_test_ratings


def pad_or_cut(seq, length):
    if len(seq) > length:
        # cut the front
        # len(seq) - 1 - x + 1 = length --> x = len(seq) - length
        return seq[len(seq) - length:]
    else:
        return np.concatenate((np.array(['<pad>'] * (length - len(seq))), seq))


def query_history_books(timestamp, user_rates, his_len):
    bool_idx = user_rates['timestamp'] < timestamp
    history_books = user_rates['book_id'].loc[bool_idx].values
    if len(history_books) != his_len:
        history_books = pad_or_cut(history_books, his_len)

    return history_books


def build_samples(
        rates, 
        all_user_rates, 
        valid_books, 
        user_encoder,
        book_encoder,
        all_user_capsules_num,
        args
    ):
    samples = []
    candidates = list(valid_books)
    for rate in rates:
        his = query_history_books(rate[2], all_user_rates[rate[0]], args.his_len)
        his = [book_encoder[book_id] for book_id in his]
        tar = book_encoder[rate[1]]
        samples.append({'user': user_encoder[rate[0]], 'his': his, 'tar': tar, 'label': 1, 'cap_num': all_user_capsules_num[rate[0]]})
        # negative sampling
        neg_tar = random.choices(candidates, k = args.n_neg)
        for tar in neg_tar:
            tar = book_encoder[tar]
            samples.append({'user': user_encoder[rate[0]], 'his': his, 'tar': tar, 'label': 0, 'cap_num': all_user_capsules_num[rate[0]]})
    
    return samples


def get_capsules_num(k, user_rating_size):
    return int(max(1, min(k, math.log(user_rating_size, 2))))


def get_user_capsules_num(k, all_user_rating_size):
    all_user_capsules_num = {}
    for user_id, ratings in all_user_rating_size.items():
        all_user_capsules_num[user_id] = get_capsules_num(k, len(ratings))

    return all_user_capsules_num


def pkl_save(filepath, obj):
    with open(filepath, "wb") as f:
        f.write(pickle.dumps(obj))


def pkl_read(filepath):
    with open(filepath, "rb") as f:
        return pickle.loads(f.read())
