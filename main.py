import itertools
from utils import *


if __name__ == '__main__':
    args = parse_args()

    valid_users, valid_books = get_valid_user_and_book_set(args)
    all_user_rates = parse_rating_data(valid_users, valid_books)
    book_cates = read_book_meta(valid_books)
    valid_cates = set(itertools.chain.from_iterable(list(book_cates.values())))
    all_user_capsules_num = get_user_capsules_num(args.k, all_user_rates)

    user_encoder, user_decoder = encode_users(valid_users)
    book_encoder, book_decoder = encode_books(valid_books)
    cate_encoder, cate_decoder = encode_cates(valid_cates)

    all_train_rates, all_test_rates = split_train_and_test_rates(all_user_rates, args.train_frac)

    train_samples = build_samples(
        all_train_rates, 
        all_user_rates, 
        valid_books, 
        user_encoder,
        book_encoder,
        all_user_capsules_num,
        args
    )
    test_samples = build_samples(
        all_test_rates, 
        all_user_rates, 
        valid_books, 
        user_encoder,
        book_encoder,
        all_user_capsules_num,
        args
    )

    pkl_save("data/train_samples.pkl", train_samples)
    pkl_save("data/test_samples.pkl", test_samples)
