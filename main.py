import random
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader

from utils import *
from model import MIND


def train(args, model, trainData):
    BCELoss = th.nn.BCELoss()
    for epoch in range(args.epochs):
        epochTotalLoss = 0
        for step, (his, tar) in enumerate(trainData):
            bs = his.shape[0]
            caps = model.B2IRouting(his, bs)
            logits, labels = model.sampledSoftmax(caps, tar, bs)

            loss = BCELoss(logits, labels)
            loss.backward()
            model.opt.step()
            model.opt.zero_grad()
            epochTotalLoss += loss
            if (step % args.print_steps == 0):
                print('Epoch {:02d} | Step {:05d} | Loss {:.6f}'.format(
                    epoch,
                    step,
                    epochTotalLoss / (step + 1),
                ))


if __name__ == '__main__':
    args = parseArgs()
    print("preparing data")
    # prepare data
    ratings = pd.read_csv("data/Books.csv", header=None)
    ratings.columns = ['userId', 'itemId', 'rate', 'timestamp']
    # item filtering
    itemFreq = ratings.groupby(['itemId'])['itemId'].count()
    validSet = set(itemFreq.loc[itemFreq >= args.min_item_freq].index)
    ratings = ratings.loc[ratings['itemId'].apply(lambda x: x in validSet), :]
    # user filtering
    userFreq = ratings.groupby(['userId'])['userId'].count()
    validSet = set(userFreq.loc[userFreq >= args.min_user_freq].index)
    ratings = ratings.loc[ratings['userId'].apply(lambda x: x in validSet), :]
    # encode user
    ukv, ikv = list(enumerate(ratings['userId'].unique())), list(enumerate(ratings['itemId'].unique()))
    userRawId = {encId: rawId for encId, rawId in ukv}
    userEncId = {rawId: encId for encId, rawId in ukv}
    # encode item, id 0 is for padding, item encode id start from 1
    itemRawId = {encId + 1: rawId for encId, rawId in ikv}
    itemEncId = {rawId: encId + 1 for encId, rawId in ikv}
    # encode
    ratings['userId'] = ratings['userId'].apply(lambda x: userEncId[x])
    ratings['itemId'] = ratings['itemId'].apply(lambda x: itemEncId[x])
    ratings.sort_values(by=['userId', 'timestamp'], inplace=True, ignore_index=True)
    # split train and test users
    trainUsers, testUsers = set(), set()
    for userId in range(len(userRawId)):
        if (random.random() <= args.train_user_frac): trainUsers.add(userId)
        else: testUsers.add(userId)
    boolIdx = ratings['userId'].apply(lambda x: x in trainUsers)
    trainRatings = ratings.loc[boolIdx, :]
    testRatings = ratings.loc[~boolIdx, :]
    # generate train samples
    trainSamples = trainRatings.groupby('userId').apply(lambda x: genUserTrainSamples(args, x))
    trainHis = np.concatenate(trainSamples.apply(lambda x: x[0]).values).astype(np.int32)
    trainTar = np.concatenate(trainSamples.apply(lambda x: x[1]).values).astype(np.int32)
    # generate test samples
    testSamples = testRatings.groupby('userId').apply(lambda x: genUserTestSamples(args, x))
    testHis = np.stack(testSamples.apply(lambda x: x[0]).values).astype(np.int32)
    _testTar = testSamples.apply(lambda x: x[1]).values
    testTar = np.arange(0, _testTar.shape[0], 1).astype(np.int32)
    trainData = DataLoader(
        Dataset(trainHis, trainTar),
        batch_size = args.train_batch_size,
        shuffle=True
    )
    testData = DataLoader(
        Dataset(testHis, testTar),
        batch_size = args.test_batch_size,
        shuffle=True
    )
    print("start training")
    model = MIND(args, embedNum=len(itemEncId) + 1)
    train(args, model, trainData)
    print("start testing")
    test(model, testData, _testTar)