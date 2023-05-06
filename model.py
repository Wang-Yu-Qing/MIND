import torch as th


class MIND(th.nn.Module):
    def __init__(self, args, embedNum):
        super(MIND, self).__init__()
        self.D = args.D
        self.K = args.K
        self.R = args.R
        self.L = args.seq_len
        self.nNeg = args.n_neg
        # weights initialization
        self.itemEmbeds = th.nn.Embedding(embedNum, self.D, padding_idx=0)
        self.dense1 = th.nn.Linear(self.D, 4 * self.D)
        self.dense2 = th.nn.Linear(4 * self.D, self.D)
        # one S for all routing operations, first dim is for batch broadcasting
        S = th.empty(self.D, self.D)
        th.nn.init.normal_(S, mean=0.0, std=1.0)
        self.S = th.nn.Parameter(S) # don't forget to make S as model parameter
        # fixed routing logits once initialized
        self.B = th.nn.init.normal_(th.empty(self.K, self.L), mean=0.0, std=1.0)
        self.opt = th.optim.Adam(self.parameters(), lr=args.lr)

    # output caps' length is in (0, 1)
    def squash(self, caps, bs):
        n = th.norm(caps, dim=2).view(bs, self.K, 1)
        nSquare = th.pow(n, 2)

        return (nSquare / ((1 + nSquare) * n + 1e-9)) * caps
    
    def B2IRouting(self, his, bs):
        """B2I dynamic routing, input behaviors, output caps
        """
        # init b, bji = b[j][i] rather than b[i][j] for matmul convinience
        # no grad for b: https://github.com/Ugenteraan/CapsNet-PyTorch/blob/master/CapsNet-PyTorch.ipynb
        B = self.B.detach()
        # except for first routing round, each sample's w is different, so need a dim for batch
        B = th.tile(B, (bs, 1, 1)) # (bs, K, L)

        # masking, make padding indices' routing logit as INT_MAX so that softmax result is 0
        # (bs, L) -> (bs, 1, L) -> (bs, K, L)
        mask = (his != 0).unsqueeze(1).tile(1, self.K, 1)
        drop = (th.ones_like(mask) * -(1 << 31)).type(th.float32)

        his = self.itemEmbeds(his) # (bs, L, D)
        his = th.matmul(his, self.S)

        for i in range(self.R):
            BMasked = th.where(mask, B, drop)
            W = th.softmax(BMasked, dim=2) # (bs, K, L)
            if i < self.R - 1:
                with th.no_grad():
                    # weighted sum all i to each j
                    caps = th.matmul(W, his) # (bs, K, D)
                    caps = self.squash(caps, bs)
                    B += th.matmul(caps, th.transpose(his, 1, 2)) # (bs, K, L)
            else:
                caps = th.matmul(W, his) # (bs, K, D)
                caps = self.squash(caps, bs)
                # skip routing logits update in last round

        # mlp
        caps = self.dense2(th.relu(self.dense1(caps))) # (bs, K, D)
        ## l2 norm
        #caps = caps / (th.norm(caps, dim=2).view(bs, self.K, 1) + 1e-9)
        
        return caps
    
    def labelAwareAttation(self, caps, tar, p=2):
        """label-aware attention, input caps and targets, output logits
            caps: (bs, K, D)
            tar: (bs, cnt, D)
            for postive tar, cnt = 1
            for negative tar, cnt = self.nNeg
        """
        tar = tar.transpose(1, 2) # (bs, D, cnt)
        w = th.softmax(
                # (bs, K, D) X (bs, D, cnt) -> (bs, K, cnt) -> (bs, cnt, K)
                th.pow(th.transpose(th.matmul(caps, tar), 1, 2), p),
                dim=2
            )
        w = w.unsqueeze(2) # (bs, cnt, K) -> (bs, cnt, 1, K)

        # (bs, cnt, 1, K) X (bs, 1, K, D) -> (bs, cnt, 1, D) -> (bs, cnt, D)
        caps = th.matmul(w, caps.unsqueeze(1)).squeeze(2)

        return caps

    def sampledSoftmax(self, caps, tar, bs, tmp=0.01):
        tarPos = self.itemEmbeds(tar) # (bs, D)
        capsPos = self.labelAwareAttation(caps, tarPos.unsqueeze(1)).squeeze(1) # (bs, D)
        # pos logits
        #his = his / (th.norm(his, dim=1).view(bs, 1) + 1e-9)
        #tar = tar / (th.norm(tar, dim=1).view(bs, 1) + 1e-9)
        # (bs, D) dot (bs, D) -> (bs, D) - sum > (bs, )
        posLogits = th.sigmoid(th.sum(capsPos * tarPos, dim=1) / tmp)

        # neg logits
        # in-batch negative sampling
        tarNeg = tarPos[th.multinomial(th.ones(bs), self.nNeg * bs, replacement=True)].view(bs, self.nNeg, self.D) # (batch_size, nNeg, D)
        capsNeg = self.labelAwareAttation(caps, tarNeg)
        # (bs, nNeg, dim) -> (bs, nNeg, 1) -> (bs * nNeg, )
        negLogits = th.sigmoid(th.sum(capsNeg * tarNeg, dim=2).view(bs * self.nNeg) / tmp)

        logits = th.concat([posLogits, negLogits])
        labels = th.concat([th.ones(bs, ), th.zeros(bs * self.nNeg)])

        return logits, labels