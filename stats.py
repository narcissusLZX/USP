import numpy as np
import random



def alias_transform(prob_list):
    N = len(prob_list)
    prob_list = prob_list * N
    small, large = [], []
    accept_small, accept_large_idx = [0]*N, [-1]*N
    for i, prob in enumerate(prob_list):
        if (prob < 1.0):
            small.append(i)
        else:
            large.append(i)
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept_small[small_idx] = prob_list[small_idx]
        accept_large_idx[small_idx] = large_idx
        prob_list[large_idx] -= 1-prob_list[small_idx]
        if (prob_list[large_idx] < 1.0):
            small.append(large_idx)
        else:
            large.append(large_idx)
    while large:
        large_idx = large.pop()
        accept_small[large_idx] = 1
    return accept_small, accept_large_idx

def alias_sample(accept_args):
    accept_small_ratio, accept_large_idx = accept_args
    N = len(accept_small_ratio)
    i = np.random.randint(low=0, high=N)
    r = np.random.random()
    if r < accept_small_ratio[i]:
        return i
    return accept_large_idx[i]

class Stats:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.n_occ = len(dataset.idx2occ)
        vectors = np.zeros((self.n_occ, dataset.vector_dim))
        for idx in range(self.n_occ):
            vectors[idx] = dataset.idx2occ[idx+1].featureVector
        cos = np.dot(vectors, vectors.T)
        self.distribute = [0]*self.n_occ
        random.seed(dataset.args.seed)
        np.random.seed(dataset.args.seed)
        self.alias_distribution = []
        for idx in range(self.n_occ): #distribute_i：similarity distribution of i-th occurrence
            self.distribute[idx] = np.concatenate((cos[idx][:idx], cos[idx][idx+1:]), axis=0)
            self.distribute[idx] = np.exp(self.distribute[idx])
            self.distribute[idx] /= np.sum(self.distribute[idx])
            self.distribute[idx] = np.cumsum(self.distribute[idx])
            self.alias_distribution.append(alias_transform(self.distribute[idx]))

    def getOcc(self, occ1_idx):
        alpha = random.uniform(0, 1)
        left, right = 0, self.n_occ-2
        while left < right:
            mid = (left+right) // 2
            if (self.distribute[occ1_idx][mid] >= alpha):
                right = mid
            else:
                left = mid+1

        return left

    