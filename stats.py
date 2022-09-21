import numpy as np
import random

class Stats:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.n_occ = len(dataset.idx2occ)
        vectors = np.zeros(self.n_occ, dataset.vector_dim)
        for idx in range(self.n_occ):
            vectors[idx] = dataset.idx2occ[idx].featureVector #ensure normalized
        cos = np.dot(vectors, vectors.T)
        self.distribute = []
        random.seed(dataset.args.seed)
        for idx in range(self.n_occ):
            self.distribute[idx] = np.concatenate((cos[idx][:idx], cos[idx][idx+1:]), axis=0)
            self.distribute[idx] = np.exp(self.distribute[idx])
            self.distribute[idx] /= np.sum(self.distribute[idx])
            self.distribute[idx] = np.cumsum(self.distribute[idx])

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
    