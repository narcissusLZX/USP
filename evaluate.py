from mydataset import Mydataset
import numpy as np

class Evaluation():
    def __init__(self, parameters, dataset = None) -> None:
        self.parameters = parameters
        if (dataset == None):
            self.parameters["init"] = False
            self.dataset = Mydataset(self.parameters)
        else:
            self.dataset = dataset
        self.getMatch()
        self.evaluate()
        self.n_match = 0

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    def getMatch(self):
        size = len(self.dataset.word2cluster)
        words = list(self.dataset.word2cluster.keys())
        print(self.dataset.n_sentence, size)
        vectors = np.zeros((self.dataset.n_sentence, size))
        for idx in range(self.dataset.n_sentence):
            for tok in self.dataset.sentences[idx]:
                tokidx = words.index(tok)
                vectors[idx][tokidx] += 1
        
        bound = self.dataset.evalStart
        self.match = [[] for _ in range(self.dataset.n_sentence-bound)]
        for idx in range(bound, self.dataset.n_sentence):
            for idx2 in range(bound):
                cos = self.cos_sim(vectors[idx2], vectors[idx])
                print(self.dataset.sentences[idx], self.dataset.sentences[idx2], cos)
                if (cos>= self.parameters["eval_threshold"]):
                    self.match[idx-bound].append(idx2)


    def evaluate(self):
        bound = self.dataset.evalStart
        ans_cnt, correct_cnt = 0, 0
        for idx in range(bound, self.dataset.n_sentence):
            for idx2 in self.match[idx-bound]:
                ans = self.dataset.qry(idx, idx2)
                ans_cnt += len(ans[0])
                correct_cnt += ans[1]
        if (ans_cnt == 0):
            print(0, 0, 0)
        else:
            print(ans_cnt, correct_cnt, correct_cnt / ans_cnt)
