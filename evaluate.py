from mydataset import Mydataset
import numpy as np

class Evaluation():
    def __init__(self, parameters) -> None:
        self.parameters = parameters
        self.parameters["init"] = False
        self.dataset = Mydataset(self.parameters)
        self.getMatch()
        self.evaluate()
        self.n_match = 0

    def cos_sim(self, v1, v2):
        return v1.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    def getMatch(self):
        size = len(self.dataset.word2cluster)
        words = list(self.dataset.word2cluster.keys())
        vectors = np.zeros(self.dataset.n_sentence, size)
        for idx in range(self.dataset.n_sentence):
            for tok in self.dataset.sentences[idx]:
                tokidx = words.index(tok)
                vectors[idx][tokidx] += 1
        
        bound = self.parameters["evalidx"]
        self.match = [0 for _ in range(self.dataset.n_sentence-bound)]
        for idx in range(bound, self.dataset.n_sentence):
            mx_cos = self.cos_sim(vectors[0], vectors[idx])
            for idx2 in range(bound):
                cos = self.cos_sim(vectors[idx2], vectors[idx])
                if (cos>mx_cos):
                    mx_cos = cos
                    self.match[idx-bound] = idx2

            if (mx_cos < self.parameters["eval_threshold"]):
                self.match[idx-bound] = -1
            else:
                self.n_match += 1


    def evaluate(self):
        bound = self.parameters["evalidx"]
        ans_cnt, correct_cnt = 0, 0
        for idx in range(bound, self.dataset.n_sentence):
            if (self.match[idx-bound] == -1):
                continue
            ans_cnt += 1
            ans = self.dataset.qry(idx)
            if (ans[1] == True):
                correct_cnt += 1
        print(ans_cnt, correct_cnt, correct_cnt / ans_cnt)
