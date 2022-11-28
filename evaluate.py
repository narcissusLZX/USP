from mydataset import Mydataset
import numpy as np
import argparse

class Evaluation():
    def __init__(self, args, dataset = None) -> None:
        self.args = args
        if (dataset == None):
            self.args.init = False
            self.dataset = Mydataset(self.args)
        else:
            self.dataset = dataset
        self.sentences = []
        self.n_sentence = 0
        self.readin(self.args.data_path)
        self.evalStart = self.n_sentence
        self.readin(self.args.eval_path)
        print(self.n_sentence, self.evalStart)
        #print(self.sentences[self.n_sentence-1], self.sentences[self.evalStart])
        self.getMatch()
        #self.evaluate()
        self.n_match = 0

    def readin(self, path):
        with open(path+".tok", "r", encoding='utf-8') as f:
            self.sentences.append([])
            occidx = 0
            for line in f.readlines():
                tok = line.strip()
                if not tok:
                    if (len(self.sentences[self.n_sentence]) > 0):
                        self.n_sentence += 1
                        self.sentences.append([])
                else:
                    self.sentences[self.n_sentence].append(tok)
                    occidx += 1
                    self.dataset.pos2occ[self.dataset.pos2hash([self.n_sentence,len(self.sentences[self.n_sentence])])] = self.dataset.idx2occ[occidx]
        #self.n_sentence += 1

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2)

    def getMatch(self):
        print("Matching Questions and Sentences ...")
        size = len(self.dataset.idx2cluster)
        clusters = list(self.dataset.idx2cluster.keys())
        print(self.n_sentence, size)
        #print(self.sentences)
        '''
        vectors = np.zeros((self.n_sentence, size))
        for idx in range(self.n_sentence):
            #print(idx)
            for tok_idx in range(len(self.sentences[idx])):
                occ = self.dataset.pos2occ[self.dataset.pos2hash([idx, tok_idx+1])]
                if (occ.label != occ.idx):
                    continue
                vectors[idx][clusters.index(occ.clusteridx)] += 1
            vectors[idx] /= np.linalg.norm(vectors[idx])
        
        self.match = [[] for _ in range(self.n_sentence-self.evalStart)]
        print(self.n_sentence, self.evalStart)
        
        questions = vectors[self.evalStart:]
        sentences = vectors[:self.evalStart]
        #sentences = vectors[:100]
        co = questions.dot(sentences.T)
        for idx in range(questions.shape[0]):
            for s_idx in range(sentences.shape[0]):
                if (co[idx][s_idx] > self.args.eval_threshold):
                    self.match[idx].append(s_idx)
        '''

        #print(self.match[577])
        ans = self.dataset.qry(577+self.evalStart, 6530)
        print(ans[0])
        print(self.sentences[577+self.evalStart], self.sentences[2084], self.sentences[6530])
        question_ids = [577]
        sentence_ids = [2084, 6530]
        vectors = np.zeros((len(question_ids)+len(sentence_ids), size))
        vector_idx = 0
        for idx in question_ids+sentence_ids:
            for tok_idx in range(len(self.sentences[idx])):
                occ = self.dataset.pos2occ[self.dataset.pos2hash([idx, tok_idx+1])]
                if (occ.label != occ.idx):
                    continue
                vectors[vector_idx][clusters.index(occ.clusteridx)] += 1
            vectors[vector_idx] /= np.linalg.norm(vectors[vector_idx])
            vector_idx += 1
        
        questions = vectors[:len(question_ids)]
        sentences = vectors[len(question_ids):]
        #sentences = vectors[:100]
        co = questions.dot(sentences.T)
        print(co)
        #print(co[577][2082:2085], co[577][6522:6525])
        print("Match Done.")

    def evaluate(self):
        bound = self.evalStart
        ans_cnt, correct_cnt = 0, 0
        for idx in range(bound, self.n_sentence):
            #print(self.sentences[idx])
            for idx2 in self.match[idx-bound]:
                ans = self.dataset.qry(idx, idx2)
                if (len(ans[0]) > 0):
                    ans_string = [self.sentences[idx2][pair[0]:pair[1]] for pair in ans[0]]
                    print(self.sentences[idx], ans_string, idx, idx2)
                ans_cnt += len(ans[0])
                correct_cnt += ans[1]
        if (ans_cnt == 0):
            print(0, 0, 0)
        else:
            print(ans_cnt, correct_cnt, correct_cnt / ans_cnt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./dataset/geniaquarter", type=str)
    parser.add_argument("--output_path", default="./dataset", type=str)
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed.")
    parser.add_argument("--Df", action="store_true", help="Dynamically adjust feature vectors of occurrences?")
    parser.add_argument("--ExtVector", action="store_true", help="Extern precomputed feature vectors?")
    parser.add_argument("--ClusterDistrConc", default=1.5, type=float)
    parser.add_argument("--VectorDim", default=0, type=int)
    parser.add_argument("--init", action="store_true", help="Initiallize dataset?")
    parser.add_argument("--eval", action="store_true", help="Evaluation?")
    parser.add_argument("--eval_path", default=None, type=str)
    parser.add_argument("--eval_ans", action="store_true", help="Match ans?")
    parser.add_argument("--Faiss", action="store_true")
    parser.add_argument("--eval_threshold", default=0.8, type=float)
    parser.add_argument("--model_path", default="./model/", type=str)
    parser.add_argument("--save_per_epoch", default=10000, type=int)
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--sim_threshold", default=0.8, type=float)
    parser.add_argument("--Distributed", action="store_true")
    model_args = parser.add_argument_group(title="Parameters of model")
    model_args.add_argument("--superConc", default=591, type=float, help="gamma. Larger the value: larger total number of clusters, smaller the clusters.")
    model_args.add_argument("--cluster_Conc", default=1.5, type=float)
    model_args.add_argument("--arg_Conc", default=0.001, type=float)
    model_args.add_argument("--gen_first_eta", nargs='+', default=[0.01, 0.001], type=float, help="Parameters of generating first argument")
    model_args.add_argument("--gen_more_eta", nargs='+', default=[0.01, 0.001], type=float,
                        help="Parameters of generating more arguments")
    model_args.add_argument("--cluster_alpha", default=0.75, type=float)
    model_args.add_argument("--arg_alpha", default=0.5, type=float)
    args = parser.parse_args()
    eva = Evaluation(args)
