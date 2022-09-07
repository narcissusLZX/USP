import random
import numpy as np
import os
import faiss

from cluster import Cluster
from occurrence import Occurrence
from argument import Argument


class Mydataset():
    def __init__(self, parameters):
        self.clusterIdx = 0
        self.idx2cluster = {} #idx->cluster
        self.word2cluster = {} # span->clusteridx  只在读入数据时使用

        self.occIdx = 0
        self.idx2occ = {}
        self.pos2occ = {}  #pos->occ
        self.parameters = parameters
        self.idx2root = {}
        self.tok2occ = {}

        self.argIdx = 0
        self.idx2arg = {}
        self.argtype2idx = {}

        random.seed(parameters["seed"])
        self.n_sentence = 0
        self.sentences = []
        self.argType2Arg = {}   
        self.TokPair2FaSon = {}
        self.pair_cnt = 0
        self.proposal = ""

        self.evalStart = 0
        self.evalAns = []

        self.n_argType = 0
        self.dynamic_features = parameters["DF"]

        if (self.parameters["init"]):
            self.loaddata()
            if (self.parameters["ExtVector"]):
                self.loadVector()

            self.precomputeVector()
            if self.parameters["Faiss"]:
                self.faiss_quantizer = faiss.IndexFlatIP(self.parameters["VectorDim"])
                self.span_index = faiss.IndexIVFFlat(self.faiss_quantizer, self.parameters["VectorDim"], 100, faiss.METRIC_INNER_PRODUCT) 
                data = [occ.featureVector for occ in self.idx2occ.values()]
                data = np.array(data).astype('float32')
                self.span_index.train(data)
                ids = np.arrange(len(self.idx2occ)).astype('int64')
                self.span_index.add_with_ids(data, ids)
                '''
                self.faiss_quantizer_cluster = faiss.IndexFlatIP(self.parameters["VectorDim"])
                self.cluster_index = faiss.IndexIVFFlat(self.faiss_quantizer_cluster, self.parameters["VectorDim"], 100, faiss.METRIC_INNER_PRODUCT)
                data = []
                for cluster in self.idx2cluster.values():
                    vector = self.GetFeatureVector(cluster)
                    vector = self.normalize(vector)
                    data.append(vector)
                data = np.array(data).astype('float32')
                ids = np.arrage(len(self.idx2cluster)).astype('int64')
                self.cluster_index.train(data)
                self.cluster_index.add_with_ids(data, ids)
                
                '''
        else:
            self.load()

    def __str__(self) -> str:
        ret = "Clusters:\n"
        for idx, cluster in self.idx2cluster.items():
            ret += str(idx)+":\n"+str(cluster)+"\n"

        '''
        ret += "Occurrence:\n"
        for idx, occ in self.idx2occ.items():
            ret += str(idx)+":\n"+str(occ)+"\n"

        ret += "Argument:\n"
        for idx, arg in self.idx2arg.items():
            ret += str(idx)+":\n"+str(arg)+"\n"
            
        '''

        return ret

    def normalize(self, vector):
        vector = np.array(vector).astype('float32')
        return vector / np.linalg.norm(vector)

    def readin(self, path):
        idx, idx2 = self.n_sentence, self.n_sentence
        with open(path+".tok", "r", encoding='utf-8') as f:
            self.sentences.append([])
            for line in f.readlines():
                tok = line.strip()
                if not tok:
                    if (len(self.sentences[self.n_sentence]) > 0):
                        self.n_sentence += 1
                        self.sentences.append([])
                else:
                    self.sentences[self.n_sentence].append(tok)
                    if tok not in self.word2cluster:
                        self.word2cluster[tok] = Cluster(self)
                    occ = Occurrence(tok, self, [self.n_sentence, len(self.sentences[self.n_sentence])], self.word2cluster[tok].idx)
                    self.word2cluster[tok].ins(occ)
                    if tok not in self.tok2occ:
                        self.tok2occ[tok] = []
                    self.tok2occ[tok].append(occ)

            if (len(self.sentences[self.n_sentence])==0):
                self.n_sentence -= 1
                self.sentences.pop()

        #dep
        with open(path+".dep", "r", encoding='utf-8') as f:
            for line in f.readlines():
                a = line.strip().split()
                if (len(a)==0):
                    idx += 1
                    continue
                if (len(a) != 3):
                    #print("warning!")
                    continue
                dep, fa, so = a
                if (dep == 'root'):
                    self.idx2root[idx] = so
                    continue
                fa_pos = fa.split("-")[-1]
                son_pos = so.split("-")[-1]
                if (fa_pos[-1]=='\''):
                    fa_pos = fa_pos[:-1]
                if (son_pos[-1]=='\''):
                    son_pos = son_pos[:-1]
                if (fa_pos==son_pos):
                    continue
                if (self.pos2hash([idx, fa_pos]) not in self.pos2occ) or (self.pos2hash([idx, son_pos]) not in self.pos2occ):
                    #print("warning: pos not in")
                    continue
                fa = self.pos2occ[self.pos2hash([idx, fa_pos])]
                son = self.pos2occ[self.pos2hash([idx, son_pos])]
                arg = Argument(self, fa, son, dep)
                fa.addSonArg(arg)
                son.addFaArg(arg)
                if dep not in self.argType2Arg:
                    self.argType2Arg[dep] = []
                self.argType2Arg[dep].append(arg)

                tokPair = fa.token+","+son.token
                if tokPair not in self.TokPair2FaSon:
                    self.TokPair2FaSon[tokPair] = []
                self.TokPair2FaSon[tokPair].append([fa, son])
            
        if self.parameters["ExtVector"]:
            tmp_read = np.load(path+".npz", allow_pickle = True)
            for i in range(idx2, self.n_sentence):
                arr = tmp_read['arr_0'][i-idx2]
                if (arr.shape[0] != len(self.sentences[i])):
                    for j in range(len(self.sentences[i])):
                        occ = self.pos2occ(self.pos2hash(i, j+1))
                        occ.featureVector = arr[j]


    def loaddata(self):
        path = self.parameters["path"]
        '''
        数据格式：
        连续若干行为一个句子的输入，句子间空行(严格1行)隔开
        .tok 依次保存各个词
        .dep 依次给出依赖关系
            一行的格式为：关系 主体-位置 客体-位置，位置从1开始计数
            如： det FKBP51-14 the-12
        '''
        self.readin(path)
        #读入tok
        self.evalStart = self.n_sentence
        if (self.parameters["eval"]):
            eval_path = self.parameters["eval_path"]
            self.readin(eval_path)
            self.n_eval = self.n_sentence - self.evalStart
            if self.parameters["eval_ans"]:
                with open(eval_path+".ans", "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        self.evalAns.append(line.strip())

        for fasons in self.TokPair2FaSon.values():
            self.pair_cnt += len(fasons)
            
        if self.parameters["Distributed"]:
            # todo
            return



    def precomputeVector(self):
        self.parameters["VectorDim"] += self.n_argType
        for occ in self.idx2occ.values():
            occ.precomputeVector(self.n_argType)

    def pos2hash(self, pos):
        return str(pos[1])+","+str(pos[0])
    
    def hash2pos(self, hash):
        return hash.split(",")

    def newOccurenceidx(self, occ:Occurrence):
        self.occIdx += 1
        self.idx2occ[self.occIdx] = occ
        self.pos2occ[self.pos2hash(occ.pos)] = occ
        return self.occIdx

    def newClusteridx(self, cluster:Cluster):
        self.clusterIdx += 1
        self.idx2cluster[self.clusterIdx] = cluster
        return self.clusterIdx

    def newArgumentIdx(self, arg:Argument):
        self.argIdx += 1
        self.idx2arg[self.argIdx] = arg
        if arg.argType not in self.argtype2idx:
            self.n_argType += 1
            self.argtype2idx[arg.argType] = self.n_argType
        return self.argIdx

    def removeCluster(self, cluster:Cluster):
        idx = cluster.idx
        if (idx not in self.idx2cluster):
            print("warning")
            return
        self.idx2cluster.pop(idx)
        
    def getRandomCluster(self):
        clusters = list(self.idx2cluster.values())
        return clusters[random.randint(0,len(clusters)-1)]

    def getRandomPair(self):
        TokPair = []
        for fason in self.TokPair2FaSon.values():
            TokPair.extend(fason)
        return TokPair[random.randint(0,len(TokPair)-1)]

    def GetFeatureVector(self, cluster:Cluster): #cluster中所有occ的feature_vector平均值
        ret = []
        for occs in cluster.Span2Occurr.values():
            for occ in occs:
                ret.append(occ.getFeatureVector())
        return np.mean(ret)

    def CalcSimilarity(self, cluster1:Cluster, cluster2:Cluster): 
        if cluster1.idx == cluster2.idx:
            return 0.0
        else:
            #todo
            vector1 = self.GetFeatureVector(cluster1) #动态计算
            vector2 = self.GetFeatureVector(cluster2) #
            cos_sim = np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)
            return cos_sim


    def getClusterbyClusterSimilarity(self, Cluster1:Cluster):
        if self.parameters["Faiss"]:
            #todo

            return 
        else:
            clusters = list(set(self.idx2cluster.values()))
            clusters.remove(Cluster1)
            prob = [self.CalcSimilarity(cluster, Cluster1) for cluster in clusters]

            scores = np.array(prob)
            scores = np.exp(scores) / np.sum(np.exp(scores))

            selectProb = random.random()
            sum = 0.0
            for i in range(len(scores)):
                sum += scores[i]
                if (sum > selectProb):
                    return clusters[i]

            return clusters[-1]

    def update(self, clusters):
        #print(clusters)
        if isinstance(clusters, list): 
            for cluster in clusters:
                self.update(cluster)
        else:
            if len(clusters.Span2Occurr) == 0:
                self.removeCluster(clusters)

    def check(self):
        for idx, cluster in self.idx2cluster.items():
            if (len(cluster.Span2Occurr) == 0):
                print(idx)

    def faiss_remove(self, occ:Occurrence):
        occ = occ.getTop()
        self.span_index.remove_ids(np.array([occ.idx]).astype('int64'))

    def faiss_calcLprob(self, vector, K=20):
        D, I = self.span_index.search(np.array(vector).astype("float32"), K)
        ret = 0
        for distance in D:
            if (distance >= self.parameters["sim_threshold"]):
                ret += 1
        return ret
    
    def faiss_insert(self, occ:Occurrence):
        occ = occ.getTop()
        vector = occ.getFeatureVector()
        self.span_index.add_with_ids(np.array([vector]).astype('float32'), np.array([occ.idx]).astype('int64'))
        
    def store(self):
        path = self.parameters["model_path"]
        if not os.path.exists(path):
            os.mkdir("path")
        path += "model.txt"
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self)

    def load(self):
        #todo
        path = self.parameters["model_path"]
        if not os.path.exists(path):
            print("Load Model Error!")
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if (line == "Cluster:"):
                    continue
                a = line.split(":")
                cluster = Cluster(a)
                self.newClusteridx(cluster)
                
    def qry(self, idx, tgt_idx):
        what_pos = self.pos2hash([idx, 1])
        occ = self.pos2occ[what_pos]
        occ = occ.getTop()
        faArg = occ.faArg
        
        tgt_occ = self.idx2root[tgt_idx]
        result = tgt_occ.qry(faArg.father.token, faArg.argType)
        ans = []
        correct_cnt = 0
        for span in result: 
            bounds = span[1]
            ans.append(self.sentences[bounds[0]-1:bounds[1]])
            if (self.parameters["eval_ans"]):
                if (ans[-1] == self.evalAns[idx-self.evalStart]):
                    correct_cnt += 1
        return ans, correct_cnt

        
