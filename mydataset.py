import random
import numpy as np
import os
import json
import faiss

from cluster import Cluster
from occurrence import Occurrence
from argument import Argument


class Mydataset():
    def __init__(self, args):
        self.clusterIdx = 0
        self.idx2cluster = {} #idx->cluster
        self.word2cluster = {} # span->clusteridx  只在读入数据时使用

        self.occIdx = 0
        self.idx2occ = {}
        self.pos2occ = {}  #pos->occ
        self.args = args
        self.idx2root = {}
        self.tok2occ = {}

        self.argIdx = 0
        self.idx2arg = {}
        self.argtype2idx = {}

        random.seed(args.seed)
        self.n_sentence = 0
        self.sentences = []
        self.argType2Arg = {}   
        self.TokPair2FaSon = {}
        self.pair_cnt = 0
        self.proposal = ""

        self.evalStart = 0
        self.evalAns = []
        self.n_eval = 0

        self.n_argType = 0
        self.dynamic_features = args.Df
        self.vector_dim = args.VectorDim

        if (self.args.init):
            self.loaddata()
            if (self.args.ExtVector):
                self.loadVector()
            
            self.precomputeVector()
            if self.args.Faiss:
                self.buildFaiss()
                
        else:
            self.load(self.args.load_model_path)
        print("n_sentence:", self.n_sentence)
        print("n_test_sentence:", self.n_eval)
        print("n_occurrence:", len(self.idx2occ))

    def load(self, path):
        if not os.path.exists(path):
            print("Load Model Error!")
            return
        print("Loading Model ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.n_argType = int(data["n_argType"])
        self.n_sentence = int(data["n_sentence"])
        self.n_eval = int(data["n_eval"])
        self.evalAns = data["evalAns"]
        self.argtype2idx = map(lambda x:int(x), data["argtype2idx"])
        self.pair_cnt = int(data["pair_cnt"])
        self.argIdx = int(data["argIdx"])
        self.occIdx = int(data["occIdx"])
        self.clusterIdx = int(data["clusterIdx"])
        self.vector_dim = data["vector_dim"]

        for idx, clustidx in data["idx2cluster"].items():
            self.idx2cluster[int(idx)] = Cluster(self, int(clustidx))
        for idx, occidx in data["idx2occ"].items():
            self.idx2occ[int(idx)] = Occurrence("", self, int(occidx))
        for idx, argidx in data["idx2arg"].items():
            self.idx2arg[int(idx)] = Argument(self, None, None, None, int(argidx))
        for idx, root in data["idx2root"].items():
            self.idx2root[int(idx)] = self.idx2occ[int(root)]
        self.argtype2idx = data["argtype2idx"]
        
        for cluster in data["clusters"]:
            self.idx2cluster[int(cluster["idx"])].load(cluster)
        for occurrence in data["occs"]:
            self.idx2occ[int(occurrence["idx"])].load(occurrence)
        for argument in data["args"]:
            self.idx2arg[int(argument["idx"])].load(argument)

        for tokpair, fason_idxs in data["TokPair2FaSon"].items():
            self.TokPair2FaSon[tokpair] = [[self.idx2occ[int(pair[0])], self.idx2occ[int(pair[1])]] for pair in fason_idxs]

        if self.args.Faiss:
            self.buildFaiss()

    def buildFaiss(self):
        print("Building Faiss Index ...")
        self.faiss_quantizer = faiss.IndexFlatIP(self.vector_dim)
        self.span_index = faiss.IndexIVFFlat(self.faiss_quantizer, self.vector_dim, 50, faiss.METRIC_INNER_PRODUCT) 
        data = [occ.featureVector for occ in self.idx2occ.values()]
        data = np.array(data).astype('float32')
        print(data.shape, self.vector_dim)
        self.span_index.train(data)
        ids = np.arange(1, len(self.idx2occ)+1).astype('int64')
        self.span_index.add_with_ids(data, ids)
        
        self.faiss_quantizer_cluster = faiss.IndexFlatIP(self.vector_dim)
        self.cluster_index = faiss.IndexIVFFlat(self.faiss_quantizer_cluster, self.vector_dim, 50, faiss.METRIC_INNER_PRODUCT)
        data = []
        ids = []
        for cluster in self.idx2cluster.values():
            vector = self.GetFeatureVector(cluster)
            vector = self.normalize(vector)
            assert vector.shape[0]==self.vector_dim
            data.append(vector)
            ids.append(cluster.idx)
        data = np.array(data).astype('float32')
        ids = np.array(ids).astype('int64')
        print(data.shape, ids.shape)
        self.cluster_index.train(data)
        self.cluster_index.add_with_ids(data, ids)
        
        print("Build Faiss Index Done.")

    def normalize(self, vector):
        vector = np.array(vector).astype('float32')
        if (np.linalg.norm(vector) == 0.):
            print("here")
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
                    occ = Occurrence(tok, self, -1, [self.n_sentence, len(self.sentences[self.n_sentence])], self.word2cluster[tok].idx)
                    self.word2cluster[tok].ins(occ)
                    if tok not in self.tok2occ:
                        self.tok2occ[tok] = []
                    self.tok2occ[tok].append(occ)

            if (len(self.sentences[self.n_sentence])==0):
                self.sentences.pop()
            else:
                self.n_sentence+=1

        #dep
        flag = False
        with open(path+".dep", "r", encoding='utf-8') as f:
            for line in f.readlines():
                a = line.strip().split()
                if (len(a)==0):
                    idx += 1
                    continue
                if (len(a) != 3):
                    print("warning!")
                    continue
                dep, fa, so = a
                fa_pos = fa.split("-")[-1]
                son_pos = so.split("-")[-1]
                if (fa_pos[-1]=='\''):
                    fa_pos = fa_pos[:-1]
                if (son_pos[-1]=='\''):
                    son_pos = son_pos[:-1]
                if (fa_pos==son_pos):
                    print("here")
                    continue
                if (dep == 'root'):
                    self.idx2root[idx] = self.pos2occ[self.pos2hash([idx, son_pos])]
                    continue
                if (self.pos2hash([idx, fa_pos]) not in self.pos2occ) or (self.pos2hash([idx, son_pos]) not in self.pos2occ):
                    print("warning: pos not in")
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
            
        if self.args.ExtVector:
            tmp_read = np.load(path+".npz", allow_pickle = True)
            for i in range(idx2, self.n_sentence):
                arr = tmp_read['arr_0'][i-idx2]
                if (arr.shape[0] != len(self.sentences[i])):
                    for j in range(len(self.sentences[i])):
                        occ = self.pos2occ(self.pos2hash(i, j+1))
                        occ.featureVector = arr[j]


    def loaddata(self):
        print("Loading Data ...")
        path = self.args.data_path
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
        if (self.args.eval):
            print("Loading eval data ... ")
            eval_path = self.args.eval_path
            self.readin(eval_path)
            self.n_eval = self.n_sentence - self.evalStart
            if self.args.eval_ans:
                with open(eval_path+".ans", "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        self.evalAns.append(line.strip())

        for fasons in self.TokPair2FaSon.values():
            self.pair_cnt += len(fasons)
            
        print("Load Data done.")
        if self.args.Distributed:
            # todo
            return



    def precomputeVector(self):
        self.vector_dim += self.n_argType*2+1
        for occ in self.idx2occ.values():
            occ.precomputeVector(self.n_argType)

    def pos2hash(self, pos):
        return str(pos[0])+","+str(pos[1])
    
    def hash2pos(self, hash):
        return hash.split(",")

    def newOccurrenceidx(self, occ:Occurrence):
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
            #print("warning")
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
        return np.mean(ret, axis=0)

    def CalcSimilarity(self, cluster1:Cluster, cluster2:Cluster): 
        if cluster1.idx == cluster2.idx:
            return -10
        else:
            #todo
            vector1 = self.GetFeatureVector(cluster1) #动态计算
            vector2 = self.GetFeatureVector(cluster2) #
            cos_sim = np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)
            return cos_sim


    def getClusterbyClusterSimilarity(self, Cluster1:Cluster):
        if self.args.Faiss:
            #todo
            #print(len(self.idx2occ), len(self.idx2cluster))
            #print(self.cluster_index.nprobe)
            d, i = self.cluster_index.search(np.array([self.GetFeatureVector(Cluster1)]).astype('float32'), 30)
            clusters, prob = [], []
            for idx in range(len(i[0])):
                if (i[0][idx] == Cluster1.idx):
                    continue
                if (i[0][idx] == -1):
                    break
                clusters.append(i[0][idx])
                prob.append(d[0][idx])
            d_min = d[0].min()
            for idx in self.idx2cluster.keys():
                if (idx not in clusters):
                    clusters.append(idx)
                    prob.append(d_min)
            clusters = [self.idx2cluster[idx] for idx in clusters]
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
        self.faiss_cluster_remove(clusters)
        new_clusters = []
        for cluster in clusters:
            if len(cluster.Span2Occurr) == 0:
                self.removeCluster(cluster)
            else:
                new_clusters.append(cluster)
        self.faiss_cluster_insert(new_clusters)

    def check(self):
        for idx, cluster in self.idx2cluster.items():
            if (len(cluster.Span2Occurr) == 0):
                print(idx)

    def faiss_cluster_remove(self, clusters):
        self.cluster_index.remove_ids(np.array([cluster.idx for cluster in clusters]).astype('int64'))

    def faiss_cluster_insert(self, clusters):
        vector = [self.normalize(self.GetFeatureVector(cluster)) for cluster in clusters]
        idx = [cluster.idx for cluster in clusters]
        self.cluster_index.add_with_ids(np.array(vector).astype('float32'), np.array(idx).astype('int64'))



    def faiss_remove(self, occ:Occurrence):
        occ = occ.getTop()
        self.span_index.remove_ids(np.array([occ.idx]).astype('int64'))

    def faiss_calcLprob(self, vector, K=20):
        D, I = self.span_index.search(np.array([vector]).astype("float32"), K)
        ret = 0
        for distance in D[0]:
            if (distance >= self.args.sim_threshold):
                ret += 1
        return ret
    
    def faiss_insert(self, occ:Occurrence):
        occ = occ.getTop()
        vector = occ.getFeatureVector()
        self.span_index.add_with_ids(np.array([vector]).astype('float32'), np.array([occ.idx]).astype('int64'))
        
    def transform2idx(self, old_dict):
        newdict = {}
        for idx, obj in old_dict.items():
            newdict[idx] = obj.idx
        return newdict

    def store(self, path): #不保存faiss
        print("Storing Model ...")
        #path = self.parameters["model_path"]
        #if not os.path.exists(path):
            #os.mkdir("path")
        data = {}
        data["idx2cluster"]=self.transform2idx(self.idx2cluster)
        data["idx2occ"] = self.transform2idx(self.idx2occ)
        data["idx2arg"] = self.transform2idx(self.idx2arg)
        data["idx2root"] = self.transform2idx(self.idx2root)
        data["argtype2idx"] = self.argtype2idx
        #data["argType2Arg"] = self.transform2idx(self.argType2Arg)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                continue
            if k == "span_index" or k == "cluster_index" or k == "faiss_quantizer" or k == "faiss_quantizer_cluster":
                continue
            data[k] = v
        
        data["TokPair2FaSon"] = {}
        for tokpair, fasons in self.TokPair2FaSon.items():
            data["TokPair2FaSon"][tokpair] = [[pair[0].idx, pair[1].idx] for pair in fasons]
        
        clusters = list(self.idx2cluster.values())
        data["clusters"] = [cluster.store() for cluster in clusters]
        occs = list(self.idx2occ.values())
        data["occs"] = [occ.store() for occ in occs]
        args = list(self.idx2arg.values())
        data["args"] = [arg.store() for arg in args]


        #path += "model.json"    
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        print("Store Model Done.")

        
                
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
            if (self.args.eval_ans):
                if (ans[-1] == self.evalAns[idx-self.evalStart]):
                    correct_cnt += 1
        return ans, correct_cnt

        
