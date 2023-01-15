import numpy as np
import math

class Occurrence():
    '''
        token: 
        pos: x-th sentence, y-th word
    '''
    def __init__(self, token, dataset, idx = -1, pos=[-1,-1], clusteridx = 0):
        self.dataset = dataset
        self.token = token
        self.clusteridx = clusteridx
        self.pos = pos
        self.sonArgType2Arg = {}  #ArgType->Arg List
        self.faArg = None
        if (idx == -1):
            self.idx = dataset.newOccurrenceidx(self)
        else:
            self.idx = idx
        self.featureVector = np.zeros(self.dataset.args.VectorDim)
        self.label = self.idx
    
    def store(self):
        data = {}
        data["token"] = self.token
        data["clusteridx"] = self.clusteridx
        if (self.faArg == None):
            data["faArg"] = -1
        else:
            data["faArg"] = self.faArg.idx
        data["pos"] = self.pos
        data["idx"] = self.idx
        data["label"] = self.label
        data["featureVector"] = self.featureVector.tolist() #另外保存？
        data["sonArgType2Arg"] = {}
        for ArgType, Args in self.sonArgType2Arg.items():
            data["sonArgType2Arg"][ArgType] = [arg.idx for arg in Args]
        return data
    
    def load(self, data):
        self.clusteridx = int(data["clusteridx"])
        if (int(data["faArg"]) == -1):
            self.faArg = None
        else:
            self.faArg = self.dataset.idx2arg[int(data["faArg"])]
        self.token = data["token"]
        self.pos = data["pos"] #check
        self.label = int(data["label"])
        self.featureVector = np.array(data["featureVector"])
        for ArgType, arg_idxs in data["sonArgType2Arg"].items():
            self.sonArgType2Arg[ArgType] = [self.dataset.idx2arg[int(arg_idx)] for arg_idx in arg_idxs]

    def precomputeVector(self, n_argType, scale_factor):
        featureVector = np.zeros(2*n_argType+1)
        if self.faArg != None:
            featureVector[self.dataset.argtype2idx[self.faArg.argType]-1] += 1
        else:
            featureVector[2*n_argType] = 1
        for argType, arg in self.sonArgType2Arg.items():
            featureVector[self.dataset.argtype2idx[argType]+n_argType-1] += len(arg)
        featureVector /= np.linalg.norm(featureVector)
        self.featureVector = np.concatenate((scale_factor*self.featureVector, (1-scale_factor)*featureVector), axis=0)


    def addSonArg(self, arg):
        if arg.argType not in self.sonArgType2Arg:
            self.sonArgType2Arg[arg.argType] = []
        self.sonArgType2Arg[arg.argType].append(arg)

    def addFaArg(self, arg):
        if (self.faArg != None):
            print("Warning: faArg already exist!")
        self.faArg = arg

    def setlabel(self, newtop):
        ret = ""
        for argType, args in self.sonArgType2Arg.items():
            for arg in args:
                son = arg.son
                if son.label == self.label:
                    son.setlabel(newtop)
        self.label = newtop.idx
        self.clusteridx = newtop.clusteridx

    def resumelabel(self, oldlabel, newtop):
        ret = ""
        for argType, args in self.sonArgType2Arg.items():
            for arg in args:
                son = arg.son
                if son.label == oldlabel:
                    son.resumelabel(oldlabel, newtop)
        self.label = newtop.idx
        self.clusteridx = newtop.clusteridx

    def getNowSon(self): #返回现在的Compose后的sonArg的编号
        ret = []
        for args in self.sonArgType2Arg.values():
            for arg in args:
                son = arg.son
                if (son.label == self.label):
                    ret.extend(son.getNowSon())
                else:
                    ret.append(arg.idx)
        return ret
    
    def getNowSpan(self):
        ret = [self.token,[self.pos[1], self.pos[1]]]
        for args in self.sonArgType2Arg.values():
            for arg in args:
                son = arg.son
                if (son.label == self.label):
                    son_result = son.getNowSon()
                    print(son_result)
                    ret[0] += "->"+son_result[0]
                    ret[1][0] = min(ret[1][0], son_result[1][0])
                    ret[1][1] = max(ret[1][1], son_result[1][1])
        return ret

    def getTop(self):
        if self.label == self.idx:
            return self
        else:
            return self.dataset.idx2occ[self.label].getTop()

    def sumFeatureVector(self):
        ret = self.featureVector
        for args in self.sonArgType2Arg.values():
            for arg in args:
                son = arg.son
                if (son.label == self.label):
                    ret += son.sumFeatureVector()
        return ret

    def PhraseLprob(self):
        ret = 0.
        for args in self.sonArgType2Arg.values():
            for arg in args:
                son = arg.son
                if (son.label == self.label):
                    ret += math.log(len(self.dataset.TokPair2FaSon[self.token+","+son.token]) / self.dataset.pair_cnt)
                    ret -= math.log(len(self.dataset.tok2occ[son.token])*len(self.dataset.tok2occ[self.token]))
                    ret += son.PhraseLprob()

        return ret

    def getFeatureVector(self):
        if self.dataset.dynamic_features:
            ret = self.sumFeatureVector()
            if (np.linalg.norm(ret) == 0):
                print("here")
            ret = ret / np.linalg.norm(ret)
            return ret
        else:
            return self.featureVector
    
    def qry(self, father, argType):
        sonArgs = self.getNowSon()
        sonArgs = [self.dataset.idx2arg[idx] for idx in sonArgs]
        ret = []
        if (self.label == self.idx):
            if self.clusteridx == father.clusteridx:
                for arg in sonArgs:
                    if (arg.argType == argType):
                        ret.append(arg.son.getNowSpan())
        #print(sonArgs)
        for arg in sonArgs:
            ret.extend(arg.son.qry(father, argType))
        return ret
