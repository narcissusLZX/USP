from sys import getwindowsversion
import numpy as np

class Occurrence():
    '''
        token: 
        clusteridx:类别序号
        pos: 出现位置[x,y]第x个样本第y个位置
    '''
    def __init__(self, token, dataset, pos=[-1,-1], clusteridx = 0):
        self.dataset = dataset
        self.token = token
        self.clusteridx = clusteridx
        self.pos = pos
        self.sonArgType2Arg = {}  #ArgType->Arg List
        self.faArg = None
        self.idx = dataset.newOccurenceidx(self)
        self.featureVector = [0]
        self.label = self.idx

    def precomputeVector(self, n_argType):
        self.featureVector = np.zeros(2*n_argType)
        if self.faArg != None:
            self.featureVector[self.dataset.argtype2idx[self.faArg.argType]-1] += 1
        for argType, arg in self.sonArgType2Arg.items():
            self.featureVector[self.dataset.argtype2idx[argType]+n_argType-1] += len(arg)

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
        ret = (self.token,[self.pos[1], self.pos[1]])
        for args in self.sonArgType2Arg.values():
            for arg in args:
                son = arg.son
                if (son.label == self.label):
                    son_result = son.getNowSon()
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

    def getFeatureVector(self):
        if self.dataset.dynamic_features:
            ret = self.sumFeatureVector()
            ret = ret / ret.sum()
            return ret
        else:
            return self.featureVector
    
    def qry(self, token, argType):
        sonArgs = self.getNowSon()
        ret = []
        if (self.token == token):
            for arg in sonArgs:
                if (arg.argType == argType):
                    ret.append(arg.son.getNowSpan())
        for arg in sonArgs:
            ret.extend(arg.son.qry(token, argType))
