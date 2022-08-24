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
        ret = self.token
        for args in self.sonArgType2Arg.values():
            for arg in args:
                son = arg.son
                if (son.label == self.label):
                    ret += "->"+son.getNowSon()
        return ret

    def getTop(self):
        if self.label == self.idx:
            return self
        else:
            return self.dataset.idx2occ[self.label].getTop()
