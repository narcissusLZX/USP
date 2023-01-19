from occurrence import Occurrence
import numpy as np    

class Cluster():
    def __init__(self, dataset, idx = -1): #idx==-1:新建类

        self.dataset = dataset
        self.idx2Occ = {} #Occidx->Occ

        '''
        self.SonCluster2Num = {} # clusteridx->2num 每次采样后要重新更新？
        self.FaCluster2Num = {}
        '''
        if (idx == -1):
            idx = dataset.newClusteridx(self)
        self.idx = idx
        return

    def store(self):
        #todo
        return

    def load(self, data):
        #todo
        return

    def GetAllOcc(self): #返回Occurrence列表
        return list(self.idx2Occ.values())
    
    def GetRandomOcc(self):
        return self.idx2Occ[np.random.randint(0, len(self.idx2Occ))]
    
    def GetSonCluster(self): #返回clusterlist
        ret = []
        occs = self.GetAllOcc()
        for occ in occs:
            SonArgs = occ.getNowSon()
            for argidx in SonArgs:
                son = self.dataset.idx2arg[argidx].son
                ret.append(self.dataset.idx2cluster[son.clusteridx])
                '''
                if (son.clusteridx in ret):
                    ret[son.clusteridx] += 1
                else:
                    ret[son.clusteridx] = 0
                '''
        return ret

    def GetSonArgs(self): #返回子Arg list
        SonArgs = []
        occs = self.GetAllOcc()
        for occ in occs:
            SonArgAfterCompose = occ.getTop().getNowSon()
            SonArgs.extend(SonArgAfterCompose)
        return SonArgs
 
    def ins(self, Occ:Occurrence): #会在这里更改Occ的cluster
        self.idx2Occ[Occ.idx] = Occ
        '''
        for argType, args in Occ.sonArgType2Arg.items():
            for arg in args:
                son = arg.son.idx
                if son not in self.SonCluster2Num:
                    self.SonCluster2Num[son] = 0
                self.SonCluster2Num[son] += 1
        
        for argType, arg in Occ.faArgType2Arg.items():
            fa = arg[0].father.idx
            if fa not in self.FaCluster2Num:
                self.FaCluster2Num[fa] = 0
            self.FaCluster2Num[fa] += 1
        '''
        Occ.clusteridx = self.idx
    
    def remove(self, Occ:Occurrence):
        self.idx2Occ.pop(Occ.idx)

        '''
        for argType, arg in Occ.sonArgType2Arg.items():
            son = arg.son.idx
            self.SonCluster2Num[son] -= 1
        
        for argType, arg in Occ.faArgType2Arg.items():
            fa = arg.fa.idx
            self.FaCluster2Num[fa] -= 1
        '''