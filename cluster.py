from occurrence import Occurrence
import numpy as np    

class Cluster():
    def __init__(self, dataset, idx = -1): #idx==-1

        self.dataset = dataset
        self.idx2Occ = {} #Occidx->Occ

        '''
        self.SonCluster2Num = {} # clusteridx->2num 
        self.FaCluster2Num = {}
        '''
        if (idx == -1):
            idx = dataset.newClusteridx(self)
        self.idx = idx
        return

    def store(self):
        ret = {'idx':self.idx, 'idx2Occ':{}}
        for key, val in self.idx2Occ.items():
            ret['idx2Occ'][key] = val.idx
        return ret

    def load(self, data):
        #todo
        self.idx = data['idx']
        self.idx2Occ = {}
        for key, val in data['idx2Occ'].items():
            self.idx2Occ[key] = self.dataset.idx2Occ[val]
        return

    def GetAllOcc(self):
        return list(self.idx2Occ.values())
    
    def GetRandomOcc(self):
        return self.idx2Occ[np.random.randint(0, len(self.idx2Occ))]
    
    def GetSonCluster(self): 
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

    def GetSonArgs(self): 
        SonArgs = []
        occs = self.GetAllOcc()
        for occ in occs:
            SonArgAfterCompose = occ.getTop().getNowSon()
            SonArgs.extend(SonArgAfterCompose)
        return SonArgs
 
    def ins(self, Occ:Occurrence): 
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