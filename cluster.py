from occurrence import Occurrence
    

class Cluster():
    def __init__(self, dataset, idx = -1): #idx==-1:新建类

        self.dataset = dataset
        self.Span2Occurr = {} #span(string)->occurrenceList

        '''
        self.SonCluster2Num = {} # clusteridx->2num 每次采样后要重新更新？
        self.FaCluster2Num = {}
        '''
        if (idx == -1):
            idx = dataset.newClusteridx(self)
        self.idx = idx
        return

    def store(self):
        data = {"idx":self.idx, "Span2Occurr":{}}
        for span, occs in self.Span2Occurr.items():
            occs_idx = [occ.idx for occ in occs]
            data["Span2Occurr"][span] = occs_idx
        return data

    def load(self, data):
        for span, occs_idx in data["Span2Occurr"].items():
            self.Span2Occurr[span] = [self.dataset.idx2occ[int(occ_idx)] for occ_idx in occs_idx]

    def GetAllOcc(self): #返回Occurrence列表
        ret = []
        for Span in self.Span2Occurr.keys():
            ret.extend(self.Span2Occurr[Span])
        return ret
    
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
        span = Occ.token
        if (span not in self.Span2Occurr):
            self.Span2Occurr[span] = []
        self.Span2Occurr[span].append(Occ)

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
        span = Occ.token
        if (span not in self.Span2Occurr ) or (Occ not in self.Span2Occurr[span]):
            print("Warning: Remove Occurrence "+Occ.token+" in Cluster "+str(self.idx))
            return
        self.Span2Occurr[span].remove(Occ)
        if (len(self.Span2Occurr[span])==0):
            self.Span2Occurr.pop(span)

        '''
        for argType, arg in Occ.sonArgType2Arg.items():
            son = arg.son.idx
            self.SonCluster2Num[son] -= 1
        
        for argType, arg in Occ.faArgType2Arg.items():
            fa = arg.fa.idx
            self.FaCluster2Num[fa] -= 1
        '''