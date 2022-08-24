from asyncio.windows_events import NULL
from contextlib import nullcontext


class Tree():
    class Node():
        def __init__(self, word, wordidx):
            self.word = word
            self.wordidx = wordidx
            self.father = NULL
            

        def setfather(self, father):
            self.father = father

    
    def loaddata(self):
        #读入tok
        with open(path+".tok", "r", encoding='utf-8') as f:
            self.sentences.append([])
            for line in f.readlines():
                tok = line.strip()
                if not tok:
                    if (len(self.sentences[self.n_sentence]) > 0):
                        self.n_sentence += 1
                        self.sentences.append([])
                else:
                    self.sentenecs[self.n_sentence].append(tok)
                    if tok not in self.word2cluster:
                        self.word2cluster[tok] = Cluster(mydataset)
                    occ = Occurrence(tok, self, [self.n_sentence, len(self.sentences[self.n_sentence])], self.word2cluster[tok].idx)
                    self.word2cluster[tok].ins(occ)

            if (len(self.sentences[self.n_sentence])==0):
                self.n_sentence -= 1
                self.sentences.pop()

        #dep
        with open(path+".dep", "r", encoding='utf-8') as f:
            idx = 0
            for line in f.readlines():
                a = line.strip().split()
                if (len(a)==0):
                    idx += 1
                    continue
                dep, fa, so = a
                fa_pos = fa.split("-")[-1]
                son_pos = so.split("-")[-1]
                fa = self.pos2occ[[idx, fa_pos]]
                son = self.pos2occ[[idx, son_pos]]
                arg = Argument(self, fa, son, dep)
                fa.addSonArg(arg)
                son.addFaArg(arg)
                if dep not in self.argType2Arg:
                    self.argType2Arg[dep] = []
                self.argType2Arg[dep].append(arg)

                tokPair = [fa.token, son.token]
                if tokPair not in self.TokPair2FaSon:
                    self.TokPair2FaSon[tokPair] = []
                self.TokPair2FaSon[tokPair].append([fa, son])


        return

    def newOccurenceidx(self, occ:Occurrence):
        self.occIdx += 1
        self.idx2occ[self.occIdx] = occ
        self.pos2occ[occ.pos] = occ
        return self.occIdx

    def newClusteridx(self, cluster:Cluster):
        self.clusterIdx += 1
        self.idx2cluster[self.clusterIdx] = cluster
        return self.clusterIdx

    def removeCluster(self, cluster:Cluster):
        idx = cluster.idx
        self.idx2cluster.pop(idx)
        
    def getRandomCluster(self):
        clusters = self.word2cluster.values()
        return clusters[random.choice(len(clusters))]

    def CalcSimilarity(self, cluster1:Cluster, cluster2:Cluster):
        if cluster1.idx == cluster2.idx:
            return 0.0
        else:
            #todo
            return 1


    def getClusterbyClusterSimilarity(self, Cluster1:Cluster):
        clusters = list(set(self.word2cluster.values()))
        prob = [self.CalcSimilarity(clusters, Cluster1) for cluster in clusters]

        scores = np.array(prob)
        scores = np.exp(scores) / np.sum(np.exp(scores))

        selectProb = random.random()
        sum = 0.0
        for i in range(len(scores)):
            sum += scores[i]
            if (sum > selectProb):
                return i
    