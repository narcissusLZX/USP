import random
import math
from turtle import title
from tqdm import tqdm
import argparse
import numpy as np

from mydataset import Mydataset
from cluster import Cluster
from occurrence import Occurrence
from argument import Argument
from evaluate import Evaluation


def LogCalcCompoundBinomial(success_num, fail_num, parameters):

    p1, p2 = parameters
    return math.lgamma(p1+p2)-math.lgamma(p1)-math.lgamma(p2)+ \
        math.lgamma(success_num+p1)+math.lgamma(fail_num+p2)-math.lgamma(success_num+fail_num-p1-p2)
    

def List2Countdict(lists):
    count_dict = {}
    for item in lists:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1    
    return count_dict

def extract(listinlist):
    ret = []
    for x in listinlist:
        if isinstance(x, list):
            ret.extend(x)
        else:
            ret.append(x)
    return list(set(ret))

def CalcProb(dataset:Mydataset, newClusters):
    newClusters = extract(newClusters)
    Lprob = 0.0
    LprobNum = 0.0
    for cluster in newClusters:
        occs = cluster.GetAllOcc()
        t_occ = len(occs)
        SonArgs = cluster.GetSonArgs()
        SonArgs = [dataset.idx2arg[idx].argType for idx in SonArgs]
        SonArgCount = List2Countdict(SonArgs)

        for argType, t_arg in SonArgCount.items():
            t_diff = 0
            for occ in occs:
                SonArgAfterCompose = occ.getTop().getNowSon()
                SonArgtype = [dataset.idx2arg[x].argType for x in SonArgAfterCompose]
                if argType in SonArgtype:
                    t_diff += 1
            LprobNum += LogCalcCompoundBinomial(t_diff, t_occ-t_diff, dataset.args.gen_first_eta)
            LprobNum += LogCalcCompoundBinomial(t_arg-t_diff, t_diff, dataset.args.gen_more_eta)
    Lprob += LprobNum

    LprobOut1 = 0.
    for cluster in newClusters:
        SonClusters = cluster.GetSonCluster()
        for sonCluster in SonClusters:
            if sonCluster in newClusters:
                continue
            occs = sonCluster.GetAllOcc()
            faArgType = set([occ.faArg.argType for occ in occs if occ.faArg != None])
            LprobOut1 += math.lgamma(len(faArgType)-dataset.args.arg_alpha)-math.lgamma(1-dataset.args.arg_alpha)

    LprobInto1 = 0.
    
    intoChildrenBefore = len(dataset.idx2cluster) - len(newClusters)
    for cluster in newClusters:
        occs = cluster.GetAllOcc()
        faArgType = set([occ.faArg.argType for occ in occs if occ.faArg != None])
        LprobInto1 += math.log(dataset.args.superConc + intoChildrenBefore * dataset.args.arg_alpha)
        LprobInto1 += math.lgamma(len(faArgType) - dataset.args.arg_alpha) - math.lgamma(1-dataset.args.arg_alpha)
        intoChildrenBefore += 1
    	
    Lprob += LprobInto1 - math.lgamma(dataset.n_argType + dataset.args.superConc)

    LprobOut2 = 0.
    for cluster in newClusters:
        SonArgs = cluster.GetSonArgs()
        SonArgs = [dataset.idx2arg[idx] for idx in SonArgs]
        SonArgType2Cluster = {}
        for sonarg in SonArgs:
            if (sonarg.argType not in SonArgType2Cluster):
                SonArgType2Cluster[sonarg.argType] = []
            SonArgType2Cluster[sonarg.argType].append(sonarg.son.clusteridx)
        SonArgTypes = [SonArg.argType for SonArg in SonArgs]
        SonArgCount = List2Countdict(SonArgTypes)
        for argType, num in SonArgCount.items():
            LprobOut2 += math.lgamma(dataset.args.cluster_Conc)
            SonClusters = List2Countdict(SonArgType2Cluster[argType])
            ChildrenBefore = len(SonClusters)
		    #isConjArg = _isConjArg(at, hyp)
            for soncluster, clusternum in SonClusters.items():
                if soncluster in newClusters:
                    continue
                LprobOut2 += math.log(dataset.args.cluster_Conc+ChildrenBefore*dataset.args.cluster_alpha)
                LprobOut2 += math.lgamma(clusternum-dataset.args.cluster_alpha)-math.lgamma(1-dataset.args.cluster_alpha)
                ChildrenBefore += 1
                '''
                if (isConjArg and soncluster.)
                '''
            LprobOut2 -= math.lgamma(dataset.args.arg_Conc + num)

    LprobInto2 = 0.
    for cluster in newClusters:
        occs = cluster.GetAllOcc()
        faArgType = [occ.faArg.argType for occ in occs if occ.faArg != None]
        faArgCount = List2Countdict(faArgType)
        for argType, argNum in faArgCount.items():
            #ChildrenBefore = dataset.argType2cnt[argType] - argNum
            #isConj = 
            #LprobInto2 += math.log(dataset.args.cluster_Conc+ChildrenBefore*dataset.args.cluster_alpha)
            LprobInto2 += math.lgamma(argNum-dataset.args.cluster_alpha)-math.lgamma(1-dataset.args.cluster_alpha)
            LprobInto2 += math.lgamma(dataset.args.cluster_Conc+dataset.argType2cnt[argType]-argNum) - math.lgamma(dataset.args.cluster_Conc+dataset.argType2cnt[argType])

    '''
    for cluster in newClusters:
        SonClusters = cluster.GetSonCluster()
        for sonCluster, num in SonClusters.items():
            LprobOut1 += math.lgamma(num-dataset.args.cluster_alpha)-math.lgamma(1-dataset.args.cluster_alpha)

    Lprob += LprobOut1

    Lprob2 = 0.
    for cluster in newClusters:
        SonArgs = []
        occs = cluster.GetAllOcc()
        for occ in occs:
            SonArgAfterCompose = occ.getTop().getNowSon()
            SonArgs.extend(SonArgAfterCompose)
        SonArgCount = List2Countdict(SonArgs)
        for argType in SonArgCount.keys():
            Lprob2 += math.lgamma(dataset.args.ClusterDistrConc)
            SonCluster = []
            for occ in occs:
                if argType not in occ.sonArgType2Arg:
                    continue
                for arg in occ.sonArgType2Arg[argType]:
                    son = arg.son
                    cluster_idx = dataset.idx2occ[son.label].clusteridx
                    SonCluster.append(cluster_idx)
            SonClusterCount = List2Countdict(SonCluster)
            for clusteridx, num in SonClusterCount.items():
                Lprob2 += math.lgamma(num-dataset.args.arg_alpha)-math.lgamma(1-dataset.args.arg_alpha)

    Lprob += Lprob2

    '''
    Lprob_phrase = 0.
    if (dataset.proposal == "compose") or (dataset.proposal == "decompose"):
        for cluster in newClusters:
            occs = cluster.GetAllOcc()
            for occ in occs:
                if (dataset.args.Distributed):
                    vector = occ.getFeatureVector()
                    norm1 = np.linalg.norm(vector)
                    if (dataset.args.Faiss):
                        #todo
                        Lprob_phrase += dataset.faiss_calcLprob(vector)
                    else:
                        cnt = 0
                        for other_occ in dataset.idx2occ.values():
                            other_vector = other_occ.getFeatureVector()
                            cos_sim = vector.dot(other_vector) / norm1 / np.linalg.norm(other_vector)
                            if (cos_sim >= dataset.args.sim_threshold):
                                cnt += 1
                        Lprob_phrase += math.log(cnt) - math.log(len(dataset.idx2occ))
                else:
                    Lprob_phrase += occ.PhraseLprob()

    Lprob += Lprob_phrase

    return Lprob



#不会多状态并存，在resume时会还原，所以accept要更新

#compose后应该是新的类还是同父结点的类？ 目前是新的类

def Merge(cluster1:Cluster, cluster2:Cluster, mergeCluster:Cluster): #merge (cluster1 & cluster2) into mergeCluster
    spans = list(cluster1.Span2Occurr.keys())
    for span in spans:
        occs = list(cluster1.Span2Occurr[span])
        for occ in occs:
            cluster1.remove(occ)
            mergeCluster.ins(occ)
    spans = list(cluster2.Span2Occurr.keys())
    for span in spans:
        occs = list(cluster2.Span2Occurr[span])
        for occ in occs:
            cluster2.remove(occ)
            mergeCluster.ins(occ)

def generate_merge(Cluster1:Cluster, dataset:Mydataset):

    #print("Merge")
    dataset.proposal = "merge"
   
    Cluster2 = dataset.getClusterbyClusterSimilarity(Cluster1)
    if (type(Cluster1)!=type(Cluster2)):
        print("warning!") 

    occ_1 = Cluster1.GetAllOcc()
    occ_2 = Cluster2.GetAllOcc()

    newCluster = Cluster(dataset, idx=-1)
    Merge(Cluster1, Cluster2, newCluster)

    
    return {"new":[newCluster], "old":[Cluster1, Cluster2], "occ":[occ_1, occ_2]}

def generate_split(oldcluster:Cluster, dataset:Mydataset): #需提前判定oldcluster内的数量>1

    dataset.proposal = "split"
    
    #print("Split")
    def createSplit(cluster:Cluster):
        Spans = list(cluster.Span2Occurr.keys())
        span1, span2 = random.sample(Spans, 2)
        cluster1 = Cluster(dataset, idx=-1)
        cluster2 = Cluster(dataset, idx=-1)
        newClusterMap = {}
        
        occs = list(cluster.Span2Occurr[span1])
        for occ in occs:
            cluster.remove(occ)
            cluster1.ins(occ)

        occs = list(cluster.Span2Occurr[span2])
        for occ in occs:
            cluster.remove(occ)
            cluster2.ins(occ)

        for span in Spans: #将每个span划分到两个新类中的一个
            if (span == span1) or (span == span2):
                continue

            occs = list(cluster.Span2Occurr[span])
            
            for occ in occs: 
                cluster.remove(occ)
                cluster1.ins(occ)
            Lprob1 = CalcProb(dataset, [cluster1, cluster2])

            for occ in occs:
                cluster1.remove(occ)
                cluster2.ins(occ)
            Lprob2 = CalcProb(dataset, [cluster1, cluster2])

            if (Lprob1 > Lprob2):
                for occ in occs:
                    cluster2.remove(occ)
                    cluster1.ins(occ)
                newClusterMap[span] = 0
            else:
                newClusterMap[span] = 1
        return cluster1, cluster2


    cluster1, cluster2 = createSplit(oldcluster)
        
    return {"new":[cluster1, cluster2], "old":[oldcluster], "occ":[cluster1.GetAllOcc(), cluster2.GetAllOcc()]}

def generate_mergesplit(dataset:Mydataset):
    Cluster1 = dataset.getRandomCluster()
    rd = random.random()
    if ((rd < 0.5) or (len(Cluster1.Span2Occurr) == 1)) and (not len(dataset.idx2cluster) == 1):
        return generate_merge(Cluster1, dataset)
    else:
        return generate_split(Cluster1, dataset)


def Decompose(occ1:Occurrence, occ2:Occurrence, cluster1:Cluster, cluster2:Cluster, dataset:Mydataset):
    occ1 = occ1.getTop()
    old_cluster_idx = occ1.clusteridx
    oldCluster = dataset.idx2cluster[old_cluster_idx]
    oldCluster.remove(occ1)
    cluster1.ins(occ1) #父节点还是原来的cluster
    cluster2.ins(occ2) #子节点加入新的cluster
    occ2.resumelabel(occ1.label, occ2)
    return oldCluster

# compose后只以根结点的属性为key出现在cluster中，自身不会出现在cluster中

def createDecompose(occ1:Occurrence, occ2:Occurrence, dataset:Mydataset): #occ1:fater, occ2:son
    #label = occ1.label
    tokPair = occ1.token+","+occ2.token
    cluster1 = Cluster(dataset, -1)
    cluster2 = Cluster(dataset, -1)
    oldClusters = []
    for pair in dataset.TokPair2FaSon[tokPair]: #对所有同样的父子节点对都要进行操作
        oldClusters.append(Decompose(pair[0], pair[1], cluster1, cluster2, dataset))
    return cluster1, cluster2, oldClusters

def Compose(occ1:Occurrence, occ2:Occurrence, newCluster:Cluster, dataset:Mydataset):
    occ1 = occ1.getTop()
    cluster1_idx = occ1.clusteridx
    cluster2_idx = occ2.clusteridx
    cluster1 = dataset.idx2cluster[cluster1_idx]
    cluster2 = dataset.idx2cluster[cluster2_idx]

    cluster1.remove(occ1)
    cluster2.remove(occ2)
    newCluster.ins(occ1) #加入新的cluster
    
    occ2.setlabel(occ1)

    return [cluster1, cluster2]

def createCompose(occ1:Occurrence, occ2:Occurrence, dataset:Mydataset):
    tokPair = occ1.token+","+occ2.token

    newCluster = Cluster(dataset, -1)
    oldClusters = []
  
    for pair in dataset.TokPair2FaSon[tokPair]:
        oldClusters.append(Compose(pair[0], pair[1], newCluster, dataset))
        
    return oldClusters, newCluster


def generate_composedecompose(dataset:Mydataset):
    occ1, occ2 = dataset.getRandomPair()

    if (occ1.label == occ2.label):
        
        #print("Decompose")
        cluster1, cluster2, oldClusters = createDecompose(occ1, occ2, dataset)
        dataset.proposal = "decompose"
        return {"new":[cluster1, cluster2], "old":oldClusters, "occ":[occ1, occ2]}
    else:
        #print("Compose")
        dataset.proposal = "compose"
        oldClusters, newCluster = createCompose(occ1, occ2, dataset)
        return {"new":[newCluster], "old":oldClusters, "occ":[occ1, occ2]}

def accept(hyp, dataset:Mydataset):
    #print("Accepet "+dataset.proposal)
    #dataset.faiss_cluster_remove(extract(hyp["new"]+hyp["old"]))
    if dataset.proposal == "merge":
        Merge(hyp["old"][0], hyp["old"][1], hyp["new"][0])
        dataset.removeCluster(hyp["old"][0])
        dataset.removeCluster(hyp["old"][1])
    elif dataset.proposal == "split":
        cluster1, cluster2 = hyp["new"]
        occs1, occs2 = hyp["occ"]
        cluster = hyp["old"][0]
        for occ in occs1:
            cluster.remove(occ)
            cluster1.ins(occ)
        for occ in occs2:
            cluster.remove(occ)
            cluster2.ins(occ)
        dataset.removeCluster(hyp["old"][0])
    elif dataset.proposal == "compose":
        occ1, occ2 = hyp["occ"]
        tokPair = occ1.token+","+occ2.token
        idx = 0
        for pair in dataset.TokPair2FaSon[tokPair]:
            if (dataset.args.Faiss):
                dataset.faiss_remove(pair[0])
                dataset.faiss_remove(pair[1])
            Compose(pair[0], pair[1], hyp["new"][0], dataset)
            idx += 1
            if (dataset.args.Faiss):
                dataset.faiss_insert(pair[0])
    elif dataset.proposal == "decompose":
        occ1, occ2 = hyp["occ"]
        tokPair = occ1.token+","+occ2.token
        idx = 0
        for pair in dataset.TokPair2FaSon[tokPair]: #对所有同样的父子节点对都要进行操作
            if (dataset.args.Faiss):
                dataset.faiss_remove(pair[0])
            Decompose(pair[0], pair[1], hyp["new"][0], hyp["new"][1], dataset)
            idx += 1
            if (dataset.args.Faiss):
                dataset.faiss_insert(pair[0])
                dataset.faiss_insert(pair[1])
    dataset.update(extract(hyp["old"] +hyp["new"]))
    return

def reject(hyp, dataset:Mydataset):
    dataset.update(extract(hyp["new"]+ hyp["old"]))

    return 

def resume(hyp, dataset:Mydataset):
    if dataset.proposal == "merge":
        cluster1, cluster2 = hyp["old"]
        occs1, occs2 = hyp["occ"]
        cluster = hyp["new"][0]
        for occ in occs1:
            cluster.remove(occ)
            cluster1.ins(occ)
        for occ in occs2:
            cluster.remove(occ)
            cluster2.ins(occ)
    elif dataset.proposal == "split": 
        Merge(hyp["new"][0], hyp["new"][1], hyp["old"][0])
    elif dataset.proposal == "compose":
        occ1, occ2 = hyp["occ"]
        tokPair = occ1.token+","+occ2.token
        idx = 0
        for pair in dataset.TokPair2FaSon[tokPair]: #对所有同样的父子节点对都要进行操作
            Decompose(pair[0], pair[1], hyp["old"][idx][0], hyp["old"][idx][1], dataset)
            idx += 1
    else:
        occ1, occ2 = hyp["occ"]
        tokPair = occ1.token+","+occ2.token
        idx = 0
        for pair in dataset.TokPair2FaSon[tokPair]:
            Compose(pair[0], pair[1], hyp["old"][idx], dataset)
            idx += 1



def CalcAcceptRatio(hyp, dataset):
    NewLprob = CalcProb(dataset, hyp["new"])
    resume(hyp, dataset)
    OldLprob = CalcProb(dataset, hyp["old"])
    if (NewLprob > OldLprob):
        return 1
    if (NewLprob - OldLprob < -20):
        return 20
    return math.exp(NewLprob-OldLprob)

def onestep(idx, dataset):
    rd = random.random()
    if (rd < 0.1):
        hyp = generate_mergesplit(dataset)
    else:
        hyp = generate_composedecompose(dataset)
    acceptratio = CalcAcceptRatio(hyp, dataset)
    rd = random.random()
    if rd < acceptratio:
        accept(hyp, dataset)
        #dataset.check()
        return True
    else:
        reject(hyp, dataset)
        #dataset.check()
        return False

def main(args):
    Max_epoch, success_cnt = args.n_epoch, 0

    random.seed(args.seed)
    dataset = Mydataset(args)
    for i in tqdm(range(args.start_epoch, Max_epoch)):
        if (onestep(i, dataset)):
            success_cnt += 1
        #if (i % 1000 == 0):
            #print("Step {} done".format(i))
        if (success_cnt * 10 < (i-args.start_epoch)):
            print("Converge after {} steps.".format(i))
            break
        if ((i+1) % args.save_per_epoch == 0):
            dataset.store(args.model_path+str(i)+".json")
        #dataset.check()
    dataset.store(args.model_path+"final.json")
    if dataset.args.eval:
        eval = Evaluation(dataset.args, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./dataset/geniaquarter", type=str)
    parser.add_argument("--output_path", default="./dataset", type=str)
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed.")
    parser.add_argument("--Df", action="store_true", help="Dynamically adjust feature vectors of occurrences?")
    parser.add_argument("--ExtVector", action="store_true", help="Extern precomputed feature vectors?")

    parser.add_argument("--n_epoch", default=1500000, type=int)
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
    parser.add_argument("--load_model_path", type=str)
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
    main(args)