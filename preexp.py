import numpy as np
from tqdm import tqdm

path = "./dataset/genia"
word2idx = {}
sentences = [[]]
words = []
with open(path+".tok", "r", encoding='utf-8') as f:
    for line in f.readlines():
        tok = line.strip()
        if not tok:
            sentences.append([])
            continue
        else:
            sentences[-1].append(tok)
            if tok not in word2idx:
                word2idx[tok] = len(word2idx)
                words.append(tok)
edge = []
st = [-1]*len(word2idx)

def link(u, v):
    edge.append((v, st[u]))
    st[u] = len(edge)-1



tmp_read = np.load(path+".npz", allow_pickle = True)
Vecs = np.concatenate(tmp_read['arr_0'], axis=0) 
mu = Vecs.mean(axis=0)
cov = np.cov(Vecs.T)
u, s, vh = np.linalg.svd(cov)
W = np.dot(u, np.diag(1 / np.sqrt(s)))
Vecs = (Vecs - mu).dot(W[:, :64])
V_idx = 0
word2vec = np.zeros((len(word2idx), 64))
for i in tqdm(range(len(sentences))):
    sentence = sentences[i]
    for tok in sentence:
        idx = word2idx[tok]
        word2vec[idx] += Vecs[V_idx]
        V_idx += 1
        #print(tok)

for i in range(len(word2idx)):
    word2vec[i] /= np.linalg.norm(word2vec[i])



cos_sim = np.dot(word2vec, word2vec.T)
threshold = 0.8
merge = cos_sim >= threshold
x,y = np.nonzero(merge)

f = [i for i in range(len(word2idx))]
#print(f)
def getfa(x):
    #print(x)
    if f[x] != x:
        f[x] = int(getfa(f[x]))
    return int(f[x])
for i in range(len(x)):
    if (x[i] > y[i]):
        print(words[x[i]], words[y[i]])
        f[getfa(x[i])] = getfa(y[i])
    

#print((count-len(word2idx))/2, len(word2idx))
#print(count, len(sentences)**2, cos_sim[30, 30], cos_sim[20, 182], cos_sim[182,20])
'''
sim = []
for i in tqdm(range(len(word2idx))):
    for j in tqdm(range(i+1, len(word2idx))):
        sim.append(cos_sim[i][j])
sim = sorted(sim)

print(sim[100], sim[-100])

'''

#dep
flag = False
idx = 0
with open(path+".dep", "r", encoding='utf-8') as fop:
    for line in fop.readlines():
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
            continue
        fa = sentences[idx][int(fa_pos)-1]
        son = sentences[idx][int(son_pos)-1]
        #print(fa, son, word2idx[fa], word2idx[son])
        #print(type(f))
        link(int(getfa(word2idx[fa])), int(getfa(word2idx[son])))
        link(int(getfa(word2idx[son])), int(getfa(word2idx[fa])))

visit = [False for _ in range(len(word2idx))]
cnt = 0

def bfs(u):
    que = [u]
    visit[u] = True
    ret = 0
    while (len(que) > 0):
        u = que.pop(0)
        edge_idx = st[u]
        ret += 1
        while (edge_idx != -1):
            e = edge[edge_idx]
            v, edge_idx = e
            if (not visit[v]):
                visit[v] = True
                que.append(v)
    return ret

for i in range(len(word2idx)):
    if (getfa(i) == i and not visit[i]):
        cnt += 1
        #print(words[i])
        print(bfs(i))

print(cnt)