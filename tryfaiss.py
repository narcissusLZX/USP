import faiss
import numpy as np
import time

d = 768  # 维数
 
data = [[0]*d for _ in range(500000)]
for i in range(500):
    data[i][i] = 1
data = np.array(data).astype('float32')
#data = np.random(data.shape).astype('float32')
# centers = int(8 * math.sqrt(len(data)))
ids = np.arange(500000).astype('int64')
 
 

nlist = 100
m = 8

cfg = faiss.GpuIndexFlatConfig()
cfg.useFloat16=False
cfg.device=0
flat_config = [cfg]
resources = [faiss.StandardGpuResources()]
index = faiss.GpuIndexFlatIP(resources[0], d, flat_config[0])
#quantizer = faiss.IndexFlatIP(d) 
#index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
 
#quantizer = faiss.IndexFlatIP(d)  # 内部的索引方式依然不变
# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # PQ方式，每个向量都被编码为8个字节大小
#index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # 这个索引支持add_with_ids
index.train(data)
index.add_with_ids(data,ids)

print("start")
time1 = time.time()
query_vector = np.array([[0,0]+[3]+[0]*765]).astype('float32')
dis, ind = index.search(query_vector, 10)
print(ind, dis)
#print(data)

print(index.is_trained)
 
# index.nprobe = 10  # 选择n个维诺空间进行索引,

data[1] = data[0]

query_vector = np.array([[0]+[3]+[0]*766]).astype('float32')
dis, ind = index.search(query_vector, 1)
print(f'初始状态全1向量的最近的向量id为{ind}')
print(dis)
 
query_vector = np.array([[1000] * 768]).astype('float32')
dis, ind = index.search(query_vector, 1)
print(f'\n初始状态全1000向量的最近的向量id为{ind}')
print(dis)
 
 
query_vector = np.array([[0] * 768]).astype('float32')
dis, ind = index.search(query_vector, 1)
print(f'\n删除全0向量之前全0向量最近的index为{ind}')
print(dis)
 
index.remove_ids(np.array([2000]).astype('int64'))
print('\n注意删除的向量id为2000，将全0向量进行删除')
print(f'样本的总数为{index.ntotal}')
 
 
query_vector = np.array([[0] * 768]).astype('float32')
dis, ind = index.search(query_vector, 1)
print(f'\n删除全0向量之后全0向量最近的index为{ind}')
print(dis)
 
add_data = np.array([[1000] * 768]).astype('float32')
add_id = np.array([2000]).astype('int64')
index.add_with_ids(add_data, add_id)
print(f'\n注意，此时将index为2000的数据进行了更新，更新的数据为全1000，插入数据后的样本总数为{index.ntotal}')
 
 
query_vector = np.array([[1000] * 768]).astype('float32')
dis, ind = index.search(query_vector, 1)
print(f'\n更新后的全1000向量的最近的向量id为{ind}')
print(dis)
 
query_vector = np.array([[1] * 768]).astype('float32')
dis, ind = index.search(query_vector, 1)
print(f'\n全1向量的最近的向量id为{ind}')
print(dis)
time2 = time.time()

print(time2-time1)