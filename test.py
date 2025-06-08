import faiss

print(faiss.get_num_gpus())
index2 = faiss.index_factory(128, "PCA64,IVF16384_HNSW32,Flat")

# get some training data
xt = faiss.rand((50000000, 256))

import time
start = time.time()
# baseline training without GPU
index_ivf = faiss.extract_index_ivf(index2)
clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
index_ivf.clustering_index = clustering_index

end = time.time()
print(end - start)