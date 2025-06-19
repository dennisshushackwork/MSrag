# Works

import faiss
import numpy as np
import time

# --- Parameters ---
d = 256  # Dimension of vectors
nb = 1_000_000  # Number of database vectors (10 million)
nq = nb  # Number of query vectors (using the same set for this example)
k = 10  # Number of nearest neighbors to find

# --- IVFPQ Parameters ---
nlist = 16384  # Number of IVF cells (Voronoi cells).
# Rule of thumb: 4*sqrt(nb) to 16*sqrt(nb). For 10M, sqrt(nb) ~ 3162.
# So 16384 is a reasonable choice.
m_pq = 32  # Number of subquantizers for PQ. d must be divisible by m_pq.
# 128 / 32 = 4 dimensions per subquantizer.
# This means codes will be 32 bytes per vector (if nbits_pq=8).
nbits_pq = 8  # Number of bits per subquantizer code. 2^8=256 centroids per sub-codebook.
# Often left at default 8. If specified in string: PQMxNBITS, e.g. PQ32x8

# For training, you can use all nb vectors or a subset.
# Using a subset can speed up training if nb is extremely large.
# For 10M, training on all is feasible but will take some minutes.
# Let's use all for this example.
nt = nb  # Number of vectors to use for training

# Search parameter
nprobe = 64  # Number of IVF cells to probe during search. Tune for speed/accuracy.

# Batch sizes for progress reporting
add_batch_size = 1_000_000
search_batch_size = 100_000

# --- Generate Sample Data ---
print(f"Generating {nb:,} random vectors of dimension {d}...")
try:
    xb = np.random.rand(nb, d).astype('float32')
    # For queries, we'll use the same data in this example
    xq = xb
    # For training data, we'll use xb (or a subset specified by nt)
    xt = xb[:nt]
except MemoryError:
    print("MemoryError: Not enough RAM to generate all data at once.")
    exit()
print(f"Generated database data shape: {xb.shape}, size: {xb.nbytes / (1024 ** 3):.2f} GB")
print(f"Using {xt.shape[0]:,} vectors for training.")

# --- Normalize Vectors for Cosine Similarity ---
print("Normalizing training, database, and query vectors (L2 normalization)...")
faiss.normalize_L2(xt)
if not np.isclose(xq, xb).all():  # Only normalize xq if it's not the same as xb
    faiss.normalize_L2(xq)
faiss.normalize_L2(xb)  # xb is normalized last if xq or xt are views of it
print("Normalization complete.")

# --- FAISS Index Setup ---
num_gpus = faiss.get_num_gpus()
print(f"Found {num_gpus} GPU(s).")

# IVF string: e.g., "IVF16384,PQ32"
# If nbits_pq is not 8, use "PQ{m_pq}x{nbits_pq}"
ivf_pq_string = f"IVF{nlist},PQ{m_pq}"
if nbits_pq != 8:
    ivf_pq_string = f"IVF{nlist},PQ{m_pq}x{nbits_pq}"

print(f"Creating CPU IndexIVFPQ with string: '{ivf_pq_string}' and METRIC_INNER_PRODUCT...")
# For cosine similarity on normalized vectors, use METRIC_INNER_PRODUCT
cpu_index = faiss.index_factory(d, ivf_pq_string, faiss.METRIC_INNER_PRODUCT)

# Handle GPU resources and index transfer
gpu_index = None
res = None  # For single GPU resources

if num_gpus > 0:
    print("Transferring index to GPU(s) for training and searching...")
    try:
        if num_gpus == 1:
            gpu_id = 0
            print(f"  Using single GPU ID: {gpu_id}")
            res = faiss.StandardGpuResources()
            # Convert the UNTRAINED CPU index to a GPU index
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        else:  # num_gpus > 1
            print(f"  Using {num_gpus} GPUs with data sharding.")
            options = faiss.GpuMultipleClonerOptions()
            options.shard = True  # Shard data after training
            # options.shard_type = 2 # SHARE_IVF_PQ for IVFPQ - balances memory
            # Training IVFPQ on multiple GPUs:
            # The GpuCloner will typically replicate the index for training or use a specific GPU.
            # The most straightforward way is to train, then distribute the trained index.
            # However, faiss.index_cpu_to_all_gpus can take an untrained index.
            # The training will then happen on the GpuIndex (often on GPU 0 or coordinated).
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=options)
        print("  Index structure successfully transferred/replicated to GPU(s).")
    except Exception as e:
        print(f"Error during GPU transfer: {e}. Will attempt to proceed on CPU.")
        gpu_index = None  # Ensure fallback to CPU
else:
    print("No GPUs found. Proceeding with CPU.")

# Select the index to use (GPU or CPU)
index_to_use = gpu_index if gpu_index else cpu_index
index_type = "GPU" if gpu_index else "CPU"

# --- Train the Index ---
if not index_to_use.is_trained:
    print(f"Training {index_type} IndexIVFPQ on {xt.shape[0]:,} vectors... This may take some minutes.")
    train_start_time = time.time()
    index_to_use.train(xt)
    train_end_time = time.time()
    print(f"Training complete. Time taken: {train_end_time - train_start_time:.2f} seconds.")
else:
    print("Index is already trained (e.g., loaded from disk or already processed).")

# --- Add Vectors to the Index (with batching for progress) ---
print(f"\nAdding {nb:,} vectors to the {index_type} index in batches of {add_batch_size:,}...")
add_overall_start_time = time.time()
num_add_batches = (nb + add_batch_size - 1) // add_batch_size
for i in range(num_add_batches):
    start_idx = i * add_batch_size
    end_idx = min((i + 1) * add_batch_size, nb)
    current_batch_data = xb[start_idx:end_idx]

    batch_add_start_time = time.time()
    index_to_use.add(current_batch_data)
    batch_add_end_time = time.time()

    print(f"  Added batch {i + 1}/{num_add_batches} ({end_idx:,}/{nb:,} vectors). "
          f"Batch took: {batch_add_end_time - batch_add_start_time:.2f}s.")
add_overall_end_time = time.time()
print(
    f"Finished adding all vectors. Total time for adding: {add_overall_end_time - add_overall_start_time:.2f} seconds.")
print(f"Index size on {index_type}: {index_to_use.ntotal:,}")

# --- Perform the k-NN Search (with batching for progress) ---
# Set nprobe for IVFPQ search
index_to_use.nprobe = nprobe
print(f"\nSet nprobe to: {index_to_use.nprobe}")
print(f"Performing k-NN search (k={k}) for {nq:,} queries on {index_type} "
      f"in batches of {search_batch_size:,}...")

num_search_batches = (nq + search_batch_size - 1) // search_batch_size
D_list = []
I_list = []
total_search_time_for_batches = 0.0
overall_search_start_time = time.time()

for i in range(num_search_batches):
    query_start_idx = i * search_batch_size
    query_end_idx = min((i + 1) * search_batch_size, nq)
    current_query_batch = xq[query_start_idx:query_end_idx]

    batch_search_start_time = time.time()
    Di, Ii = index_to_use.search(current_query_batch, k)
    batch_search_end_time = time.time()

    D_list.append(Di)
    I_list.append(Ii)

    current_batch_time = batch_search_end_time - batch_search_start_time
    total_search_time_for_batches += current_batch_time

    processed_queries = query_end_idx
    print(f"  Processed query batch {i + 1}/{num_search_batches} ({processed_queries:,}/{nq:,} queries). "
          f"Batch took: {current_batch_time:.2f}s.")

    if i + 1 < num_search_batches:
        avg_time_per_batch = total_search_time_for_batches / (i + 1)
        batches_remaining = num_search_batches - (i + 1)
        est_remaining_time = batches_remaining * avg_time_per_batch
        print(f"    Estimated time remaining: {est_remaining_time / 60:.2f} minutes.")

if D_list:
    D = np.concatenate(D_list, axis=0)
    I = np.concatenate(I_list, axis=0)
else:
    D = np.empty((0, k), dtype=np.float32)
    I = np.empty((0, k), dtype=np.int64)

overall_search_end_time = time.time()
actual_total_search_duration = overall_search_end_time - overall_search_start_time

print(f"\n--- Search Complete ({index_type}) ---")
print(f"Total time for k-NN search (including batching): {actual_total_search_duration:.2f} seconds "
      f"(approx {actual_total_search_duration / 60:.2f} minutes).")
print(f"Sum of individual batch search times: {total_search_time_for_batches:.2f} seconds.")

if actual_total_search_duration > 0:
    print(f"Approximate throughput: {nq / actual_total_search_duration:.2f} queries per second.")

# Since METRIC_INNER_PRODUCT returns dot products, and vectors are normalized, D contains cosine similarities.
# Higher values in D mean more similar.
# print("\nDistances (cosine similarities) matrix shape:", D.shape)
# print("Indices matrix shape:", I.shape)
# if nq > 0 and k > 0 and D.size > 0:
#     print("\nFor the first query vector, top k similarities:", D[0])
#     print("For the first query vector, top k indices:", I[0])

print("\nScript finished.")
