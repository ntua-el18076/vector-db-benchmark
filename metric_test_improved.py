import time
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, SearchRequest
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# --- Configuration ---
NUM_VECTORS = 100_000   
DIMENSIONS = 128        
BATCH_SIZE = 1000       
SEARCH_BATCH_SIZE = 100 
QDRANT_HOST = "localhost"
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334
MILVUS_HOST = "localhost"
OUTPUT_FILE = "benchmark_metrics_results.txt"

# --- Logging Helper ---
# Clears the file at the start, then appends output to both screen and file
def initialize_log():
    with open(OUTPUT_FILE, "w") as f:
        f.write("=== VECTOR DB METRIC BENCHMARK RESULTS ===\n")
        f.write(f"Vectors: {NUM_VECTORS}, Dimensions: {DIMENSIONS}\n")
        f.write("==========================================\n\n")

def log(message):
    print(message)  # Print to console
    with open(OUTPUT_FILE, "a") as f:
        f.write(message + "\n")  # Write to file

# --- 1. Generate Synthetic Data ---
log(f"Generating {NUM_VECTORS} random vectors of {DIMENSIONS} dimensions...")
vectors = np.random.rand(NUM_VECTORS, DIMENSIONS).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]

categories = ["news", "sports", "finance", "tech"]
payloads = [{"id": i, "category": categories[i % 4]} for i in range(NUM_VECTORS)]

# Milvus Data Prep
milvus_ids = list(range(NUM_VECTORS))
milvus_vectors = vectors.tolist()
milvus_categories = [categories[i % 4] for i in range(NUM_VECTORS)]


# --- QDRANT BENCHMARK ---
def benchmark_qdrant():
    log("\n" + "="*40)
    log("      TESTING QDRANT")
    log("="*40)
    
    try:
        # Hybrid Client Approach: gRPC for Setup/Sequential, HTTP for Batching (Stability)
        client_grpc = QdrantClient(
            host=QDRANT_HOST, 
            port=QDRANT_HTTP_PORT, 
            grpc_port=QDRANT_GRPC_PORT, 
            prefer_grpc=True,
            check_compatibility=False
        )
        client_http = QdrantClient(
            host=QDRANT_HOST, 
            port=QDRANT_HTTP_PORT, 
            prefer_grpc=False 
        )
    except Exception as e:
        log(f"Failed to connect to Qdrant: {e}")
        return

    metrics = {
        "Cosine": Distance.COSINE,
        "Euclidean": Distance.EUCLID
    }

    query_set_np = vectors[:1000]
    query_set_list = query_set_np.tolist()

    for metric_name, metric_val in metrics.items():
        collection_name = f"qdrant_{metric_name.lower()}"
        log(f"\n--- [Qdrant] Testing Metric: {metric_name} ---")
        
        # Recreate Collection
        if client_grpc.collection_exists(collection_name):
            client_grpc.delete_collection(collection_name)
            
        client_grpc.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=DIMENSIONS, distance=metric_val),
        )

        # Upload Data
        log("  Uploading data...")
        start_upload = time.time()
        client_grpc.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payloads,
            ids=list(range(NUM_VECTORS)),
            batch_size=BATCH_SIZE,
            wait=True 
        )
        log(f"  Upload finished in {time.time() - start_upload:.2f}s")

        # --- TEST 1: Sequential Search (gRPC) ---
        log("  Running Sequential Search (100 queries) [via gRPC]...")
        start_seq = time.time()
        for q in query_set_list[:100]:
            client_grpc.search(collection_name=collection_name, query_vector=q, limit=5)
        rps_seq = 100 / (time.time() - start_seq)
        log(f"  -> Sequential RPS: {rps_seq:.2f} (Network Bound)")

        # --- TEST 2: Batch Search (HTTP) ---
        log("  Running BATCH Search (1,000 queries) [via HTTP]...")
        start_batch = time.time()
        for i in range(0, len(query_set_list), SEARCH_BATCH_SIZE):
            batch = query_set_list[i : i + SEARCH_BATCH_SIZE]
            
            # Using HTTP client + SearchRequest objects avoids version conflicts
            requests_list = [
                SearchRequest(vector=v, limit=5) for v in batch
            ]
            
            client_http.search_batch(
                collection_name=collection_name,
                requests=requests_list
            )
        rps_batch = 1000 / (time.time() - start_batch)
        log(f"  -> BATCH RPS: {rps_batch:.2f} (Engine Throughput)")

        # --- TEST 3: Filtered Search (HTTP) ---
        if metric_name == "Cosine":
            log("  Running Filtered Search (category='tech') [via HTTP]...")
            tech_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value="tech"))])
            
            start_filter = time.time()
            for i in range(0, len(query_set_list), SEARCH_BATCH_SIZE):
                batch = query_set_list[i : i + SEARCH_BATCH_SIZE]
                
                requests_list = [
                    SearchRequest(vector=v, filter=tech_filter, limit=5) for v in batch
                ]
                
                client_http.search_batch(
                    collection_name=collection_name,
                    requests=requests_list
                )
            rps_filter = 1000 / (time.time() - start_filter)
            log(f"  -> Filtered RPS: {rps_filter:.2f}")


# --- MILVUS BENCHMARK ---
def benchmark_milvus():
    log("\n" + "="*40)
    log("      TESTING MILVUS")
    log("="*40)
    
    try:
        connections.connect("default", host=MILVUS_HOST, port="19530")
    except Exception as e:
        log(f"Skipping Milvus (Not running or connection failed): {e}")
        return

    metrics = {
        "Cosine": "COSINE",
        "Euclidean": "L2"
    }
    
    query_set_list = vectors[:1000].tolist()

    for metric_name, metric_val in metrics.items():
        collection_name = f"milvus_{metric_name.lower()}"
        log(f"\n--- [Milvus] Testing Metric: {metric_name} ---")
        
        # 1. Define Schema
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIMENSIONS),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50) 
        ]
        schema = CollectionSchema(fields, "Testing metrics")
        collection = Collection(collection_name, schema)

        # 2. Upload Data 
        log("  Uploading data...")
        collection.insert([milvus_ids, milvus_vectors, milvus_categories])
        collection.flush()
        
        # 3. Build Index
        log("  Building Index...")
        index_params = {
            "metric_type": metric_val,
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 128}
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        collection.load()

        # 4. Search
        log("  Running BATCH Search (1,000 queries)...")
        search_params = {"metric_type": metric_val, "params": {"ef": 128}}
        
        start_batch = time.time()
        collection.search(
            data=query_set_list, 
            anns_field="embeddings", 
            param=search_params, 
            limit=5
        )
        rps_batch = 1000 / (time.time() - start_batch)
        log(f"  -> BATCH RPS: {rps_batch:.2f}")
        
        # 5. Filtered Search
        if metric_name == "Cosine":
            log("  Running Filtered Search (category='tech')...")
            start_filter = time.time()
            collection.search(
                data=query_set_list, 
                anns_field="embeddings", 
                param=search_params, 
                limit=5,
                expr="category == 'tech'"
            )
            rps_filter = 1000 / (time.time() - start_filter)
            log(f"  -> Filtered RPS: {rps_filter:.2f}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1" 
    initialize_log() # Create/Clear the log file
    benchmark_qdrant()
    benchmark_milvus()