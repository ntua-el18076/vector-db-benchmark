import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# --- Configuration ---
NUM_VECTORS = 100_000   # Size of dataset (Small enough to run fast, big enough to measure)
DIMENSIONS = 128        # Keep dimensions constant
BATCH_SIZE = 1000       # Upload batch size
QDRANT_URL = "localhost"
MILVUS_HOST = "localhost"

# Generate Synthetic Data (Random Floats)
print(f"Generating {NUM_VECTORS} random vectors of {DIMENSIONS} dimensions...")
vectors = np.random.rand(NUM_VECTORS, DIMENSIONS).astype(np.float32)
# Normalize vectors for fair Cosine comparison (optional, but good practice)
vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
payloads = [{"id": i} for i in range(NUM_VECTORS)]

def benchmark_qdrant():
    print("\n--- Testing Qdrant ---")
    client = QdrantClient(url=QDRANT_URL)
    
    metrics = {
        "Cosine": Distance.COSINE,
        "Euclidean": Distance.EUCLID
    }

    for metric_name, metric_val in metrics.items():
        collection_name = f"qdrant_{metric_name.lower()}"
        print(f"Setting up {collection_name} ({metric_name})...")
        
        # 1. Recreate Collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=DIMENSIONS, distance=metric_val),
        )

        # 2. Upload Data
        print("  Uploading data...")
        start_upload = time.time()
        client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payloads,
            ids=list(range(NUM_VECTORS)),
            batch_size=BATCH_SIZE
        )
        print(f"  Upload finished in {time.time() - start_upload:.2f}s")

        # 3. Search Test
        print("  Running 1,000 queries...")
        query_vectors = vectors[:1000] # Use first 1000 vectors as queries
        
        start_search = time.time()
        for q in query_vectors:
            client.search(
                collection_name=collection_name,
                query_vector=q,
                limit=5
            )
        duration = time.time() - start_search
        rps = 1000 / duration
        print(f"  Result: {metric_name} RPS: {rps:.2f}")

def benchmark_milvus():
    print("\n--- Testing Milvus ---")
    try:
        connections.connect("default", host=MILVUS_HOST, port="19530")
    except Exception as e:
        print("Skipping Milvus (Not running or connection failed)")
        return

    metrics = {
        "Cosine": "COSINE",
        "Euclidean": "L2"
    }

    for metric_name, metric_val in metrics.items():
        collection_name = f"milvus_{metric_name.lower()}"
        print(f"Setting up {collection_name} ({metric_name})...")
        
        # 1. Define Schema
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIMENSIONS)
        ]
        schema = CollectionSchema(fields, "Testing metrics")
        collection = Collection(collection_name, schema)

        # 2. Upload Data
        print("  Uploading data...")
        # Milvus expects columns: [ids, vectors]
        collection.insert([list(range(NUM_VECTORS)), vectors])
        collection.flush()
        
        # 3. Build Index (REQUIRED for Milvus search)
        index_params = {
            "metric_type": metric_val,
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 128}
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        collection.load()

        # 4. Search Test
        print("  Running 1,000 queries...")
        query_vectors = vectors[:1000].tolist()
        
        search_params = {"metric_type": metric_val, "params": {"ef": 128}}
        
        start_search = time.time()
        collection.search(
            data=query_vectors, 
            anns_field="embeddings", 
            param=search_params, 
            limit=5
        )
        duration = time.time() - start_search
        rps = 1000 / duration
        print(f"  Result: {metric_name} RPS: {rps:.2f}")

if __name__ == "__main__":
    # Ensure standard numpy speed
    import os
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    benchmark_qdrant()
    # Uncomment the line below if Milvus is running
    benchmark_milvus()