import requests
import os
import json
import chromadb

def create_embeddings_robust(text_list):
    all_embeddings = []
    valid_indices = []
    batch_size = 50 
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        batch_indices = list(range(i, i+len(batch)))
        
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        })
        
        data = r.json()
        if "embeddings" in data:
            all_embeddings.extend(data["embeddings"])
            valid_indices.extend(batch_indices)
        else:
            print(f"\nBatch failed. Falling back to isolating the bad chunk 1-by-1...")
            # If the batch fails, process 1-by-1 to isolate and skip the problematic string
            for idx, text in zip(batch_indices, batch):
                r_single = requests.post("http://localhost:11434/api/embed", json={
                    "model": "bge-m3",
                    "input": [text]
                })
                data_single = r_single.json()
                if "embeddings" in data_single:
                    all_embeddings.append(data_single["embeddings"][0])
                    valid_indices.append(idx)
                else:
                    err = data_single.get('error', 'Unknown Error')
                    print(f"--> Successfully bypassed problematic chunk: '{text}' (Error: {err})")
                    
    return all_embeddings, valid_indices

def index_json_into_chroma(json_path, chroma_dir):
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="transcripts")
    
    with open(json_path, encoding="utf-8") as f:
        content = json.load(f)
        
    json_file = os.path.basename(json_path)
    print(f"Creating Embeddings for {json_file}")
    
    # Pre-filter explicitly empty chunks
    valid_chunks = [c for c in content.get('segments', []) if c.get('text', '').strip()]
    
    if not valid_chunks:
        print(f"Skipping {json_file} because it has no valid text chunks.")
        return 0

    # Extract text
    texts = [c['text'].strip() for c in valid_chunks]
    
    # Use robust embedding generator that skips bad strings
    embeddings, valid_indices = create_embeddings_robust(texts)
       
    ids = []
    metadatas = []
    final_texts = []
    
    # Only keep the texts and metadata for the chunks that successfully generated an embedding
    for idx in valid_indices:
        chunk = valid_chunks[idx]
        ids.append(f"{json_file}_chunk_{idx}")
        
        metadata = {
            "file_name": json_file,
            "start": chunk.get("start", 0),
            "end": chunk.get("end", 0)
        }
        metadatas.append(metadata)
        final_texts.append(texts[idx])
        
    if final_texts:
        # Insert or update data in ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=final_texts,
            metadatas=metadatas
        )
        print(f"Successfully added {len(final_texts)} chunks from {json_file} into ChromaDB.\n")
        return len(final_texts)
    else:
        print(f"No valid embeddings could be generated for {json_file}.\n")
        return 0

def get_collection_count(chroma_dir):
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="transcripts")
    return collection.count()

def index_all_transcripts():
    # Initialize ChromaDB client (this creates a 'chroma_db' folder in your directory to save data)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get a collection (a table/index in vector DB terminology)
    collection = client.get_or_create_collection(name="transcripts")

    # List all the jsons safely
    jsons = [f for f in os.listdir("transcripts") if f.endswith(".json")]

    for json_file in jsons:
        index_json_into_chroma(f"transcripts/{json_file}", "./chroma_db")

    print(f"All done! Total chunks in your ChromaDB database: {collection.count()}")

if __name__ == "__main__":
    if not os.path.exists("transcripts"):
        print("Directory not found: transcripts")
    else:
        index_all_transcripts()