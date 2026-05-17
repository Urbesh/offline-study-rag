import chromadb
import requests
import json

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    return response

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="transcripts")

incoming_query = input("Ask a Question: ")
print("\nSearching database for matches...")
question_embedding = create_embedding([incoming_query])[0] 

top_results = 9
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=top_results
)

# Extract and format the results into a JSON-like list of dictionaries so the LLM can read it
context_chunks = []
for i in range(len(results['documents'][0])):
    text = results['documents'][0][i]
    metadata = results['metadatas'][0][i]
    
    # Remove the .json extension so it reads as a normal video title
    file_name = metadata.get('file_name', 'Unknown')
    if file_name.endswith('.json'):
        file_name = file_name[:-5]
    
    context_chunks.append({
        "video title": file_name,
        "start": metadata.get('start', 0),
        "end": metadata.get('end', 0),
        "text": text
    })

# Convert the list of dicts to a JSON string for the prompt (ensure_ascii=False fixes the \uff5c issue)
formatted_context = json.dumps(context_chunks, indent=2, ensure_ascii=False)

prompt = f"""You are a highly knowledgeable teaching assistant for the 'Cloud Computing Masterclass'. 

### Instructions:
1. Answer the user's question comprehensively and conversationally. You are fully encouraged to use your own broad knowledge to clearly explain concepts (like 'What is CIDR addressing?').
2. After explaining the concept, ALWAYS use the provided 'Video Subtitle Chunks' to tell the user exactly where they can learn more about this in the course.
3. Clearly state the video (file name) and provide the exact timestamp (e.g., 'at the 45-second mark') so the user can easily find it.
4. Do NOT explain your process or mention that you are reading from a JSON format. Just give the answer directly.
5. If the user's question is completely unrelated to cloud computing or the course material, politely say that you can only answer questions related to the Cloud Computing Masterclass.

### Video Subtitle Chunks (Context):
{formatted_context}

### User Question:
{incoming_query}

### Your Answer:
"""

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

print("Generating answer using local LLM...\n")
response_data = inference(prompt)
final_answer = response_data.get("response", "No response generated.")

print("\n--- AI Response ---")
print(final_answer)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(final_answer)