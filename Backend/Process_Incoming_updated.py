import chromadb
import requests
import json
import re

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

def detect_course_subject(context_chunks):
    """Infer the course subject from video titles in the retrieved chunks.
    
    Extracts common keywords from video file names to build a short,
    human-readable course-subject string (e.g. 'Cloud Computing',
    'Database Management Systems', 'Machine Learning').
    """
    if not context_chunks:
        return "this course"

    # Collect all video titles
    titles = [chunk.get("video title", "") for chunk in context_chunks if chunk.get("video title")]
    if not titles:
        return "this course"

    # Combine all titles into one blob and do lightweight cleanup
    blob = " ".join(titles)
    # Remove leading numbering like "1_", "23_", emojis, and special chars
    blob = re.sub(r'\d+_', ' ', blob)
    blob = re.sub(r'[^\w\s]', ' ', blob)  # strip emojis / punctuation
    blob = blob.lower()

    # Count word frequency (skip very common stop words)
    stop_words = {
        'the', 'a', 'an', 'in', 'of', 'for', 'and', 'or', 'to', 'is', 'it',
        'with', 'on', 'at', 'by', 'from', 'this', 'that', 'are', 'was', 'be',
        'has', 'had', 'have', 'will', 'can', 'do', 'does', 'did', 'what',
        'how', 'why', 'when', 'where', 'who', 'which', 'vs', 'made', 'simple',
        'explained', 'complete', 'guide', 'beginners', 'introduction', 'hindi',
        'easiest', 'understanding', 'step', 'model', 'type', 'seven', 'note',
        'various', 's', 'not', 'your', 'you', 'all', 'its'
    }
    words = blob.split()
    freq = {}
    for w in words:
        w = w.strip()
        if len(w) > 2 and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        return "this course"

    # Pick the top keywords by frequency
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    # Take top 2-3 keywords to form a subject phrase
    top_keywords = [w.title() for w, _ in sorted_words[:3]]
    return " ".join(top_keywords)

def inference_stream(prompt):
    """Yields response tokens one-by-one for streaming UI display."""
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": True
    }, stream=True)
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break

def answer_question(query, chroma_dir):
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name="transcripts")

    question_embedding = create_embedding([query])[0] 

    top_results = 9
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_results
    )

    context_chunks = []
    sources = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            
            file_name = metadata.get('file_name', 'Unknown')
            if file_name.endswith('.json'):
                file_name = file_name[:-5]
            
            context_chunks.append({
                "video title": file_name,
                "start": metadata.get('start', 0),
                "end": metadata.get('end', 0),
                "text": text
            })
            
            # Simple format for frontend sources
            sources.append(f"{file_name} (around {int(metadata.get('start', 0))}s)")

    formatted_context = json.dumps(context_chunks, indent=2, ensure_ascii=False)

    # Auto-detect the course subject from retrieved video titles
    course_subject = detect_course_subject(context_chunks)

    prompt = f"""You are a highly knowledgeable teaching assistant for a course on '{course_subject}'. 

### Instructions:
1. Answer the user's question comprehensively and conversationally. You are fully encouraged to use your own broad knowledge to clearly explain concepts.
2. After explaining the concept, ALWAYS use the provided 'Video Subtitle Chunks' to tell the user exactly where they can learn more about this in the course.
3. Clearly state the video (file name) and provide the exact timestamp (e.g., 'at the 45-second mark') so the user can easily find it.
4. Do NOT explain your process or mention that you are reading from a JSON format. Just give the answer directly.
5. If the user's question is completely unrelated to {course_subject} or the course material, politely say that you can only answer questions related to this course.

### Video Subtitle Chunks (Context):
{formatted_context}

### User Question:
{query}

### Your Answer:
"""

    response_data = inference(prompt)
    final_answer = response_data.get("response", "No response generated.")

    return final_answer, sources

def answer_question_stream(query, chroma_dir):
    """Like answer_question, but returns a token generator for streaming UI display."""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name="transcripts")

    question_embedding = create_embedding([query])[0]

    top_results = 9
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_results
    )

    context_chunks = []
    sources = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]

            file_name = metadata.get('file_name', 'Unknown')
            if file_name.endswith('.json'):
                file_name = file_name[:-5]

            context_chunks.append({
                "video title": file_name,
                "start": metadata.get('start', 0),
                "end": metadata.get('end', 0),
                "text": text
            })

            sources.append(f"{file_name} (around {int(metadata.get('start', 0))}s)")

    formatted_context = json.dumps(context_chunks, indent=2, ensure_ascii=False)

    # Auto-detect the course subject from retrieved video titles
    course_subject = detect_course_subject(context_chunks)

    prompt = f"""You are a highly knowledgeable teaching assistant for a course on '{course_subject}'. 

### Instructions:
1. Answer the user's question comprehensively and conversationally. You are fully encouraged to use your own broad knowledge to clearly explain concepts.
2. After explaining the concept, ALWAYS use the provided 'Video Subtitle Chunks' to tell the user exactly where they can learn more about this in the course.
3. Clearly state the video (file name) and provide the exact timestamp (e.g., 'at the 45-second mark') so the user can easily find it.
4. Do NOT explain your process or mention that you are reading from a JSON format. Just give the answer directly.
5. If the user's question is completely unrelated to {course_subject} or the course material, politely say that you can only answer questions related to this course.

### Video Subtitle Chunks (Context):
{formatted_context}

### User Question:
{query}

### Your Answer:
"""

    token_generator = inference_stream(prompt)
    return token_generator, sources

def main():
    incoming_query = input("Ask a Question: ")
    print("\nSearching database for matches...")
    answer, sources = answer_question(incoming_query, "./chroma_db")
    print("\n--- AI Response ---")
    print(answer)
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(answer)

if __name__ == "__main__":
    main()