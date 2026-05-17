This project is a Flask-based educational assistant designed to make video lectures fully searchable and interactive. The system processes uploaded videos or YouTube links by extracting the audio, transcribing it via OpenAI's Whisper, and storing the chunked text data in a vector database. Using a Retrieval-Augmented Generation (RAG) pipeline, the frontend interface allows students to seamlessly upload course materials, generate automated summaries, and ask natural language questions to find precise, context-aware help based directly on their lectures. This project is built using opensource tools and models. This project is made to run locally in any machine without internet connection after the initial setup and model downloads. This project is in it's initial stages and is being developed, more features are planned and will be added in the future. 

## Components

### Backend
- **Video Processing**: Uses `yt-dlp` to download YouTube videos and `ffmpeg` to extract audio tracks.
- **Transcription**: Converts audio to text using `Faster-Whisper`. And uses `ollama`'s `bge-m3` model for generating embeddings.
- **Vector Database**: Stores and retrieves text embeddings using `ChromaDB`.
- **LLM Integration**: Generates summaries and answers questions using `ollama` (defaulting to `llama3.1`).


## Architecture

The system follows a standard RAG architecture:
1.  **Ingestion Pipeline**: Video -> Audio Extraction -> Transcription -> Chunking -> Embedding -> Vector DB.
2.  **Query Pipeline**: User Query -> Embedding -> Vector DB Search (Retrieval) -> LLM Context Augmentation -> Answer Generation.

## Prograssion
**Backend**  [Done]
**Frontend** [Pending...]
**connect backend and frontend** [Pending...]
**Proper Readme file with technical details and stuff** [Pending...]

