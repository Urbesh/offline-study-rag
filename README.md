# Study RAG — Offline Video Lecture Assistant

An AI-powered application that transforms recorded video lectures into an **interactive, searchable knowledge base**. Upload or download course videos, and the system will automatically transcribe, embed, and index them — letting you ask natural-language questions and get precise, context-aware answers complete with video timestamps. Everything runs **100 % locally** on your machine after the initial setup — no internet connection, no API keys, no cloud services required.

> **Status:** This project is under active development. The core pipeline and frontend are functional; additional features are planned for future releases.

---

## Table of Contents

- [How It Works](#-how-it-works)
- [Project Components](#-project-components)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## How It Works

Study RAG uses a **Retrieval-Augmented Generation (RAG)** pipeline to turn video lectures into an interactive Q&A system:

1. **You provide videos** — either by pasting a YouTube URL or uploading local video files through the web interface.
2. **The system processes them** — it extracts audio, transcribes speech to text, splits the transcript into chunks, and stores vector embeddings in a local database.
3. **You ask questions** — type any question about your lecture content, and the system retrieves the most relevant transcript chunks, feeds them as context to a local LLM, and streams back a detailed answer with exact video timestamps so you can jump to the right spot in your lecture.

---

## Project Components

### Backend

The backend is a collection of Python modules, each responsible for one stage of the data pipeline:

#### 1. `Youtube_video_downloader.py` — Video Acquisition
Downloads videos from YouTube using `yt-dlp`. Accepts a URL and a target directory, supports a progress-hook callback for real-time download progress in the UI, and saves the video in its best available quality.

#### 2. `Video_to_MP3.py` — Audio Extraction
Extracts the audio track from video files (`.mp4`, `.webm`, `.mkv`) using `ffmpeg` via a subprocess call. It sanitises filenames to avoid filesystem issues and supports both batch conversion (an entire directory) and single-file conversion used by the frontend.

#### 3. `MP3_to_JSON.py` — Speech-to-Text Transcription
Transcribes audio files into structured JSON using **Faster-Whisper** (a CTranslate2-optimised port of OpenAI's Whisper). Key details:
- Uses the `large-v3` model for high accuracy.
- Runs on **CUDA (GPU)** by default and automatically falls back to **CPU** if no compatible GPU is detected.
- The `translate` task is used, so non-English lectures are automatically translated to English during transcription.
- Output JSON includes the full transcript text, per-segment timestamps, detected language, and language probability.

#### 4. `Create_embeddings.py` — Vector Embedding & Indexing
Generates vector embeddings for each transcript chunk using Ollama's `bge-m3` embedding model and stores them in **ChromaDB** (a local, persistent vector database). Features include:
- **Robust batch processing** — sends chunks in batches of 50; if a batch fails, it falls back to one-by-one processing to isolate and skip problematic chunks.
- **Upsert logic** — uses `collection.upsert()` so re-processing a video won't create duplicate entries.
- Each stored chunk includes metadata (source filename, start/end timestamps) for accurate citation.

#### 5. `Process_Incoming_updated.py` — Query & Answer Generation
Handles the RAG query pipeline end-to-end:
1. Embeds the user's question with `bge-m3`.
2. Queries ChromaDB for the top 9 most relevant transcript chunks.
3. **Dynamically detects the course subject** from the retrieved video titles (using keyword frequency analysis), so the system prompt adapts to any course — Cloud Computing, DBMS, Machine Learning, etc.
4. Constructs a detailed prompt with the context chunks and sends it to **Ollama's `llama3.1:8b`** model.
5. Supports both **blocking** (`answer_question`) and **streaming** (`answer_question_stream`) response modes; the frontend uses streaming for a real-time typing effect.

---

### Frontend

#### `app.py` — Streamlit Web Interface
A single-page **Streamlit** application that ties all backend modules together into a polished, user-friendly interface:

- **Sidebar Panel:**
  - **Database Status** — shows the number of indexed chunks and processed videos at a glance, with a green "Ready" or yellow "No Data Yet" badge.
  - **YouTube Download** — paste a URL and download with a live progress bar.
  - **Local Upload** — drag-and-drop or browse for `.mp4` / `.webm` / `.mkv` files.
  - **Process Pipeline** — one-click button to run the full Convert → Transcribe → Embed pipeline on all unprocessed videos, with per-step status updates.

- **Main Chat Area:**
  - A conversational chat interface with suggestion chips for first-time users.
  - Streamed AI responses with **source citation pills** showing the exact video name and timestamp.
  - Full chat history persistence within a session.

- **Design:**
  - Custom CSS with Inter font, glassmorphism effects, gradient accents, and a dark-mode colour scheme.
  - The Whisper model is loaded once via `@st.cache_resource` and shared across all reruns for performance.

---

## Architecture

The system follows a standard two-pipeline RAG architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                         │
│                                                                 │
│  YouTube URL ──┐                                                │
│                ├──► Video File ──► ffmpeg ──► Audio (.mp3)      │
│  Local Upload ─┘                                                │
│                                                                 │
│  Audio (.mp3) ──► Faster-Whisper ──► Transcript (.json)         │
│                                                                 │
│  Transcript ──► bge-m3 (Ollama) ──► Embeddings ──► ChromaDB     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                            │
│                                                                 │
│  User Question ──► bge-m3 ──► ChromaDB (similarity search)      │
│                                                                 │
│  Top-9 Chunks + Question ──► llama3.1:8b (Ollama) ──► Answer    │
│                                                     + Sources   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.13+ | Core application language |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) | Fast Python package and project manager |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive web UI |
| **Transcription** | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) | Speech-to-text (with GPU acceleration) |
| **Embeddings** | [Ollama](https://ollama.com/) + `bge-m3` | Text → vector embeddings |
| **LLM** | [Ollama](https://ollama.com/) + `llama3.1:8b` | Answer generation |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Local persistent vector storage & retrieval |
| **Video Download** | [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube video downloading |
| **Audio Extraction** | [ffmpeg](https://ffmpeg.org/) | Video → audio conversion |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed on your system before proceeding:

1. **Python 3.13+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify: `python --version`

2. **uv** (Python package manager)
   - Install via the official script:
     ```bash
     # Windows (PowerShell)
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

     # macOS / Linux
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - Verify: `uv --version`

3. **ffmpeg**
   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your system `PATH`.
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt install ffmpeg`
   - Verify: `ffmpeg -version`

4. **Ollama**
   - Download and install from [ollama.com](https://ollama.com/download)
   - Verify: `ollama --version`

5. **NVIDIA GPU + CUDA** *(optional but highly recommended)*
   - A CUDA-capable GPU dramatically speeds up Whisper transcription. Without one, the system falls back to CPU (much slower but still functional).

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/UrbesLab/offline-study-rag.git
   cd offline-study-rag
   ```

2. **Install Python dependencies with uv:**
   ```bash
   uv sync
   ```
   This creates a virtual environment (`.venv`) and installs all dependencies from `pyproject.toml` and `uv.lock` automatically.

3. **Pull the required Ollama models:**
   ```bash
   ollama pull llama3.1:8b
   ollama pull bge-m3
   ```
   > **Note:** `llama3.1:8b` is approximately 4.7 GB and `bge-m3` is approximately 1.2 GB. These only need to be downloaded once.

---

### Running the Application

1. **Start the Ollama server** (if it isn't already running):
   ```bash
   ollama serve
   ```

2. **Launch the Streamlit app:**
   ```bash
   uv run streamlit run Frontend/app.py
   ```

3. **Open the app** in your browser at the URL shown in the terminal (usually `http://localhost:8501`).

4. **Start using it:**
   - Use the sidebar to download a YouTube video or upload a local video file.
   - Click **"🚀 Process All Videos"** to run the full pipeline.
   - Once processing is complete, ask questions in the chat!

---

## Project Structure

```
offline-study-rag/
├── Backend/
│   ├── Youtube_video_downloader.py   # Downloads YouTube videos via yt-dlp
│   ├── Video_to_MP3.py               # Extracts audio using ffmpeg
│   ├── MP3_to_JSON.py                # Transcribes audio with Faster-Whisper
│   ├── Create_embeddings.py          # Generates embeddings & indexes in ChromaDB
│   ├── Process_Incoming_updated.py   # RAG query pipeline (retrieval + LLM)
│   ├── Videos/                       # Downloaded / uploaded video files
│   └── transcripts/                  # Generated JSON transcripts
├── Frontend/
│   └── app.py                        # Streamlit web application
├── pyproject.toml                    # Project metadata & dependencies
├── uv.lock                          # Locked dependency versions
├── .python-version                   # Python version pin (3.13)
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

© 2026 Urbesh Mondal
