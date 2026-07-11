import streamlit as st
import sys
import os
import json
import time
import tempfile

# ---------------------------------------------------------------------------
# Path setup — make Backend importable regardless of where streamlit is run from
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "Backend")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Backend imports
from Backend.Youtube_video_downloader import download_youtube_video
from Backend.Video_to_MP3 import video_to_mp3
from Backend.Create_embeddings import index_json_into_chroma, get_collection_count
from Backend.Process_Incoming_updated import answer_question_stream

# Directories (relative to Backend/)
VIDEOS_DIR = os.path.join(_BACKEND_DIR, "Videos")
AUDIOS_DIR = os.path.join(_BACKEND_DIR, "audios")
TRANSCRIPTS_DIR = os.path.join(_BACKEND_DIR, "transcripts")
CHROMA_DIR = os.path.join(_BACKEND_DIR, "chroma_db")

# Ensure directories exist
for d in [VIDEOS_DIR, AUDIOS_DIR, TRANSCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Whisper model caching ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load Whisper model once and keep it in memory across reruns."""
    from faster_whisper import WhisperModel
    MODEL_SIZE = "large-v3"
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
    except Exception:
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return model


# ── Transcription helper ──────────────────────────────────────────────────
def transcribe_audio(mp3_path, model):
    """Transcribe a single MP3 file and save the JSON transcript."""
    file_name = os.path.basename(mp3_path)
    base_name = os.path.splitext(file_name)[0]
    json_path = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.json")

    if os.path.exists(json_path):
        return json_path  # already transcribed

    segments, info = model.transcribe(mp3_path, task="translate", beam_size=5)

    transcription_data = {
        "file_name": file_name,
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "segments": [],
    }
    full_text = []
    for segment in segments:
        seg_text = segment.text.strip()
        transcription_data["segments"].append(
            {"start": segment.start, "end": segment.end, "text": seg_text}
        )
        full_text.append(seg_text)

    transcription_data["text"] = " ".join(full_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcription_data, f, ensure_ascii=False, indent=4)

    return json_path


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Study RAG — Video Lecture Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import Google Font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base overrides ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar styling ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}

/* ── Chat message styling ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.1);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
    padding: 1rem 1.25rem;
}

/* ── Source citation pills ──────────────────────────────────────────── */
.source-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15));
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px 4px;
    font-size: 0.78rem;
    color: #c4b5fd;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ── Status / expander styling ──────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    background: rgba(15, 15, 26, 0.4);
}

/* ── Metric styling ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(59, 130, 246, 0.08));
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 12px;
    padding: 1rem;
}

/* ── Primary button ─────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
}

/* ── Welcome header gradient ────────────────────────────────────────── */
.welcome-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.welcome-header h1 {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 40%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.welcome-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* ── Divider ────────────────────────────────────────────────────────── */
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(139, 92, 246, 0.18);
    margin: 1.2rem 0;
}

/* ── Status badge ───────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-badge.ready {
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.25);
}
.status-badge.empty {
    background: rgba(250, 204, 21, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(250, 204, 21, 0.25);
}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: count processed items
# ═══════════════════════════════════════════════════════════════════════════
def _count_files(directory, ext):
    """Count files with a given extension in a directory."""
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(ext)])


def _get_processed_videos():
    """Return list of transcript basenames already in the transcripts folder."""
    if not os.path.isdir(TRANSCRIPTS_DIR):
        return []
    return sorted(
        [os.path.splitext(f)[0] for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json")]
    )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Study RAG")
    st.caption("Your offline video lecture assistant")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Database status ───────────────────────────────────────────────
    try:
        doc_count = get_collection_count(CHROMA_DIR)
    except Exception:
        doc_count = 0

    processed_videos = _get_processed_videos()

    if doc_count > 0:
        st.markdown(
            f'<span class="status-badge ready">● Database Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-badge empty">○ No Data Yet</span>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    col1.metric("Chunks", f"{doc_count:,}")
    col2.metric("Videos", len(processed_videos))

    if processed_videos:
        with st.expander("📚 Processed Videos", expanded=False):
            for v in processed_videos:
                st.markdown(f"- {v}")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── 1. YouTube Download ───────────────────────────────────────────
    st.markdown("### ⬇️ Add from YouTube")
    yt_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    btn_download = st.button("⬇️  Download Video", use_container_width=True, disabled=st.session_state.processing)

    if btn_download and yt_url:
        st.session_state.processing = True
        with st.status("Downloading YouTube video…", expanded=True) as status:
            try:
                st.write("🔗 Connecting to YouTube…")

                # Progress hook for yt-dlp
                progress_bar = st.progress(0, text="Starting download…")

                def _yt_progress(d):
                    if d["status"] == "downloading":
                        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                        downloaded = d.get("downloaded_bytes", 0)
                        if total > 0:
                            pct = downloaded / total
                            progress_bar.progress(
                                min(pct, 1.0),
                                text=f"Downloading… {pct:.0%}",
                            )
                    elif d["status"] == "finished":
                        progress_bar.progress(1.0, text="Download complete ✓")

                video_path = download_youtube_video(yt_url, VIDEOS_DIR, progress_hook=_yt_progress)
                status.update(label="✅ Video downloaded!", state="complete")
                st.toast(f"Downloaded: {os.path.basename(video_path)}", icon="✅")
            except Exception as e:
                status.update(label="❌ Download failed", state="error")
                st.error(f"Error: {e}")
        st.session_state.processing = False

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── 2. Local File Upload ──────────────────────────────────────────
    st.markdown("### 📁 Upload Local Video")
    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "webm", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        save_path = os.path.join(VIDEOS_DIR, uploaded_file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.toast(f"Saved: {uploaded_file.name}", icon="📁")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── 3. Process Pipeline ───────────────────────────────────────────
    st.markdown("### ⚙️ Process Videos")
    st.caption("Convert → Transcribe → Embed")

    # Gather un-processed videos
    video_files = []
    if os.path.isdir(VIDEOS_DIR):
        video_files = [
            f
            for f in os.listdir(VIDEOS_DIR)
            if f.lower().endswith((".mp4", ".webm", ".mkv"))
        ]

    unprocessed_count = 0
    for vf in video_files:
        base = os.path.splitext(vf)[0]
        safe_base = "".join(c for c in base if c not in r'<>:"/\|?*')
        json_path = os.path.join(TRANSCRIPTS_DIR, f"{safe_base}.json")
        if not os.path.exists(json_path):
            unprocessed_count += 1

    if unprocessed_count > 0:
        st.info(f"**{unprocessed_count}** video(s) ready to process")
    elif len(video_files) > 0:
        st.success("All videos are processed ✓")
    else:
        st.warning("No videos found. Download or upload one first.")

    btn_process = st.button(
        "🚀  Process All Videos",
        use_container_width=True,
        disabled=st.session_state.processing or unprocessed_count == 0,
    )

    if btn_process:
        st.session_state.processing = True

        # Load whisper model (cached)
        with st.status("Loading Whisper model…", expanded=True) as model_status:
            st.write("⏳ This may take a moment on first run…")
            whisper_model = load_whisper_model()
            model_status.update(label="✅ Whisper model ready", state="complete")

        overall_progress = st.progress(0, text="Starting pipeline…")
        total_videos = len(video_files)

        for idx, vf in enumerate(video_files):
            video_path = os.path.join(VIDEOS_DIR, vf)
            base = os.path.splitext(vf)[0]
            safe_base = "".join(c for c in base if c not in r'<>:"/\|?*')

            overall_pct = idx / total_videos
            overall_progress.progress(overall_pct, text=f"Processing {idx + 1}/{total_videos}: {vf[:40]}…")

            with st.status(f"Processing: {vf[:50]}…", expanded=True) as file_status:
                # Step 1 — Audio extraction
                mp3_path = os.path.join(AUDIOS_DIR, f"{safe_base}.mp3")
                if not os.path.exists(mp3_path):
                    st.write("🎵 Extracting audio with ffmpeg…")
                    try:
                        mp3_path = video_to_mp3(video_path, AUDIOS_DIR)
                        st.write("✅ Audio extracted")
                    except Exception as e:
                        st.error(f"ffmpeg error: {e}")
                        file_status.update(label=f"❌ Failed: {vf[:40]}", state="error")
                        continue
                else:
                    st.write("✅ Audio already exists — skipped")

                # Step 2 — Transcription
                json_out = os.path.join(TRANSCRIPTS_DIR, f"{safe_base}.json")
                if not os.path.exists(json_out):
                    st.write("🗣️ Transcribing with Whisper (this takes a while)…")
                    try:
                        json_out = transcribe_audio(mp3_path, whisper_model)
                        st.write("✅ Transcript saved")
                    except Exception as e:
                        st.error(f"Whisper error: {e}")
                        file_status.update(label=f"❌ Failed: {vf[:40]}", state="error")
                        continue
                else:
                    st.write("✅ Transcript already exists — skipped")

                # Step 3 — Embeddings
                st.write("📊 Generating embeddings and storing in ChromaDB…")
                try:
                    chunks_added = index_json_into_chroma(json_out, CHROMA_DIR)
                    st.write(f"✅ {chunks_added} chunks indexed")
                except Exception as e:
                    st.error(f"Embedding error: {e}")
                    file_status.update(label=f"❌ Failed: {vf[:40]}", state="error")
                    continue

                file_status.update(label=f"✅ Done: {vf[:50]}", state="complete")

        overall_progress.progress(1.0, text="Pipeline complete ✓")
        st.session_state.processing = False
        st.toast("All videos processed!", icon="🎉")
        st.rerun()

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.caption("Built with Streamlit · Runs 100% locally")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA — CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

# Welcome header (only when chat is empty)
if not st.session_state.chat_history:
    st.markdown(
        """
    <div class="welcome-header">
        <h1>Study RAG</h1>
        <p>Ask anything about your video lectures — powered by local AI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Suggestion chips
    st.markdown("")
    cols = st.columns(3)
    suggestions = [
        "📝 Summarise this course",
        "🔑 What are the key concepts?",
        "📖 Explain the fundamentals",
    ]
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, use_container_width=True, key=f"sug_{i}"):
            # Strip emoji prefix for the actual query
            query_text = suggestion.split(" ", 1)[1] if " " in suggestion else suggestion
            st.session_state.chat_history.append({"role": "user", "content": query_text})
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            pills_html = "".join(
                f'<span class="source-pill">📎 {s}</span>' for s in msg["sources"]
            )
            st.markdown(
                f"<div style='margin-top:8px'>{pills_html}</div>",
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your lectures…")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    # Check if database has data
    try:
        count = get_collection_count(CHROMA_DIR)
    except Exception:
        count = 0

    if count == 0:
        no_data_msg = "⚠️ The knowledge base is empty. Please add and process some videos first using the sidebar."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": no_data_msg}
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(no_data_msg)
    else:
        # Stream response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                token_gen, sources = answer_question_stream(user_input, CHROMA_DIR)
                full_response = st.write_stream(token_gen)

                # Deduplicate sources
                unique_sources = list(dict.fromkeys(sources))

                # Show source pills
                if unique_sources:
                    pills_html = "".join(
                        f'<span class="source-pill">📎 {s}</span>'
                        for s in unique_sources
                    )
                    st.markdown(
                        f"<div style='margin-top:8px'>{pills_html}</div>",
                        unsafe_allow_html=True,
                    )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "sources": unique_sources,
                    }
                )
            except Exception as e:
                error_msg = f"❌ Error generating response: {e}\n\nMake sure **Ollama** is running with `llama3.1:8b` and `bge-m3` models loaded."
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )
