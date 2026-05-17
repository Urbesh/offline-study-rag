import os
import json
from faster_whisper import WhisperModel

# Configuration
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "int8_float16"
DEVICE = "cuda"
INPUT_DIR = "audios"
OUTPUT_DIR = "transcripts"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading Whisper model '{MODEL_SIZE}' with compute type '{COMPUTE_TYPE}' on {DEVICE}...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to CPU...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    mp3_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp3")]
    
    if not mp3_files:
        print(f"No MP3 files found in '{INPUT_DIR}' directory.")
        return

    print(f"Found {len(mp3_files)} MP3 files. Starting transcription and translation to English...")

    for i, file_name in enumerate(mp3_files, 1):
        file_path = os.path.join(INPUT_DIR, file_name)
        base_name = os.path.splitext(file_name)[0]
        json_file_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        if os.path.exists(json_file_path):
            print(f"[{i}/{len(mp3_files)}] Skipping '{file_name}' (already processed).")
            continue
        print(f"[{i}/{len(mp3_files)}] Processing '{file_name}'...")
        try:
            segments, info = model.transcribe(
                file_path, 
                task="translate", 
                beam_size=5
            )
            
            print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            transcription_data = {
                "file_name": file_name,
                "detected_language": info.language,
                "language_probability": info.language_probability,
                "segments": []
            }
            
            full_text = []
            for segment in segments:
                segment_text = segment.text.strip()
                transcription_data["segments"].append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text
                })
                full_text.append(segment_text)
            
            transcription_data["text"] = " ".join(full_text)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=4)
                
            print(f"Saved transcript to '{json_file_path}'")
            
        except Exception as e:
            print(f"Error processing '{file_name}': {e}")

if __name__ == "__main__":
    main()
