# Converts the videos to mp3 
import os 
import subprocess
import shutil
import sys
sys.stdout.reconfigure(encoding='utf-8')

def _get_ffmpeg_path():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        if os.path.exists(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"):
            ffmpeg_path = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
        elif os.path.exists(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
            ffmpeg_path = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        else:
            ffmpeg_path = "ffmpeg"
    return ffmpeg_path

def convert_videos_in_dir(videos_dir="Videos", audios_dir="audios"):
    if not os.path.exists(audios_dir):
        os.makedirs(audios_dir)
        print(f"Created directory: {audios_dir}")

    ffmpeg_path = _get_ffmpeg_path()
    
    try:
        files = os.listdir(videos_dir)
    except FileNotFoundError:
        print(f"Directory not found: {videos_dir}")
        return

    for file in files:
        if not file.endswith((".mp4", ".webm", ".mkv")):
            continue

        try:
            if file.startswith("Lec-") and (":" in file or "：" in file):
                colon_char = ":" if ":" in file else "："
                tutorial_number = file.split("Lec-")[1].split(colon_char)[0]
                file_name = file.split(colon_char, 1)[1].strip().rsplit(".", 1)[0]
            elif " #" in file:
                tutorial_number = file.split(" [")[0].split(" #")[1]
                file_name = file.split(" ｜ ")[0] if " ｜ " in file else file.split(" | ")[0]
            else:
                tutorial_number = "vid"
                file_name = file.rsplit(".", 1)[0]

            safe_file_name = "".join(c for c in file_name if c not in r'<>:"/\|?*')
            
            print(f"Converting: {tutorial_number} - {safe_file_name}")
            subprocess.run([ffmpeg_path, "-y", "-i", os.path.join(videos_dir, file), os.path.join(audios_dir, f"{tutorial_number}_{safe_file_name}.mp3")], check=True)
            
        except FileNotFoundError:
            print("Error: Could not find ffmpeg executable. Make sure it is installed and added to your system PATH.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Skipping {file} - FFmpeg Error: {e}")
        except Exception as e:
            print(f"Skipping {file} - Error: {e}")

def video_to_mp3(video_path, audios_dir):
    if not os.path.exists(audios_dir):
        os.makedirs(audios_dir)
        print(f"Created directory: {audios_dir}")

    ffmpeg_path = _get_ffmpeg_path()
    file_name_with_ext = os.path.basename(video_path)
    file_name = os.path.splitext(file_name_with_ext)[0]
    safe_file_name = "".join(c for c in file_name if c not in r'<>:"/\|?*')
    output_mp3_path = os.path.join(audios_dir, f"{safe_file_name}.mp3")
    
    try:
        subprocess.run([ffmpeg_path, "-y", "-i", video_path, output_mp3_path], check=True)
        return output_mp3_path
    except Exception as e:
        print(f"Error converting video: {e}")
        raise e

if __name__ == "__main__":
    convert_videos_in_dir()