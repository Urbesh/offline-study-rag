# Converts the videos to mp3 
import os 
import subprocess
import shutil
import sys
sys.stdout.reconfigure(encoding='utf-8')
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    if os.path.exists(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"):
        ffmpeg_path = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
    elif os.path.exists(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
        ffmpeg_path = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
    else:
        ffmpeg_path = "ffmpeg"

if not os.path.exists("audios"):
    os.makedirs("audios")
    print("Created directory: audios")

files = os.listdir("Videos") 
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
        subprocess.run([ffmpeg_path, "-y", "-i", f"Videos/{file}", f"audios/{tutorial_number}_{safe_file_name}.mp3"], check=True)
        
    except FileNotFoundError:
        print("Error: Could not find ffmpeg executable. Make sure it is installed and added to your system PATH.")
        break
    except subprocess.CalledProcessError as e:
        print(f"Skipping {file} - FFmpeg Error: {e}")
    except Exception as e:
        print(f"Skipping {file} - Error: {e}")