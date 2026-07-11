import yt_dlp
import os

def download_youtube_video(url, download_path, progress_hook=None):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        print(f"Created directory: {download_path}")

    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
        'quiet': False,
    }
    
    if progress_hook:
        ydl_opts['progress_hooks'] = [progress_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading to: {download_path}")
            info = ydl.extract_info(url, download=True)
            print("\nDone! Check your folder.")
            return ydl.prepare_filename(info)
            
    except Exception as e:
        print(f"Something went wrong: {e}")
        raise e

if __name__ == "__main__":
    video_url = input("Enter the YouTube URL: ")
    target_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Videos")
    download_youtube_video(video_url, target_folder)