# backend/app.py

import os
import traceback
import yt_dlp
import numpy as np
import whisper
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from moviepy.editor import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
from PIL import Image, ImageDraw, ImageFont
import textwrap
import uuid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gc
import requests # NEW: Added for API calls

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TEMP_VIDEO_FOLDER = 'temp_videos'
GIF_OUTPUT_FOLDER = 'static/gifs'
app.config['UPLOAD_FOLDER'] = TEMP_VIDEO_FOLDER
# NEW: Get RapidAPI Key from server environment for security
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") 

# --- Proxy Configuration ---
PROXY_URLS = [
    # Add your Bright Data proxy URL here if you have one
    # 'http://brd-customer-hl_9555f995-zone-residential_proxy1:xjd7xb637zjl@brd.superproxy.io:33335'
]

# --- AI Models (Lazy Loading) ---
whisper_model = None
sentence_model = None

# --- Ensure Folders Exist ---
if not os.path.exists(TEMP_VIDEO_FOLDER):
    os.makedirs(TEMP_VIDEO_FOLDER)
if not os.path.exists(GIF_OUTPUT_FOLDER):
    os.makedirs(GIF_OUTPUT_FOLDER)

# --- Helper Functions for AI and GIF Creation ---
# (These are your existing functions and do not need to be changed)
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model (tiny.en)...")
        whisper_model = whisper.load_model("tiny.en")
        print("Whisper model loaded.")
    return whisper_model

def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        print("Loading Sentence Transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence Transformer model loaded.")
    return sentence_model
    
# ... (Paste the rest of your helper functions here: get_existing_transcript, generate_transcript_with_whisper, find_relevant_segments, add_text_to_frame, create_gif_from_segment) ...


# --- NEW DOWNLOAD FUNCTIONS ---

def download_youtube_via_api(youtube_url, video_id):
    if not RAPIDAPI_KEY:
        return None # API Key not configured, skip this method
    
    print("Attempt 1: Trying to download via RapidAPI...")
    api_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
    querystring = {"url": youtube_url}
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
    }
    
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json()

        video_to_download = None
        if data.get("videos") and data["videos"].get("items"):
            for item in data["videos"]["items"]:
                if "mp4" in item.get("mimeType", "") and item.get("height", 0) <= 480:
                    video_to_download = item
                    break
        
        if not video_to_download:
            print("API did not return a suitable video format.")
            return None

        video_url = video_to_download["url"]
        video_path = os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")

        video_response = requests.get(video_url, timeout=120)
        video_response.raise_for_status()

        with open(video_path, 'wb') as f:
            f.write(video_response.content)
        
        print("Video downloaded successfully via API.")
        return video_path
    except Exception as e:
        print(f"!!! RapidAPI download failed: {e}")
        return None

def download_youtube_with_proxy(youtube_url, video_id):
    if not PROXY_URLS:
        return None # No proxies configured, skip this method

    print("Attempt 2: Trying to download via proxy...")
    for proxy in PROXY_URLS:
        try:
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]',
                'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"),
                'merge_output_format': 'mp4',
                'proxy': proxy, 'socket_timeout': 30, 'nocheckcertificate': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                return os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")
        except Exception as e:
            print(f"Proxy failed: {e}")
            continue
    return None

def download_youtube_directly(youtube_url, video_id):
    print("Attempt 3: Trying direct download as a final fallback...")
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]',
            'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"),
            'merge_output_format': 'mp4', 'socket_timeout': 30
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            return os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")
    except Exception as e:
        print(f"Direct download failed: {e}")
    return None

def download_youtube_fallback(youtube_url, video_id):
    # The ultimate fallback function
    
    # Method 1: API
    video_path = download_youtube_via_api(youtube_url, video_id)
    if video_path and os.path.exists(video_path): return video_path
        
    # Method 2: Proxies
    video_path = download_youtube_with_proxy(youtube_url, video_id)
    if video_path and os.path.exists(video_path): return video_path

    # Method 3: Direct
    video_path = download_youtube_directly(youtube_url, video_id)
    if video_path and os.path.exists(video_path): return video_path
    
    return None

# --- Main Processing Logic ---
def process_video_and_generate_gifs(video_path, prompt, video_id):
    # ... (This function remains the same as your previous version) ...
    pass

# --- Flask Routes ---
@app.route('/api/generate-gifs', methods=['POST'])
def generate_gifs_route():
    data = request.get_json()
    prompt, youtube_url = data.get('prompt'), data.get('youtube_url')
    if not prompt or not youtube_url:
        return jsonify({'error': 'Prompt and YouTube URL are required.'}), 400
    
    video_id = None
    video_path = None
    try:
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                video_id = info.get('id')
        except Exception:
            return jsonify({'error': 'Invalid YouTube URL or video is unavailable.'}), 400
        
        if not video_id:
            return jsonify({'error': 'Could not extract video ID from URL'}), 400
        
        video_path = download_youtube_fallback(youtube_url, video_id)
        
        if not video_path:
            return jsonify({'error': 'Failed to download video using all available methods. The video may be private or restricted.'}), 400
        
        gif_paths, error = process_video_and_generate_gifs(video_path, prompt, video_id)
        if error:
            return jsonify({'error': error}), 400
            
        base_url = request.host_url
        gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
        return jsonify({'gifs': gif_urls})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'A critical server error occurred: {e}'}), 500
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

# ... (Your generate_gifs_from_upload_route and other routes remain the same) ...
# ... PASTE YOUR UPLOAD ROUTE AND STATIC FILE ROUTES HERE ...

if __name__ == '__main__':
    app.run(debug=True, port=5000)