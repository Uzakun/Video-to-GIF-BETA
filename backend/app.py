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
import requests

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TEMP_VIDEO_FOLDER = 'temp_videos'
GIF_OUTPUT_FOLDER = 'static/gifs'
app.config['UPLOAD_FOLDER'] = TEMP_VIDEO_FOLDER
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") # Gets key from server environment

# --- Proxy Configuration ---
PROXY_URLS = [
    # Add your Bright Data (or other) residential proxy URL here
    'http://brd-customer-hl_9555f995-zone-residential_proxy1:xjd7xb637zjl@brd.superproxy.io:33335'
]

# --- AI Models (Lazy Loading) ---
whisper_model = None
sentence_model = None

# --- Ensure Folders Exist ---
if not os.path.exists(TEMP_VIDEO_FOLDER):
    os.makedirs(TEMP_VIDEO_FOLDER)
if not os.path.exists(GIF_OUTPUT_FOLDER):
    os.makedirs(GIF_OUTPUT_FOLDER)

# --- Helper Functions for Lazy Model Loading ---
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

# --- Helper Functions for Video Processing ---

def get_existing_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        return transcript.fetch()
    except Exception:
        return None

def generate_transcript_with_whisper(video_path):
    try:
        print("Generating transcript with Whisper...")
        model = get_whisper_model()
        result = model.transcribe(video_path, fp16=False)
        formatted_transcript = []
        for segment in result["segments"]:
            formatted_transcript.append({
                'text': segment['text'],
                'start': segment['start'],
                'duration': segment['end'] - segment['start']
            })
        gc.collect()
        return formatted_transcript
    except Exception as e:
        print(f"!!! Whisper transcription failed: {e}")
        traceback.print_exc()
        return None

def find_relevant_segments(transcript, prompt, count=3):
    try:
        print("Performing local smart search...")
        segment_texts = [s['text'] for s in transcript]
        s_model = get_sentence_model()
        prompt_embedding = s_model.encode([prompt])
        segment_embeddings = s_model.encode(segment_texts)
        similarities = cosine_similarity(prompt_embedding, segment_embeddings)[0]
        scored_segments = sorted(zip(transcript, similarities), key=lambda x: x[1], reverse=True)
        candidate_pool = [segment for segment, score in scored_segments[:20]]
        if not candidate_pool:
            return random.sample(transcript, min(len(transcript), count))
        return random.sample(candidate_pool, min(len(candidate_pool), count))
    except Exception as e:
        print(f"!!! Local smart search failed: {e}.")
        return random.sample(transcript, min(len(transcript), count))
    finally:
        gc.collect()

def add_text_to_frame(frame, text):
    try:
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        # --- MODIFIED: Increased font size significantly ---
        font_size = 80
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        img_width, img_height = pil_image.size
        # --- MODIFIED: Adjusted text wrap for the larger font ---
        wrapped_text = textwrap.fill(text, width=int(img_width / 35))

        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # --- MODIFIED: Adjusted y-position for the larger font ---
        x, y = (img_width - text_width) / 2, img_height - text_height - 60

        # --- REMOVED: The for-loop that created the black outline has been deleted ---

        # This part draws the final white text
        draw.text((x, y), wrapped_text, font=font, fill="white")
        return np.array(pil_image)
    except Exception:
        return frame

def create_gif_from_segment(video_path, segment, output_filename):
    start_time, duration, text = segment['start'], segment['duration'], segment['text']
    end_time = start_time + min(duration, 5)
    video_clip = None
    try:
        video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
        frames = [add_text_to_frame(video_clip.get_frame(t), text) for t in np.arange(0, video_clip.duration, 1/10)]
        if frames:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(output_filename, save_all=True, append_images=pil_frames[1:], duration=100, loop=0, optimize=True)
            return output_filename
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        if video_clip:
            video_clip.close()
        gc.collect()

def process_video_and_generate_gifs(video_path, prompt, video_id=None):
    transcript = get_existing_transcript(video_id) if video_id else None
    if not transcript:
        transcript = generate_transcript_with_whisper(video_path)
    if not transcript:
        if os.path.exists(video_path): os.remove(video_path)
        return None, "Could not get or generate a transcript for this video."
    segments = find_relevant_segments(transcript, prompt, count=3)
    if not segments:
        if os.path.exists(video_path): os.remove(video_path)
        return None, "No relevant segments found for that prompt."
    gif_paths = []
    output_prefix = video_id if video_id else f"upload_{uuid.uuid4().hex[:6]}"
    for i, segment in enumerate(segments):
        unique_id = uuid.uuid4().hex[:8]
        output_filename = os.path.join(GIF_OUTPUT_FOLDER, f"{output_prefix}_{i}_{unique_id}.gif")
        gif_path = create_gif_from_segment(video_path, segment, output_filename)
        if gif_path:
            gif_paths.append(gif_path)
    if os.path.exists(video_path):
        os.remove(video_path)
    return gif_paths, None

# --- DOWNLOAD FUNCTIONS ---

def download_youtube_via_api(youtube_url, video_id):
    if not RAPIDAPI_KEY:
        return None
    print("Attempt 1: Trying to download via RapidAPI...")
    api_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
    querystring = {"url": youtube_url}
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"}
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
        if not video_to_download: return None
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
        return None
    print("Attempt 2: Trying to download via proxy...")
    for proxy in PROXY_URLS:
        try:
            ydl_opts = {'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]', 'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"), 'merge_output_format': 'mp4', 'proxy': proxy, 'socket_timeout': 30, 'nocheckcertificate': True}
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
        ydl_opts = {'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]', 'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"), 'merge_output_format': 'mp4', 'socket_timeout': 30}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            return os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")
    except Exception as e:
        print(f"Direct download failed: {e}")
    return None

def download_youtube_fallback(youtube_url, video_id):
    video_path = download_youtube_via_api(youtube_url, video_id)
    if video_path and os.path.exists(video_path): return video_path
    video_path = download_youtube_with_proxy(youtube_url, video_id)
    if video_path and os.path.exists(video_path): return video_path
    video_path = download_youtube_directly(youtube_url, video_id)
    if video_path and os.path.exists(video_path): return video_path
    return None

# --- Main Flask Routes ---
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

@app.route('/api/generate-gifs-from-upload', methods=['POST'])
def generate_gifs_from_upload_route():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    file, prompt = request.files['video'], request.form.get('prompt')
    if file.filename == '' or not prompt:
        return jsonify({'error': 'No selected file or no prompt provided'}), 400
    video_path = None
    try:
        filename = f"upload_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        gif_paths, error = process_video_and_generate_gifs(video_path, prompt, video_id=None)
        if error:
            return jsonify({'error': error}), 400
        base_url = request.host_url
        gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
        return jsonify({'gifs': gif_urls})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during upload processing.'}), 500
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

@app.route('/static/gifs/<filename>')
def serve_gif(filename):
    return send_from_directory(GIF_OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)