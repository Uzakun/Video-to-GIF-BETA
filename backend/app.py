# backend/app.py

import os
import traceback
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
# New imports for the official YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TEMP_VIDEO_FOLDER = 'temp_videos'
GIF_OUTPUT_FOLDER = 'static/gifs'
app.config['UPLOAD_FOLDER'] = TEMP_VIDEO_FOLDER

# --- Global Model Variables (will be loaded on first use) ---
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

# --- Official YouTube API Functions ---
def get_video_details(video_id):
    """Fetches details for a given YouTube video ID using the YouTube Data API."""
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY environment variable not set.")
        return None, "Server is not configured with a YouTube API Key."

    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videos().list(
            part="snippet,contentDetails,status",
            id=video_id
        )
        response = request.execute()

        if not response.get('items'):
            return None, "Video not found or is private."
        
        video_item = response['items'][0]

        # Check for region locks or embeddability issues
        if video_item['status'].get('uploadStatus') != 'processed':
            return None, "Video is still processing or has been deleted."
        if not video_item['status'].get('embeddable'):
            return None, "This video is not embeddable and cannot be processed."

        return video_item, None
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None, f"An API error occurred: {e.resp.status}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, "An unexpected server error occurred while fetching video details."

def download_video_from_url(url, output_path):
    """Downloads a video from a direct URL."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Failed to download video from URL: {e}")
        return None

# --- Core Processing Functions ---
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
        print("Whisper transcription complete.")
        formatted_transcript = [{'text': seg['text'], 'start': seg['start'], 'duration': seg['end'] - seg['start']} for seg in result["segments"]]
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
        
        # Select the top 20 candidates and then randomly sample from them
        scored_segments = sorted(zip(transcript, similarities), key=lambda x: x[1], reverse=True)
        candidate_pool = [segment for segment, score in scored_segments[:20]]

        if not candidate_pool:
            return random.sample(transcript, min(len(transcript), count))
        
        gc.collect()
        return random.sample(candidate_pool, min(len(candidate_pool), count))
    except Exception as e:
        print(f"!!! Local smart search failed: {e}. Falling back to random segments.")
        return random.sample(transcript, min(len(transcript), count))

def add_text_to_frame(frame, text):
    try:
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arialbd.ttf", 48)
        except IOError:
            font = ImageFont.load_default()

        img_width, img_height = pil_image.size
        wrapped_text = textwrap.fill(text, width=int(img_width / 22))
        
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (img_width - text_width) / 2, img_height - text_height - 40

        # Create a black outline
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), wrapped_text, font=font, fill="black")
        
        draw.text((x, y), wrapped_text, font=font, fill="white")
        return np.array(pil_image)
    except Exception as e:
        print(f"!!! Error adding text to frame: {e}")
        return frame

def create_gif_from_segment(video_path, segment, output_filename):
    start_time, duration, text = segment['start'], segment['duration'], segment['text']
    end_time = start_time + min(duration, 5) # GIFs max 5 seconds
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

def process_video_and_generate_gifs(video_path, prompt, video_id):
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
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Error removing video file {video_path}: {e}")
    
    gc.collect()
    return gif_paths, None

# --- API Routes ---
@app.route('/api/generate-gifs', methods=['POST'])
def generate_gifs_route():
    data = request.get_json()
    prompt, youtube_url = data.get('prompt'), data.get('youtube_url')
    if not prompt or not youtube_url:
        return jsonify({'error': 'Prompt and YouTube URL are required.'}), 400
    
    video_path = None
    try:
        # Step 1: Extract video ID from URL
        video_id = None
        if "youtube.com/watch?v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL. Could not extract video ID.'}), 400
        print(f"Extracted video ID: {video_id}")

        # Step 2: Use YouTube Data API to get video details
        details, error = get_video_details(video_id)
        if error:
            return jsonify({'error': error}), 400
        
        # Step 3: Use yt-dlp to get the direct download URL (not for downloading)
        # This is a reliable way to get a streamable URL without downloading directly
        download_url = None
        try:
            ydl_opts = {'format': 'best[ext=mp4][height<=480]', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                download_url = info.get('url')
        except Exception as e:
            print(f"yt-dlp failed to get direct URL: {e}")
            return jsonify({'error': 'Could not retrieve video download link.'}), 400

        if not download_url:
             return jsonify({'error': 'Failed to get a downloadable link for the video.'}), 400

        # Step 4: Download the video from the direct URL
        filename = f"{video_id}.mp4"
        video_path = os.path.join(TEMP_VIDEO_FOLDER, filename)
        if not download_video_from_url(download_url, video_path):
            return jsonify({'error': 'Failed to download video from the retrieved link.'}), 500
        
        # Step 5: Process the downloaded video to create GIFs
        gif_paths, error = process_video_and_generate_gifs(video_path, prompt, video_id)
        if error:
            return jsonify({'error': error}), 400
            
        base_url = request.host_url
        gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
        return jsonify({'gifs': gif_urls})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'A critical server error occurred.'}), 500
    finally:
        # Final cleanup of temp video file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Error removing temp video file {video_path}: {e}")
        gc.collect()

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
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during upload processing.'}), 500
    finally:
       if video_path and os.path.exists(video_path):
           try:
               os.remove(video_path)
           except Exception as e:
               print(f"Error removing temp video file {video_path}: {e}")
       gc.collect()

@app.route('/static/gifs/<filename>')
def serve_gif(filename):
    return send_from_directory(GIF_OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)