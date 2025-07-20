# backend/app.py

import os
import traceback
import yt_dlp
import cv2
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
import json

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TEMP_VIDEO_FOLDER = 'temp_videos'
GIF_OUTPUT_FOLDER = 'static/gifs'
app.config['UPLOAD_FOLDER'] = TEMP_VIDEO_FOLDER

# --- Global Model Variables (will be loaded on first use) ---
whisper_model = None
sentence_model = None

# --- Proxy Configuration ---
# You can use free proxy services or set up your own
PROXY_URLS = [
    # Add more proxy URLs as needed
    # Format: 'http://username:password@proxy-server:port' or 'http://proxy-server:port'
    # You can get free proxies from services like ProxyScrape, ProxyList, etc.
    # Or use a paid service like Bright Data, Smartproxy, etc.
]

# --- Alternative: Use a YouTube downloader API service ---
# Services like RapidAPI offer YouTube downloader endpoints
YOUTUBE_API_SERVICE = {
    'enabled': True,  # Set to True to use API service instead of direct yt-dlp
    'api_url': 'https://youtube-media-downloader.p.rapidapi.com/v2/video/details',
    'api_key': '35026510f3msh35c1dd768d79849p137ccajsnca36f4120cdf',  # Your actual API key
    'api_host': 'youtube-media-downloader.p.rapidapi.com'
}

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
        formatted_transcript = []
        for segment in result["segments"]:
            formatted_transcript.append({
                'text': segment['text'],
                'start': segment['start'],
                'duration': segment['end'] - segment['start']
            })
        del model
        gc.collect()
        return formatted_transcript
    except Exception as e:
        print(f"!!! Whisper transcription failed: {e}")
        return None

def find_relevant_segments(transcript, prompt, count=3):
    try:
        print("Performing local smart search...")
        segment_texts = [s['text'] if isinstance(s, dict) else s.text for s in transcript]
        
        s_model = get_sentence_model()
        prompt_embedding = s_model.encode([prompt])
        segment_embeddings = s_model.encode(segment_texts)
        
        similarities = cosine_similarity(prompt_embedding, segment_embeddings)[0]
        
        scored_segments = []
        for i, segment in enumerate(transcript):
            scored_segments.append({'segment': segment, 'score': similarities[i]})
        
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        
        candidate_pool = [s['segment'] for s in scored_segments[:20]]
        
        if not candidate_pool:
            print("Smart search found no matches, falling back to random segments.")
            return random.sample(transcript, min(len(transcript), count))

        print(f"Found {len(candidate_pool)} potential segments. Randomly selecting {count}.")
        del s_model, prompt_embedding, segment_embeddings, similarities
        gc.collect()
        return random.sample(candidate_pool, min(len(candidate_pool), count))

    except Exception as e:
        print(f"!!! Local smart search failed: {e}. Falling back to random segments.")
        return random.sample(transcript, min(len(transcript), count))

def add_text_to_frame(frame, text):
    try:
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        font_size = 40
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arialbd.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

        img_width, img_height = pil_image.size
        wrapped_text = textwrap.fill(text, width=int(img_width / 25))

        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (img_width - text_width) / 2, img_height - text_height - 30

        outline_range = 2
        for dx in range(-outline_range, outline_range + 1):
            for dy in range(-outline_range, outline_range + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), wrapped_text, font=font, fill="black")

        draw.text((x, y), wrapped_text, font=font, fill="white")
        return np.array(pil_image)
    except Exception as e:
        print(f"!!! Error adding text to frame: {e}")
        return frame

def create_gif_from_segment(video_path, segment, output_filename):
    if isinstance(segment, dict):
        start_time, duration, text = segment['start'], segment['duration'], segment['text']
    else:
        start_time, duration, text = segment.start, segment.duration, segment.text

    end_time = start_time + min(duration, 5)
    
    video_clip = None
    try:
        video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
        frames = [add_text_to_frame(video_clip.get_frame(t), text) for t in np.arange(0, video_clip.duration, 1/5)]
        
        if frames:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(output_filename, save_all=True, append_images=pil_frames[1:], duration=200, loop=0)
            del frames, pil_frames
            gc.collect()
            return output_filename
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        if video_clip:
            video_clip.close()
            del video_clip
            gc.collect()

def download_youtube_with_api(youtube_url, video_id):
    """Download YouTube video using external API service"""
    if not YOUTUBE_API_SERVICE['enabled']:
        return None
    
    try:
        # Extract video ID from URL if needed
        if 'youtube.com/watch?v=' in youtube_url:
            video_id = youtube_url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in youtube_url:
            video_id = youtube_url.split('/')[-1].split('?')[0]
        
        headers = {
            'X-RapidAPI-Key': YOUTUBE_API_SERVICE['api_key'],
            'X-RapidAPI-Host': YOUTUBE_API_SERVICE['api_host']
        }
        
        # This API expects videoId parameter
        params = {'videoId': video_id}
        
        print(f"Calling API with video ID: {video_id}")
        response = requests.get(YOUTUBE_API_SERVICE['api_url'], headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"API Response received, status: {data.get('status', 'unknown')}")
            
            # Check if the response has videos
            if 'videos' in data and data['videos']:
                # Sort videos by quality and find best match <= 480p
                videos = data['videos']
                suitable_videos = [v for v in videos if v.get('height', 0) <= 480 and v.get('url')]
                
                if not suitable_videos:
                    # If no 480p or lower, just take the lowest quality available
                    suitable_videos = [v for v in videos if v.get('url')]
                
                if suitable_videos:
                    # Sort by height to get the best quality within our limit
                    suitable_videos.sort(key=lambda x: x.get('height', 0), reverse=True)
                    selected_video = suitable_videos[0]
                    
                    video_url = selected_video['url']
                    video_path = os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")
                    
                    print(f"Downloading video: {selected_video.get('height', 'unknown')}p quality")
                    
                    # Download with timeout and better error handling
                    video_response = requests.get(video_url, stream=True, timeout=120)
                    video_response.raise_for_status()
                    
                    total_size = int(video_response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(video_path, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    if int(progress) % 10 == 0:  # Print every 10%
                                        print(f"Download progress: {int(progress)}%")
                    
                    print(f"Video downloaded successfully to {video_path}")
                    return video_path
                else:
                    print("No suitable video formats found in API response")
            else:
                print(f"API response missing videos. Response: {json.dumps(data, indent=2)[:500]}")
        else:
            error_msg = response.text[:200] if response.text else "No error message"
            print(f"API request failed with status {response.status_code}: {error_msg}")
        
        return None
    except requests.exceptions.Timeout:
        print("API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error during API request: {str(e)}")
        return None
    except Exception as e:
        print(f"API download failed: {str(e)}")
        traceback.print_exc()
        return None

def download_youtube_with_proxy(youtube_url, video_id):
    """Download YouTube video using proxy"""
    for proxy in PROXY_URLS:
        try:
            print(f"Trying proxy: {proxy}")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]',
                'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"),
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
                'proxy': proxy,
                'socket_timeout': 30,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download([youtube_url])
                if error_code == 0:
                    return os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")
        except Exception as e:
            print(f"Proxy {proxy} failed: {e}")
            continue
    
    return None

def download_youtube_fallback(youtube_url, video_id):
    """Try multiple methods to download YouTube video"""
    # Method 1: Try with API service first
    video_path = download_youtube_with_api(youtube_url, video_id)
    if video_path and os.path.exists(video_path):
        print("Downloaded using API service")
        return video_path
    
    # Method 2: Try with proxies
    if PROXY_URLS:
        video_path = download_youtube_with_proxy(youtube_url, video_id)
        if video_path and os.path.exists(video_path):
            print("Downloaded using proxy")
            return video_path
    
    # Method 3: Try direct download (may work on some VPS)
    try:
        print("Trying direct download...")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]',
            'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            # Additional options that might help
            'geo_bypass': True,
            'geo_bypass_country': 'US',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([youtube_url])
            if error_code == 0:
                return os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4")
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    return None

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

@app.route('/api/generate-gifs', methods=['POST'])
def generate_gifs_route():
    data = request.get_json()
    prompt, youtube_url = data.get('prompt'), data.get('youtube_url')
    if not prompt or not youtube_url:
        return jsonify({'error': 'Prompt and YouTube URL are required.'}), 400
    
    video_id = None
    video_path = None
    try:
        # Extract video ID with better error handling
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                video_id = info.get('id')
                print(f"Extracted video ID: {video_id}")
        except Exception as e:
            print(f"Error extracting video ID: {str(e)}")
            # Try manual extraction as fallback
            if 'youtube.com/watch?v=' in youtube_url:
                video_id = youtube_url.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in youtube_url:
                video_id = youtube_url.split('/')[-1].split('?')[0]
            else:
                return jsonify({'error': 'Invalid YouTube URL format'}), 400
        
        if not video_id:
            return jsonify({'error': 'Could not extract video ID from URL'}), 400
        
        # Use fallback download method
        print(f"Attempting to download video ID: {video_id}")
        video_path = download_youtube_fallback(youtube_url, video_id)
        
        if not video_path or not os.path.exists(video_path):
            # Provide more specific error message
            error_msg = 'Failed to download video. '
            if YOUTUBE_API_SERVICE['enabled']:
                error_msg += 'The API service might be down or the video might be restricted. '
            error_msg += 'Please try using the file upload option instead.'
            return jsonify({'error': error_msg}), 400
        
        print(f"Video downloaded successfully, processing GIFs...")
        gif_paths, error = process_video_and_generate_gifs(video_path, prompt, video_id)
        if error:
            return jsonify({'error': error}), 400
            
        base_url = request.host_url
        gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
        print(f"Successfully generated {len(gif_urls)} GIFs")
        return jsonify({'gifs': gif_urls})
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"!!! Critical error in generate_gifs_route: {str(e)}")
        print(f"Full traceback:\n{error_trace}")
        
        # Return a more informative error for debugging
        return jsonify({
            'error': 'An internal server error occurred while processing your request.',
            'details': str(e) if app.debug else None  # Only show details in debug mode
        }), 500
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Error removing temporary video file {video_path}: {e}")
        gc.collect()

@app.route('/api/generate-gifs-from-upload', methods=['POST'])
def generate_gifs_from_upload_route():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file, prompt = request.files['video'], request.form.get('prompt')
    if file.filename == '' or not prompt:
        return jsonify({'error': 'No selected file or no prompt provided'}), 400

    video_path = None
    if file:
        filename = f"upload_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        gif_paths, error = process_video_and_generate_gifs(video_path, prompt, video_id=None)
        if error:
            return jsonify({'error': error}), 400
            
        base_url = request.host_url
        gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
        return jsonify({'gifs': gif_urls})
    
    return jsonify({'error': 'An unknown error occurred during upload. Please try a smaller video file.'}), 500

@app.route('/static/gifs/<filename>')
def serve_gif(filename):
    return send_from_directory(GIF_OUTPUT_FOLDER, filename)

@app.route('/api/check-youtube-support', methods=['GET'])
def check_youtube_support():
    """Endpoint to check if YouTube downloads are working"""
    return jsonify({
        'api_enabled': YOUTUBE_API_SERVICE['enabled'],
        'proxy_count': len(PROXY_URLS),
        'recommendation': 'Use file upload for best reliability' if not YOUTUBE_API_SERVICE['enabled'] and len(PROXY_URLS) == 0 else 'YouTube URLs should work'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)