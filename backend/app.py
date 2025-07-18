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
import gc # Import garbage collector

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
        # --- CHANGE: Use the smallest English-only model for memory saving ---
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
        # Use the lazily loaded model
        model = get_whisper_model()
        result = model.transcribe(video_path, fp16=False) # fp16=False for CPU compatibility/stability on low-RAM
        print("Whisper transcription complete.")
        formatted_transcript = []
        for segment in result["segments"]:
            formatted_transcript.append({
                'text': segment['text'],
                'start': segment['start'],
                'duration': segment['end'] - segment['start']
            })
        # Try to explicitly free memory after use (may not always work perfectly)
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
        
        # Use the lazily loaded model
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
        # Try to explicitly free memory after use
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
        font_size = 40 # Slightly reduced font size
        try:
            # Prefer system-available fonts
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size) # Common on Linux
        except IOError:
            try:
                font = ImageFont.truetype("arialbd.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default() # Fallback

        img_width, img_height = pil_image.size
        # Adjust text wrapping width based on typical resolutions (e.g., 480p)
        wrapped_text = textwrap.fill(text, width=int(img_width / 25)) # Adjusted wrapping

        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (img_width - text_width) / 2, img_height - text_height - 30 # Adjusted vertical position

        outline_range = 2 # Slightly reduced outline for smaller font
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

    end_time = start_time + min(duration, 5) # Keep max 5 second GIF
    
    video_clip = None # Initialize to None for cleanup
    try:
        video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
        # --- CHANGE: Reduce frame rate for GIF creation to save memory ---
        # Current: 10 FPS (1/10)
        # New: 5 FPS (1/5) - adjust lower if still OOM (e.g., 1/4 for 4 FPS)
        frames = [add_text_to_frame(video_clip.get_frame(t), text) for t in np.arange(0, video_clip.duration, 1/5)]
        
        if frames:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(output_filename, save_all=True, append_images=pil_frames[1:], duration=200, loop=0) # duration=200ms means 5 FPS
            # Try to explicitly free memory
            del frames, pil_frames
            gc.collect()
            return output_filename
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        if video_clip:
            video_clip.close() # Ensure video clip resources are released
            del video_clip
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
            
    # Clean up original video file
    if os.path.exists(video_path): 
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Error removing video file {video_path}: {e}")
    
    # Force garbage collection
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
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info.get('id')
        
        # --- CHANGE: Target a specific low resolution (e.g., 480p) to save memory ---
        # Ensure only a single best quality mp4 stream at 480p or less is downloaded
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]',
            'outtmpl': os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.%(ext)s"),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([youtube_url])
            if error_code != 0:
                raise Exception(f"yt-dlp failed to download video. Error code: {error_code}")
        
        video_path = os.path.join(TEMP_VIDEO_FOLDER, f"{video_id}.mp4") # Assume mp4 after merge
        
        gif_paths, error = process_video_and_generate_gifs(video_path, prompt, video_id)
        if error:
            return jsonify({'error': error}), 400
            
        base_url = request.host_url
        gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
        return jsonify({'gifs': gif_urls})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'A server error occurred. Please try a shorter video.'}), 500
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
        filename = f"upload_{uuid.uuid4().hex}.mp4" # Ensure mp4 extension for consistent processing
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

if __name__ == '__main__':
    # When running locally, models will load on first request
    app.run(debug=True, port=5000)