# backend/app.py

import os
import traceback
import yt_dlp
import numpy as np
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import textwrap
import uuid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gc
import replicate # The Replicate library

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TEMP_VIDEO_FOLDER = 'static/temp_videos' # Needs to be in static to be publicly accessible
GIF_OUTPUT_FOLDER = 'static/gifs'

# --- AI Models ---
# The heavy Whisper model is removed. We only keep the lightweight SentenceTransformer.
sentence_model = None

# --- Ensure Folders Exist ---
if not os.path.exists(TEMP_VIDEO_FOLDER):
    os.makedirs(TEMP_VIDEO_FOLDER)
if not os.path.exists(GIF_OUTPUT_FOLDER):
    os.makedirs(GIF_OUTPUT_FOLDER)

# --- Helper Functions ---

def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        print("Loading Sentence Transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence Transformer model loaded.")
    return sentence_model

def get_transcript_from_replicate(public_video_url):
    """
    Calls the Replicate API to get a transcript from a public video URL.
    """
    try:
        print(f"Calling Replicate Whisper API for URL: {public_video_url}")
        # This is a popular, efficient Whisper model on Replicate
        output = replicate.run(
            "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87c558621c89ec3",
            input={"audio": public_video_url, "language": "en"}
        )
        print("Received response from Replicate.")
        
        # Reformat the output to match what the rest of our code expects
        formatted_transcript = []
        for segment in output["segments"]:
             formatted_transcript.append({
                'text': segment['text'],
                'start': segment['start'],
                'duration': segment['end'] - segment['start']
            })
        return formatted_transcript
    except Exception as e:
        print(f"!!! Replicate API failed: {e}")
        return None

def find_relevant_segments(transcript, prompt, count=3):
    # This function is fast and lightweight, so we keep it on our server.
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
        print(f"!!! Local smart search failed: {e}. Falling back.")
        return random.sample(transcript, min(len(transcript), count))


def add_text_to_frame(frame, text):
    try:
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        font_size = 48
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        img_width, img_height = pil_image.size
        wrapped_text = textwrap.fill(text, width=int(img_width / 22))

        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (img_width - text_width) / 2, img_height - text_height - 40

        outline_range = 3
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
    start_time, duration, text = segment['start'], segment['duration'], segment['text']
    end_time = start_time + min(duration, 5)
    
    try:
        with VideoFileClip(video_path).subclip(start_time, end_time) as video_clip:
            frames = [add_text_to_frame(video_clip.get_frame(t), text) for t in np.arange(0, video_clip.duration, 1/10)]
            if frames:
                pil_frames = [Image.fromarray(f) for f in frames]
                pil_frames[0].save(output_filename, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
                return output_filename
        return None
    except Exception:
        traceback.print_exc()
        return None


def process_video_and_generate_gifs(video_path, prompt):
    # --- MODIFIED LOGIC ---
    # 1. Create a public URL for the uploaded video
    base_url = request.host_url
    video_filename = os.path.basename(video_path)
    public_video_url = f"{base_url}static/temp_videos/{video_filename}"
    
    # 2. Offload the heavy transcription to Replicate
    transcript = get_transcript_from_replicate(public_video_url)

    if not transcript:
        if os.path.exists(video_path): os.remove(video_path)
        return None, "Could not generate a transcript using the AI service."

    # 3. Find segments locally (this is fast)
    segments = find_relevant_segments(transcript, prompt, count=3)
    if not segments:
        if os.path.exists(video_path): os.remove(video_path)
        return None, "No relevant segments found for that prompt."

    # 4. Create GIFs locally (this is fast)
    gif_paths = []
    output_prefix = f"upload_{uuid.uuid4().hex[:6]}"
    for i, segment in enumerate(segments):
        unique_id = uuid.uuid4().hex[:8]
        output_filename = os.path.join(GIF_OUTPUT_FOLDER, f"{output_prefix}_{i}_{unique_id}.gif")
        
        gif_path = create_gif_from_segment(video_path, segment, output_filename)
        if gif_path:
            gif_paths.append(gif_path)
            
    # 5. Clean up the temporary video file
    if os.path.exists(video_path): os.remove(video_path)
    return gif_paths, None

# --- Flask Routes ---

@app.route('/api/generate-gifs', methods=['POST'])
def generate_gifs_route():
    # To use this route, you would need a similar flow: 
    # 1. Get the public video URL from yt-dlp
    # 2. Send that public URL to get_transcript_from_replicate
    # 3. Proceed as normal.
    return jsonify({'error': 'YouTube URL processing is not yet implemented in this version.'}), 400

@app.route('/api/generate-gifs-from-upload', methods=['POST'])
def generate_gifs_from_upload_route():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file, prompt = request.files['video'], request.form.get('prompt')
    if file.filename == '' or not prompt:
        return jsonify({'error': 'No selected file or no prompt provided'}), 400

    video_path = None
    try:
        if file:
            filename = f"upload_{uuid.uuid4().hex}.mp4"
            # The temp video MUST be in a publicly accessible folder
            video_path = os.path.join(TEMP_VIDEO_FOLDER, filename)
            file.save(video_path)
            
            gif_paths, error = process_video_and_generate_gifs(video_path, prompt)
            
            if error:
                return jsonify({'error': error}), 400
                
            base_url = request.host_url
            gif_urls = [f"{base_url}static/gifs/{os.path.basename(path)}" for path in gif_paths]
            return jsonify({'gifs': gif_urls})
    except Exception as e:
        traceback.print_exc()
        if video_path and os.path.exists(video_path): os.remove(video_path)
        return jsonify({'error': 'A critical server error occurred.'}), 500
    
    return jsonify({'error': 'An unknown error occurred during upload.'}), 500

# Add a route to serve the temporary videos
@app.route('/static/temp_videos/<filename>')
def serve_temp_video(filename):
    return send_from_directory(TEMP_VIDEO_FOLDER, filename)

@app.route('/static/gifs/<filename>')
def serve_gif(filename):
    return send_from_directory(GIF_OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)