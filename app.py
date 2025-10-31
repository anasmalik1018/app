from flask import Flask, render_template, request, jsonify, send_file, redirect, session
from flask_cors import CORS
import requests
import base64
import io
import sqlite3
import os
import uuid
from datetime import datetime
import json
import time
import random
import string
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime, timedelta
import urllib.parse
import numpy as np
import threading
import glob
import tempfile

# Fix for Python 3.13 - aifc module removed
import sys
if sys.version_info >= (3, 13):
    import types
    
    # Create a proper mock aifc module with necessary attributes
    class MockAifc:
        def open(self, *args, **kwargs):
            raise NotImplementedError("aifc module is not available in Python 3.13+")
        
        Error = Exception
        
    aifc_module = types.ModuleType('aifc')
    for attr in dir(MockAifc):
        if not attr.startswith('_'):
            setattr(aifc_module, attr, getattr(MockAifc, attr))
    
    sys.modules['aifc'] = aifc_module

# Now import speech_recognition
try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError as e:
    print(f"⚠️  Speech recognition not available: {e}")
    VOICE_ENABLED = False
    sr = None

try:
    from gtts import gTTS
    import pygame
    TTS_ENABLED = True
except ImportError as e:
    print(f"⚠️  Text-to-speech not available: {e}")
    TTS_ENABLED = False
    gTTS = None
    pygame = None

app = Flask(__name__)
CORS(app)

# Add these imports for admin panel
from functools import wraps
import hashlib
import secrets
import shutil
import psutil
import platform

# Admin configurations
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # Change this to a secure password
SECRET_KEY = secrets.token_hex(32)
app.secret_key = SECRET_KEY

# Configuration - Fixed API endpoint
POLLINATION_BASE_URL = "https://image.pollinations.ai"

# Create directories if they don't exist
os.makedirs('static/generated_images', exist_ok=True)
os.makedirs('static/audio', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize pygame mixer for audio playback
if TTS_ENABLED and pygame:
    try:
        pygame.mixer.init()
    except:
        print("⚠️  Could not initialize pygame mixer")
        TTS_ENABLED = False

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return jsonify({"error": "Admin login required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password):
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hashlib.sha256(password.encode()).hexdigest() == hashed

# Admin Authentication Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'GET':
        return render_template('admin_login.html')
    
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_login_time'] = datetime.now().isoformat()
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    session.pop('admin_login_time', None)
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard"""
    if 'admin_logged_in' not in session:
        return redirect('/admin/login')
    return render_template('admin_dashboard.html')

# Admin API Routes
@app.route('/api/admin/stats')
@admin_required
def admin_stats():
    """Get comprehensive system statistics"""
    try:
        # File system stats
        image_files = glob.glob('static/generated_images/*.png')
        audio_files = glob.glob('static/audio/*.mp3')
        
        # Calculate storage usage
        images_size = sum(os.path.getsize(f) for f in image_files) / (1024*1024)  # MB
        audio_size = sum(os.path.getsize(f) for f in audio_files) / (1024*1024)  # MB
        
        # Generation logs
        log_file = "generation_log.json"
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        # Recent activity (last 24 hours)
        recent_activity = []
        if logs:
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_activity = [log for log in logs if 
                             datetime.fromisoformat(log["timestamp"]) > cutoff_time]
        
        # System info
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "memory_used": psutil.virtual_memory().used / (1024**3),   # GB
            "disk_total": psutil.disk_usage('/').total / (1024**3),    # GB
            "disk_used": psutil.disk_usage('/').used / (1024**3),      # GB
        }
        
        return jsonify({
            "success": True,
            "stats": {
                "total_images": len(image_files),
                "total_audio": len(audio_files),
                "images_size_mb": round(images_size, 2),
                "audio_size_mb": round(audio_size, 2),
                "total_generations": len(logs),
                "recent_activity": len(recent_activity),
                "system_info": system_info
            },
            "recent_generations": recent_activity[:10]  # Last 10 generations
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/images')
@admin_required
def admin_list_images():
    """List all generated images with details"""
    try:
        image_folder = 'static/generated_images'
        images = []
        
        if os.path.exists(image_folder):
            for filename in os.listdir(image_folder):
                if filename.endswith('.png'):
                    file_path = os.path.join(image_folder, filename)
                    stat = os.stat(file_path)
                    
                    images.append({
                        "filename": filename,
                        "size_mb": round(stat.st_size / (1024*1024), 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "url": f"/static/generated_images/{filename}"
                    })
        
        # Sort by creation time (newest first)
        images.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "success": True,
            "images": images,
            "total_count": len(images)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/delete-image/<image_id>', methods=['DELETE'])
@admin_required
def admin_delete_image(image_id):
    """Delete specific image"""
    try:
        # Sanitize filename
        image_id = os.path.basename(image_id)
        image_path = f"static/generated_images/{image_id}"
        
        if os.path.exists(image_path):
            os.remove(image_path)
            return jsonify({"success": True, "message": f"Image {image_id} deleted"})
        else:
            return jsonify({"success": False, "error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/cleanup', methods=['POST'])
@admin_required
def admin_cleanup():
    """Manual cleanup of old files"""
    try:
        data = request.get_json()
        cleanup_type = data.get('type', 'all')  # 'images', 'audio', 'all'
        max_age_hours = int(data.get('max_age_hours', 24))
        
        deleted_files = {"images": 0, "audio": 0}
        current_time = datetime.now()
        
        if cleanup_type in ['images', 'all']:
            # Cleanup images
            image_files = glob.glob('static/generated_images/*.png')
            for image_file in image_files:
                file_time = datetime.fromtimestamp(os.path.getmtime(image_file))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    os.remove(image_file)
                    deleted_files["images"] += 1
        
        if cleanup_type in ['audio', 'all']:
            # Cleanup audio
            audio_files = glob.glob('static/audio/*.mp3')
            for audio_file in audio_files:
                file_time = datetime.fromtimestamp(os.path.getmtime(audio_file))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    os.remove(audio_file)
                    deleted_files["audio"] += 1
        
        return jsonify({
            "success": True,
            "deleted_files": deleted_files,
            "message": f"Cleanup completed. Deleted {deleted_files['images']} images and {deleted_files['audio']} audio files"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    """Get or update admin settings"""
    settings_file = 'admin_settings.json'
    
    if request.method == 'GET':
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
            else:
                settings = {
                    "max_images_per_request": 4,
                    "max_image_size": 2048,
                    "auto_cleanup_hours": 24,
                    "enable_voice": VOICE_ENABLED,
                    "default_model": "flux",
                    "default_quality": "standard"
                }
            
            return jsonify({"success": True, "settings": settings})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    elif request.method == 'POST':
        try:
            new_settings = request.get_json()
            
            # Validate settings
            if new_settings.get('max_images_per_request', 0) > 10:
                return jsonify({"success": False, "error": "Max images per request cannot exceed 10"}), 400
            
            # Save settings
            with open(settings_file, 'w') as f:
                json.dump(new_settings, f, indent=2)
            
            return jsonify({"success": True, "message": "Settings updated successfully"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/logs')
@admin_required
def admin_logs():
    """Get generation logs with pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        log_file = "generation_log.json"
        logs = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_logs = logs[start_idx:end_idx]
        
        return jsonify({
            "success": True,
            "logs": paginated_logs,
            "total_count": len(logs),
            "page": page,
            "per_page": per_page,
            "total_pages": (len(logs) + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/backup', methods=['POST'])
@admin_required
def admin_backup():
    """Create backup of important files"""
    try:
        backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup generation logs
        if os.path.exists('generation_log.json'):
            shutil.copy2('generation_log.json', backup_dir)
        
        # Backup settings
        if os.path.exists('admin_settings.json'):
            shutil.copy2('admin_settings.json', backup_dir)
        
        # Backup recent images (last 24 hours)
        image_backup_dir = os.path.join(backup_dir, 'images')
        os.makedirs(image_backup_dir, exist_ok=True)
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        image_files = glob.glob('static/generated_images/*.png')
        
        backed_up_images = 0
        for image_file in image_files:
            file_time = datetime.fromtimestamp(os.path.getmtime(image_file))
            if file_time > cutoff_time:
                shutil.copy2(image_file, image_backup_dir)
                backed_up_images += 1
        
        # Create backup info file
        backup_info = {
            "created": datetime.now().isoformat(),
            "files_backed_up": {
                "logs": 1 if os.path.exists('generation_log.json') else 0,
                "settings": 1 if os.path.exists('admin_settings.json') else 0,
                "images": backed_up_images
            }
        }
        
        with open(os.path.join(backup_dir, 'backup_info.json'), 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        return jsonify({
            "success": True,
            "backup_dir": backup_dir,
            "backup_info": backup_info,
            "message": f"Backup created successfully in {backup_dir}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/users')
@admin_required
def admin_users():
    """Get user activity statistics"""
    try:
        log_file = "generation_log.json"
        logs = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        # Analyze user activity
        daily_activity = {}
        hourly_activity = {}
        
        for log in logs:
            timestamp = datetime.fromisoformat(log['timestamp'])
            date_key = timestamp.strftime('%Y-%m-%d')
            hour_key = timestamp.strftime('%H')
            
            daily_activity[date_key] = daily_activity.get(date_key, 0) + 1
            hourly_activity[hour_key] = hourly_activity.get(hour_key, 0) + 1
        
        return jsonify({
            "success": True,
            "daily_activity": daily_activity,
            "hourly_activity": hourly_activity,
            "total_sessions": len(logs),
            "unique_days": len(daily_activity)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/system-info')
@admin_required
def admin_system_info():
    """Get detailed system information"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get running processes (simplified)
        process_count = len(psutil.pids())
        
        # Get network info
        network = psutil.net_io_counters()
        
        return jsonify({
            "success": True,
            "system_info": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process_count": process_count
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/audio')
@admin_required
def admin_list_audio():
    """List all generated audio files with details"""
    try:
        audio_folder = 'static/audio'
        audio_files = []
        
        if os.path.exists(audio_folder):
            for filename in os.listdir(audio_folder):
                if filename.endswith('.mp3'):
                    file_path = os.path.join(audio_folder, filename)
                    stat = os.stat(file_path)
                    
                    audio_files.append({
                        "filename": filename,
                        "size_mb": round(stat.st_size / (1024*1024), 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "url": f"/static/audio/{filename}"
                    })
        
        # Sort by creation time (newest first)
        audio_files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "success": True,
            "audio_files": audio_files,
            "total_count": len(audio_files)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/delete-audio/<audio_id>', methods=['DELETE'])
@admin_required
def admin_delete_audio(audio_id):
    """Delete specific audio file"""
    try:
        # Sanitize filename
        audio_id = os.path.basename(audio_id)
        audio_path = f"static/audio/{audio_id}"
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            return jsonify({"success": True, "message": f"Audio file {audio_id} deleted"})
        else:
            return jsonify({"success": False, "error": "Audio file not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/gallery')
def gallery():
    try:
        image_folder = os.path.join('static', 'generated_images')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        images.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)), reverse=True)
        
        return render_template('gallery.html', images=images)
    except Exception as e:
        print(f"❌ Error in gallery route: {str(e)}")
        return render_template('gallery.html', images=[])

class VoiceHandler:
    def __init__(self):
        if not VOICE_ENABLED or sr is None:
            print("⚠️  Voice recognition disabled")
            self.recognizer = None
            self.microphone = None
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            print("🎤 Adjusting microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            print("✅ Microphone setup complete")
        except Exception as e:
            print(f"⚠️  Could not initialize voice handler: {e}")
            self.recognizer = None
            self.microphone = None
    
    def speech_to_text(self, audio_data=None, language='en-US'):
        """Convert speech to text"""
        if not VOICE_ENABLED or self.recognizer is None:
            return {"success": False, "error": "Voice recognition not available"}
            
        try:
            if audio_data is None:
                # Record from microphone
                with self.microphone as source:
                    print("🎤 Listening... Speak now!")
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
            else:
                # Use provided audio data
                audio = audio_data
            
            print("🔄 Processing speech...")
            
            # Try Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio, language=language)
                print(f"✅ Speech recognized: {text}")
                return {"success": True, "text": text}
            except sr.UnknownValueError:
                return {"success": False, "error": "Could not understand audio"}
            except sr.RequestError as e:
                return {"success": False, "error": f"Speech recognition error: {str(e)}"}
                    
        except Exception as e:
            return {"success": False, "error": f"Voice processing error: {str(e)}"}
    
    def text_to_speech(self, text, language='en', slow=False):
        """Convert text to speech"""
        if not TTS_ENABLED or gTTS is None:
            return {"success": False, "error": "Text-to-speech not available"}
            
        try:
            if not text.strip():
                return {"success": False, "error": "No text to convert"}
            
            # Create TTS object
            tts = gTTS(text=text, lang=language, slow=slow)
            
            # Save to temporary file
            audio_id = self.generate_audio_id()
            audio_path = f"static/audio/{audio_id}.mp3"
            tts.save(audio_path)
            
            print(f"🔊 Audio saved: {audio_path}")
            
            return {
                "success": True,
                "audio_url": f"/static/audio/{audio_id}.mp3",
                "audio_id": audio_id,
                "text": text
            }
            
        except Exception as e:
            return {"success": False, "error": f"TTS error: {str(e)}"}
    
    def play_audio(self, audio_path):
        """Play audio file"""
        if not TTS_ENABLED or pygame is None:
            return {"success": False, "error": "Audio playback not available"}
            
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return {"success": True, "message": "Audio played successfully"}
        except Exception as e:
            return {"success": False, "error": f"Audio playback error: {str(e)}"}
    
    def generate_audio_id(self):
        """Generate unique audio ID"""
        timestamp = str(int(time.time()))
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"voice_{timestamp}_{random_str}"

class LAHgenAPI:
    def __init__(self):
        self.base_url = POLLINATION_BASE_URL
        
    def generate_multiple_images(self, prompt, negative_prompt="", style="auto", width=1024, height=1024, 
                                seed=None, enhance_prompt=True, quality="standard", model="flux", 
                                remove_watermark=True, num_images=1):
        """Generate multiple images using Pollination.ai API"""
        results = []
        base_seed = seed if seed else random.randint(1, 1000000)
        
        for i in range(num_images):
            current_seed = base_seed + i
            
            result = self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                style=style,
                width=width,
                height=height,
                seed=current_seed,
                enhance_prompt=enhance_prompt,
                quality=quality,
                model=model,
                remove_watermark=remove_watermark
            )
            
            results.append(result)
        
        return results
        
    def generate_image(self, prompt, negative_prompt="", style="auto", width=1024, height=1024, 
                      seed=None, enhance_prompt=True, quality="standard", model="flux", remove_watermark=True):
        """Generate image using Pollination.ai API"""
        try:
            # Clean and prepare prompt
            prompt = prompt.strip()
            if not prompt:
                return {"success": False, "error": "Empty prompt"}
            
            # Prepare the prompt with style
            if style != "auto" and enhance_prompt:
                style_prompts = {
                    "realistic": "photorealistic, high detail, professional photography",
                    "anime": "anime style, manga, japanese art style",
                    "digital-art": "digital art, concept art, artistic",
                    "3d": "3D render, CGI, three dimensional",
                    "cinematic": "cinematic lighting, movie scene, dramatic"
                }
                if style in style_prompts:
                    prompt = f"{prompt}, {style_prompts[style]}"
            
            # Add quality enhancement
            if quality == "high":
                prompt += ", high quality, detailed"
            elif quality == "ultra":
                prompt += ", ultra high quality, extremely detailed, masterpiece"
            
            # Add watermark removal keywords to prompt
            if remove_watermark:
                prompt += ", no watermark, clean image, no logo, no text"
            
            # Generate seed if not provided
            if seed is None:
                seed = random.randint(1, 1000000)
            
            # Prepare API request
            encoded_prompt = urllib.parse.quote(prompt)
            api_url = f"{self.base_url}/prompt/{encoded_prompt}"
            
            params = {
                "width": min(width, 2048),
                "height": min(height, 2048),
                "seed": seed,
                "model": model,
                "enhance": str(enhance_prompt).lower(),
                "nologo": "true"
            }
            
            # Add negative prompt
            if negative_prompt:
                negative_prompt += ", watermark, logo, text, signature"
                params["negative"] = negative_prompt
            else:
                params["negative"] = "watermark, logo, text, signature"
            
            print(f"🎨 Generating image...")
            print(f"📝 Prompt: {prompt[:100]}...")
            
            # Make API request with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(api_url, params=params, timeout=120)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        
                        if content_type.startswith('image/'):
                            # Save image
                            image_id = self.generate_image_id()
                            image_path = f"static/generated_images/{image_id}.png"
                            
                            with open(image_path, 'wb') as f:
                                f.write(response.content)
                            
                            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                print(f"✅ Image saved: {image_path}")
                                
                                if remove_watermark:
                                    self.remove_watermark_from_image(image_path, image_path)
                                
                                return {
                                    "success": True,
                                    "image_url": f"/static/generated_images/{image_id}.png",
                                    "image_id": image_id,
                                    "seed": seed
                                }
                            else:
                                return {"success": False, "error": "Failed to save image"}
                        else:
                            return {"success": False, "error": f"Invalid response type: {content_type}"}
                    else:
                        print(f"❌ API Error {response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return {"success": False, "error": f"API error {response.status_code}"}
                        
                except requests.exceptions.Timeout:
                    print(f"⏱️  Request timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {"success": False, "error": "Request timeout"}
                except requests.exceptions.RequestException as e:
                    print(f"🌐 Network error: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {"success": False, "error": f"Network error: {str(e)}"}
            
            return {"success": False, "error": "Max retries exceeded"}
                
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def remove_watermark_from_image(self, input_path, output_path):
        """Remove watermark from image using image processing techniques"""
        try:
            img = Image.open(input_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            
            # Common watermark locations
            watermark_regions = [
                (int(width * 0.7), int(height * 0.85), width, height),
                (0, int(height * 0.85), int(width * 0.3), height),
                (int(width * 0.7), 0, width, int(height * 0.15)),
                (0, 0, int(width * 0.3), int(height * 0.15)),
                (int(width * 0.3), int(height * 0.9), int(width * 0.7), height)
            ]
            
            processed_img = img.copy()
            
            for region in watermark_regions:
                x1, y1, x2, y2 = region
                if x2 > x1 and y2 > y1:
                    region_img = img.crop((x1, y1, x2, y2))
                    blurred_region = region_img.filter(ImageFilter.GaussianBlur(radius=1.5))
                    processed_img.paste(blurred_region, (x1, y1))
            
            processed_img.save(output_path, 'PNG', quality=95)
            return True
            
        except Exception as e:
            print(f"⚠️  Error removing watermark: {str(e)}")
            return False
    
    def generate_image_id(self):
        """Generate unique image ID"""
        timestamp = str(int(time.time()))
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"lahgen_{timestamp}_{random_str}"

class ImageCleanup:
    def __init__(self):
        self.cleanup_interval = 3600  # 1 hour
        self.max_age_hours = 24  # Keep images for 24 hours
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    self.cleanup_old_images()
                    self.cleanup_old_audio()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    print(f"🧹 Cleanup error: {str(e)}")
                    time.sleep(self.cleanup_interval)
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def cleanup_old_images(self):
        """Remove old generated images"""
        try:
            image_dir = 'static/generated_images'
            current_time = datetime.now()
            deleted_count = 0
            
            image_files = glob.glob(os.path.join(image_dir, '*.png'))
            
            for image_file in image_files:
                file_time = datetime.fromtimestamp(os.path.getmtime(image_file))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > self.max_age_hours:
                    os.remove(image_file)
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old images")
                
        except Exception as e:
            print(f"🧹 Error during image cleanup: {str(e)}")
    
    def cleanup_old_audio(self):
        """Remove old audio files"""
        try:
            audio_dir = 'static/audio'
            current_time = datetime.now()
            deleted_count = 0
            
            audio_files = glob.glob(os.path.join(audio_dir, '*.mp3'))
            
            for audio_file in audio_files:
                file_time = datetime.fromtimestamp(os.path.getmtime(audio_file))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > self.max_age_hours:
                    os.remove(audio_file)
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old audio files")
                
        except Exception as e:
            print(f"🧹 Error during audio cleanup: {str(e)}")

# Initialize handlers
voice_handler = VoiceHandler()
lahgen_api = LAHgenAPI()
image_cleanup = ImageCleanup()

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template error: {str(e)}")
        return f"""
        <html>
        <head><title>LAHgen - Image Generator</title></head>
        <body>
            <h1>LAHgen - Image Generator</h1>
            <p>Voice Enabled: {'Yes' if VOICE_ENABLED else 'No'}</p>
            <p>Template error: {str(e)}</p>
            <p>Please make sure the templates/index.html file exists.</p>
        </body>
        </html>
        """, 500

@app.route('/api/voice/listen', methods=['POST'])
def voice_listen():
    """Listen to voice input and convert to text"""
    if not VOICE_ENABLED:
        return jsonify({"success": False, "error": "Voice recognition not available"}), 503
        
    try:
        data = request.get_json()
        language = data.get('language', 'en-US')
        
        print("🎤 Starting voice recognition...")
        result = voice_handler.speech_to_text(language=language)
        
        if result["success"]:
            # Also create audio response
            response_text = f"I heard: {result['text']}"
            tts_result = voice_handler.text_to_speech(response_text, language=language[:2])
            
            return jsonify({
                "success": True,
                "text": result["text"],
                "audio_response": tts_result if tts_result["success"] else None
            })
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/voice/upload', methods=['POST'])
def voice_upload():
    """Upload audio file and convert to text"""
    if not VOICE_ENABLED:
        return jsonify({"success": False, "error": "Voice recognition not available"}), 503
        
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en-US')
        
        if audio_file.filename == '':
            return jsonify({"success": False, "error": "No audio file selected"}), 400
        
        # Save uploaded file temporarily
        temp_path = tempfile.mktemp(suffix='.wav')
        audio_file.save(temp_path)
        
        # Process audio file
        try:
            with sr.AudioFile(temp_path) as source:
                audio_data = voice_handler.recognizer.record(source)
            
            result = voice_handler.speech_to_text(audio_data, language=language)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if result["success"]:
                return jsonify({
                    "success": True,
                    "text": result["text"]
                })
            else:
                return jsonify(result), 400
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/voice/speak', methods=['POST'])
def voice_speak():
    """Convert text to speech"""
    if not TTS_ENABLED:
        return jsonify({"success": False, "error": "Text-to-speech not available"}), 503
        
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        slow = data.get('slow', False)
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        result = voice_handler.text_to_speech(text, language=language, slow=slow)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/voice/generate', methods=['POST'])
def voice_generate():
    """Voice-to-image generation (complete workflow)"""
    if not VOICE_ENABLED:
        return jsonify({"success": False, "error": "Voice recognition not available"}), 503
        
    try:
        data = request.get_json()
        
        # First, get voice input
        voice_language = data.get('voice_language', 'en-US')
        print("🎤 Listening for image generation prompt...")
        
        # Listen for voice input
        voice_result = voice_handler.speech_to_text(language=voice_language)
        
        if not voice_result["success"]:
            return jsonify({
                "success": False,
                "error": "Could not understand voice input",
                "voice_error": voice_result["error"]
            }), 400
        
        prompt = voice_result["text"]
        print(f"🎨 Voice prompt received: {prompt}")
        
        # Generate image with voice prompt
        image_params = {
            'negative_prompt': data.get('negative_prompt', ''),
            'style': data.get('style', 'auto'),
            'model': data.get('model', 'flux'),
            'num_images': min(int(data.get('num_images', 1)), 4),
            'width': int(data.get('width', 1024)),
            'height': int(data.get('height', 1024)),
            'seed': data.get('seed'),
            'enhance_prompt': data.get('enhance_prompt', True),
            'quality': data.get('quality', 'standard'),
            'remove_watermark': data.get('remove_watermark', True)
        }
        
        # Generate images
        results = lahgen_api.generate_multiple_images(
            prompt=prompt,
            **image_params
        )
        
        # Process results
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        if successful_results:
            # Create voice confirmation
            response_text = f"I've generated {len(successful_results)} image{'s' if len(successful_results) > 1 else ''} based on your prompt: {prompt}"
            tts_result = voice_handler.text_to_speech(response_text, language=voice_language[:2])
            
            # Log generation
            for result in successful_results:
                log_generation(prompt, result["image_id"], result["seed"])
            
            return jsonify({
                "success": True,
                "voice_prompt": prompt,
                "images": successful_results,
                "failed_count": len(failed_results),
                "total_generated": len(successful_results),
                "voice_response": tts_result if tts_result["success"] else None
            })
        else:
            error_text = f"Sorry, I couldn't generate images from your prompt: {prompt}"
            tts_result = voice_handler.text_to_speech(error_text, language=voice_language[:2])
            
            return jsonify({
                "success": False,
                "voice_prompt": prompt,
                "error": "Failed to generate images",
                "failed_results": failed_results,
                "voice_response": tts_result if tts_result["success"] else None
            }), 500
            
    except Exception as e:
        error_text = f"Sorry, there was an error processing your request: {str(e)}"
        try:
            tts_result = voice_handler.text_to_speech(error_text)
        except:
            tts_result = None
        
        return jsonify({
            "success": False,
            "error": str(e),
            "voice_response": tts_result
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Handle image generation requests"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        # Extract parameters
        prompt = data.get('prompt', '').strip()
        negative_prompt = data.get('negative_prompt', '')
        style = data.get('style', 'auto')
        model = data.get('model', 'flux')
        num_images = min(int(data.get('num_images', 1)), 4)
        
        # Image dimensions
        size = data.get('size', '1024x1024')
        if size == 'custom':
            width = int(data.get('width', 1024))
            height = int(data.get('height', 1024))
        else:
            try:
                width, height = map(int, size.split('x'))
            except:
                width, height = 1024, 1024
        
        # Other parameters
        seed = data.get('seed')
        if seed:
            try:
                seed = int(seed)
            except:
                seed = None
        
        enhance_prompt = data.get('enhance_prompt', True)
        quality = data.get('quality', 'standard')
        remove_watermark = data.get('remove_watermark', True)
        
        # Check for voice response request
        voice_response = data.get('voice_response', False)
        voice_language = data.get('voice_language', 'en')
        
        # Validate prompt
        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"}), 400
        
        print(f"🎨 Generating {num_images} images with prompt: {prompt[:100]}...")
        
        # Generate images
        results = lahgen_api.generate_multiple_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            width=width,
            height=height,
            seed=seed,
            enhance_prompt=enhance_prompt,
            quality=quality,
            model=model,
            remove_watermark=remove_watermark,
            num_images=num_images
        )
        
        # Process results
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        response_data = {
            "success": len(successful_results) > 0,
            "images": successful_results,
            "failed_count": len(failed_results),
            "total_generated": len(successful_results),
            "prompt": prompt
        }
        
        # Add voice response if requested
        if voice_response and successful_results and TTS_ENABLED:
            response_text = f"Successfully generated {len(successful_results)} image{'s' if len(successful_results) > 1 else ''} for: {prompt}"
            tts_result = voice_handler.text_to_speech(response_text, language=voice_language)
            if tts_result["success"]:
                response_data["voice_response"] = tts_result
        
        if successful_results:
            # Log generation
            for result in successful_results:
                log_generation(prompt, result["image_id"], result["seed"])
            
            return jsonify(response_data)
        else:
            error_messages = [r.get("error", "Unknown error") for r in failed_results]
            response_data["error"] = f"Failed to generate images: {', '.join(error_messages)}"
            response_data["failed_results"] = failed_results
            
            return jsonify(response_data), 500
            
    except Exception as e:
        print(f"💥 Generate endpoint error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test API connectivity"""
    try:
        test_url = f"{POLLINATION_BASE_URL}/prompt/test"
        response = requests.get(test_url, params={"width": 512, "height": 512}, timeout=30)
        
        return jsonify({
            "success": True,
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', 'unknown'),
            "api_url": test_url,
            "response_size": len(response.content),
            "voice_enabled": VOICE_ENABLED,
            "tts_enabled": TTS_ENABLED
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download/<image_id>')
def download_image(image_id):
    """Download generated image"""
    try:
        # Sanitize image_id to prevent directory traversal
        image_id = os.path.basename(image_id)
        image_path = f"static/generated_images/{image_id}.png"
        
        if os.path.exists(image_path):
            return send_file(image_path, as_attachment=True, download_name=f"{image_id}.png")
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/enhance-prompt', methods=['POST'])
def enhance_prompt():
    """Enhance user prompt with AI"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        enhanced_prompt = enhance_prompt_simple(prompt)
        
        return jsonify({
            "success": True,
            "enhanced_prompt": enhanced_prompt
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def enhance_prompt_simple(prompt):
    """Simple prompt enhancement function"""
    enhancement_words = [
        "highly detailed", "professional", "stunning", "beautiful",
        "masterpiece", "award winning", "cinematic lighting",
        "perfect composition", "sharp focus", "vibrant colors",
        "8k resolution", "ultra realistic", "photorealistic"
    ]
    
    selected_enhancements = random.sample(enhancement_words, min(3, len(enhancement_words)))
    enhanced = f"{prompt}, {', '.join(selected_enhancements)}"
    
    return enhanced

def log_generation(prompt, image_id, seed):
    """Log image generation for analytics"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "image_id": image_id,
            "seed": seed
        }
        
        log_file = "generation_log.json"
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        logs = logs[-1000:]  # Keep only last 1000 entries
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"⚠️  Logging error: {str(e)}")

@app.route('/api/stats')
def get_stats():
    """Get generation statistics"""
    try:
        log_file = "generation_log.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            total_generations = len(logs)
            recent_generations = len([log for log in logs if 
                                    (datetime.now() - datetime.fromisoformat(log["timestamp"])).days <= 7])
            
            image_count = len(glob.glob('static/generated_images/*.png'))
            audio_count = len(glob.glob('static/audio/*.mp3'))
            
            return jsonify({
                "total_generations": total_generations,
                "recent_generations": recent_generations,
                "current_images": image_count,
                "current_audio_files": audio_count,
                "voice_enabled": VOICE_ENABLED,
                "tts_enabled": TTS_ENABLED,
                "success": True
            })
        else:
            return jsonify({
                "total_generations": 0,
                "recent_generations": 0,
                "current_images": 0,
                "current_audio_files": 0,
                "voice_enabled": VOICE_ENABLED,
                "tts_enabled": TTS_ENABLED,
                "success": True
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/test-voice')
@admin_required
def admin_test_voice():
    """Test voice functionality"""
    try:
        if not TTS_ENABLED:
            return jsonify({
                "success": False,
                "error": "Text-to-speech not available",
                "voice_enabled": VOICE_ENABLED,
                "tts_enabled": TTS_ENABLED
            }), 503
            
        # Test TTS
        test_text = "Admin panel voice test successful"
        tts_result = voice_handler.text_to_speech(test_text)
        
        return jsonify({
            "success": True,
            "tts_test": tts_result,
            "voice_enabled": VOICE_ENABLED,
            "tts_enabled": TTS_ENABLED,
            "message": "Voice system is working correctly"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "voice_enabled": VOICE_ENABLED,
            "tts_enabled": TTS_ENABLED
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 LAHgen Server Starting...")
    print("📸 Text-to-Image Generator")
    print(f"🎤 Voice Input: {'Enabled' if VOICE_ENABLED else 'Disabled'}")
    print(f"🔊 Voice Output: {'Enabled' if TTS_ENABLED else 'Disabled'}")
    print("🖼️  Multiple Image Generation (Up to 4 images)")
    print("🧹 Automatic Cleanup (Images & Audio - 24 hours)")
    print("🌐 Server will be available at: http://localhost:5000")
    print("✅ API endpoint: https://image.pollinations.ai")
    print("🧪 Test API connectivity at: http://localhost:5000/api/test")
    print("🔐 Admin panel: http://localhost:5000/admin")
    print(f"   Username: {ADMIN_USERNAME}")
    print(f"   Password: {ADMIN_PASSWORD}")
    print("")
    
    if not VOICE_ENABLED:
        print("⚠️  Voice features disabled - install: pip install SpeechRecognition pyaudio")
    if not TTS_ENABLED:
        print("⚠️  TTS features disabled - install: pip install gtts pygame")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)