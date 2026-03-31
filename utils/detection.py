import cv2
import os
import time
import uuid
import numpy as np
from ultralytics import YOLO
import requests
import re
import boto3
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from utils.db import SessionLocal, Detection, Camera, RecordingSession, AnalysisSession
from sqlalchemy.sql import func

# Load environment variables
load_dotenv()

# R2 Configuration
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME")
# Optional: Public URL (e.g., https://yourdomain.com or pub-X.r2.dev)
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

r2_client = None
if all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET]):
    try:
        r2_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name='auto'
        )
    except Exception as e:
        print(f"Debug: R2 Client Init Error: {e}")

def upload_to_r2(img, trigger_name, filename):
    if r2_client is None or img is None or img.size == 0:
        return None
    try:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        clean_trigger = re.sub(r'[^a-zA-Z0-9]', '_', trigger_name)
        key = f"{clean_trigger}/{date_str}/{time_str}/{filename}"
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        r2_client.put_object(Bucket=R2_BUCKET, Key=key, Body=buffer.tobytes(), ContentType='image/jpeg')
        # Construct R2 Public URL (Use R2_PUBLIC_URL if set, else fallback to API endpoint)
        if R2_PUBLIC_URL:
            public_url = f"{R2_PUBLIC_URL.rstrip('/')}/{key}"
        else:
            public_url = f"{R2_ENDPOINT}/{R2_BUCKET}/{key}"
        print(f"DEBUG: SUCCESS! Saved to R2: {public_url}")
        return public_url
    except Exception as e:
        print(f"DEBUG: R2 Upload FAILED: {e}")
        return None

# DB Configuration
DB_HOST = os.getenv("DB_HOST", "dpg-d72j4spr0fns73ebi470-a.ohio-postgres.render.com")
DB_NAME = os.getenv("DB_NAME", "db_3rdai")
DB_USER = os.getenv("DB_USER", "db_3rdai_user")
DB_PASS = os.getenv("DB_PASS", "WHbW4G3mT0qzgGmPODeLCWwnVwlcR6xO")

def save_to_db(data):
    trigger = data.get('trigger')
    plate_number = data.get('plate_number')
    image_plate_url = data.get('image_plate_url')
    image_object_url = data.get('image_object_url')

    if trigger == "Number Plate Detection":
        if not plate_number or plate_number in [None, "", "N/A", "SCANNING...", "UNREADABLE"] or not is_valid_indian_plate(plate_number):
            print(f"DEBUG: Skipping DB save - invalid plate: {plate_number}")
            return
        if not image_plate_url and not image_object_url:
            print("DEBUG: Skipping DB save - missing images for plate detection.")
            return

    if not image_plate_url and not image_object_url and not plate_number:
        print("DEBUG: Skipping DB save - all key fields are null.")
        return

    db = SessionLocal()
    try:
        new_detection = Detection(
            task_id=data.get('task_id'),
            filename=data.get('filename'),
            timestamp=data.get('timestamp'),
            trigger=data.get('trigger'),
            event=data.get('event'),
            image_plate_url=data.get('image_plate_url'),
            image_object_url=data.get('image_object_url'),
            plate_number=data.get('plate_number'),
            vehicle_color=data.get('vehicle_color')
        )
        db.add(new_detection)
        db.commit()
        print(f"DEBUG: SUCCESS! Saved Detection to Database via SQLAlchemy (Task: {data.get('task_id')})")
    except Exception as e:
        db.rollback()
        print(f"DEBUG: Database Save FAILED: {e}")
    finally:
        db.close()

# FORCE TCP for RTSP globally before any CV2 operations
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# PLATE RECOGNIZER CLOUD API CONFIGURATION
PLATE_RECOGNIZER_TOKEN = "cabf3c65d1ec04ff52c1d5d0489fb083cdd2e305"
PLATE_RECOGNIZER_URL = 'https://api.platerecognizer.com/v1/plate-reader/'

# OCR INITIALIZATION
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True) 
except ImportError:
    reader = None
    print("Warning: easyocr not installed.")

paddle_reader = None

try:
    import webcolors
    from sklearn.cluster import KMeans
except ImportError:
    webcolors = None
    KMeans = None

MODEL_MAP = {
    "Number Plate Detection": "ANPR.pt",
    "Helmet Detection": "helmet.pt",
    "Triple Riding Detection": "triple.pt",
    "Seatbelt Detection": "seatbelt.pt"
}

def get_vehicle_color(img):
    if img is None or img.size == 0:
        return "Unknown"
    try:
        img_small = cv2.resize(img, (50, 50))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        if KMeans is not None:
            pixels = img_rgb.reshape((-1, 3))
            kmeans = KMeans(n_clusters=3, n_init=5)
            kmeans.fit(pixels)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[np.argmax(counts)]
            dominant_rgb = kmeans.cluster_centers_[dominant_idx].astype(int)
        else:
            dominant_rgb = np.median(img_rgb, axis=(0, 1)).astype(int)
        r, g, b = dominant_rgb
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        if max_val < 50: return "Black"
        if min_val > 200: return "White"
        if diff < 20: return "Grey"
        if r > g and r > b: return "Red"
        if g > r and g > b: return "Green"
        if b > r and b > g: return "Blue"
        if r > 150 and g > 150 and b < 100: return "Yellow"
        return "Silver"
    except Exception as e:
        print(f"Debug: Color error: {e}")
        return "Unknown"

def is_valid_indian_plate(plate_text):
    if not plate_text: return False
    # Standard Indian Plate Regex: e.g., MH12AB1234, DL3CAP1234, KA011234
    # Pattern: 2 letters, 1-2 digits, optional 1-2 letters, 4 digits
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$'
    return bool(re.match(pattern, plate_text))

def get_best_ocr(crop_img):
    if crop_img is None or crop_img.size == 0:
        return ""
    try:
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_bytes = buffer.tobytes()
        response = requests.post(PLATE_RECOGNIZER_URL,
            data=dict(regions=['in']),
            files=dict(upload=('plate.jpg', img_bytes)),
            headers={'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'},
            timeout=10)
        if response.status_code in [200, 201]:
            res_data = response.json()
            if res_data.get('results'):
                plate_text = res_data['results'][0].get('plate', '').upper()
                clean_text = re.sub(r'[^A-Z0-9]', '', plate_text)
                if is_valid_indian_plate(clean_text):
                    print(f"Debug: Valid Plate Recognizer Result: {clean_text}")
                    return clean_text
        else:
            print(f"Debug: Plate Recognizer API Error: {response.status_code}")
    except Exception as e:
        print(f"Debug: Plate Recognizer API Connection Error: {e}")
    try:
        if reader:
            res = reader.readtext(crop_img)
            if res:
                all_text = "".join([r[1] for r in res]).upper()
                clean_local = re.sub(r'[^A-Z0-9]', '', all_text)
                if is_valid_indian_plate(clean_local):
                    print(f"Debug: Valid Local OCR Result: {clean_local}")
                    return clean_local
    except Exception as local_err:
        print(f"Debug: Local OCR Fallback Error: {local_err}")
    return ""

_loaded_models_cache = {}

def get_model(trigger_name):
    model_filename = MODEL_MAP.get(trigger_name, "yolov8n.pt")
    cache_key = model_filename
    if cache_key not in _loaded_models_cache:
        models_dir = "models"
        model_path = os.path.join(models_dir, model_filename)
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
        try:
            _loaded_models_cache[cache_key] = YOLO(model_path)
            print(f"Debug: Loaded model {model_filename} for {trigger_name}")
        except Exception as e:
            print(f"Debug: Error loading model {model_filename}: {e}")
            return None
    return _loaded_models_cache[cache_key]

class LiveCameraProcessor:
    def __init__(self, camera_id, camera_link, selected_triggers):
        self.camera_id = camera_id
        self.camera_link = camera_link
        self.selected_triggers = selected_triggers
        self.is_running = False
        self.status = "connecting"
        self.latest_frame = None
        self.raw_frame_buffer = None
        self.logs = []
        self.processed_track_ids = set()
        self.seen_plate_numbers = set()
        self.models = {t: get_model(t) for t in selected_triggers}
        self.vehicle_model = YOLO("yolov8n.pt")
        self.frame_count = 0
        self.process_every_n_frames = 2 # Process every 2nd frame for speed
        
        self.latest_jpeg = None # Pre-encoded JPEG for WebSocket delivery
        self.cap = None
        self.reader_thread = None
        self.processor_thread = None
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_session_id = None
        self.analysis_session_id = None
        self.recording_start_time = None
        self.recording_file_path = None
        self.recording_source = "manual" # manual | auto | analysis
        
        self.add_log("System", "Initiating camera connection sequence...")
        
        import threading
        self.orchestrator = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestrator.start()

    def _update_latest_frame(self, frame):
        if frame is None: return
        h, w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, int(h * (1280/w))))
        self.latest_frame = frame
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        self.latest_jpeg = buffer.tobytes()

    def add_log(self, trigger, event, plate=None, obj=None, p_num=None, color=None, r2=False):
        log_entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "trigger": trigger,
            "event": event,
            "image_plate": plate,
            "image_object": obj,
            "plate_number": p_num,
            "vehicle_color": color,
            "saved_to_r2": r2
        }
        self.logs.append(log_entry)
        if len(self.logs) > 100: self.logs.pop(0)

    def _orchestration_loop(self):
        self.is_running = True
        connection_configs = [
            {"name": "FFmpeg TCP", "options": "rtsp_transport;tcp"},
            {"name": "FFmpeg UDP", "options": "rtsp_transport;udp"},
            {"name": "OpenCV Default", "options": None}
        ]
        
        connected = False
        for config in connection_configs:
            c_name = config['name']
            self.add_log("System", f"Trying connection via {c_name}...")
            
            c_options = config.get('options')
            if c_options:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = str(c_options)
            else:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            
            self.cap = cv2.VideoCapture(self.camera_link, cv2.CAP_FFMPEG if "FFmpeg" in c_name else cv2.CAP_ANY)
            
            # Set a small timeout for connection check
            start_t = time.time()
            while time.time() - start_t < 5: # 5 second timeout per method
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        connected = True
                        break
                time.sleep(0.5)
            
            if connected:
                self.add_log("System", f"Successfully connected via {c_name}")
                break
            else:
                self.add_log("System", f"Method {c_name} failed or timed out.")
                self.cap.release()

        if not connected:
            self.status = "failed"
            self.is_running = False
            self.add_log("System", "All connection methods failed. Please check the camera link.")
            return

        self.status = "connected"
        import threading
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.reader_thread.start()
        self.processor_thread.start()

    def _reader_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_link)
                continue
            self.raw_frame_buffer = frame

    def _process_loop(self):
        crops_dir = os.path.join("static", "crops", self.camera_id)
        os.makedirs(crops_dir, exist_ok=True)
        
        while self.is_running:
            if self.raw_frame_buffer is None or self.raw_frame_buffer.size == 0:
                time.sleep(0.01)
                continue
            
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames != 0:
                # Update latest frame for streaming but skip heavy AI
                self._update_latest_frame(self.raw_frame_buffer)
                continue

            try:
                frame = self.raw_frame_buffer
                h, w = frame.shape[:2]
                if w > 1280:
                    frame = cv2.resize(frame, (1280, int(h * (1280/w))))
                raw_frame = frame.copy()
                
                for trigger_name, model in self.models.items():
                    if model is None: continue
                    results = model.track(frame, persist=True, verbose=False, iou=0.5, conf=0.4)[0]
                    
                    if results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                        ids = results.boxes.id.cpu().numpy().astype(int)
                        confs = results.boxes.conf.cpu().numpy()
                        
                        for box, obj_id, conf in zip(boxes, ids, confs):
                            x1, y1, x2, y2 = box
                            unique_track_key = f"{trigger_name}_{obj_id}"
                            
                            # Draw visual feedback
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1-5), 0, 0.4, (0,255,0), 1)
                            
                            if unique_track_key not in self.processed_track_ids:
                                p_fname = f"{trigger_name.replace(' ', '_')}_{obj_id}_{int(time.time())}.jpg"
                                
                                # Take a TIGHT crop for plates, pad for others
                                if trigger_name == "Number Plate Detection":
                                    p_pad = 5
                                    x1_p, y1_p = max(0, x1-p_pad), max(0, y1-p_pad)
                                    x2_p, y2_p = min(frame.shape[1], x2+p_pad), min(frame.shape[0], y2+p_pad)
                                    crop = raw_frame[y1_p:y2_p, x1_p:x2_p]
                                else:
                                    pad = 20
                                    x1_c, y1_c = max(0, x1-pad), max(0, y1-pad)
                                    x2_c, y2_c = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                                    crop = raw_frame[y1_c:y2_c, x1_c:x2_c]
                                
                                if crop.size == 0: continue
                                
                                plate_text = ""
                                v_color = "Unknown"
                                r2_plate_url = None
                                r2_object_url = None
                                local_plate_path = None
                                local_object_path = None

                                if trigger_name == "Number Plate Detection":
                                    plate_text = get_best_ocr(crop)
                                    if plate_text and plate_text in self.seen_plate_numbers: continue
                                    if plate_text: self.seen_plate_numbers.add(plate_text)
                                    
                                    cv2.imwrite(os.path.join(crops_dir, p_fname), crop)
                                    local_plate_path = f"/static/crops/{self.camera_id}/{p_fname}"
                                    r2_plate_url = upload_to_r2(crop, trigger_name, p_fname)
                                    
                                    # Find clear vehicle crop for the plate
                                    v_res = self.vehicle_model.predict(raw_frame, verbose=False, classes=[2,3,5,7], conf=0.4)[0]
                                    for v_box in v_res.boxes:
                                        vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                                        px, py = (x1+x2)/2, (y1+y2)/2
                                        if vx1 <= px <= vx2 and vy1 <= py <= vy2:
                                            v_crop = raw_frame[vy1:vy2, vx1:vx2]
                                            v_color = get_vehicle_color(v_crop)
                                            v_fname = f"v_{obj_id}_{int(time.time())}.jpg"
                                            cv2.imwrite(os.path.join(crops_dir, v_fname), v_crop)
                                            local_object_path = f"/static/crops/{self.camera_id}/{v_fname}"
                                            r2_object_url = upload_to_r2(v_crop, trigger_name, v_fname)
                                            break
                                else:
                                    # For other objects (Helmet, etc.), just crop the object carefully
                                    cv2.imwrite(os.path.join(crops_dir, p_fname), crop)
                                    local_object_path = f"/static/crops/{self.camera_id}/{p_fname}"
                                    r2_object_url = upload_to_r2(crop, trigger_name, p_fname)

                                self.processed_track_ids.add(unique_track_key)
                                
                                save_data = {
                                    "task_id": self.camera_id,
                                    "filename": self.camera_link,
                                    "timestamp": 0.0,
                                    "trigger": trigger_name,
                                    "event": f"Detection (ID: {obj_id})",
                                    "image_plate_url": r2_plate_url,
                                    "image_object_url": r2_object_url,
                                    "plate_number": plate_text or None,
                                    "vehicle_color": v_color if v_color != "Unknown" else None
                                }
                                
                                # Only save and log if we actually got a clear detection (esp for plates)
                                if trigger_name == "Number Plate Detection" and not plate_text:
                                    continue
                                
                                save_to_db(save_data)
                                
                                self.add_log(
                                    trigger_name, 
                                    f"Detected {trigger_name} (ID: {obj_id})",
                                    plate=local_plate_path,
                                    obj=local_object_path,
                                    p_num=plate_text,
                                    color=v_color,
                                    r2=True if r2_plate_url or r2_object_url else False
                                )

                self._update_latest_frame(frame)
                
                # Write to recording if active
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                    
            except Exception as e:
                print(f"[DEBUG] Error in processing loop: {e}")
                time.sleep(0.1)

    def start_recording(self, initiated_by="System", note=None, source="manual", analysis_session_id=None):
        if self.is_recording:
            return False, "Already recording"
        
        try:
            self.recording_source = source
            self.analysis_session_id = analysis_session_id
            self.recording_session_id = str(uuid.uuid4())
            
            # Setup output path
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', self.camera_id)
            filename = f"{safe_name}_{timestamp_str}.mp4"
            
            output_dir = os.path.join("static", "recordings", self.camera_id)
            os.makedirs(output_dir, exist_ok=True)
            self.recording_file_path = os.path.join(output_dir, filename)
            
            # Create DB record
            db = SessionLocal()
            new_session = RecordingSession(
                id=self.recording_session_id,
                camera_id=self.camera_id,
                video_name=filename,
                file_path=f"/static/recordings/{self.camera_id}/{filename}",
                source=source,
                initiated_by=initiated_by,
                description=note,
                started_at=func.now()
            )
            db.add(new_session)
            
            # If it's an analysis session, update it
            if analysis_session_id:
                analysis = db.query(AnalysisSession).filter(AnalysisSession.id == analysis_session_id).first()
                if analysis:
                    analysis.recording_session_id = self.recording_session_id
                    analysis.capture_started_at = func.now()
            
            db.commit()
            db.close()
            
            # Initialize VideoWriter (we'll start writing in the processor loop)
            # We need width and height from first frame or cap
            if self.latest_frame is not None:
                h, w = self.latest_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.recording_file_path, fourcc, 20, (w, h))
            
            self.is_recording = True
            self.recording_start_time = time.time()
            self.add_log("Recording", f"Started recording (Source: {source})", r2=False)
            return True, self.recording_session_id
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False, str(e)

    def stop_recording(self, stopped_by="System"):
        if not self.is_recording:
            return False, "Not recording"
        
        try:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            duration = int(time.time() - self.recording_start_time) if self.recording_start_time else 0
            
            db = SessionLocal()
            session = db.query(RecordingSession).filter(RecordingSession.id == self.recording_session_id).first()
            if session:
                session.stopped_at = func.now()
                session.duration_secs = duration
                session.stopped_by = stopped_by
            
            # If it was an analysis session
            if self.analysis_session_id:
                analysis = db.query(AnalysisSession).filter(AnalysisSession.id == self.analysis_session_id).first()
                if analysis:
                    analysis.capture_ended_at = func.now()
                    analysis.stopped_by = stopped_by

            db.commit()
            db.close()
            
            self.add_log("Recording", f"Stopped recording. Duration: {duration}s", r2=False)
            return True, self.recording_session_id
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False, str(e)

    def stop(self):
        if self.is_recording:
            self.stop_recording()
        self.is_running = False
        if hasattr(self, 'cap') and self.cap: self.cap.release()

def process_video(task_id, input_path, output_path, selected_triggers):
    logs = []
    crops_dir = os.path.join("static", "crops", task_id)
    os.makedirs(crops_dir, exist_ok=True)
    
    loaded_models = {t: get_model(t) for t in selected_triggers if get_model(t)}
    vehicle_model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return [{"frame": 0, "event": "Error opening video", "type": "error"}]
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Scale down for faster processing if very large
    if width > 1280:
        scale = 1280 / width
        width, height = 1280, int(height * scale)
    
    temp_output = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    processed_track_ids = set()
    seen_plate_numbers = set()
    frame_count = 0
    process_every_n = 3 # Fast mode: process every 3rd frame (matches ~10fps analysis)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame = cv2.resize(frame, (width, height))
        
        if frame_count % process_every_n != 0:
            out.write(frame)
            continue
            
        raw_frame = frame.copy()
        for trigger_name, model in loaded_models.items():
            results = model.track(frame, persist=True, verbose=False, iou=0.5, conf=0.45)[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                
                for box, obj_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = box
                    unique_track_key = f"{trigger_name}_{obj_id}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1-5), 0, 0.5, (0,255,0), 2)
                    
                    if unique_track_key not in processed_track_ids:
                        p_fname = f"{trigger_name.replace(' ', '_')}_{obj_id}_{frame_count}.jpg"
                        
                        # Take a TIGHT crop for plates, pad for others
                        if trigger_name == "Number Plate Detection":
                            p_pad = 5
                            x1_p, y1_p = max(0, x1-p_pad), max(0, y1-p_pad)
                            x2_p, y2_p = min(width, x2+p_pad), min(height, y2+p_pad)
                            crop = raw_frame[y1_p:y2_p, x1_p:x2_p]
                        else:
                            pad = 20
                            x1_c, y1_c = max(0, x1-pad), max(0, y1-pad)
                            x2_c, y2_c = min(width, x2+pad), min(height, y2+pad)
                            crop = raw_frame[y1_c:y2_c, x1_c:x2_c]
                        
                        if crop.size == 0: continue
                        
                        plate_text = ""
                        v_color = "Unknown"
                        r2_plate_url = None
                        r2_object_url = None
                        local_plate_path = None
                        local_object_path = None

                        if trigger_name == "Number Plate Detection":
                            plate_text = get_best_ocr(crop)
                            if plate_text and plate_text in seen_plate_numbers: continue
                            if plate_text: seen_plate_numbers.add(plate_text)
                            
                            cv2.imwrite(os.path.join(crops_dir, p_fname), crop)
                            local_plate_path = f"/static/crops/{task_id}/{p_fname}"
                            r2_plate_url = upload_to_r2(crop, trigger_name, p_fname)
                            
                            # Find matching vehicle
                            v_res = vehicle_model.predict(raw_frame, verbose=False, classes=[2,3,5,7], conf=0.4)[0]
                            for v_box in v_res.boxes:
                                vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                                px, py = (x1+x2)/2, (y1+y2)/2
                                if vx1<=px<=vx2 and vy1<=py<=vy2:
                                    v_crop = raw_frame[vy1:vy2, vx1:vx2]
                                    v_color = get_vehicle_color(v_crop)
                                    v_fname = f"v_{obj_id}_{frame_count}.jpg"
                                    cv2.imwrite(os.path.join(crops_dir, v_fname), v_crop)
                                    local_object_path = f"/static/crops/{task_id}/{v_fname}"
                                    r2_object_url = upload_to_r2(v_crop, trigger_name, v_fname)
                                    break
                            if v_color == "Unknown": v_color = get_vehicle_color(crop)
                        else:
                            # Other objects
                            cv2.imwrite(os.path.join(crops_dir, p_fname), crop)
                            local_object_path = f"/static/crops/{task_id}/{p_fname}"
                            r2_object_url = upload_to_r2(crop, trigger_name, p_fname)

                        processed_track_ids.add(unique_track_key)
                        
                        save_data = {
                            "task_id": task_id,
                            "filename": input_path.split("/")[-1],
                            "timestamp": round(frame_count / fps, 2),
                            "trigger": trigger_name,
                            "event": f"Detection (ID: {obj_id})",
                            "image_plate_url": r2_plate_url,
                            "image_object_url": r2_object_url,
                            "plate_number": plate_text or None,
                            "vehicle_color": v_color if v_color != "Unknown" else None
                        }
                        
                        # Only save and log if we actually got a clear detection (esp for plates)
                        if trigger_name == "Number Plate Detection" and not plate_text:
                            continue
                            
                        save_to_db(save_data)

                        logs.append({
                            "timestamp": round(frame_count / fps, 2),
                            "trigger": trigger_name,
                            "event": f"Detected {trigger_name} (ID: {obj_id})",
                            "image_plate": local_plate_path,
                            "image_object": local_object_path,
                            "plate_number": plate_text,
                            "vehicle_color": v_color,
                            "saved_to_r2": True if r2_plate_url or r2_object_url else False
                        })
        out.write(frame)
        
    cap.release()
    out.release()
    
    try:
        # Optimized FFmpeg command for speed
        subprocess.run(['ffmpeg', '-y', '-i', temp_output, '-c:v', 'libx264', '-crf', '28', '-preset', 'ultrafast', '-movflags', '+faststart', '-pix_fmt', 'yuv420p', output_path], check=True, capture_output=True)
        if os.path.exists(temp_output): os.remove(temp_output)
    except Exception as e:
        print(f"DEBUG: Video optimization FAILED: {e}")
        if os.path.exists(temp_output): os.rename(temp_output, output_path)
        
    return logs

