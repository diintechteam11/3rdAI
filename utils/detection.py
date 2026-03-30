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
from utils.db import SessionLocal, Detection, Camera, RecordingSession, AnalysisSession
from sqlalchemy.sql import func
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# R2 Configuration
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME")
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
        if R2_PUBLIC_URL:
            public_url = f"{R2_PUBLIC_URL.rstrip('/')}/{key}"
        else:
            public_url = f"{R2_ENDPOINT}/{R2_BUCKET}/{key}"
        return public_url
    except Exception as e:
        print(f"DEBUG: R2 Upload FAILED: {e}")
        return None

# OCR INITIALIZATION
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True) 
except ImportError:
    reader = None

MODEL_MAP = {
    "Number Plate Detection": "ANPR.pt",
    "Helmet Detection": "helmet.pt",
    "Triple Riding Detection": "triple_riding.pt",
    "Seatbelt Detection": "seatbelt.pt"
}

def get_vehicle_color(img):
    try:
        if img is None or img.size == 0: return "Unknown"
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (50, 50))
        pixels = img_rgb.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        r, g, b = kmeans.cluster_centers_[0].astype(int)
        if max(r, g, b) < 50: return "Black"
        if min(r, g, b) > 200: return "White"
        if r > g and r > b: return "Red"
        if g > r and g > b: return "Green"
        if b > r and b > g: return "Blue"
        if r > 150 and g > 150 and b < 100: return "Yellow"
        return "Silver"
    except: return "Unknown"

def is_valid_indian_plate(plate_text):
    if not plate_text: return False
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$'
    return bool(re.match(pattern, plate_text))

def get_best_ocr(crop_img):
    if crop_img is None or crop_img.size == 0: return ""
    try:
        # 1. Plate Recognizer (Cloud)
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        response = requests.post('https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=['in']),
            files=dict(upload=('plate.jpg', buffer.tobytes())),
            headers={'Authorization': 'Token cabf3c65d1ec04ff52c1d5d0489fb083cdd2e305'},
            timeout=10)
        if response.status_code in [200, 201]:
            res_data = response.json()
            if res_data.get('results'):
                plate = res_data['results'][0].get('plate', '').upper()
                return re.sub(r'[^A-Z0-9]', '', plate)
    except: pass
    
    try:
        # 2. EasyOCR (Local Fallback)
        if reader:
            res = reader.readtext(crop_img)
            if res:
                all_text = "".join([r[1] for r in res]).upper()
                return re.sub(r'[^A-Z0-9]', '', all_text)
    except: pass
    return ""

_loaded_models_cache = {}
def get_model(trigger_name):
    fname = MODEL_MAP.get(trigger_name, "yolov8n.pt")
    if fname not in _loaded_models_cache:
        path = os.path.join("models", fname)
        if not os.path.exists(path): path = "yolov8n.pt"
        _loaded_models_cache[fname] = YOLO(path)
    return _loaded_models_cache[fname]

class LiveCameraProcessor:
    def __init__(self, camera_id, camera_link, selected_triggers):
        self.camera_id, self.camera_link, self.selected_triggers = camera_id, camera_link, selected_triggers
        self.is_running, self.status = False, "connecting"
        self.latest_frame, self.latest_jpeg = None, None
        self.logs, self.processed_track_ids, self.seen_plate_numbers = [], set(), set()
        self.models = {t: get_model(t) for t in selected_triggers}
        self.vehicle_model = YOLO("yolov8n.pt")
        self.is_recording, self.recording_writer = False, None
        self.recording_start_time, self.recording_session_id = 0, None
        import threading
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.is_running = True
        self.thread.start()

    def _run(self):
        cap = cv2.VideoCapture(self.camera_link)
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break
            
            raw_frame = frame.copy()
            for trigger_name, model in self.models.items():
                if model is None: continue
                results = model.track(frame, persist=True, verbose=False, conf=0.3)[0]
                if not results.boxes or results.boxes.id is None: continue
                
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    track_key = f"{trigger_name}_{obj_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if track_key not in self.processed_track_ids:
                        crop = raw_frame[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        
                        plate_text = ""
                        if trigger_name == "Number Plate Detection":
                            plate_text = get_best_ocr(crop)
                        
                        fname = f"{track_key}_{int(time.time())}.jpg"
                        r2_url = upload_to_r2(crop, trigger_name, fname)
                        self.processed_track_ids.add(track_key)
                        
                        self.logs.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "trigger": trigger_name,
                            "event": f"Detected {trigger_name} (ID: {obj_id})",
                            "plate_number": plate_text or "Scanning...",
                            "image_plate": r2_url,
                            "saved_to_r2": True if r2_url else False
                        })
            
            self.latest_frame = frame
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            self.latest_jpeg = jpeg.tobytes()
            
            if self.is_recording and self.recording_writer:
                self.recording_writer.write(frame)
        cap.release()
        if self.recording_writer: self.recording_writer.release()

    def start_recording(self, initiated_by="System", note=None, source="manual", analysis_session_id=None):
        if self.is_recording: return False, "Already"
        self.recording_session_id = str(uuid.uuid4())
        fname = f"{self.camera_id}_{int(time.time())}.mp4"
        path = os.path.join("static", "recordings", fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = self.latest_frame.shape[:2] if self.latest_frame is not None else (720, 1280)
        self.recording_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
        self.is_recording = True
        self.recording_start_time = time.time()
        return True, self.recording_session_id

    def stop_recording(self, stopped_by="System"):
        if not self.is_recording: return False, "Not recording"
        self.is_recording = False
        if self.recording_writer:
            self.recording_writer.release()
            self.recording_writer = None
        return True, self.recording_session_id

def save_to_db(data):
    db = SessionLocal()
    try:
        new_det = Detection(
            camera_id=data.get("camera_id", "external"),
            trigger_type=data.get("trigger"),
            event_description=data.get("event"),
            plate_number=data.get("plate_number"),
            image_plate_url=data.get("image_plate_url"),
            image_object_url=data.get("image_object_url"),
            vehicle_color=data.get("vehicle_color"),
            saved_to_r2=True if (data.get("image_plate_url") or data.get("image_object_url")) else False
        )
        db.add(new_det); db.commit()
    except Exception as e:
        db.rollback(); print(f"DB Error: {e}")
    finally: db.close()

def process_video(task_id, input_path, output_path, selected_triggers):
    logs = []
    crops_dir = os.path.join("static", "crops", task_id); os.makedirs(crops_dir, exist_ok=True)
    models = {t: get_model(t) for t in selected_triggers}
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); temp_out = output_path + ".tmp.mp4"; out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        raw_frame = frame.copy()
        for trigger_name, model in models.items():
            if model is None: continue
            results = model.track(frame, persist=True, verbose=False, conf=0.3)[0]
            if not results.boxes or results.boxes.id is None: continue
            for box, obj_id in zip(results.boxes.xyxy.cpu().numpy().astype(int), results.boxes.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Capture log for Number Plate
                if trigger_name == "Number Plate Detection":
                    crop = raw_frame[y1:y2, x1:x2]
                    plate_text = get_best_ocr(crop)
                    if plate_text: # Found something
                        fname = f"plate_{obj_id}_{frame_count}.jpg"
                        cv2.imwrite(os.path.join(crops_dir, fname), crop)
                        r2_url = upload_to_r2(crop, trigger_name, fname)
                        logs.append({
                            "timestamp": round(frame_count / fps, 2), "trigger": trigger_name,
                            "event": f"Detected Plate (ID: {obj_id})", "plate_number": plate_text,
                            "image_plate": f"/static/crops/{task_id}/{fname}", "saved_to_r2": True if r2_url else False
                        })
        out.write(frame)
    cap.release(); out.release()
    subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
    if os.path.exists(temp_out): os.remove(temp_out)
    return logs
