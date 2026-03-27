import cv2
import os
import time
import uuid
import numpy as np
from ultralytics import YOLO
import requests
import re
import boto3
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import subprocess

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
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        # Initialize table if not exists with correct columns
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id SERIAL PRIMARY KEY,
                task_id VARCHAR(100),
                filename VARCHAR(255),
                timestamp FLOAT,
                trigger VARCHAR(100),
                event TEXT,
                image_plate_url TEXT,
                image_object_url TEXT,
                plate_number VARCHAR(100),
                vehicle_color VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Ensure all columns exist for users with older versions of the table
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS task_id VARCHAR(100);")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS filename VARCHAR(255);")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS timestamp FLOAT;")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS trigger VARCHAR(100);")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS event TEXT;")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS image_plate_url TEXT;")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS image_object_url TEXT;")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS plate_number VARCHAR(100);")
        cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS vehicle_color VARCHAR(100);")
        
        cur.execute("""
            INSERT INTO detections 
            (task_id, filename, timestamp, trigger, event, image_plate_url, image_object_url, plate_number, vehicle_color)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data.get('task_id'),
            data.get('filename'),
            data.get('timestamp'),
            data.get('trigger'),
            data.get('event'),
            data.get('image_plate_url'),
            data.get('image_object_url'),
            data.get('plate_number'),
            data.get('vehicle_color')
        ))
        conn.commit()
        print(f"DEBUG: SUCCESS! Saved Detection to Database (Task: {data.get('task_id')})")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DEBUG: Database Save FAILED: {e}")

# FORCE TCP for RTSP globally before any CV2 operations
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# PLATE RECOGNIZER CLOUD API CONFIGURATION
PLATE_RECOGNIZER_TOKEN = "b7352531317f55e343ae5b91a74df3a79bf7bbd5"
PLATE_RECOGNIZER_URL = 'https://api.platerecognizer.com/v1/plate-reader/'

# OCR INITIALIZATION
try:
    import easyocr
    # Using EasyOCR as primary for stability and better multi-line handling
    reader = easyocr.Reader(['en'], gpu=True) # Set gpu=False if no CUDA available
except ImportError:
    reader = None
    print("Warning: easyocr not installed.")

# PaddleOCR is disabled due to internal error: ConvertPirAttribute2RuntimeAttribute
paddle_reader = None

# Required libraries
try:
    import webcolors
    from sklearn.cluster import KMeans
except ImportError:
    webcolors = None
    KMeans = None

# Mapping triggers to model filenames
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
                if clean_text:
                    print(f"Debug: Plate Recognizer Result: {clean_text}")
                    return clean_text
        else:
            print(f"Debug: Plate Recognizer API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Debug: Plate Recognizer API Connection Error: {e}")
    try:
        if reader:
            res = reader.readtext(crop_img)
            if res:
                all_text = "".join([r[1] for r in res]).upper()
                clean_local = re.sub(r'[^A-Z0-9]', '', all_text)
                if clean_local:
                    print(f"Debug: LOCAL FALLBACK Result: {clean_local}")
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
        import threading
        self.orchestrator = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestrator.start()

    def _orchestration_loop(self):
        self.is_running = True
        connection_configs = [
            {"name": "FFmpeg TCP", "options": "rtsp_transport;tcp"},
            {"name": "FFmpeg UDP", "options": "rtsp_transport;udp"},
            {"name": "Default Backend", "options": None}
        ]
        connected = False
        for config in connection_configs:
            c_name = config.get('name', 'Unknown')
            c_options = config.get('options')
            if c_options:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = str(c_options)
            else:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            if c_name.startswith("FFmpeg"):
                self.cap = cv2.VideoCapture(self.camera_link, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.camera_link)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    connected = True
                    break
                else:
                    self.cap.release()
        if not connected:
            self.status = "failed"
            self.is_running = False
            return
        self.status = "connected"
        import threading
        self.raw_frame_buffer = None
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.reader_thread.start()
        self.processor_thread.start()

    def _reader_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.camera_link)
                continue
            self.raw_frame_buffer = frame

    def _process_loop(self):
        crops_dir = os.path.join("static", "crops", self.camera_id)
        os.makedirs(crops_dir, exist_ok=True)
        while self.is_running:
            if self.raw_frame_buffer is None:
                time.sleep(0.01)
                continue
            try:
                frame = self.raw_frame_buffer
                h, w = frame.shape[:2]
                if w > 1280:
                    frame = cv2.resize(frame, (1280, int(h * (1280/w))))
                raw_frame = frame.copy()
                for trigger_name, model in self.models.items():
                    if model is None: continue
                    results = model.track(frame, persist=True, verbose=False, iou=0.5, conf=0.35)[0]
                    if results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                        ids = results.boxes.id.cpu().numpy().astype(int)
                        confs = results.boxes.conf.cpu().numpy()
                        for box, obj_id, conf in zip(boxes, ids, confs):
                            x1, y1, x2, y2 = box
                            unique_track_key = f"{trigger_name}_{obj_id}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1-5), 0, 0.4, (0,255,0), 1)
                            if unique_track_key not in self.processed_track_ids:
                                plate_text = ""
                                if trigger_name == "Number Plate Detection":
                                    pad = 15
                                    x1_p, y1_p = max(0, x1-pad), max(0, y1-pad)
                                    x2_p, y2_p = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                                    plate_crop = raw_frame[y1_p:y2_p, x1_p:x2_p]
                                    if plate_crop.size > 0:
                                        plate_text = get_best_ocr(plate_crop)
                                        if plate_text and plate_text in self.seen_plate_numbers:
                                            continue
                                        if plate_text: self.seen_plate_numbers.add(plate_text)
                                        self.processed_track_ids.add(unique_track_key)
                                        p_fname = f"plate_{obj_id}_{int(time.time())}.jpg"
                                        cv2.imwrite(os.path.join(crops_dir, p_fname), plate_crop)
                                        image_plate_url = f"/static/crops/{self.camera_id}/{p_fname}"
                                        r2_plate_url = upload_to_r2(plate_crop, trigger_name, p_fname)
                                        v_color = get_vehicle_color(plate_crop)
                                        r2_object_url = None
                                        local_object_path = None
                                        v_res = self.vehicle_model.predict(raw_frame, verbose=False, classes=[2,3,5,7], conf=0.3)[0]
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
                                        save_to_db({
                                            "task_id": self.camera_id,
                                            "filename": self.camera_link,
                                            "timestamp": 0.0,
                                            "trigger": trigger_name,
                                            "event": f"Detection (ID: {obj_id})",
                                            "image_plate_url": r2_plate_url,
                                            "image_object_url": r2_object_url,
                                            "plate_number": plate_text or "UNREADABLE",
                                            "vehicle_color": v_color
                                        })
                                        self.logs.append({
                                            "timestamp": time.strftime("%H:%M:%S"),
                                            "trigger": trigger_name,
                                            "event": f"Detection (ID: {obj_id})",
                                            "image_plate": image_plate_url,
                                            "image_object": local_object_path,
                                            "plate_number": plate_text or "UNREADABLE",
                                            "vehicle_color": v_color,
                                            "saved_to_r2": True if r2_plate_url else False
                                        })
                                else:
                                    self.processed_track_ids.add(unique_track_key)
                                    save_to_db({
                                        "task_id": self.camera_id,
                                        "filename": self.camera_link,
                                        "timestamp": 0.0,
                                        "trigger": trigger_name,
                                        "event": f"{trigger_name} (ID: {obj_id})",
                                        "image_plate_url": None, "image_object_url": None, "plate_number": None, "vehicle_color": None
                                    })
                                    self.logs.append({
                                        "timestamp": time.strftime("%H:%M:%S"),
                                        "trigger": trigger_name,
                                        "event": f"{trigger_name} (ID: {obj_id})",
                                        "image_plate": None, "image_object": None, "plate_number": None
                                    })
                self.latest_frame = frame
            except Exception as e:
                print(f"[DEBUG] Error in processing loop: {e}")
                time.sleep(0.1)

    def stop(self):
        self.is_running = False
        if self.cap: self.cap.release()

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
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)
    temp_output = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    processed_track_ids = set()
    seen_plate_numbers = set()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        frame = cv2.resize(frame, (width, height))
        raw_frame = frame.copy()
        for trigger_name, model in loaded_models.items():
            results = model.track(frame, persist=True, verbose=False, iou=0.5)[0]
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                for box, obj_id, conf in zip(boxes, ids, confs):
                    if conf < 0.4: continue
                    x1, y1, x2, y2 = box
                    unique_track_key = f"{trigger_name}_{obj_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1-5), 0, 0.5, (0,255,0), 2)
                    if unique_track_key not in processed_track_ids:
                        plate_text = ""
                        if trigger_name == "Number Plate Detection":
                            pad = 15
                            x1_p, y1_p = max(0, x1-pad), max(0, y1-pad)
                            x2_p, y2_p = min(width, x2+pad), min(height, y2+pad)
                            plate_crop = raw_frame[y1_p:y2_p, x1_p:x2_p]
                            if plate_crop.size > 0:
                                plate_text = get_best_ocr(plate_crop)
                                if plate_text and plate_text in seen_plate_numbers:
                                    continue
                                if plate_text: seen_plate_numbers.add(plate_text)
                                processed_track_ids.add(unique_track_key)
                                p_fname = f"plate_{obj_id}_{frame_count}.jpg"
                                cv2.imwrite(os.path.join(crops_dir, p_fname), plate_crop)
                                image_plate_url = f"/static/crops/{task_id}/{p_fname}"
                                r2_plate_url = upload_to_r2(plate_crop, trigger_name, p_fname)
                                v_color = "Unknown"
                                vehicle_found = False
                                r2_object_url = None
                                local_object_path = None
                                v_res = vehicle_model.predict(raw_frame, verbose=False, classes=[2,3,5,7], conf=0.3)[0]
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
                                        vehicle_found = True
                                        break
                                if not vehicle_found:
                                    v_color = get_vehicle_color(plate_crop)
                                # Save to PostgreSQL (R2 Backup)
                                save_to_db({
                                    "task_id": task_id,
                                    "filename": input_path.split("/")[-1],
                                    "timestamp": round(frame_count / fps, 2),
                                    "trigger": trigger_name,
                                    "event": f"New Detection (ID: {obj_id})",
                                    "image_plate_url": r2_plate_url,
                                    "image_object_url": r2_object_url,
                                    "plate_number": plate_text or "UNREADABLE",
                                    "vehicle_color": v_color
                                })

                                # Update UI Logs (Local URLs work immediately)
                                logs.append({
                                    "timestamp": round(frame_count / fps, 2),
                                    "trigger": trigger_name,
                                    "event": f"New Detection (ID: {obj_id})",
                                    "image_plate": image_plate_url,
                                    "image_object": local_object_path,
                                    "plate_number": plate_text or "UNREADABLE",
                                    "vehicle_color": v_color,
                                    "saved_to_r2": True if r2_plate_url else False
                                })
                        else:
                            processed_track_ids.add(unique_track_key)
                            save_to_db({
                                "task_id": task_id,
                                "filename": input_path.split("/")[-1],
                                "timestamp": round(frame_count / fps, 2),
                                "trigger": trigger_name,
                                "event": f"{trigger_name} detected (ID: {obj_id})",
                                "image_plate_url": None, "image_object_url": None, "plate_number": None, "vehicle_color": None
                            })
                            logs.append({
                                "timestamp": round(frame_count / fps, 2),
                                "trigger": trigger_name,
                                "event": f"{trigger_name} detected (ID: {obj_id})",
                                "image_plate": None, "image_object": None, "plate_number": None
                            })
        out.write(frame)
    cap.release()
    out.release()
    try:
        subprocess.run(['ffmpeg', '-y', '-i', temp_output, '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', '-movflags', '+faststart', '-pix_fmt', 'yuv420p', output_path], check=True, capture_output=True)
        if os.path.exists(temp_output): os.remove(temp_output)
    except Exception as e:
        print(f"DEBUG: Video optimization FAILED: {e}")
        if os.path.exists(temp_output): os.rename(temp_output, output_path)
    return logs
