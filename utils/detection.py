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
from utils.db import db, DETECTIONS, CAMERAS, RECORDINGS, get_iso_now
from datetime import datetime
from dotenv import load_dotenv

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
            return f"{R2_PUBLIC_URL.rstrip('/')}/{key}"
        return f"{R2_ENDPOINT}/{R2_BUCKET}/{key}"
    except Exception as e:
        print(f"DEBUG: R2 Upload FAILED: {e}")
        return None

# OCR INITIALIZATION
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True) 
except: reader = None

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
        img_rgb = cv2.resize(img_rgb, (32, 32)) # smaller for speed
        pixels = img_rgb.reshape(-1, 3)
        counts = np.bincount(pixels[:, 0]) # Simple heuristic for speed
        # For simplicity, returning silver for this demo
        return "Silver"
    except: return "Unknown"

def get_best_ocr(crop_img):
    if crop_img is None or crop_img.size == 0: return ""
    # 1. Plate Recognizer (Cloud)
    try:
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        response = requests.post('https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=['in']),
            files=dict(upload=('plate.jpg', buffer.tobytes())),
            headers={'Authorization': 'Token cabf3c65d1ec04ff52c1d5d0489fb083cdd2e305'},
            timeout=10)
        if response.status_code < 300:
            res_data = response.json()
            if res_data.get('results'):
                return re.sub(r'[^A-Z0-9]', '', res_data['results'][0].get('plate', '').upper())
    except: pass
    
    # 2. Local Fallback
    try:
        if reader:
            res = reader.readtext(crop_img)
            if res:
                return re.sub(r'[^A-Z0-9]', '', "".join([r[1] for r in res]).upper())
    except: pass
    return ""

_models_cache = {}
def get_model(t):
    fname = MODEL_MAP.get(t, "yolov8n.pt")
    if fname not in _models_cache:
        path = os.path.join("models", fname)
        if not os.path.exists(path): path = "yolov8n.pt"
        _models_cache[fname] = YOLO(path)
    return _models_cache[fname]

class LiveCameraProcessor:
    def __init__(self, camera_id, camera_link, selected_triggers):
        self.camera_id, self.camera_link = camera_id, camera_link
        self.selected_triggers = selected_triggers
        self.is_running = True
        self.latest_jpeg = None
        self.logs, self.processed_track_ids, self.seen_plates = [], set(), set()
        self.models = {t: get_model(t) for t in selected_triggers}
        self.v_model = YOLO("yolov8n.pt")
        
        self.is_recording = False
        self.rec_writer = None
        self.rec_name = "Untitled_Rec"
        self.rec_id = None
        
        import threading
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cap = cv2.VideoCapture(self.camera_link)
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break
            
            # Analyze
            raw_frame = frame.copy()
            h, w = frame.shape[:2]
            
            for trigger_name, model in self.models.items():
                if not model: continue
                results = model.track(frame, persist=True, verbose=False, conf=0.3)[0]
                if not (results.boxes and results.boxes.id is not None): continue
                
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                
                for box, obj_id in zip(boxes, ids):
                    track_key = f"{trigger_name}_{obj_id}"
                    x1, y1, x2, y2 = box
                    
                    # Highlight
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if track_key not in self.processed_track_ids:
                        self.processed_track_ids.add(track_key)
                        
                        # TIGHT CROP for plate
                        p_pad = 5
                        px1, py1 = max(0, x1-p_pad), max(0, y1-p_pad)
                        px2, py2 = min(w, x2+p_pad), min(h, y2+p_pad)
                        p_crop = raw_frame[py1:py2, px1:px2]
                        
                        plate_text = get_best_ocr(p_crop) if trigger_name == "Number Plate Detection" else ""
                        
                        # Find Vehicle (LARGE CROP)
                        v_crop, v_url, v_color = None, None, "Unknown"
                        v_res = self.v_model.predict(raw_frame, verbose=False, classes=[2,3,5,7], conf=0.4)[0]
                        for v_box in v_res.boxes:
                            vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                            if vx1<=(x1+x2)/2<=vx2 and vy1<=(y1+y2)/2<=vy2:
                                v_crop = raw_frame[vy1:vy2, vx1:vx2]
                                v_url = upload_to_r2(v_crop, "Vehicle", f"v_{obj_id}.jpg")
                                v_color = get_vehicle_color(v_crop)
                                break
                        
                        # Upload Plate
                        p_url = upload_to_r2(p_crop, trigger_name, f"p_{obj_id}.jpg")
                        
                        # Create Log
                        log_entry = {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "trigger": trigger_name,
                            "event": f"Detected {trigger_name} (ID: {obj_id})",
                            "plate_number": plate_text or "No text",
                            "image_plate": p_url,
                            "image_object": v_url,
                            "vehicle_color": v_color,
                            "saved_to_r2": True if p_url else False
                        }
                        self.logs.append(log_entry)
                        # Save to MongoDB async (done in main.py)
            
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            self.latest_jpeg = jpeg.tobytes()
            
            if self.is_recording and self.rec_writer:
                self.rec_writer.write(frame)
                
        cap.release()
        if self.rec_writer: self.rec_writer.release()

    def start_rec(self, name):
        if self.is_recording: return
        self.rec_name = name
        self.rec_id = str(uuid.uuid4())
        fname = f"{self.rec_id}_{int(time.time())}.mp4"
        path = os.path.join("static", "recordings", fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Using simpler codec for compatibility
        self.rec_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))
        self.is_recording = True
        return self.rec_id

    def stop_rec(self):
        self.is_recording = False
        if self.rec_writer:
            self.rec_writer.release()
            self.rec_writer = None
        return self.rec_id

def process_video(task_id, input_path, output_path, triggers):
    logs = []
    crops_dir = os.path.join("static", "crops", task_id); os.makedirs(crops_dir, exist_ok=True)
    models = {t: get_model(t) for t in triggers}
    cap = cv2.VideoCapture(input_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30
    temp_out = output_path + ".tmp.mp4"; out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_count = 0
    seen_ids = set()
    v_model = YOLO("yolov8n.pt")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        raw_frame = frame.copy()
        
        for t, m in models.items():
            if not m: continue
            res = m.track(frame, persist=True, verbose=False, conf=0.3)[0]
            if not (res.boxes and res.boxes.id is not None): continue
            
            for box, obj_id in zip(res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.id.cpu().numpy().astype(int)):
                track_key = f"{t}_{obj_id}"
                if track_key not in seen_ids:
                    seen_ids.add(track_key)
                    x1, y1, x2, y2 = box
                    
                    # CROP TIGHT
                    pad = 5
                    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                    cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                    crop = raw_frame[cy1:cy2, cx1:cx2]
                    
                    plate_text = get_best_ocr(crop) if t == "Number Plate Detection" else ""
                    fname = f"p_{obj_id}_{frame_count}.jpg"
                    cv2.imwrite(os.path.join(crops_dir, fname), crop)
                    
                    # Vehicle detect for LARGE crop
                    v_url = None
                    v_res = v_model.predict(raw_frame, verbose=False, classes=[2,3,5,7], conf=0.4)[0]
                    for v_box in v_res.boxes:
                        vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                        if vx1<=(x1+x2)/2<=vx2 and vy1<=(y1+y2)/2<=vy2:
                            v_crop = raw_frame[vy1:vy2, vx1:vx2]
                            v_fname = f"v_{obj_id}_{frame_count}.jpg"
                            cv2.imwrite(os.path.join(crops_dir, v_fname), v_crop)
                            v_url = f"/static/crops/{task_id}/{v_fname}"
                            break
                    
                    logs.append({
                        "timestamp": round(frame_count / fps, 2),
                        "trigger": t,
                        "event": f"Detection (ID: {obj_id})",
                        "plate_number": plate_text or "No text",
                        "image_plate": f"/static/crops/{task_id}/{fname}",
                        "image_object": v_url or f"/static/crops/{task_id}/{fname}",
                        "saved_to_r2": False
                    })
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        out.write(frame)
    cap.release(); out.release()
    subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
    if os.path.exists(temp_out): os.remove(temp_out)
    return logs
