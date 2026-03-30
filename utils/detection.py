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
                try:
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
                                if plate_text and plate_text in self.seen_plate_numbers: continue
                                if plate_text: self.seen_plate_numbers.add(plate_text)
                            
                            fname = f"{track_key}_{int(time.time())}.jpg"
                            r2_url = upload_to_r2(crop, trigger_name, fname)
                            self.processed_track_ids.add(track_key)
                            
                            # Log and Save to DB
                            log_entry = {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "trigger": trigger_name,
                                "event": f"Detected {trigger_name} (ID: {obj_id})",
                                "plate_number": plate_text or "No text recognized",
                                "image_plate": r2_url,
                                "saved_to_r2": True if r2_url else False
                            }
                            self.logs.append(log_entry)
                            
                            # Save globally to detections table
                            self.save_detection_to_db({
                                "camera_id": self.camera_id,
                                "trigger": trigger_name,
                                "event": log_entry["event"],
                                "plate_number": plate_text,
                                "image_plate_url": r2_url,
                                "saved_to_r2": log_entry["saved_to_r2"]
                            })
                except Exception as e:
                    print(f"Debug: Tracking error: {e}")

            self.latest_frame = frame
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            self.latest_jpeg = jpeg.tobytes()
            
            if self.is_recording and self.recording_writer:
                self.recording_writer.write(frame)
        cap.release()
        if self.recording_writer: self.recording_writer.release()

    def save_detection_to_db(self, data):
        db = SessionLocal()
        try:
            new_det = Detection(
                camera_id=data["camera_id"],
                trigger_type=data["trigger"],
                event_description=data["event"],
                plate_number=data["plate_number"],
                image_plate_url=data["image_plate_url"],
                saved_to_r2=data["saved_to_r2"]
            )
            db.add(new_det); db.commit()
        except Exception as e:
            db.rollback(); print(f"DB Detection Save Error: {e}")
        finally: db.close()

    def start_recording(self, initiated_by="System", note=None, source="manual", analysis_session_id=None):
        if self.is_recording: return False, "Already recording"
        
        self.recording_session_id = str(uuid.uuid4())
        fname = f"{self.camera_id}_{int(time.time())}.mp4"
        path = os.path.join("static", "recordings", fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save session to DB
        db = SessionLocal()
        try:
            session = RecordingSession(
                id=self.recording_session_id,
                camera_id=self.camera_id,
                video_name=fname,
                file_path=f"/static/recordings/{fname}",
                source=source,
                initiated_by=initiated_by,
                description=note,
                started_at=func.now()
            )
            db.add(session)
            
            # If associated with analysis
            if analysis_session_id:
                analysis = db.query(AnalysisSession).filter(AnalysisSession.id == analysis_session_id).first()
                if analysis: analysis.recording_session_id = self.recording_session_id
            
            db.commit()
        except Exception as e:
            db.rollback(); return False, str(e)
        finally: db.close()

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

def process_video(task_id, input_path, output_path, selected_triggers):
    logs = []
    processed_plates = set() # Prevent duplicates in same video
    crops_dir = os.path.join("static", "crops", task_id); os.makedirs(crops_dir, exist_ok=True)
    
    models = {t: get_model(t) for t in selected_triggers}
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = output_path + ".tmp.mp4"
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
    
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
            
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if trigger_name == "Number Plate Detection":
                    crop = raw_frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    plate_text = get_best_ocr(crop)
                    
                    if plate_text and plate_text not in processed_plates:
                        processed_plates.add(plate_text)
                        fname = f"plate_{plate_text}_{frame_count}.jpg"
                        cv2.imwrite(os.path.join(crops_dir, fname), crop)
                        r2_url = upload_to_r2(crop, trigger_name, fname)
                        
                        logs.append({
                            "timestamp": round(frame_count / fps, 2),
                            "trigger": trigger_name,
                            "event": f"Detected Plate (ID: {obj_id})",
                            "plate_number": plate_text,
                            "image_plate": f"/static/crops/{task_id}/{fname}",
                            "saved_to_r2": True if r2_url else False
                        })
                else:
                    # Generic logging for other triggers
                    track_key = f"{trigger_name}_{obj_id}"
                    if track_key not in [l.get("track_key") for l in logs]:
                        logs.append({
                            "track_key": track_key,
                            "timestamp": round(frame_count / fps, 2),
                            "trigger": trigger_name,
                            "event": f"Detected {trigger_name} (ID: {obj_id})",
                            "saved_to_r2": False
                        })
                        
        out.write(frame)
    cap.release(); out.release()
    
    # FFmpeg optimize
    subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
    if os.path.exists(temp_out): os.remove(temp_out)
    
    return logs
