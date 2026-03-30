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

# OCR & Cloud Storage Setup
load_dotenv()
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

r2 = None
if all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET]):
    try:
        r2 = boto3.client('s3', endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY)
    except: r2 = None

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)
except:
    reader = None

MODEL_MAP = {
    "Number Plate Detection": "ANPR.pt",
    "Helmet Detection": "helmet.pt",
    "Triple Riding Detection": "triple_riding.pt",
    "Seatbelt Detection": "seatbelt.pt"
}

def ocr(crop):
    if crop is None or crop.size == 0: return ""
    try:
        _, buf = cv2.imencode('.jpg', crop)
        res = requests.post('https://api.platerecognizer.com/v1/plate-reader/',
            files={'upload': ('p.jpg', buf.tobytes())},
            headers={'Authorization': 'Token cabf3c65d1ec04ff52c1d5d0489fb083cdd2e305'},
            timeout=8)
        if res.status_code in [200, 201]:
            d = res.json()
            if d.get('results'):
                return re.sub(r'[^A-Z0-9]', '', d['results'][0].get('plate', '').upper())
    except: pass
    if reader:
        try:
            res = reader.readtext(crop)
            if res: return re.sub(r'[^A-Z0-9]', '', "".join([r[1] for r in res]).upper())
        except: pass
    return ""

def upload(img, trigger, session_id):
    # LOCAL SAVE ALWAYS as fallback
    try:
        ts = int(time.time()*1000)
        local_dir = os.path.join("static", "crops", str(session_id))
        os.makedirs(local_dir, exist_ok=True)
        fname = f"{trigger}_{ts}.jpg"
        fpath = os.path.join(local_dir, fname)
        cv2.imwrite(fpath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        local_url = f"/static/crops/{session_id}/{fname}"
        
        # Cloud attempt
        if r2:
            key = f"rec_{session_id}/{trigger}/{fname}"
            _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            r2.put_object(Bucket=R2_BUCKET, Key=key, Body=buf.tobytes(), ContentType='image/jpeg')
            return f"{R2_PUBLIC_URL.rstrip('/')}/{key}" if R2_PUBLIC_URL else key
            
        return local_url
    except: return None

def get_model(name):
    path = os.path.join("models", MODEL_MAP.get(name, "yolov8n.pt"))
    if not os.path.exists(path): path = "yolov8n.pt"
    return YOLO(path)

class LiveCameraProcessor:
    def __init__(self, camera_id, camera_link, selected_triggers):
        self.camera_id, self.camera_link = camera_id, camera_link
        self.latest_frame, self.latest_jpeg = None, None
        self.is_running = True
        self.logs, self.seen_plates = [], set()
        self.is_recording, self.writer = False, None
        self.recording_session_id = None
        self.current_recording_path = None
        
        self.trigger_list = selected_triggers
        self.models = {t: get_model(t) for t in self.trigger_list if t in MODEL_MAP}
        self.vehicle_model = YOLO("yolov8n.pt")
        
        import threading
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def update_triggers(self, new_list):
        self.trigger_list = new_list
        self.models = {t: get_model(t) for t in self.trigger_list if t in MODEL_MAP}

    def _run(self):
        cap = cv2.VideoCapture(self.camera_link)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        while self.is_running:
            ret, frame = cap.read()
            if not ret: time.sleep(1); cap = cv2.VideoCapture(self.camera_link); continue
            
            self.latest_frame = frame.copy()
            vehicle_res = self.vehicle_model.track(frame, persist=True, verbose=False, classes=[2,3,5,7])[0]
            
            for t_name, model in self.models.items():
                res = model.track(frame, persist=True, verbose=False, conf=0.4)[0]
                if not res.boxes or res.boxes.id is None: continue
                
                for box, obj_id, cls_id in zip(res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.id.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int)):
                    label = model.names[cls_id]
                    x1, y1, x2, y2 = box
                    
                    plate_text = ""
                    if t_name == "Number Plate Detection":
                        crop_p = self.latest_frame[y1:y2, x1:x2]
                        plate_text = ocr(crop_p)
                        if not plate_text or plate_text in self.seen_plates: continue
                        self.seen_plates.add(plate_text)
                    elif f"{t_name}_{obj_id}" in self.seen_plates: continue
                    else: self.seen_plates.add(f"{t_name}_{obj_id}")

                    v_crop = self.latest_frame[y1:y2, x1:x2]
                    if vehicle_res.boxes and vehicle_res.boxes.id is not None:
                        for v_box in vehicle_res.boxes.xyxy.cpu().numpy().astype(int):
                            vx1, vy1, vx2, vy2 = v_box
                            if x1 >= vx1 and x2 <= vx2 and y1 >= vy1 and y2 <= vy2:
                                v_crop = self.latest_frame[vy1:vy2, vx1:vx2]
                                break
                    
                    url_p = upload(self.latest_frame[y1:y2, x1:x2], "Plate", self.recording_session_id or "live")
                    url_v = upload(v_crop, "Vehicle", self.recording_session_id or "live")
                    
                    log = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "trigger": t_name, "event": f"Detected {label}",
                        "plate_number": plate_text,
                        "image_plate": url_p, "image_object": url_v,
                        "saved_to_r2": "http" in str(url_p)
                    }
                    self.logs.append(log)
                    self._persist(log)

            if self.is_recording and self.writer:
                cv2.putText(frame, f"REC: {datetime.now().strftime('%H:%M:%S')}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                self.writer.write(frame)

            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            self.latest_jpeg = jpeg.tobytes()
        cap.release()

    def _persist(self, log):
        db = SessionLocal()
        try:
            d = Detection(
                camera_id=self.camera_id, trigger_type=log["trigger"],
                event_description=log["event"], plate_number=log["plate_number"],
                image_plate_url=log["image_plate"], image_object_url=log["image_object"],
                saved_to_r2=log["saved_to_r2"]
            )
            db.add(d); db.commit()
        except: db.rollback()
        finally: db.close()

    def start_recording(self, initiated_by, video_name, source):
        if self.is_recording: return False, "Already"
        self.recording_session_id = str(uuid.uuid4())
        fname = f"{video_name}_{int(time.time())}.mp4"
        self.current_recording_path = os.path.join("static", "recordings", fname)
        h, w = self.latest_frame.shape[:2] if self.latest_frame is not None else (720, 1280)
        # AVC1 is better for web but might need a re-encode to be 100% safe
        self.writer = cv2.VideoWriter(self.current_recording_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
        db = SessionLocal()
        try:
            s = RecordingSession(id=self.recording_session_id, camera_id=self.camera_id, video_name=fname, file_path=f"/static/recordings/{fname}", source=source, started_at=func.now())
            db.add(s); db.commit()
        except: db.rollback(); return False, "DB Error"
        finally: db.close()
        self.is_recording = True
        return True, self.recording_session_id

    def stop_recording(self, stopped_by):
        self.is_recording = False
        if self.writer: self.writer.release(); self.writer = None
        # RE-ENCODE for WEB PLAYBACK
        if self.current_recording_path:
            out_path = self.current_recording_path + ".mp4"
            try:
                subprocess.run(['ffmpeg', '-y', '-i', self.current_recording_path, '-vcodec', 'libx264', '-preset', 'ultrafast', out_path], check=True)
                os.replace(out_path, self.current_recording_path)
            except: pass
        return True, self.recording_session_id

def process_video(tid, inp, out, trigs):
    logs = []
    seen = set()
    models = {t: get_model(t) for t in trigs if t in MODEL_MAP}
    cap = cv2.VideoCapture(inp)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS) or 20
    writer = cv2.VideoWriter(out + ".tmp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    f_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        f_count += 1
        for t_name, model in models.items():
            res = model.track(frame, persist=True, verbose=False)[0]
            if not res.boxes or res.boxes.id is None: continue
            for box, obj_id, cls_id in zip(res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.id.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int)):
                x1,y1,x2,y2 = box
                plate = ""
                if t_name == "Number Plate Detection":
                    plate = ocr(frame[y1:y2, x1:x2])
                    if not plate or plate in seen: continue
                    seen.add(plate)
                elif f"{t_name}_{obj_id}" in seen: continue
                else: seen.add(f"{t_name}_{obj_id}")
                logs.append({"timestamp": round(f_count/fps, 2), "trigger": t_name, "plate_number": plate, "event": f"Detected {model.names[cls_id]}"})
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        writer.write(frame)
    cap.release(); writer.release()
    subprocess.run(['ffmpeg', '-y', '-i', out + ".tmp.mp4", '-vcodec', 'libx264', out])
    return logs
