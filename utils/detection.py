import cv2
import os
import time
import uuid
import numpy as np
from ultralytics import YOLO
import requests
import re
import boto3
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# R2 Configuration
r2_endpoint = os.getenv("R2_ENDPOINT_URL")
r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
r2_bucket = os.getenv("R2_BUCKET_NAME")

r2 = None
if all([r2_endpoint, r2_access_key, r2_secret_key, r2_bucket]):
    try:
        r2 = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            region_name='auto'
        )
        print("Debug: Cloudflare R2 Client Initialized Successfully.")
    except Exception as e:
        print(f"Debug: Error initializing R2 client: {e}")

def upload_to_r2(img, trigger_name, filename):
    """
    Upload an image to Cloudflare R2, organized by trigger, date, and time.
    """
    if r2 is None or img is None or img.size == 0:
        return None
        
    try:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Clean trigger name for key
        clean_trigger = re.sub(r'[^a-zA-Z0-9]', '_', trigger_name)
        
        # Key structure: section/date/time/unique_id.jpg
        key = f"{clean_trigger}/{date_str}/{time_str}/{filename}"
        
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        r2.put_object(
            Bucket=r2_bucket,
            Key=key,
            Body=buffer.tobytes(),
            ContentType='image/jpeg'
        )
        full_url = f"{r2_endpoint}/{r2_bucket}/{key}"
        print(f"Debug: SUCCESS! Saved to R2: {full_url}")
        return full_url
    except Exception as e:
        print(f"Debug: R2 Upload Error: {e}")
        return None

def upload_video_to_r2(local_path, filename):
    """
    Upload a video file to Cloudflare R2.
    """
    if r2 is None:
        return None
    try:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        key = f"Videos/{date_str}/{filename}"
        
        with open(local_path, "rb") as f:
            r2.put_object(
                Bucket=r2_bucket,
                Key=key,
                Body=f,
                ContentType='video/mp4'
            )
        full_url = f"{r2_endpoint}/{r2_bucket}/{key}"
        print(f"Debug: SUCCESS! Video saved to R2: {full_url}")
        return full_url
    except Exception as e:
        print(f"Debug: R2 Video Upload Error: {e}")
        return None

# FORCE TCP for RTSP globally before any CV2 operations
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# PLATE RECOGNIZER CLOUD API CONFIGURATION
PLATE_RECOGNIZER_TOKEN = "b7352531317f55e343ae5b91a74df3a79bf7bbd5"
PLATE_RECOGNIZER_URL = 'https://api.platerecognizer.com/v1/plate-reader/'

# OCR INITIALIZATION
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True) 
except ImportError:
    reader = None
    print("Warning: easyocr not installed.")

# Required libraries
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

def get_best_ocr(crop_img):
    if crop_img is None or crop_img.size == 0:
        return ""
    try:
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_bytes = buffer.tobytes()
        response = requests.post(
            PLATE_RECOGNIZER_URL,
            data=dict(regions=['in']),
            files=dict(upload=('plate.jpg', img_bytes)),
            headers={'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'},
            timeout=10
        )
        if response.status_code in [200, 201]:
            res_data = response.json()
            if res_data.get('results'):
                plate_text = res_data['results'][0].get('plate', '').upper()
                clean_text = re.sub(r'[^A-Z0-9]', '', plate_text)
                if clean_text:
                    return clean_text
    except Exception as e:
        print(f"Debug: Plate Recognizer API Connection Error: {e}")
    try:
        if reader:
            res = reader.readtext(crop_img)
            if res:
                all_text = "".join([r[1] for r in res]).upper()
                clean_local = re.sub(r'[^A-Z0-9]', '', all_text)
                if clean_local:
                    return clean_local
    except Exception as local_err:
        print(f"Debug: Local OCR Fallback Error: {local_err}")
    return ""

_loaded_models_cache = {}

def get_model(trigger_name):
    if trigger_name == "General": model_filename = "yolov8n.pt"
    else: model_filename = MODEL_MAP.get(trigger_name, "yolov8n.pt")
    
    cache_key = model_filename
    if cache_key not in _loaded_models_cache:
        models_dir = "models"
        model_path = os.path.join(models_dir, model_filename)
        if not os.path.exists(model_path):
            model_path = model_filename # Let Ultralytics download it
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
        self.general_model = get_model("General")
        import threading
        self.orchestrator = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestrator.start()

    def _orchestration_loop(self):
        self.is_running = True
        configs = [
            {"name": "FFmpeg TCP", "options": "rtsp_transport;tcp"},
            {"name": "FFmpeg UDP", "options": "rtsp_transport;udp"},
            {"name": "Default", "options": None}
        ]
        connected = False
        for config in configs:
            opt = config.get('options')
            if opt: os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opt
            elif "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ: del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            
            self.cap = cv2.VideoCapture(self.camera_link, cv2.CAP_FFMPEG if config['name'].startswith("FFmpeg") else 0)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    connected = True
                    break
                else: self.cap.release()
        if not connected:
            self.status = "failed"
            self.is_running = False
            return
        self.status = "connected"
        try:
            from utils.database import save_task_to_db
            save_task_to_db(self.camera_id, self.camera_link, "streaming", None)
        except: pass
        import threading
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
                frame = self.raw_frame_buffer.copy()
                h, w = frame.shape[:2]
                if w > 1280: frame = cv2.resize(frame, (1280, int(h * (1280/w))))
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
                            key = f"{trigger_name}_{obj_id}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID:{obj_id}", (x1, y1-5), 0, 0.4, (0,255,0), 1)
                            if key not in self.processed_track_ids:
                                log_entry = None
                                if trigger_name == "Number Plate Detection":
                                    plate_crop = raw_frame[max(0, y1-15):min(h, y2+15), max(0, x1-15):min(w, x2+15)]
                                    if plate_crop.size > 0:
                                        plate_text = get_best_ocr(plate_crop)
                                        if plate_text and plate_text in self.seen_plate_numbers: continue
                                        if plate_text: self.seen_plate_numbers.add(plate_text)
                                        self.processed_track_ids.add(key)
                                        p_fname = f"plate_{obj_id}_{int(time.time())}.jpg"
                                        cv2.imwrite(os.path.join(crops_dir, p_fname), plate_crop)
                                        r2_u = upload_to_r2(plate_crop, trigger_name, p_fname)
                                        v_col, v_r2, local_v_url = "Unknown", None, None
                                        if self.general_model:
                                            v_res = self.general_model.predict(raw_frame, verbose=False, classes=[2,3,5,7])[0]
                                            for v_box in v_res.boxes:
                                                vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                                                if vx1<=(x1+x2)/2<=vx2 and vy1<=(y1+y2)/2<=vy2:
                                                    v_crop = raw_frame[vy1:vy2, vx1:vx2]
                                                    v_col = get_vehicle_color(v_crop)
                                                    v_fn = f"v_{obj_id}_{int(time.time())}.jpg"
                                                    cv2.imwrite(os.path.join(crops_dir, v_fn), v_crop)
                                                    v_r2 = upload_to_r2(v_crop, trigger_name, v_fn)
                                                    local_v_url = f"/static/crops/{self.camera_id}/{v_fn}"
                                                    break
                                        if not v_r2: v_col = get_vehicle_color(plate_crop)
                                        log_entry = {
                                            "timestamp": time.strftime("%H:%M:%S"), "trigger": trigger_name,
                                            "event": f"Detection (ID: {obj_id})", "image_plate": f"/static/crops/{self.camera_id}/{p_fname}",
                                            "image_object": local_v_url, "plate_number": plate_text or "UNREADABLE",
                                            "vehicle_color": v_col, "r2_saved": r2_u is not None, "r2_url": r2_u,
                                            "image_plate_r2": r2_u, "image_object_r2": v_r2
                                        }
                                else:
                                    self.processed_track_ids.add(key)
                                    o_crop = raw_frame[y1:y2, x1:x2]
                                    o_fn = f"{re.sub(r'[^a-zA-Z0-9]', '_', trigger_name)}_{obj_id}_{int(time.time())}.jpg"
                                    cv2.imwrite(os.path.join(crops_dir, o_fn), o_crop)
                                    r2_u = upload_to_r2(o_crop, trigger_name, o_fn)
                                    log_entry = {
                                        "timestamp": time.strftime("%H:%M:%S"), "trigger": trigger_name,
                                        "event": f"{trigger_name} (ID: {obj_id})", "image_plate": None,
                                        "image_object": f"/static/crops/{self.camera_id}/{o_fn}",
                                        "plate_number": None, "r2_saved": r2_u is not None, "r2_url": r2_u,
                                        "image_object_r2": r2_u
                                    }
                                if log_entry:
                                    self.logs.append(log_entry)
                                    try:
                                        from utils.database import save_logs_to_db
                                        save_logs_to_db(self.camera_id, [log_entry])
                                    except: pass
                self.latest_frame = frame
            except Exception as e:
                print(f"[DEBUG] Processing error: {e}")
                time.sleep(0.1)
    def stop(self):
        self.is_running = False
        if hasattr(self, 'cap') and self.cap: self.cap.release()

def process_video(task_id, input_path, output_path, selected_triggers):
    logs, crops_dir = [], os.path.join("static", "crops", task_id)
    os.makedirs(crops_dir, exist_ok=True)
    models = {t: get_model(t) for t in selected_triggers if get_model(t)}
    gen_model = get_model("General")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return {"logs": [], "video_url_r2": None}
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30
    if width > 1920: scale = 1920/width; width, height = 1920, int(height*scale)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    proc_ids, seen_plates, f_count = set(), set(), 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        f_count += 1
        frame = cv2.resize(frame, (width, height))
        raw = frame.copy()
        for t_name, model in models.items():
            results = model.track(frame, persist=True, verbose=False, iou=0.5)[0]
            if results.boxes.id is not None:
                boxes, ids, confs = results.boxes.xyxy.cpu().numpy().astype(int), results.boxes.id.cpu().numpy().astype(int), results.boxes.conf.cpu().numpy()
                for box, obj_id, conf in zip(boxes, ids, confs):
                    if conf < 0.4: continue
                    x1, y1, x2, y2 = box
                    key = f"{t_name}_{obj_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if key not in proc_ids:
                        if t_name == "Number Plate Detection":
                            p_crop = raw[max(0, y1-15):min(height, y2+15), max(0, x1-15):min(width, x2+15)]
                            if p_crop.size > 0:
                                p_text = get_best_ocr(p_crop)
                                if p_text and p_text in seen_plates: continue
                                if p_text: seen_plates.add(p_text)
                                proc_ids.add(key)
                                p_fn = f"plate_{obj_id}_{f_count}.jpg"
                                cv2.imwrite(os.path.join(crops_dir, p_fn), p_crop)
                                r2_u = upload_to_r2(p_crop, t_name, p_fn)
                                v_col, v_r2, v_fn = "Unknown", None, None
                                if gen_model:
                                    v_res = gen_model.predict(raw, verbose=False, classes=[2,3,5,7])[0]
                                    for v_box in v_res.boxes:
                                        vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                                        if vx1<=(x1+x2)/2<=vx2 and vy1<=(y1+y2)/2<=vy2:
                                            v_crop = raw[vy1:vy2, vx1:vx2]
                                            v_col = get_vehicle_color(v_crop)
                                            v_fn = f"v_{obj_id}_{f_count}.jpg"
                                            cv2.imwrite(os.path.join(crops_dir, v_fn), v_crop)
                                            v_r2 = upload_to_r2(v_crop, t_name, v_fn)
                                            break
                                if not v_r2: v_col = get_vehicle_color(p_crop)
                                logs.append({
                                    "timestamp": round(f_count/fps, 2), "trigger": t_name, "event": f"New Detection (ID: {obj_id})",
                                    "image_plate": f"/static/crops/{task_id}/{p_fn}", "image_object": f"/static/crops/{task_id}/{v_fn}" if v_fn else None,
                                    "plate_number": p_text or "UNREADABLE", "vehicle_color": v_col, "r2_saved": r2_u is not None,
                                    "r2_url": r2_u, "image_plate_r2": r2_u, "image_object_r2": v_r2
                                })
                        else:
                            proc_ids.add(key)
                            o_crop = raw[y1:y2, x1:x2]
                            o_fn = f"{t_name}_{obj_id}_{f_count}.jpg"
                            cv2.imwrite(os.path.join(crops_dir, o_fn), o_crop)
                            r2_u = upload_to_r2(o_crop, t_name, o_fn)
                            logs.append({
                                "timestamp": round(f_count/fps, 2), "trigger": t_name, "event": f"{t_name} (ID: {obj_id})",
                                "image_plate": None, "image_object": f"/static/crops/{task_id}/{o_fn}",
                                "plate_number": None, "r2_saved": r2_u is not None, "r2_url": r2_u,
                                "image_object_r2": r2_u
                            })
        out.write(frame)
    cap.release(); out.release()
    v_url = upload_video_to_r2(output_path, os.path.basename(output_path))
    return {"logs": logs, "video_url_r2": v_url}
