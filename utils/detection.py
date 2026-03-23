import cv2
import os
import time
import uuid
import numpy as np
from ultralytics import YOLO
import requests
import re

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
    """
    Detect the dominant color of the vehicle crop.
    Uses KMeans if available, otherwise falls back to median color.
    """
    if img is None or img.size == 0:
        return "Unknown"
        
    try:
        # Resize to speed up
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
            # Fallback to median color
            dominant_rgb = np.median(img_rgb, axis=(0, 1)).astype(int)

        r, g, b = dominant_rgb
        # ... (rest of classification same as before)

        # Simple manual classification for speed and common names
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
    """
    Perform OCR using Plate Recognizer Cloud API for maximum accuracy.
    Falls back to EasyOCR if API fails.
    """
    if crop_img is None or crop_img.size == 0:
        return ""
    
    try:
        # Convert OpenCV image (numpy array) to bytes
        # High quality JPEG for better OCR resolution
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_bytes = buffer.tobytes()
        
        # 1. PRIMARY: Plate Recognizer API
        response = requests.post(
            PLATE_RECOGNIZER_URL,
            data=dict(regions=['in']), # Set to 'in' for India
            files=dict(upload=('plate.jpg', img_bytes)),
            headers={'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'},
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            res_data = response.json()
            if res_data.get('results'):
                # Extract first plate result
                plate_text = res_data['results'][0].get('plate', '').upper()
                # Clean characters using regex
                clean_text = re.sub(r'[^A-Z0-9]', '', plate_text)
                if clean_text:
                    print(f"Debug: Plate Recognizer Result: {clean_text}")
                    return clean_text
        else:
            print(f"Debug: Plate Recognizer API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Debug: Plate Recognizer API Connection Error: {e}")
        
    # 2. FALLBACK: Use local OCR (EasyOCR/Paddle) if API fails
    try:
        # Simplified local extraction as backup
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

# MODEL CACHE TO AVOID REDUNDANT LOADING
_loaded_models_cache = {}

def get_model(trigger_name):
    """
    Get or load a model from cache.
    """
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
    """
    Handles live camera stream processing with async-first connection architecture.
    Does NOT block during initialization, allowing the UI to remain responsive.
    """
    def __init__(self, camera_id, camera_link, selected_triggers):
        self.camera_id = camera_id
        self.camera_link = camera_link
        self.selected_triggers = selected_triggers
        self.is_running = False
        self.status = "connecting" # Track status for UI feedback
        self.latest_frame = None
        self.raw_frame_buffer = None
        self.logs = []
        self.processed_track_ids = set()
        self.seen_plate_numbers = set()
        
        # Load models from cache immediately (fast if cached)
        self.models = {t: get_model(t) for t in selected_triggers}
        self.general_model = get_model("Number Plate Detection") if "Number Plate Detection" in selected_triggers else None
        
        # Start only the main orchestration thread immediately
        import threading
        self.orchestrator = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestrator.start()
        print(f"[DEBUG] Camera {camera_id}: Connection task started in background.")

    def _orchestration_loop(self):
        """Orchestrates the connection lifecycle with robust fallbacks."""
        self.is_running = True
        
        # Connection Tiers: 
        # 1. TCP (Most stable, good for poor networks)
        # 2. UDP (Fastest, works on some restrictive cameras)
        # 3. Default (Let OpenCV decide)
        
        connection_configs = [
            {"name": "FFmpeg TCP", "options": "rtsp_transport;tcp"},
            {"name": "FFmpeg UDP", "options": "rtsp_transport;udp"},
            {"name": "Default Backend", "options": None}
        ]
        
        connected = False
        for config in connection_configs:
            c_name = config.get('name', 'Unknown')
            c_options = config.get('options')
            
            print(f"[DEBUG] Attempting {c_name} connection to: {self.camera_link}")
            
            if c_options:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = str(c_options)
            else:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            
            # Use appropriate backend
            if c_name.startswith("FFmpeg"):
                self.cap = cv2.VideoCapture(self.camera_link, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.camera_link)
                
            if self.cap.isOpened():
                # Verify we can actually read a frame
                ret, frame = self.cap.read()
                if ret:
                    print(f"[DEBUG] SUCCESS: Camera {self.camera_id} connected via {config['name']}.")
                    connected = True
                    break
                else:
                    print(f"[DEBUG] isOpened() was True but read() failed for {config['name']}. Moving to next tier.")
                    self.cap.release()
            else:
                print(f"[DEBUG] Failed to open via {config['name']}.")

        if not connected:
            print(f"[DEBUG] FATAL Error: All connection attempts failed for {self.camera_id}")
            self.status = "failed"
            self.is_running = False
            return

        self.status = "connected"
        import threading
        # Ensure we are currently in reading state before processing
        self.raw_frame_buffer = None
        
        # Start internal Reader and Processor threads
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.reader_thread.start()
        self.processor_thread.start()

    def _reader_loop(self):
        """High-speed frame pulling thread."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[DEBUG] Stream interrupt for {self.camera_id}. Re-syncing...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.camera_link)
                continue
            self.raw_frame_buffer = frame

    def _process_loop(self):
        """Detection and logic processing thread."""
        crops_dir = os.path.join("static", "crops", self.camera_id)
        os.makedirs(crops_dir, exist_ok=True)
        
        while self.is_running:
            if self.raw_frame_buffer is None:
                time.sleep(0.01)
                continue

            try:
                loop_start = time.time()
                
                # Fetch latest frame and clear buffer to signal we are processing
                frame = self.raw_frame_buffer
                # (Optional: Only resize if needed to save time)
                h, w = frame.shape[:2]
                if w > 1280:
                    frame = cv2.resize(frame, (1280, int(h * (1280/w))))
                
                raw_frame = frame.copy()
                
                # Run detections
                for trigger_name, model in self.models.items():
                    if model is None: continue
                    
                    det_start = time.time()
                    results = model.track(frame, persist=True, verbose=False, iou=0.5, conf=0.35)[0]
                    det_time = time.time() - det_start
                    
                    if results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                        ids = results.boxes.id.cpu().numpy().astype(int)
                        confs = results.boxes.conf.cpu().numpy()
                        
                        for box, obj_id, conf in zip(boxes, ids, confs):
                            x1, y1, x2, y2 = box
                            unique_track_key = f"{trigger_name}_{obj_id}"
                            
                            # Draw UI
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1-5), 0, 0.4, (0,255,0), 1)

                            if unique_track_key not in self.processed_track_ids:
                                ocr_start = time.time()
                                plate_text = ""
                                image_plate_url = None
                                image_object_url = None
                                
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
                                        
                                        # Color detection
                                        v_color = get_vehicle_color(plate_crop)

                                        self.logs.append({
                                            "timestamp": time.strftime("%H:%M:%S"),
                                            "trigger": trigger_name,
                                            "event": f"Detection (ID: {obj_id})",
                                            "image_plate": image_plate_url,
                                            "image_object": None,
                                            "plate_number": plate_text or "UNREADABLE",
                                            "vehicle_color": v_color
                                        })
                                        print(f"[DEBUG] Cam {self.camera_id}: OCR Result {plate_text} in {time.time()-ocr_start:.3f}s")
                                else:
                                    self.processed_track_ids.add(unique_track_key)
                                    self.logs.append({
                                        "timestamp": time.strftime("%H:%M:%S"),
                                        "trigger": trigger_name,
                                        "event": f"{trigger_name} (ID: {obj_id})",
                                        "image_plate": None, "image_object": None, "plate_number": None
                                    })
                
                self.latest_frame = frame
                # Optional: print(f"[DEBUG] Loop took {time.time()-loop_start:.3f}s")
                
            except Exception as e:
                print(f"[DEBUG] Error in processing loop: {e}")
                time.sleep(0.1)

    def stop(self):
        self.is_running = False
        if self.cap: self.cap.release()

def process_video(task_id, input_path, output_path, selected_triggers):
    """
    High-performance video processing with:
    - Persistent tracking for deduplication
    - Improved OCR with pre-processing
    - Cross-frame plate stabilization
    """
    logs = []
    crops_dir = os.path.join("static", "crops", task_id)
    os.makedirs(crops_dir, exist_ok=True)
    
    # Load/Get models from cache
    loaded_models = {}
    for trigger in selected_triggers:
        model = get_model(trigger)
        if model:
            loaded_models[trigger] = model

    # Vehicle detection model for associating plates with cars
    general_model = get_model("Number Plate Detection") if "Number Plate Detection" in selected_triggers else None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return [{"frame": 0, "event": "Error opening video", "type": "error"}]

    # Pre-calculate video dimensions for speed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Avoid excessive downscaling for better OCR quality
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)

    # FIXED: Use mp4v directly for maximum browser compatibility and no H264 dependency
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # DEDUPLICATION MECHANISMS
    # 1. Tracked Object IDs already processed
    processed_track_ids = set()
    # 2. Plate numbers already seen (to handle ID switching)
    seen_plate_numbers = set()
    
    frame_count = 0
    # PERFORMANCE: Frame skipping (process 1 out of every N frames)
    # 1 means process all, 2 means skip every 2nd. Setting to 1 for maximum accuracy 
    # but the loop is now optimized for speed.
    SKIP_FRAMES = 1 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        # Resize frame once at the start of loop
        frame = cv2.resize(frame, (width, height))
        raw_frame = frame.copy()

        # Run detection and tracking
        for trigger_name, model in loaded_models.items():
            results = model.track(frame, persist=True, verbose=False, iou=0.5)[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, obj_id, conf, cls in zip(boxes, ids, confs, classes):
                    if conf < 0.4: continue
                    
                    x1, y1, x2, y2 = box
                    unique_track_key = f"{trigger_name}_{obj_id}"
                    
                    # Always draw on video
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1-5), 0, 0.5, (0,255,0), 2)

                    # LOGGING LOGIC
                    if unique_track_key not in processed_track_ids:
                        
                        plate_text = ""
                        image_plate_url = None
                        image_object_url = None
                        
                        if trigger_name == "Number Plate Detection":
                            # IMPROVE CROPPING: Add padding (10-20 pixels)
                            pad = 15
                            x1_p = max(0, x1 - pad)
                            y1_p = max(0, y1 - pad)
                            x2_p = min(width, x2 + pad)
                            y2_p = min(height, y2 + pad)
                            
                            plate_crop = raw_frame[y1_p:y2_p, x1_p:x2_p]
                            if plate_crop.size > 0:
                                plate_text = get_best_ocr(plate_crop)
                                
                                # SECOND LAYER DEDUPLICATION: By OCR result
                                if plate_text and plate_text in seen_plate_numbers:
                                    continue # Already logged this plate number via a different track ID
                                
                                # Valid new plate
                                if plate_text:
                                    seen_plate_numbers.add(plate_text)
                                
                                # Keep this track ID logged
                                processed_track_ids.add(unique_track_key)

                                # Save Images
                                p_fname = f"plate_{obj_id}_{frame_count}.jpg"
                                cv2.imwrite(os.path.join(crops_dir, p_fname), plate_crop)
                                image_plate_url = f"/static/crops/{task_id}/{p_fname}"
                                
                                # Associate with vehicle & detect color
                                v_color = "Unknown"
                                vehicle_found = False
                                if general_model:
                                    v_res = general_model.predict(raw_frame, verbose=False, classes=[2,3,5,7])[0]
                                    for v_box in v_res.boxes:
                                        vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                                        px, py = (x1+x2)/2, (y1+y2)/2
                                        if vx1<=px<=vx2 and vy1<=py<=vy2:
                                            v_crop = raw_frame[vy1:vy2, vx1:vx2]
                                            v_color = get_vehicle_color(v_crop)
                                            v_fname = f"v_{obj_id}_{frame_count}.jpg"
                                            cv2.imwrite(os.path.join(crops_dir, v_fname), v_crop)
                                            image_object_url = f"/static/crops/{task_id}/{v_fname}"
                                            vehicle_found = True
                                            break
                                
                                if not vehicle_found:
                                    # Fallback: estimate color from plate crop if car not found
                                    v_color = get_vehicle_color(plate_crop)
                                    print(f"Debug: Vehicle not found for plate ID {obj_id}, using plate crop for color")

                                logs.append({
                                    "timestamp": round(frame_count / fps, 2),
                                    "trigger": trigger_name,
                                    "event": f"New Detection (ID: {obj_id})",
                                    "image_plate": image_plate_url,
                                    "image_object": image_object_url,
                                    "plate_number": plate_text or "UNREADABLE",
                                    "vehicle_color": v_color
                                })
                        else:
                            # Non-ANPR triggers (e.g. Helmet)
                            processed_track_ids.add(unique_track_key)
                            logs.append({
                                "timestamp": round(frame_count / fps, 2),
                                "trigger": trigger_name,
                                "event": f"{trigger_name} detected (ID: {obj_id})",
                                "image_plate": None,
                                "image_object": None,
                                "plate_number": None
                            })

        out.write(frame)

    cap.release()
    out.release()
    return logs
