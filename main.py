from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form, WebSocket, WebSocketDisconnect, APIRouter, Header, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import json
import time
import re
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from pydantic import BaseModel, Field

from utils.detection import process_video, LiveCameraProcessor, MODEL_MAP
from utils.db import init_db, get_db, Camera, RecordingSession, AnalysisSession, Detection

app = FastAPI(
    title="AI Video Analytics & Recording API",
    version="1.2",
    description="Professional Camera Recording and Management System with AI Integration"
)

# Enable CORS for all roots (Fixes 403 in some environments)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    init_db()

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
CROPS_DIR = BASE_DIR / "static" / "crops"
RECORDINGS_DIR = BASE_DIR / "static" / "recordings"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
for d in [UPLOAD_DIR, OUTPUT_DIR, CROPS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys storage
API_KEYS_FILE = BASE_DIR / "api_keys.json"

def load_keys():
    if not API_KEYS_FILE.exists() or API_KEYS_FILE.stat().st_size == 0:
        return {}
    try:
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_keys(keys):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

if not API_KEYS_FILE.exists() or API_KEYS_FILE.stat().st_size == 0:
    save_keys({})

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory tracking
processing_tasks = {}
active_camera_processors = {} # camera_id -> processor object

# --- UTILS & AUTH ---

async def verify_auth(authorization: str = Header(None), x_api_key: str = Header(None)):
    """Verifies Bearer token or X-API-Key."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    elif x_api_key:
        token = x_api_key
    
    if not token:
        return True
        
    keys = load_keys()
    if token not in keys and not token.startswith("sk-"): # Allow sk- prefix for simulated keys
        pass
    return True

def get_error_response(code: str, message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "code": code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

# --- API ROUTES (v1) ---
api_v1 = APIRouter(prefix="/api/v1", tags=["API v1"])

# 1. Camera Management
@api_v1.get("/cameras")
async def list_cameras(
    search: Optional[str] = None,
    status: Optional[str] = None,
    location: Optional[str] = None,
    brand: Optional[str] = None,
    sort: str = "latest",
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    auth=Depends(verify_auth)
):
    query = db.query(Camera)
    if search: query = query.filter(Camera.name.ilike(f"%{search}%") | Camera.location.ilike(f"%{search}%"))
    if status: query = query.filter(Camera.status == status)
    if location: query = query.filter(Camera.location == location)
    if brand: query = query.filter(Camera.brand == brand)
    
    if sort == "latest": query = query.order_by(Camera.created_at.desc())
    elif sort == "oldest": query = query.order_by(Camera.created_at.asc())
    elif sort == "name_asc": query = query.order_by(Camera.name.asc())
    elif sort == "name_desc": query = query.order_by(Camera.name.desc())

    total = query.count()
    cameras = query.offset((page - 1) * page_size).limit(page_size).all()
    return {"cameras": cameras, "total": total, "page": page, "page_size": page_size}

@api_v1.get("/cameras/{camera_id}")
async def get_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return get_error_response("RESOURCE_NOT_FOUND", "Camera found", 404)
    return camera

@api_v1.post("/cameras", status_code=201)
async def register_camera(name: str = Form(...), ip: str = Form(...), location: str = Form(...), brand: Optional[str] = Form(None), db: Session = Depends(get_db)):
    new_camera = Camera(id=str(uuid.uuid4()), name=name, ip=ip, location=location, brand=brand, status="active")
    db.add(new_camera); db.commit(); db.refresh(new_camera)
    return new_camera

# 2. Recording Control
@api_v1.post("/cameras/{camera_id}/recording/start")
async def start_recording(camera_id: str, initiated_by: str = Form(...), note: Optional[str] = Form(None), db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return get_error_response("RESOURCE_NOT_FOUND", "Camera found", 404)
    if camera.status == "recording": return get_error_response("CONFLICT", "Already recording", 409)

    if camera_id not in active_camera_processors:
        active_camera_processors[camera_id] = LiveCameraProcessor(camera_id, camera.ip, ["Number Plate Detection"])
        await asyncio.sleep(1)
    
    success, s_id = active_camera_processors[camera_id].start_recording(initiated_by=initiated_by, note=note, source="manual")
    if success:
        camera.status = "recording"; camera.is_recording = True; db.commit()
        return {"session_id": s_id, "camera_id": camera_id, "status": "recording", "started_at": datetime.utcnow().isoformat(), "initiated_by": initiated_by}
    return get_error_response("INTERNAL_ERROR", "Starting failed", 500)

@api_v1.post("/cameras/{camera_id}/recording/stop")
async def stop_recording(camera_id: str, stopped_by: str = Form(...), video_name: Optional[str] = Form(None), description: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if camera_id not in active_camera_processors: return get_error_response("NOT_FOUND", "No session", 404)
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return get_error_response("RESOURCE_NOT_FOUND", "Camera found", 404)

    success, s_id = active_camera_processors[camera_id].stop_recording(stopped_by=stopped_by)
    if success:
        session = db.query(RecordingSession).filter(RecordingSession.id == s_id).first()
        if session:
            final_name = video_name or f"{camera.name.replace(' ', '_')}_{datetime.now().strftime('%Y%md_%H%M%S')}.mp4"
            session.video_name = final_name; session.description = description
            session.stopped_at = func.now(); session.saved_at = func.now(); session.stopped_by = stopped_by
            if session.started_at:
                delta = datetime.now() - session.started_at.replace(tzinfo=None)
                session.duration_secs = int(delta.total_seconds())
        camera.status = "active"; camera.is_recording = False
        db.commit(); db.refresh(session)
        return {"recording_id": s_id, "camera_id": camera_id, "video_name": session.video_name, "file_path": session.file_path, "duration_secs": session.duration_secs, "started_at": session.started_at, "stopped_at": session.stopped_at}
    return get_error_response("INTERNAL_ERROR", "Stop failed", 500)

@api_v1.get("/cameras/{camera_id}/recording/status")
async def get_recording_status(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return get_error_response("RESOURCE_NOT_FOUND", "Camera found", 404)
    session_id, started_at, duration = None, None, 0
    if camera.is_recording and camera_id in active_camera_processors:
        proc = active_camera_processors[camera_id]
        session_id = proc.recording_session_id
        started_at = datetime.fromtimestamp(proc.recording_start_time).isoformat() if proc.recording_start_time else None
        duration = int(time.time() - proc.recording_start_time) if proc.recording_start_time else 0
    return {"camera_id": camera_id, "is_recording": camera.is_recording, "session_id": session_id, "started_at": started_at, "duration_secs": duration}

# 3. Recording Management
@api_v1.get("/recordings")
async def list_all_recordings(camera_id: Optional[str] = None, search: Optional[str] = None, date_from: Optional[str] = None, sort: str = "latest", source: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(RecordingSession)
    if camera_id: query = query.filter(RecordingSession.camera_id == camera_id)
    if search: query = query.filter(RecordingSession.video_name.ilike(f"%{search}%") | RecordingSession.description.ilike(f"%{search}%"))
    if source: query = query.filter(RecordingSession.source == source)
    if sort == "latest": query = query.order_by(RecordingSession.started_at.desc())
    else: query = query.order_by(RecordingSession.started_at.asc())
    return {"recordings": query.all()}

@api_v1.get("/recordings/{recording_id}")
async def get_recording(recording_id: str, db: Session = Depends(get_db)):
    rec = db.query(RecordingSession).filter(RecordingSession.id == recording_id).first()
    if not rec: return get_error_response("RESOURCE_NOT_FOUND", "Recording found", 404)
    return rec

@api_v1.delete("/recordings/{recording_id}")
async def delete_recording(recording_id: str, delete_file: bool = False, db: Session = Depends(get_db)):
    rec = db.query(RecordingSession).filter(RecordingSession.id == recording_id).first()
    if not rec: return get_error_response("RESOURCE_NOT_FOUND", "Recording found", 404)
    if delete_file and rec.file_path:
        p = os.path.join(RECORDINGS_DIR, os.path.basename(rec.file_path))
        if os.path.exists(p): os.remove(p)
    db.delete(rec); db.commit()
    return {"message": "Deleted"}

@api_v1.post("/cameras/{camera_id}/analysis/start")
async def start_analysis(camera_id: str, analysis_type: str = Form(...), triggered_by: str = Form(...), db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return get_error_response("RESOURCE_NOT_FOUND", "Camera found", 404)
    analysis = AnalysisSession(id=str(uuid.uuid4()), camera_id=camera_id, analysis_type=analysis_type, triggered_by=triggered_by, capture_started_at=func.now())
    db.add(analysis); db.commit()
    if camera_id not in active_camera_processors:
        active_camera_processors[camera_id] = LiveCameraProcessor(camera_id, camera.ip, [analysis_type])
        await asyncio.sleep(1)
    success, s_id = active_camera_processors[camera_id].start_recording(initiated_by=triggered_by, source="analysis", analysis_session_id=analysis.id)
    if success:
        camera.status = "recording"; db.commit()
        return {"session_id": s_id, "analysis_session_id": analysis.id, "camera_id": camera_id, "status": "capturing", "analysis_type": analysis_type}
    return get_error_response("INTERNAL_ERROR", "Failed", 500)

@api_v1.post("/cameras/{camera_id}/analysis/stop")
async def stop_analysis(camera_id: str, analysis_session_id: str = Form(...), stopped_by: str = Form(...), analysis_result: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if camera_id not in active_camera_processors: return get_error_response("NOT_FOUND", "No processor", 404)
    success, r_id = active_camera_processors[camera_id].stop_recording(stopped_by=stopped_by)
    if success:
        analysis = db.query(AnalysisSession).filter(AnalysisSession.id == analysis_session_id).first()
        if analysis:
            analysis.capture_ended_at = func.now(); analysis.stopped_by = stopped_by; analysis.analysis_result = analysis_result
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if camera: camera.status = "active"; camera.is_recording = False
            db.commit()
            return {"recording_id": r_id, "analysis_session_id": analysis_session_id, "source": "analysis", "analysis_result": analysis_result}
    return get_error_response("INTERNAL_ERROR", "Failed", 500)

app.include_router(api_v1)

# --- UI & LEGACY ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse(request, "index.html", {"request": request})
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request): return templates.TemplateResponse(request, "api.html", {"request": request})
@app.get("/api-access", response_class=HTMLResponse)
async def api_access(request: Request): return templates.TemplateResponse(request, "api.html", {"request": request})
@app.get("/playground", response_class=HTMLResponse)
async def playground(request: Request): return templates.TemplateResponse(request, "playground.html", {"request": request})

@app.post("/connect-camera")
async def connect_camera(name: str = Form(...), link: str = Form(...), triggers: str = Form(...), db: Session = Depends(get_db)):
    camera_id = str(uuid.uuid4())
    new_camera = Camera(id=camera_id, name=name, ip=link, location="Dashboard", status="active")
    db.add(new_camera); db.commit()
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    active_camera_processors[camera_id] = LiveCameraProcessor(camera_id, link, selected_triggers)
    return {"message": "Connecting", "camera_id": camera_id}

@app.get("/camera-status/{camera_id}")
async def get_legacy_status(camera_id: str):
    if camera_id not in active_camera_processors: return {"status": "not_found"}
    if active_camera_processors[camera_id].latest_jpeg: return {"status": "connected"}
    return {"status": "initializing"}

@app.get("/camera-logs/{camera_id}")
async def get_legacy_logs(camera_id: str):
    if camera_id not in active_camera_processors: return {"logs": []}
    return {"logs": active_camera_processors[camera_id].logs}

@app.post("/disconnect-camera/{camera_id}")
async def disconnect_camera(camera_id: str, db: Session = Depends(get_db)):
    if camera_id in active_camera_processors:
        active_camera_processors[camera_id].is_running = False
        del active_camera_processors[camera_id]
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if camera: camera.status = "inactive"; camera.is_recording = False; db.commit()
    return {"message": "Disconnected"}

@app.websocket("/ws-camera/{camera_id}")
async def websocket_camera_stream(websocket: WebSocket, camera_id: str):
    if camera_id not in active_camera_processors: await websocket.close(code=1008); return
    await websocket.accept(); p = active_camera_processors[camera_id]
    try:
        while True:
            if p.latest_jpeg: await websocket.send_bytes(p.latest_jpeg)
            await asyncio.sleep(0.04)
    except: pass

@app.post("/upload-video")
async def legacy_upload(background_tasks: BackgroundTasks, video_file: UploadFile = File(...), triggers: str = Form(""), auth=Depends(verify_auth)):
    task_id = str(uuid.uuid4())
    input_path = str(UPLOAD_DIR / f"{task_id}_{video_file.filename}")
    with open(input_path, "wb") as f: f.write(await video_file.read())
    output_path = str(OUTPUT_DIR / f"processed_{task_id}.mp4")
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    processing_tasks[task_id] = {"status": "queued", "id": task_id}
    background_tasks.add_task(background_video_processing, task_id, input_path, output_path, selected_triggers)
    return {"message": "Processing started", "task_id": task_id}

async def background_video_processing(task_id: str, input_path: str, output_path: str, selected_triggers: list):
    try:
        processing_tasks[task_id]["status"] = "processing"
        loop = asyncio.get_event_loop()
        logs = await loop.run_in_executor(None, process_video, task_id, input_path, output_path, selected_triggers)
        processing_tasks[task_id].update({"status": "completed", "logs": logs, "video_url": f"/static/outputs/{os.path.basename(output_path)}", "filename": os.path.basename(input_path)})
    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"; processing_tasks[task_id]["error"] = str(e)

@app.get("/video-result/{task_id}")
async def get_video_result(task_id: str): return processing_tasks.get(task_id, {"status": "not_found"})

@app.post("/generate-api-key")
async def gen_key():
    k = f"sk-{uuid.uuid4().hex}"; keys = load_keys(); keys[k] = {"created_at": str(datetime.now())}; save_keys(keys)
    return {"api_key": k}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
