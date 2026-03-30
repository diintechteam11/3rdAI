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
    title="3rdAI | Enterprise Analytics",
    version="1.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trackers
active_camera_processors = {} 
log_connections = {} # camera_id -> list of websockets

@app.on_event("startup")
async def on_startup():
    init_db()
    # Persistence: Reconnect cameras that were active
    db = next(get_db())
    all_cams = db.query(Camera).all()
    for cam in all_cams:
        if cam.status == "active" or cam.status == "recording":
            # Don't auto-start all unless requested, but marked as available
            # We'll just ensure the UI fetches them
            pass
    db.close()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
CROPS_DIR = BASE_DIR / "static" / "crops"
RECORDINGS_DIR = BASE_DIR / "static" / "recordings"

for d in [UPLOAD_DIR, OUTPUT_DIR, CROPS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- API v1 ---
api_v1 = APIRouter(prefix="/api/v1", tags=["API v1"])

@api_v1.get("/cameras")
async def list_cameras(db: Session = Depends(get_db)):
    return {"cameras": db.query(Camera).all()}

@api_v1.post("/cameras/{camera_id}/recording/start")
async def start_recording(camera_id: str, initiated_by: str = Form(...), video_name: Optional[str] = Form(None), triggers: Optional[str] = Form(None), db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return JSONResponse(status_code=404, content={"error": "Not Found"})
    
    if camera_id not in active_camera_processors:
        active_camera_processors[camera_id] = LiveCameraProcessor(camera_id, camera.ip, ["Number Plate Detection"])
        await asyncio.sleep(1)
    
    if triggers:
        active_camera_processors[camera_id].update_triggers([t.strip() for t in triggers.split(",") if t.strip()])

    success, s_id = active_camera_processors[camera_id].start_recording(initiated_by, video_name, "manual")
    if success:
        camera.status = "recording"; camera.is_recording = True; db.commit()
        return {"session_id": s_id}
    return JSONResponse(status_code=500, content={"error": "Start failed"})

@api_v1.post("/cameras/{camera_id}/recording/stop")
async def stop_recording(camera_id: str, stopped_by: str = Form(...), db: Session = Depends(get_db)):
    if camera_id not in active_camera_processors: return JSONResponse(status_code=404, content={"error": "No Session"})
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    success, s_id = active_camera_processors[camera_id].stop_recording(stopped_by)
    if success:
        camera.status = "active"; camera.is_recording = False
        db.commit()
        return {"ok": True}
    return JSONResponse(status_code=500, content={"error": "Stop failed"})

@api_v1.get("/recordings")
async def list_recordings(db: Session = Depends(get_db)):
    recs = db.query(RecordingSession).order_by(RecordingSession.started_at.desc()).all()
    return {"recordings": recs}

@api_v1.delete("/recordings/{id}")
async def delete_recording(id: str, db: Session = Depends(get_db)):
    rec = db.query(RecordingSession).filter(RecordingSession.id == id).first()
    if rec:
        if rec.file_path:
            p = BASE_DIR / rec.file_path.lstrip("/")
            if p.exists(): p.unlink()
        db.delete(rec); db.commit()
        return {"ok": True}
    return JSONResponse(status_code=404, content={"error": "Rec not found"})

@api_v1.get("/recordings/{id}/logs")
async def get_recording_logs(id: str, db: Session = Depends(get_db)):
    rec = db.query(RecordingSession).filter(RecordingSession.id == id).first()
    if not rec: return {"logs": []}
    # Fetch detections for this session or time range
    dets = db.query(Detection).filter(
        Detection.camera_id == rec.camera_id,
        Detection.created_at >= rec.started_at,
        (Detection.created_at <= rec.stopped_at if rec.stopped_at else True)
    ).all()
    return {"logs": dets}

app.include_router(api_v1)

# --- WEB UI & AD-HOC ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/connect-camera")
async def connect(name: str = Form(...), link: str = Form(...), db: Session = Depends(get_db)):
    # Check if camera exists in DB
    cam = db.query(Camera).filter(Camera.ip == link).first()
    if not cam:
        cam = Camera(id=str(uuid.uuid4()), name=name, ip=link, status="active")
        db.add(cam); db.commit()
    else:
        cam.status = "active"; db.commit()
    
    if cam.id not in active_camera_processors:
        active_camera_processors[cam.id] = LiveCameraProcessor(cam.id, link, ["Number Plate Detection"])
    return {"camera_id": cam.id}

@app.post("/disconnect-camera/{id}")
async def disconnect(id: str, db: Session = Depends(get_db)):
    if id in active_camera_processors:
        active_camera_processors[id].is_running = False
        del active_camera_processors[id]
    cam = db.query(Camera).filter(Camera.id == id).first()
    if cam: cam.status = "inactive"; db.commit()
    return {"ok": True}

@app.get("/camera-status/{id}")
async def status(id: str):
    if id in active_camera_processors and active_camera_processors[id].latest_jpeg: return {"status": "connected"}
    return {"status": "initializing"}

@app.get("/camera-logs/{id}")
async def legacy_logs(id: str):
    if id in active_camera_processors:
        l = active_camera_processors[id].logs[:]
        active_camera_processors[id].logs = []
        return {"logs": l}
    return {"logs": []}

# WebSockets
@app.websocket("/ws-camera/{id}")
async def ws_stream(websocket: WebSocket, id: str):
    if id not in active_camera_processors: return
    await websocket.accept()
    p = active_camera_processors[id]
    try:
        while True:
            if p.latest_jpeg: await websocket.send_bytes(p.latest_jpeg)
            await asyncio.sleep(0.04) # ~25fps
    except: pass

@app.websocket("/ws-logs/{id}")
async def ws_logs(websocket: WebSocket, id: str):
    await websocket.accept()
    if id not in log_connections: log_connections[id] = []
    log_connections[id].append(websocket)
    try:
        while True:
            # We fetch from processor and broadcast
            if id in active_camera_processors:
                p = active_camera_processors[id]
                if p.logs:
                    l = p.logs[:]; p.logs = []
                    for ws in log_connections[id]:
                        await ws.send_json({"logs": l})
            await asyncio.sleep(1)
    except:
        if id in log_connections: log_connections[id].remove(websocket)

@app.post("/upload-video")
async def upload_proc(background_tasks: BackgroundTasks, video_file: UploadFile = File(...), triggers: str = Form("")):
    tid = str(uuid.uuid4())
    inp = str(UPLOAD_DIR / f"{tid}_{video_file.filename}")
    with open(inp, "wb") as f: f.write(await video_file.read())
    out = str(OUTPUT_DIR / f"proc_{tid}.mp4")
    trigs = [t.strip() for t in triggers.split(",") if t.strip()]
    
    background_tasks.add_task(run_background, tid, inp, out, trigs)
    return {"task_id": tid}

async def run_background(tid, inp, out, trigs):
    logs = await asyncio.get_event_loop().run_in_executor(None, process_video, tid, inp, out, trigs)
    # Store result in a temporary file for the UI to fetch
    with open(OUTPUT_DIR / f"{tid}.json", "w") as f: json.dump({"status": "completed", "logs": logs, "video_url": f"/static/outputs/{os.path.basename(out)}"}, f)

@app.get("/video-result/{tid}")
async def get_res(tid: str):
    path = OUTPUT_DIR / f"{tid}.json"
    if path.exists():
        with open(path, "r") as f: return json.load(f)
    return {"status": "processing"}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
