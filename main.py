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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
CROPS_DIR = BASE_DIR / "static" / "crops"
RECORDINGS_DIR = BASE_DIR / "static" / "recordings"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [UPLOAD_DIR, OUTPUT_DIR, CROPS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

processing_tasks = {}
active_camera_processors = {} 

async def verify_auth(authorization: str = Header(None)):
    return True

# --- API v1 ---
api_v1 = APIRouter(prefix="/api/v1", tags=["API v1"])

@api_v1.get("/cameras")
async def list_cameras(db: Session = Depends(get_db)):
    return {"cameras": db.query(Camera).all()}

@api_v1.post("/cameras/{camera_id}/recording/start")
async def start_recording(
    camera_id: str, 
    initiated_by: str = Form(...), 
    video_name: Optional[str] = Form(None), 
    triggers: Optional[str] = Form(None), 
    db: Session = Depends(get_db)
):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera: return JSONResponse(status_code=404, content={"error": "Unknown Camera"})
    
    if camera_id not in active_camera_processors:
        # Initial connect if not already there
        active_camera_processors[camera_id] = LiveCameraProcessor(camera_id, camera.ip, ["Number Plate Detection"])
        await asyncio.sleep(1)
    
    # Update triggers if provided
    if triggers:
        selected = [t.strip() for t in triggers.split(",") if t.strip()]
        active_camera_processors[camera_id].update_triggers(selected)

    success, s_id = active_camera_processors[camera_id].start_recording(
        initiated_by=initiated_by, 
        video_name=video_name, 
        source="manual"
    )
    
    if success:
        camera.status = "recording"; camera.is_recording = True; db.commit()
        return {"session_id": s_id, "video_name": video_name}
    return JSONResponse(status_code=500, content={"error": "Failed to start"})

@api_v1.post("/cameras/{camera_id}/recording/stop")
async def stop_recording(camera_id: str, stopped_by: str = Form(...), db: Session = Depends(get_db)):
    if camera_id not in active_camera_processors: return JSONResponse(status_code=404, content={"error": "No Session"})
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    
    success, s_id = active_camera_processors[camera_id].stop_recording(stopped_by=stopped_by)
    if success:
        camera.status = "active"; camera.is_recording = False
        # Update session end details
        session = db.query(RecordingSession).filter(RecordingSession.id == s_id).first()
        if session:
            session.stopped_at = func.now()
            if session.started_at:
                session.duration_secs = int((datetime.now() - session.started_at.replace(tzinfo=None)).total_seconds())
        db.commit()
        return {"session_id": s_id, "status": "stopped"}
    return JSONResponse(status_code=500, content={"error": "Stop failed"})

@api_v1.get("/recordings")
async def list_recs(db: Session = Depends(get_db)):
    return {"recordings": db.query(RecordingSession).order_by(RecordingSession.started_at.desc()).all()}

@api_v1.get("/recordings/{id}")
async def get_rec(id: str, db: Session = Depends(get_db)):
    return db.query(RecordingSession).filter(RecordingSession.id == id).first()

app.include_router(api_v1)

# --- DASHBOARD ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/connect-camera")
async def connect(name: str = Form(...), link: str = Form(...), triggers: str = Form(""), db: Session = Depends(get_db)):
    cid = str(uuid.uuid4())
    new_c = Camera(id=cid, name=name, ip=link, status="active")
    db.add(new_c); db.commit()
    trigs = [t.strip() for t in triggers.split(",") if t.strip()]
    active_camera_processors[cid] = LiveCameraProcessor(cid, link, trigs)
    return {"camera_id": cid}

@app.get("/camera-status/{id}")
async def status(id: str):
    if id in active_camera_processors and active_camera_processors[id].latest_jpeg: return {"status": "connected"}
    return {"status": "initializing"}

@app.get("/camera-logs/{id}")
async def logs(id: str):
    if id in active_camera_processors:
        l = active_camera_processors[id].logs[:]
        active_camera_processors[id].logs = [] # Clear logs after read as per user request to "remove from right side" on stop
        return {"logs": l}
    return {"logs": []}

@app.post("/disconnect-camera/{id}")
async def disconnect(id: str, db: Session = Depends(get_db)):
    if id in active_camera_processors:
        active_camera_processors[id].is_running = False
        del active_camera_processors[id]
        c = db.query(Camera).filter(Camera.id == id).first()
        if c: c.status = "inactive"; db.commit()
    return {"ok": True}

@app.websocket("/ws-camera/{id}")
async def ws(websocket: WebSocket, id: str):
    if id not in active_camera_processors: return
    await websocket.accept(); p = active_camera_processors[id]
    try:
        while True:
            if p.latest_jpeg: await websocket.send_bytes(p.latest_jpeg)
            await asyncio.sleep(0.04)
    except: pass

@app.post("/upload-video")
async def upload(background_tasks: BackgroundTasks, video_file: UploadFile = File(...), triggers: str = Form("")):
    tid = str(uuid.uuid4())
    inp = str(UPLOAD_DIR / f"{tid}_{video_file.filename}")
    with open(inp, "wb") as f: f.write(await video_file.read())
    out = str(OUTPUT_DIR / f"res_{tid}.mp4")
    trigs = [t.strip() for t in triggers.split(",") if t.strip()]
    processing_tasks[tid] = {"status": "processing"}
    background_tasks.add_task(run_proc, tid, inp, out, trigs)
    return {"task_id": tid}

async def run_proc(tid, inp, out, trigs):
    logs = await asyncio.get_event_loop().run_in_executor(None, process_video, tid, inp, out, trigs)
    processing_tasks[tid] = {"status": "completed", "logs": logs, "video_url": f"/static/outputs/{os.path.basename(out)}"}

@app.get("/video-result/{tid}")
async def res(tid: str): return processing_tasks.get(tid, {"status": "not_found"})

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
