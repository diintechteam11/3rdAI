from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form, WebSocket, WebSocketDisconnect, APIRouter, Header, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, json, time, re, asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union

from utils.detection import process_video, LiveCameraProcessor, MODEL_MAP
from utils.db import db, init_mongo, CAMERAS, DETECTIONS, RECORDINGS, SCHEDULES, get_iso_now

app = FastAPI(title="3rdAI Enterprise v2.0", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def on_startup():
    await init_mongo()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
CROPS_DIR = BASE_DIR / "static" / "crops"
RECORDINGS_DIR = BASE_DIR / "static" / "recordings"

for d in [UPLOAD_DIR, OUTPUT_DIR, CROPS_DIR, RECORDINGS_DIR]: d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

active_camera_processors = {} 
processing_tasks = {}

def get_error(code, msg, status=400):
    return JSONResponse(status_code=status, content={"error": True, "code": code, "message": msg, "timestamp": get_iso_now()})

api = APIRouter(prefix="/api/v1")

@api.get("/cameras")
async def list_cams(search:str=None, status:str=None):
    q = {}
    if search: q["$or"] = [{"name":{"$regex":search, "$options":"i"}}, {"location":{"$regex":search, "$options":"i"}}]
    if status: q["status"] = status
    cursor = db[CAMERAS].find(q).sort("created_at", -1)
    res = await cursor.to_list(100)
    for r in res: r.pop("_id", None)
    return {"cameras": res}

@api.get("/cameras/{id}")
async def get_cam(id:str):
    c = await db[CAMERAS].find_one({"id":id})
    if not c: return get_error("NOT_FOUND", "No camera", 404)
    c.pop("_id", None)
    return c

@api.post("/cameras/{id}/recording/start")
async def start_rec(id:str, name:str=Form(...)):
    if id not in active_camera_processors:
        c = await db[CAMERAS].find_one({"id":id})
        active_camera_processors[id] = LiveCameraProcessor(id, c["ip"], ["Number Plate Detection"])
        await asyncio.sleep(1)
    
    proc = active_camera_processors[id]
    rid = proc.start_rec(name)
    await db[CAMERAS].update_one({"id":id}, {"$set": {"is_recording":True, "status":"recording"}})
    return {"recording_id": rid, "status": "started", "name": name}

@api.post("/cameras/{id}/recording/stop")
async def stop_rec(id:str):
    if id not in active_camera_processors: return get_error("NOT_FOUND", "Not active", 404)
    proc = active_camera_processors[id]
    rid = proc.stop_rec()
    
    # Save recording to DB
    rec_doc = {
        "id": rid, "camera_id": id, "video_name": proc.rec_name,
        "video_url": f"/static/recordings/{rid}_{int(time.time())}.mp4", # Fixed reference
        "logs": list(proc.logs), "created_at": get_iso_now()
    }
    await db[RECORDINGS].insert_one(rec_doc)
    await db[CAMERAS].update_one({"id":id}, {"$set": {"is_recording":False, "status":"active"}})
    proc.logs = [] # Reset for next recording cycle
    return {"recording_id": rid, "status": "stopped"}

@api.get("/recordings")
async def list_recs(camera_id:str=None):
    q = {"camera_id": camera_id} if camera_id else {}
    cursor = db[RECORDINGS].find(q).sort("created_at", -1)
    res = await cursor.to_list(100)
    for r in res: r.pop("_id", None)
    return {"recordings": res}

app.include_router(api)

@app.get("/", response_class=HTMLResponse)
async def home(request:Request): return templates.TemplateResponse("index.html", {"request":request})

@app.post("/connect-camera")
async def connect(name:str=Form(...), link:str=Form(...), triggers:str=Form(...)):
    id = str(uuid.uuid4())
    doc = {"id":id, "name":name, "ip":link, "location":"Deployment Zone", "status":"active", "created_at":get_iso_now(), "is_recording": False}
    await db[CAMERAS].insert_one(doc)
    active_camera_processors[id] = LiveCameraProcessor(id, link, [t.strip() for t in triggers.split(",") if t.strip()])
    return {"camera_id": id}

@app.get("/camera-status/{id}")
async def status(id:str):
    if id not in active_camera_processors: return {"status":"not_found"}
    return {"status":"connected" if active_camera_processors[id].latest_jpeg else "initializing"}

@app.get("/camera-logs/{id}")
async def logs(id:str):
    if id not in active_camera_processors: return {"logs":[]}
    return {"logs": list(active_camera_processors[id].logs)}

@app.websocket("/ws-camera/{id}")
async def ws_cam(websocket:WebSocket, id:str):
    if id not in active_camera_processors: return await websocket.close(1008)
    await websocket.accept()
    p = active_camera_processors[id]
    try:
        while True:
            if p.latest_jpeg: await websocket.send_bytes(p.latest_jpeg)
            await asyncio.sleep(0.04) # 25FPS
    except: pass

@app.post("/upload-video")
async def upload(background:BackgroundTasks, video:UploadFile=File(...), triggers:str=Form("")):
    tid = str(uuid.uuid4())
    in_p = str(UPLOAD_DIR / f"{tid}_{video.filename}")
    with open(in_p, "wb") as f: f.write(await video.read())
    out_p = str(OUTPUT_DIR / f"res_{tid}.mp4")
    processing_tasks[tid] = {"status":"queued"}
    background.add_task(run_proc, tid, in_p, out_p, [t.strip() for t in triggers.split(",") if t.strip()])
    return {"task_id": tid}

async def run_proc(tid, in_p, out_p, ts):
    processing_tasks[tid]["status"] = "processing"
    try:
        loop = asyncio.get_event_loop()
        # process_video is CPU intensive, delegating to executor
        logs = await loop.run_in_executor(None, process_video, tid, in_p, out_p, ts)
        result = {"id": tid, "status": "completed", "logs": logs, "video_url": f"/static/outputs/{os.path.basename(out_p)}", "created_at": get_iso_now()}
        processing_tasks[tid].update(result)
        await db["uploads"].insert_one(result)
    except Exception as e:
        err = {"id": tid, "status": "failed", "error": str(e), "created_at": get_iso_now()}
        processing_tasks[tid].update(err)
        await db["uploads"].insert_one(err)

@app.get("/video-result/{id}")
async def result(id:str):
    if id in processing_tasks: return processing_tasks[id]
    r = await db["uploads"].find_one({"id": id})
    if r: r.pop("_id", None); return r
    return {"status":"not_found"}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
