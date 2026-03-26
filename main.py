from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form, Header, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import json
from pathlib import Path
from utils.detection import process_video, LiveCameraProcessor
from utils.database import save_task_to_db, save_logs_to_db, get_tasks_from_db, get_task_logs_from_db
import cv2

app = FastAPI(title="AI Video Analytics Web App")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
CROPS_DIR = BASE_DIR / "static" / "crops"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys storage
API_KEYS_FILE = BASE_DIR / "api_keys.json"
if not API_KEYS_FILE.exists():
    with open(API_KEYS_FILE, "w") as f:
        json.dump({}, f)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory database for active task tracking
processing_tasks = {}
camera_processes = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/api-access", response_class=HTMLResponse)
async def api_access(request: Request):
    return templates.TemplateResponse(request, "api.html")

@app.get("/playground", response_class=HTMLResponse)
async def playground(request: Request):
    return templates.TemplateResponse(request, "playground.html")

@app.post("/generate-api-key")
async def generate_api_key():
    new_key = f"sk-{uuid.uuid4().hex}"
    with open(API_KEYS_FILE, "r") as f:
        keys = json.load(f)
    keys[new_key] = {"created_at": str(uuid.uuid1()), "usage": 0}
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f)
    return {"api_key": new_key}

async def check_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        return True
    with open(API_KEYS_FILE, "r") as f:
        keys = json.load(f)
    if x_api_key not in keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    keys[x_api_key]["usage"] = keys[x_api_key].get("usage", 0) + 1
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f)
    return True

def background_video_processing(task_id: str, input_path: str, output_path: str, selected_triggers: list, filename: str):
    """
    Function to be run in BackgroundTasks to avoid blocking.
    Now saves results to PostgreSQL.
    """
    try:
        processing_tasks[task_id]["status"] = "processing"
        save_task_to_db(task_id, filename, "processing", None)
        
        # Run AI logic
        result = process_video(task_id, input_path, output_path, selected_triggers)
        logs = result.get("logs", [])
        video_url_r2 = result.get("video_url_r2")
        
        # Build public URL fallback
        output_url = video_url_r2 or f"/static/outputs/{os.path.basename(output_path)}"
        
        processing_tasks[task_id].update({
            "status": "completed",
            "logs": logs,
            "video_url": output_url
        })
        
        # Save to PostgreSQL
        print(f"DEBUG: Attempting to save task {task_id} and {len(logs)} logs to DB...")
        save_task_to_db(task_id, filename, "completed", output_url)
        save_logs_to_db(task_id, logs)
        
        print(f"Task {task_id} completed and saved to DB.")
        
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
        save_task_to_db(task_id, filename, "failed", None)

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    triggers: str = Form(""),
    x_api_key: str = Header(None)
):
    await check_api_key(x_api_key)
    if not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return JSONResponse(content={"error": "Invalid video format"}, status_code=400)
    
    task_id = str(uuid.uuid4())
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    if not selected_triggers:
        return JSONResponse(content={"error": "No triggers selected"}, status_code=400)

    input_filename = f"{task_id}_{video_file.filename}"
    input_path = str(UPLOAD_DIR / input_filename)
    with open(input_path, "wb") as f:
        f.write(await video_file.read())

    output_filename = f"processed_{task_id}.mp4"
    output_path = str(OUTPUT_DIR / output_filename)

    processing_tasks[task_id] = {
        "status": "queued",
        "video_url": None,
        "logs": [],
        "id": task_id,
        "filename": video_file.filename
    }
    
    # Save initial task state to DB
    save_task_to_db(task_id, video_file.filename, "queued", None)

    background_tasks.add_task(
        background_video_processing, 
        task_id, 
        input_path, 
        output_path, 
        selected_triggers,
        video_file.filename
    )

    return {"message": "Upload successful, processing started.", "task_id": task_id}

@app.get("/video-result/{task_id}")
async def get_video_result(task_id: str, x_api_key: str = Header(None)):
    await check_api_key(x_api_key)
    task = processing_tasks.get(task_id)
    if not task:
        # Check DB if not in memory
        from utils.database import SessionLocal, ProcessingTask
        if SessionLocal:
            db = SessionLocal()
            db_task = db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
            if db_task:
                logs = get_task_logs_from_db(task_id)
                # Convert logs objects to dicts
                log_dicts = []
                for l in logs:
                    log_dicts.append({
                        "timestamp": l.timestamp,
                        "trigger": l.trigger,
                        "event": l.event,
                        "image_plate": l.image_plate,
                        "image_object": l.image_object,
                        "plate_number": l.plate_number,
                        "vehicle_color": l.vehicle_color
                    })
                return {
                    "id": db_task.id,
                    "filename": db_task.filename,
                    "status": db_task.status,
                    "video_url": db_task.video_url,
                    "logs": log_dicts
                }
            db.close()
        return JSONResponse(content={"error": "Task not found"}, status_code=404)
    return task

@app.get("/logs/{task_id}")
async def get_logs(task_id: str, x_api_key: str = Header(None)):
    await check_api_key(x_api_key)
    task = await get_video_result(task_id, x_api_key)
    if isinstance(task, JSONResponse): return task
    return {"logs": task.get("logs", [])}

@app.post("/connect-camera")
async def connect_camera(
    name: str = Form(...),
    link: str = Form(...),
    triggers: str = Form("")
):
    camera_id = f"cam-{uuid.uuid4().hex[:8]}"
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    if not selected_triggers:
        return JSONResponse(content={"error": "No triggers selected"}, status_code=400)

    processor = LiveCameraProcessor(camera_id, link, selected_triggers)
    camera_processes[camera_id] = {
        "id": camera_id,
        "name": name,
        "link": link,
        "triggers": selected_triggers,
        "processor": processor
    }
    return {"message": "Camera connection initiated", "camera_id": camera_id}

@app.get("/camera-status/{camera_id}")
async def get_camera_status(camera_id: str):
    if camera_id not in camera_processes:
        return JSONResponse(content={"status": "not_found"}, status_code=404)
    return {"status": camera_processes[camera_id]["processor"].status}

@app.get("/camera-stream/{camera_id}")
async def camera_stream(camera_id: str):
    if camera_id not in camera_processes:
        raise HTTPException(status_code=404, detail="Camera not found")
    processor = camera_processes[camera_id]["processor"]
    def generate():
        while True:
            if processor.latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processor.latest_frame)
                if not ret: continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                import time
                time.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera-logs/{camera_id}")
async def get_camera_logs(camera_id: str):
    if camera_id not in camera_processes:
        return JSONResponse(content={"error": "Camera not found"}, status_code=404)
    processor = camera_processes[camera_id]["processor"]
    return {"logs": processor.logs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
