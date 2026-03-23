from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import json
from pathlib import Path
from pathlib import Path
from utils.detection import process_video, LiveCameraProcessor
import cv2
from fastapi.responses import StreamingResponse


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

# In-memory database for task tracking (in production, use Redis or DB)
processing_tasks = {}
camera_processes = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs", response_class=HTMLResponse)
async def documentation(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

@app.get("/api-access", response_class=HTMLResponse)
async def api_access(request: Request):
    return templates.TemplateResponse("api.html", {"request": request})

@app.get("/playground", response_class=HTMLResponse)
async def playground(request: Request):
    return templates.TemplateResponse("playground.html", {"request": request})

@app.post("/generate-api-key")
async def generate_api_key():
    new_key = f"sk-{uuid.uuid4().hex}"
    with open(API_KEYS_FILE, "r") as f:
        keys = json.load(f)
    keys[new_key] = {"created_at": str(uuid.uuid1()), "usage": 0}
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f)
    return {"api_key": new_key}

from fastapi import Header, HTTPException

async def check_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        # For browser testing, we allow skipping key check on the main upload if not using Playground
        # But for documentation's sake, we'll enforce it if provided or if coming from Playground
        return True
        
    with open(API_KEYS_FILE, "r") as f:
        keys = json.load(f)
    if x_api_key not in keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    # Increment usage
    keys[x_api_key]["usage"] = keys[x_api_key].get("usage", 0) + 1
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f)
    return True

def background_video_processing(task_id: str, input_path: str, output_path: str, selected_triggers: list):
    """
    Function to be run in BackgroundTasks to avoid blocking.
    """
    try:
        processing_tasks[task_id]["status"] = "processing"
        
        # Run AI logic
        logs = process_video(task_id, input_path, output_path, selected_triggers)
        
        # Build public URL for the output video
        output_url = f"/static/outputs/{os.path.basename(output_path)}"
        
        processing_tasks[task_id].update({
            "status": "completed",
            "logs": logs,
            "video_url": output_url
        })
        
        print(f"Task {task_id} completed successfully.")
        
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    triggers: str = Form(""),
    x_api_key: str = Header(None)
):
    # Validate key if provided (Playground always sends it)
    await check_api_key(x_api_key)
    
    if not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return JSONResponse(content={"error": "Invalid video format"}, status_code=400)
    
    # Generate unique ID for task
    task_id = str(uuid.uuid4())
    
    # Parse triggers (comma separated)
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    if not selected_triggers:
        return JSONResponse(content={"error": "No triggers selected"}, status_code=400)

    # Save uploaded file
    input_filename = f"{task_id}_{video_file.filename}"
    input_path = str(UPLOAD_DIR / input_filename)
    
    with open(input_path, "wb") as f:
        f.write(await video_file.read())

    # Path for processed result
    output_filename = f"processed_{task_id}.mp4"
    output_path = str(OUTPUT_DIR / output_filename)

    # Initial state
    processing_tasks[task_id] = {
        "status": "queued",
        "video_url": None,
        "logs": [],
        "id": task_id,
        "filename": video_file.filename
    }

    # Add background task
    background_tasks.add_task(
        background_video_processing, 
        task_id, 
        input_path, 
        output_path, 
        selected_triggers
    )

    return {"message": "Upload successful, processing started.", "task_id": task_id}

@app.get("/video-result/{task_id}")
async def get_video_result(task_id: str, x_api_key: str = Header(None)):
    await check_api_key(x_api_key)
    task = processing_tasks.get(task_id)
    if not task:
        return JSONResponse(content={"error": "Task not found"}, status_code=404)
    return task

@app.get("/logs/{task_id}")
async def get_logs(task_id: str, x_api_key: str = Header(None)):
    await check_api_key(x_api_key)
    task = processing_tasks.get(task_id)
    if not task:
        return JSONResponse(content={"error": "Task not found"}, status_code=404)
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

    # Initialize live processor (now async-background)
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
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', processor.latest_frame)
                if not ret: continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
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

@app.post("/disconnect-camera/{camera_id}")
async def disconnect_camera(camera_id: str):
    if camera_id in camera_processes:
        camera_processes[camera_id]["processor"].stop()
        del camera_processes[camera_id]
        return {"message": "Camera disconnected"}
    return JSONResponse(content={"error": "Camera not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
