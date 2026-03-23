# AI Video Analytics Web App

This high-performance web app processes traffic videos and detects specific triggers:
- Number Plate Detection (ANPR)
- Helmet Detection
- Triple Riding Detection
- Seatbelt Detection

## ⚡ Tech Stack
- **Backend**: FastAPI (Python)
- **AI**: YOLOv8 (Ultralytics), OpenCV
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript

## 📁 Project Structure
- `main.py`: FastAPI backend and task management.
- `utils/detection.py`: AI processing engine using YOLOv8.
- `templates/index.html`: Premium glassmorphism UI.
- `models/`: Directory for `.pt` files (e.g., `ANPR.pt`, `helmet.pt`, etc.).
- `static/uploads/`: Original uploaded videos.
- `static/outputs/`: Processed videos with detections.

## 🚀 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python main.py
   ```
   *The server will run on [http://localhost:8000](http://localhost:8000).*

3. **Upload AI Models**:
   Ensure your `.pt` files are placed in the `models/` directory:
   - `ANPR.pt`
   - `helmet.pt`
   - `triple.pt`
   - `seatbelt.pt`
   *Note: If specific models are not found, the system will automatically fall back to `yolov8n.pt` downloaded from Ultralytics.*

## ⚙️ Features
- **Dynamic Model Loading**: Only the models required for the selected triggers are loaded into memory.
- **Async Processing**: Uses FastAPI's `BackgroundTasks` to ensure the frontend remains responsive during processing.
- **H264 Compatibility**: Processed videos are exported as MP4 with H264 codec for direct browser playback.
- **Premium Frontend**: Responsive glassmorphism design with real-time progress polling and log filtering.
