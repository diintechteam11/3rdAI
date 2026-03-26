from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

class ProcessingTask(Base):
    __tablename__ = "processing_tasks"
    
    id = Column(String, primary_key=True)
    filename = Column(String)
    status = Column(String)
    video_url = Column(String) # R2 URL
    created_at = Column(DateTime, default=datetime.utcnow)
    
    logs = relationship("DetectionLog", back_populates="task")

class DetectionLog(Base):
    __tablename__ = "detection_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("processing_tasks.id"))
    timestamp = Column(Float)
    trigger = Column(String)
    event = Column(String)
    image_plate = Column(String) # R2 URL
    image_object = Column(String) # R2 URL
    plate_number = Column(String)
    vehicle_color = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    task = relationship("ProcessingTask", back_populates="logs")

# Database Initialization
engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        print("Debug: PostgreSQL Database Initialized Successfully.")
    except Exception as e:
        print(f"Debug: Database Initialization Error: {e}")

def save_task_to_db(task_id, filename, status, video_url):
    if not SessionLocal: return
    db = SessionLocal()
    try:
        task = db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
        if not task:
            task = ProcessingTask(id=task_id, filename=filename, status=status, video_url=video_url)
            db.add(task)
        else:
            task.status = status
            task.video_url = video_url
        db.commit()
        print(f"DEBUG: SUCCESS! Task {task_id} saved to DB.")
    except Exception as e:
        print(f"DEBUG: ERROR! Task {task_id} NOT saved to DB: {e}")
    finally:
        db.close()

def save_logs_to_db(task_id, log_entries):
    if not SessionLocal: return
    db = SessionLocal()
    try:
        for entry in log_entries:
            # Prefer R2 URL for DB storage
            new_log = DetectionLog(
                task_id=task_id,
                timestamp=entry.get("timestamp"),
                trigger=entry.get("trigger"),
                event=entry.get("event"),
                image_plate=entry.get("image_plate_r2") or entry.get("image_plate"),
                image_object=entry.get("image_object_r2") or entry.get("image_object"),
                plate_number=entry.get("plate_number"),
                vehicle_color=entry.get("vehicle_color")
            )
            db.add(new_log)
        db.commit()
        print(f"DEBUG: SUCCESS! Saved {len(log_entries)} logs to PostgreSQL for task {task_id}")
    except Exception as e:
        print(f"DEBUG: ERROR! Logs NOT saved to DB: {e}")
    finally:
        db.close()

def get_tasks_from_db():
    if not SessionLocal: return []
    db = SessionLocal()
    try:
        return db.query(ProcessingTask).all()
    finally:
        db.close()

def get_task_logs_from_db(task_id):
    if not SessionLocal: return []
    db = SessionLocal()
    try:
        return db.query(DetectionLog).filter(DetectionLog.task_id == task_id).all()
    finally:
        db.close()
