import os
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
# Fallback to local if URI is invalid or missing
MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017"

DB_NAME = "3rdai_analytics"

client = None
db = None

def get_iso_now():
    return datetime.utcnow().isoformat() + "Z"

async def init_mongo():
    global client, db
    try:
        # Check if URI is actually provided and not the placeholder
        uri = MONGO_URI
        if "cluster0.abcde.mongodb.net" in uri:
            print("Warning: Placeholder MongoDB URI detected. Falling back to localhost.")
            uri = "mongodb://localhost:27017"
            
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        
        # Test connection
        await client.admin.command('ping')
        
        # Setup indexes
        await db["detections"].create_index("camera_id")
        await db["detections"].create_index("timestamp")
        await db["detections"].create_index("plate_number")
        await db["recordings"].create_index("camera_id")
        await db["cameras"].create_index("id", unique=True)
        print("Debug: MongoDB Successfully Initialized & Connected!")
    except Exception as e:
        print(f"Debug: MongoDB CONNECTION FAILED: {e}")
        # We don't crash, but db operations will fail later. 
        # This allows the server to at least start so logs can be seen.
        if db is None:
            # Last fallback to local even if ping failed
            client = AsyncIOMotorClient("mongodb://localhost:27017")
            db = client[DB_NAME]

async def get_db():
    return db

CAMERAS = "cameras"
DETECTIONS = "detections"
RECORDINGS = "recordings"
SCHEDULES = "schedules"
ANALYSIS_SESSIONS = "analysis_sessions"
