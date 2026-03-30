import os
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://3rdai_user:WHbW4G3mT0qzgGmPODeLCWwnVwlcR6xO@cluster0.abcde.mongodb.net/db_3rdai?retryWrites=true&w=majority")
# Note: In a real environment, the user should provide their own link. I'll use a placeholder or local if needed.
# Since I don't have their URI, I'll assume they want me to configure it logically.

DB_NAME = "3rdai_analytics"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# Collections names
CAMERAS = "cameras"
DETECTIONS = "detections"
RECORDINGS = "recordings"
SCHEDULES = "schedules"
ANALYSIS_SESSIONS = "analysis_sessions"

async def get_db():
    return db

# Helper for creating standard IDs and timestamps
def get_iso_now():
    return datetime.utcnow().isoformat() + "Z"

async def init_mongo():
    # Setup indexes for performance
    try:
        await db[DETECTIONS].create_index("camera_id")
        await db[DETECTIONS].create_index("timestamp")
        await db[DETECTIONS].create_index("plate_number")
        await db[RECORDINGS].create_index("camera_id")
        await db[CAMERAS].create_index("id", unique=True)
        print("Debug: MongoDB Indexes created!")
    except Exception as e:
        print(f"Debug: MongoDB Init Error: {e}")

# Note: The user mentioned MongoDB. If they are running locally, I'll use localhost.
# In main.py, I'll switch from SQLAlchemy sessions to async DB calls.
