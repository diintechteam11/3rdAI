import psycopg2
from datetime import datetime

# DATABASE CONFIG
DB_CONFIG = {
    "host": "dpg-d72j4spr0fns73ebi470-a.ohio-postgres.render.com",
    "database": "db_3rdai",
    "user": "db_3rdai_user",
    "password": "WHbW4G3mT0qzgGmPODeLCWwnVwlcR6xO"
}

def check_database():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("\n" + "="*80)
        print("DATABASE CONTENT OVERVIEW")
        print("="*80)

        # 1. PROCESSING TASKS
        print("\n--- [PROCESSING TASKS] ---")
        cur.execute("SELECT id, filename, status, video_url, created_at FROM processing_tasks ORDER BY created_at DESC LIMIT 5;")
        tasks = cur.fetchall()
        
        if not tasks:
            print("No tasks found.")
        else:
            print(f"{'ID':<38} | {'Filename':<30} | {'Status':<10} | {'Created At'}")
            print("-" * 100)
            for t in tasks:
                print(f"{t[0]:<38} | {str(t[1])[:30]:<30} | {t[2]:<10} | {t[4]}")
                if t[3]: print(f"   └─ URL: {t[3]}")

        # 2. DETECTION LOGS
        print("\n--- [DETECTION LOGS (Latest 10)] ---")
        cur.execute("""
            SELECT l.task_id, l.timestamp, l.trigger, l.plate_number, l.vehicle_color, l.image_plate, l.image_object
            FROM detection_logs l 
            ORDER BY l.created_at DESC 
            LIMIT 10;
        """)
        logs = cur.fetchall()
        
        if not logs:
            print("No detection logs found.")
        else:
            print(f"{'Task ID':<38} | {'Time':<6} | {'Trigger':<20} | {'Plate':<15} | {'Color'}")
            print("-" * 100)
            for l in logs:
                print(f"{l[0]:<38} | {str(l[1])[:6]:<6} | {l[2]:<20} | {str(l[3]):<15} | {l[4]}")
                if l[5]: print(f"   └─ Plate Image: {l[5]}")
                if l[6]: print(f"   └─ Vehicle Image: {l[6]}")

        # 3. COUNTS
        print("\n--- [SUMMARY] ---")
        cur.execute("SELECT COUNT(*) FROM processing_tasks;")
        task_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM detection_logs;")
        log_count = cur.fetchone()[0]
        print(f"Total Tasks: {task_count}")
        print(f"Total Detection Logs: {log_count}")

        cur.close()
        conn.close()
        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    check_database()