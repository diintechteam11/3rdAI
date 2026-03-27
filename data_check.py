import psycopg2
import os
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

# DB Configuration
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

def check_data():
    try:
        print("\n" + "="*50)
        print("DATABASE SAVED DATA CHECK")
        print("="*50)
        
        conn = psycopg2.connect(
            host=DB_HOST, 
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASS
        )
        cur = conn.cursor()
        
        # 1. Check Total Count
        cur.execute("SELECT COUNT(*) FROM detections;")
        total_count = cur.fetchone()[0]
        print(f"\n[SUMMARY] Total Detections Saved: {total_count}")
        
        # 2. Check Triggers Breakdown
        cur.execute("SELECT trigger, COUNT(*) FROM detections GROUP BY trigger;")
        triggers = cur.fetchall()
        print("\n[BREAKDOWN] Detections by Trigger Type:")
        for t, c in triggers:
            print(f" - {t}: {c}")

        # 3. Check Latest Entries
        print(f"\n[LATEST] Top 10 Most Recent Detections:")
        cur.execute("""
            SELECT id, created_at, trigger, plate_number, vehicle_color, image_plate_url 
            FROM detections 
            ORDER BY created_at DESC 
            LIMIT 10;
        """)
        rows = cur.fetchall()
        headers = ["ID", "Time", "Trigger", "Plate", "Color", "R2 URL"]
        
        # Clean up rows for better table display
        cleaned_rows = []
        for r in rows:
            # Shorten R2 URL for display
            url = str(r[5])
            if url and len(url) > 30:
                short_url = url[:15] + "..." + url[-15:]
            else:
                short_url = url
            cleaned_rows.append([r[0], r[1], r[2], r[3], r[4], short_url])

        print(tabulate(cleaned_rows, headers=headers, tablefmt="grid"))
        
        cur.close()
        conn.close()
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"Error checking data: {e}")

if __name__ == "__main__":
    check_data()