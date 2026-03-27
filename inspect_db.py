import psycopg2

try:
    conn = psycopg2.connect(
        host="dpg-d72j4spr0fns73ebi470-a.ohio-postgres.render.com",
        database="db_3rdai",
        user="db_3rdai_user",
        password="WHbW4G3mT0qzgGmPODeLCWwnVwlcR6xO"
    )

    cur = conn.cursor()

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = [t[0] for t in cur.fetchall()]
    
    for table in tables:
        print(f"\n🔍 Table: {table}\n")
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
        cols = cur.fetchall()
        for col in cols:
            print(f"- {col[0]} ({col[1]})")

    cur.close()
    conn.close()

except Exception as e:
    print("❌ Error:", e)
