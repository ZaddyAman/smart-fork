import sqlite3
import os

db_path = r"C:\Users\amans\.smartfork\metadata.db"
print(f"Connecting to DB: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n--- TABLES ---")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    for t in tables:
        print(f"- {t}")

    print("\n--- RECORD COUNTS ---")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {t};")
        print(f"{t}: {cursor.fetchone()[0]} rows")

    print("\n--- PHASE 6: SAMPLE PARENT CHUNK (SQLite Text Preservation) ---")
    cursor.execute("SELECT parent_id, chunk_index, length(text), text FROM parent_chunks LIMIT 1;")
    sample = cursor.fetchone()
    if sample:
        print(f"Parent ID: {sample[0]}")
        print(f"Chunk Index: {sample[1]}")
        print(f"Text Length: {sample[2]} chars")
        print(f"Text Preview:\n{sample[3][:300]}...\n[TRUNCATED FOR VIEWING]")
    else:
        print("No parent chunks found.")

    print("\n--- PHASE 7: SAMPLE SESSION SUMMARY (TextRank Output) ---")
    cursor.execute("SELECT session_id, length(summary_doc), summary_doc FROM sessions WHERE summary_doc IS NOT NULL LIMIT 1;")
    sess = cursor.fetchone()
    if sess:
        print(f"Session ID: {sess[0]}")
        print(f"Summary Length: {sess[1]} chars")
        print(f"Summary Preview:\n{sess[2]}")
    else:
        print("No summaries generated yet (run `smartfork summarize-v2`).")

    conn.close()
except Exception as e:
    print(f"Error accessing database: {e}")
