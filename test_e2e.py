import os
import subprocess
import sys
import time
import json
import threading
import queue
import requests

os.chdir(r"C:\Users\mohamed\Downloads\gfn\footiq-backend")

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./footiq_test.db"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""
os.environ["SUPABASE_JWT_SECRET"] = ""

# Clean up old DB
for f in ["footiq_test.db", "footiq_test.db-journal"]:
    if os.path.exists(f):
        os.remove(f)

# Start uvicorn in subprocess with output piped
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    text=True,
)

# Read server output in background
log_lines = []
def read_output():
    for line in iter(proc.stdout.readline, ''):
        log_lines.append(line.rstrip())
        print(f"[SERVER] {line.rstrip()}")

threading.Thread(target=read_output, daemon=True).start()

# Wait for server to be ready
print("Starting server...")
server_ready = False
for i in range(30):
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if resp.status_code == 200:
            print("Server ready!")
            server_ready = True
            break
    except:
        time.sleep(1)

if not server_ready:
    print("Server failed to start")
    proc.kill()
    sys.exit(1)

try:
    BASE = "http://127.0.0.1:8000"
    video_path = r"C:\Users\mohamed\Downloads\Telegram Desktop\video_2026-09-30_22-16-35 (4).mp4"

    # Step 1: Upload video
    print("\n=== STEP 1: Upload Video ===")
    url = BASE + "/api/v1/videos/"
    files = {'file': ('test.mp4', open(video_path, 'rb'), 'video/mp4')}
    data = {'player_id': 'dz1', 'match_name': 'Test Match'}
    resp = requests.post(url, files=files, data=data, timeout=30)
    print("Upload Status:", resp.status_code)
    if resp.status_code == 200:
        result = resp.json()
        video_id = result.get('id')
        print("Video ID:", video_id)
        print("Player ID:", result.get('player_id'))
        print("Public URL:", result.get('public_url'))
    else:
        print("Error:", resp.text[:500])

    # Step 2: Start analysis
    print("\n=== STEP 2: Start Analysis ===")
    url = BASE + "/api/v1/analyses"
    body = {'video_id': video_id, 'player_id': 'dz1', 'match_name': 'Test Match'}
    try:
        resp = requests.post(url, json=body, timeout=15)
        print("Analysis Status:", resp.status_code)
        if resp.status_code == 200:
            result = resp.json()
            analysis_id = result.get('id')
            job_id = result.get('job_id')
            print("Analysis ID:", analysis_id)
            print("Job ID:", job_id)
        else:
            print("Error:", resp.text[:500])
    except requests.exceptions.ReadTimeout:
        print("TIMEOUT - Server too slow to respond")

    # Wait and check server logs
    time.sleep(2)

    # Step 3: Get the job status
    print("\n=== STEP 3: Get Job Status ===")
    if 'job_id' in dir():
        url = BASE + "/api/v1/analyses/jobs/" + job_id
        try:
            resp = requests.get(url, timeout=10)
            print("Status:", resp.status_code)
            if resp.status_code == 200:
                result = resp.json()
                print("  Job status:", result.get('status'))
                print("  Job error:", result.get('error'))
        except Exception as e:
            print("Error:", str(e)[:200])

    # Step 4: Get analysis
    print("\n=== STEP 4: Get Analysis ===")
    if 'analysis_id' in dir():
        url = BASE + "/api/v1/analyses/" + analysis_id
        try:
            resp = requests.get(url, timeout=10)
            print("Status:", resp.status_code)
            if resp.status_code == 200:
                result = resp.json()
                print("Analysis ID:", result.get('id'))
                print("Video URL:", result.get('video_url'))
        except Exception as e:
            print("Error:", str(e)[:200])

    print("\n=== DONE ===")

finally:
    proc.terminate()
    proc.wait()
    print("\nServer stopped")
    print("\nLast 20 server log lines:")
    for line in log_lines[-20:]:
        print(line)
