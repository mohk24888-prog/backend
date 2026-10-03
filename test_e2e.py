import os
import subprocess
import sys
import time
import json
import threading
import requests

os.chdir(r"C:\Users\mohamed\Downloads\gfn\footiq-backend")

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./footiq_test.db"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""
os.environ["SUPABASE_JWT_SECRET"] = ""

# Clean up old DB
for f in ["footiq_test.db", "footiq_test.db-journal"]:
    if os.path.exists(f):
        os.remove(f)

# Start uvicorn in subprocess, redirect output to file
log_file = open("server_test.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

def read_log():
    log_file.flush()
    with open("server_test.log", "r") as f:
        print(f.read())

# Wait for server to be ready
print("Starting server...")
for i in range(30):
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if resp.status_code == 200:
            print("Server ready!")
            break
    except:
        time.sleep(1)
else:
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
    resp = requests.post(url, files=files, data=data, timeout=120)
    print("Upload Status:", resp.status_code)
    if resp.status_code == 200:
        result = resp.json()
        video_id = result.get('id')
        print("Video ID:", video_id)
        print("Player ID:", result.get('player_id'))
        print("Public URL:", result.get('public_url'))
    else:
        print("Error:", resp.text[:500])
        sys.exit(1)

    # Step 2: Start analysis
    print("\n=== STEP 2: Start Analysis ===")
    url = BASE + "/api/v1/analyses"
    body = {'video_id': video_id, 'player_id': 'dz1', 'match_name': 'Test Match'}
    resp = requests.post(url, json=body, timeout=60)
    print("Analysis Status:", resp.status_code)
    if resp.status_code == 200:
        result = resp.json()
        analysis_id = result.get('id')
        job_id = result.get('job_id')
        print("Analysis ID:", analysis_id)
        print("Job ID:", job_id)
        print("Player ID:", result.get('player_id'))
    else:
        print("Error:", resp.text[:500])
        sys.exit(1)

    # Step 3: Poll job status
    print("\n=== STEP 3: Poll Job Status ===")
    for i in range(5):
        url = BASE + "/api/v1/analyses/jobs/" + job_id
        resp = requests.get(url, timeout=10)
        print("Poll", i+1, "- Status:", resp.status_code)
        if resp.status_code == 200:
            result = resp.json()
            print("  Job status:", result.get('status'))
            print("  Job error:", result.get('error'))
        else:
            print("  Error:", resp.text[:200])
        time.sleep(1)

    # Step 4: Get analysis
    print("\n=== STEP 4: Get Analysis ===")
    url = BASE + "/api/v1/analyses/" + analysis_id
    resp = requests.get(url, timeout=10)
    print("Get Analysis Status:", resp.status_code)
    if resp.status_code == 200:
        result = resp.json()
        print("Analysis ID:", result.get('id'))
        print("Video URL:", result.get('video_url'))
        print("Overall Rating:", result.get('overall_rating'))
    else:
        print("Error:", resp.text[:200])

    # Step 5: Get video URL
    print("\n=== STEP 5: Get Video URL ===")
    url = BASE + "/api/v1/videos/" + video_id + "/url"
    resp = requests.get(url, timeout=10)
    print("Video URL Status:", resp.status_code)
    if resp.status_code == 200:
        result = resp.json()
        print("Public URL:", result.get('public_url'))
    else:
        print("Error:", resp.text[:200])

    print("\n=== ALL TESTS PASSED ===")

finally:
    # Kill the server
    proc.terminate()
    proc.wait()
    log_file.close()
    print("\nServer stopped")
    # Read and print server log
    read_log()
