import subprocess
import time
import sys
import os

print("Starting VoiceGuard AI Backend (FastAPI on port 8000)...")
backend_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=r"s:\major_project_documents\voiceguard_ai\backend"
)

print("Starting VoiceGuard AI Frontend (Vite on port 5173)...")
frontend_process = subprocess.Popen(
    ["cmd.exe", "/c", "npx vite"],
    cwd=r"s:\major_project_documents\voiceguard_ai"
)

time.sleep(3)
print("Servers initiated! Backend: http://localhost:8000 | Frontend: http://localhost:5173")
