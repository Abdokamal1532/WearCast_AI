# --- KAGGLE COMPATIBILITY PATCHES ---
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

try:
    import transformers
    if not hasattr(transformers, "EncoderDecoderCache"):
        class EncoderDecoderCache: pass
        transformers.EncoderDecoderCache = EncoderDecoderCache
except ImportError:
    pass

import os
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path
from typing import Dict, Optional

# ============================================================
# 1. AUTO-DEPENDENCY INSTALLATION (For Kaggle/Professional Use)
# ============================================================
def install_dependencies():
    print("[SYSTEM] Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        from pyngrok import ngrok
    except ImportError:
        print("[SYSTEM] Installing missing dependencies (fastapi, uvicorn, pyngrok, python-multipart)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "pyngrok", "python-multipart"])
        print("[SYSTEM] Dependencies installed.")

install_dependencies()

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import torch
from PIL import Image

# Ensure PROJECT_ROOT is in path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wearcast.inference_wearcast_hd import WearCastHD

# ============================================================
# 2. CONFIGURATION & INITIALIZATION
# ============================================================
app = FastAPI(title="WearCast AI Professional API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Task Storage
tasks: Dict[str, dict] = {}
gpu_lock = threading.Lock()

print("[STARTUP] Initializing WearCast Engine (GPU)...")
# Initialize the model on GPU 0
try:
    wearcast_model = WearCastHD(0)
    print("[STARTUP] WearCast Engine ready.")
except Exception as e:
    print(f"[STARTUP] ERROR: Could not initialize model: {e}")
    wearcast_model = None

# Suppress harmless CUDA/XLA warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ngrok Setup
# EXACT token from user: 37TuVJjPQ5gDI1YioSlO45Zj6WS_3fRgJK5Huea4d9pJScX2a
NGROK_TOKEN = "37TuVJjPQ5gDI1YioSlO45Zj6WS_3fRgJK5Huea4d9pJScX2a"
PORT = 8000

def start_ngrok():
    from pyngrok import conf, ngrok
    print(f"[DEBUG] Force-setting Ngrok token: {NGROK_TOKEN[:6]}...{NGROK_TOKEN[-6:]}")
    
    # 1. Update the global configuration directly
    conf.get_default().auth_token = NGROK_TOKEN
    
    try:
        # Cleanup existing tunnels
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
    except:
        pass
    
    # 2. Connect with the explicit token
    public_url = ngrok.connect(PORT).public_url
    print("\n" + "="*70)
    print(f"🚀 WEARCAST API IS LIVE!")
    print(f"🔗 Public URL: {public_url}")
    print(f"🛠️  Frontend Integration: Use this URL in your app")
    print("="*70 + "\n")
    return public_url

# ============================================================
# 3. CORE INFERENCE LOGIC (Background)
# ============================================================
def run_inference(task_id: str, vton_img: Image.Image, garm_img: Image.Image):
    global wearcast_model
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["start_time"] = time.time()
    tasks[task_id]["total_steps"] = 20
    tasks[task_id]["current_step"] = 0
    tasks[task_id]["remaining"] = 60 # Initial guess
    
    def progress_callback(step, t, latents):
        tasks[task_id]["current_step"] = step
        elapsed = time.time() - tasks[task_id]["start_time"]
        
        if step > 0:
            avg_time_per_step = elapsed / (step + 1)
            remaining_steps = tasks[task_id]["total_steps"] - (step + 1)
            # Add a small buffer for post-processing (approx 5s)
            tasks[task_id]["remaining"] = int(avg_time_per_step * remaining_steps) + 5
        else:
            # First step might be slower due to warm-up
            tasks[task_id]["remaining"] = 55 

    try:
        with gpu_lock:
            print(f"[PROCESS] Task {task_id} started on GPU.")
            # Standard settings for high quality
            with torch.no_grad():
                images = wearcast_model(
                    model_type='hd',
                    category='upperbody',
                    image_garm=garm_img.resize((768, 1024)),
                    image_vton=vton_img.resize((768, 1024)),
                    mask=None,
                    image_ori=vton_img.resize((768, 1024)),
                    num_samples=1,
                    num_steps=tasks[task_id]["total_steps"],
                    image_scale=2.5,
                    seed=-1,
                    callback=progress_callback,
                    callback_steps=1
                )
            
            if images:
                output_path = os.path.join(PROJECT_ROOT, f"run/outputs/{task_id}.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                images[0].save(output_path)
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["remaining"] = 0
                tasks[task_id]["result_path"] = output_path
                print(f"[SUCCESS] Task {task_id} finished.")
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = "Model returned no images"
                
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"[ERROR] Task {task_id} failed: {e}")

# ============================================================
# 4. API ENDPOINTS
# ============================================================

@app.post("/tryon")
async def tryon(background_tasks: BackgroundTasks, person: UploadFile = File(...), garment: UploadFile = File(...)):
    """
    Start a professional try-on task.
    Returns a task_id to track progress.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Load images into memory
        p_img = Image.open(person.file).convert("RGB")
        g_img = Image.open(garment.file).convert("RGB")
        
        tasks[task_id] = {
            "status": "queued",
            "remaining": 65,  # Default starting point
            "created_at": time.time()
        }
        
        # Start background processing
        background_tasks.add_task(run_inference, task_id, p_img, g_img)
        
        return {
            "task_id": task_id,
            "message": "Task started successfully",
            "estimated_time_seconds": 65
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid images: {e}")

@app.get("/stream/{task_id}")
async def stream_progress(task_id: str):
    """
    Professional SSE Stream: Sends real-time 'remaining time' updates.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    def event_generator():
        while True:
            current_status = tasks[task_id]["status"]
            remaining = tasks[task_id].get("remaining", 0)
            
            if current_status == "completed":
                yield f"data: {{\"status\": \"completed\", \"remaining\": 0, \"url\": \"/result/{task_id}\"}}\n\n"
                break
            elif current_status == "failed":
                yield f"data: {{\"status\": \"failed\", \"error\": \"{tasks[task_id].get('error')}\"}}\n\n"
                break
            
            # Message mapping based on status
            message = "Processing..."
            if current_status == "queued":
                message = "Waiting for GPU resources..."
            elif current_status == "processing":
                step = tasks[task_id].get("current_step", 0)
                total = tasks[task_id].get("total_steps", 20)
                message = f"Generating textures (Step {step}/{total})..."
                if step < 2:
                    message = "Analyzing images and preparing GPU..."
                elif remaining < 5:
                    message = "Finalizing pixels and saving result..."
            
            yield f"data: {{\"status\": \"{current_status}\", \"message\": \"{message}\", \"remaining\": {remaining}}}\n\n"
            
            time.sleep(1) # More frequent updates for real-time feel

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Download the final generated image.
    """
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result not ready or task not found")
    
    file_path = tasks[task_id]["result_path"]
    return FileResponse(file_path, media_type="image/png")

# ============================================================
# 5. EXECUTION
# ============================================================
if __name__ == "__main__":
    # 1. Start the Ngrok Tunnel
    public_url = start_ngrok()
    
    # 2. Run the FastAPI Server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
