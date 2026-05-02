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
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import torch
from PIL import Image

# Ensure PROJECT_ROOT is in path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wearcast.inference_wearcast_hd import WearCastHD

# Global State
tasks: Dict[str, dict] = {}
gpu_lock = threading.Lock()

app = FastAPI(title="WearCast AI Professional API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Performance Cache (to share timing across tasks)
GLOBAL_PERFORMANCE = {
    "avg_step_time": 1.5,   # Default for T4-like performance
    "preprocess_time": 20.0, # Loading and preprocessing
    "last_total_time": 65.0
}

def detect_hardware_defaults():
    global GLOBAL_PERFORMANCE
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).upper()
        print(f"[STARTUP] Detected Hardware: {name}")
        if "A100" in name:
            GLOBAL_PERFORMANCE = {"avg_step_time": 0.45, "preprocess_time": 8.0, "last_total_time": 20.0}
        elif "V100" in name:
            GLOBAL_PERFORMANCE = {"avg_step_time": 0.65, "preprocess_time": 10.0, "last_total_time": 25.0}
        elif "L4" in name:
            GLOBAL_PERFORMANCE = {"avg_step_time": 0.8, "preprocess_time": 12.0, "last_total_time": 30.0}
        elif "T4" in name:
            GLOBAL_PERFORMANCE = {"avg_step_time": 2.0, "preprocess_time": 25.0, "last_total_time": 85.0}
        elif any(x in name for x in ["3090", "4090", "3080", "4080"]):
            GLOBAL_PERFORMANCE = {"avg_step_time": 0.55, "preprocess_time": 10.0, "last_total_time": 22.0}
    else:
        print("[STARTUP] No GPU detected. Using CPU defaults.")
        GLOBAL_PERFORMANCE = {"avg_step_time": 15.0, "preprocess_time": 30.0, "last_total_time": 350.0}

detect_hardware_defaults()

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
    global wearcast_model, GLOBAL_PERFORMANCE
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["start_time"] = time.time()
    
    # 1. Detect Garment Complexity to sync step count with model logic
    print(f"[PROCESS] Task {task_id}: Detecting garment complexity...")
    is_complex = wearcast_model.detect_garment_complexity(garm_img)
    # Match WearCastHD.get_optimal_params logic (40 for complex, 30 for simple)
    total_steps = 40 if is_complex else 30
    
    tasks[task_id]["total_steps"] = total_steps
    tasks[task_id]["current_step"] = 0
    
    # 2. Refined Initial Estimate based on complexity
    preprocess_est = GLOBAL_PERFORMANCE.get("preprocess_time", 15)
    avg_step_est = GLOBAL_PERFORMANCE.get("avg_step_time", 1.5)
    buffer = 6
    # Total = Preprocess + (Steps * Avg) + Buffer
    init_total = preprocess_est + (total_steps * avg_step_est) + buffer
    tasks[task_id]["est_finish_time"] = time.time() + init_total
    
    tasks[task_id]["preprocess_start_time"] = time.time()
    tasks[task_id]["unet_start_time"] = None
    tasks[task_id]["avg_time_per_step"] = avg_step_est

    def progress_callback(step, t, latents):
        now = time.time()
        tasks[task_id]["current_step"] = step
        
        # 1. Handle end of preprocessing / start of UNet
        if tasks[task_id]["unet_start_time"] is None:
            tasks[task_id]["unet_start_time"] = now
            # Record actual preprocessing duration
            actual_preprocess = now - tasks[task_id]["preprocess_start_time"]
            GLOBAL_PERFORMANCE["preprocess_time"] = actual_preprocess
            return

        # 2. Update Step Timing (EMA)
        elapsed_unet = now - tasks[task_id]["unet_start_time"]
        safe_step = max(1, step)
        current_avg = elapsed_unet / safe_step
        
        # Smoothed Step Time
        tasks[task_id]["avg_time_per_step"] = 0.6 * tasks[task_id]["avg_time_per_step"] + 0.4 * current_avg
        GLOBAL_PERFORMANCE["avg_step_time"] = tasks[task_id]["avg_time_per_step"]
            
        # 3. Dynamic Finish Time Recalculation
        remaining_steps = max(0, tasks[task_id]["total_steps"] - safe_step)
        buffer = 6 # Post-processing buffer
        
        new_est = now + (tasks[task_id]["avg_time_per_step"] * remaining_steps) + buffer
        tasks[task_id]["est_finish_time"] = new_est
        
        # Update global total for next task
        total_so_far = now - tasks[task_id]["start_time"]
        GLOBAL_PERFORMANCE["last_total_time"] = total_so_far + (tasks[task_id]["avg_time_per_step"] * remaining_steps) + buffer

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
                    image_scale=2.5 if is_complex else 2.0,
                    seed=-1,
                    callback=progress_callback,
                    callback_steps=1
                )
            
            if images:
                output_path = os.path.join(PROJECT_ROOT, f"run/outputs/{task_id}.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                images[0].save(output_path)
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["est_finish_time"] = time.time()
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
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WearCast AI API Developer Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #8a2be2;
            --secondary: #ff1493;
            --dark: #0f0c29;
            --glass: rgba(255, 255, 255, 0.05);
            --border: rgba(255, 255, 255, 0.1);
            --method-get: #61affe;
            --method-post: #49cc90;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #fff;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .container {
            width: 100%;
            max-width: 1000px;
            background: var(--glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        h1 {
            font-size: 3rem;
            text-align: center;
            background: linear-gradient(to right, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        p.subtitle {
            text-align: center;
            color: #a0aec0;
            margin-bottom: 3rem;
            font-size: 1.1rem;
        }
        .endpoint-list {
            display: grid;
            gap: 1.5rem;
        }
        .endpoint {
            background: rgba(0,0,0,0.2);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: transform 0.2s, border-color 0.2s;
        }
        .endpoint:hover {
            transform: translateY(-2px);
            border-color: rgba(255,255,255,0.3);
        }
        .endpoint-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .method {
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-weight: 800;
            font-size: 0.9rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .method.get { background: rgba(97, 175, 254, 0.2); color: var(--method-get); border: 1px solid rgba(97, 175, 254, 0.5); }
        .method.post { background: rgba(73, 204, 144, 0.2); color: var(--method-post); border: 1px solid rgba(73, 204, 144, 0.5); }
        
        .path {
            font-size: 1.3rem;
            font-family: monospace;
            color: #e2e8f0;
        }
        .description {
            color: #a0aec0;
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }
        .params {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 1rem;
            font-size: 0.9rem;
        }
        .params h4 { color: #cbd5e0; margin-bottom: 0.5rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
        .param-item { display: flex; gap: 1rem; margin-bottom: 0.4rem; }
        .param-name { color: #4facfe; font-family: monospace; }
        .param-type { color: #ff1493; font-family: monospace; }
        .param-desc { color: #a0aec0; }
        
        .interactive-docs {
            margin-top: 3rem;
            text-align: center;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
        }
        .btn {
            display: inline-block;
            padding: 1rem 2rem;
            border-radius: 12px;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: white;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 0 0.5rem;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(138, 43, 226, 0.4);
        }
        .btn.outline {
            background: transparent;
            border: 2px solid var(--primary);
        }
        .btn.outline:hover {
            background: rgba(138, 43, 226, 0.1);
            box-shadow: 0 5px 15px rgba(138, 43, 226, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ WearCast AI API</h1>
        <p class="subtitle">Developer Portal & Endpoint Reference</p>
        
        <div class="endpoint-list">
            
            <div class="endpoint">
                <div class="endpoint-header">
                    <span class="method post">POST</span>
                    <span class="path">/tryon</span>
                </div>
                <div class="description">
                    Initiates a new Virtual Try-On task. Submits person and garment images and returns a unique Task ID for tracking.
                </div>
                <div class="params">
                    <h4>Form Data (multipart/form-data)</h4>
                    <div class="param-item">
                        <span class="param-name">person</span>
                        <span class="param-type">file</span>
                        <span class="param-desc">Image of the person (JPEG/PNG)</span>
                    </div>
                    <div class="param-item">
                        <span class="param-name">garment</span>
                        <span class="param-type">file</span>
                        <span class="param-desc">Image of the garment to try on (JPEG/PNG)</span>
                    </div>
                </div>
            </div>

            <div class="endpoint">
                <div class="endpoint-header">
                    <span class="method get">GET</span>
                    <span class="path">/stream/{task_id}</span>
                </div>
                <div class="description">
                    Server-Sent Events (SSE) endpoint to monitor real-time task progress. Provides dynamic time estimations and status updates.
                </div>
                <div class="params">
                    <h4>Path Parameters</h4>
                    <div class="param-item">
                        <span class="param-name">task_id</span>
                        <span class="param-type">string</span>
                        <span class="param-desc">The UUID returned from the /tryon endpoint</span>
                    </div>
                </div>
            </div>

            <div class="endpoint">
                <div class="endpoint-header">
                    <span class="method get">GET</span>
                    <span class="path">/result/{task_id}</span>
                </div>
                <div class="description">
                    Retrieves the final generated try-on image once the task is completed.
                </div>
                <div class="params">
                    <h4>Path Parameters</h4>
                    <div class="param-item">
                        <span class="param-name">task_id</span>
                        <span class="param-type">string</span>
                        <span class="param-desc">The UUID of the completed task</span>
                    </div>
                </div>
            </div>

        </div>

        <div class="interactive-docs">
            <h2 style="margin-bottom: 1.5rem; font-weight: 600;">Interactive Documentation</h2>
            <a href="/docs" class="btn">🚀 Swagger UI</a>
            <a href="/redoc" class="btn outline">📖 ReDoc</a>
        </div>
    </div>
</body>
</html>
"""

@app.get("/")
async def root_dashboard():
    """Returns a beautiful, interactive web dashboard to test the API."""
    return HTMLResponse(content=DASHBOARD_HTML)

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
            "est_finish_time": time.time() + GLOBAL_PERFORMANCE["last_total_time"],
            "created_at": time.time()
        }
        
        # Start background processing
        background_tasks.add_task(run_inference, task_id, p_img, g_img)
        
        return {
            "task_id": task_id,
            "message": "Task started successfully",
            "estimated_time_seconds": int(GLOBAL_PERFORMANCE["last_total_time"])
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

    async def event_generator():
        last_remaining = int(GLOBAL_PERFORMANCE["last_total_time"])
        while True:
            now = time.time()
            current_status = tasks[task_id]["status"]
            est_finish = tasks[task_id].get("est_finish_time", now + last_remaining)
            
            # Calculate remaining time dynamically
            raw_remaining = int(est_finish - now)
            
            # MONOTONIC LOGIC: During processing, the timer must ONLY go down.
            if current_status == "processing":
                # Ensure it never jumps UP, and always stays at least 1s until done
                remaining = min(last_remaining, raw_remaining)
                
                # If the raw estimate jumped up (e.g. step was slow), 
                # we just stick to the last value or count down 1s
                if raw_remaining > last_remaining:
                    remaining = max(1, last_remaining - 1)
                
                remaining = max(1, remaining)
            elif current_status == "completed":
                remaining = 0
            else:
                # Queued state: allow initial estimate
                remaining = raw_remaining
            
            last_remaining = remaining
            
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
                
                if step < 1:
                    message = "Analyzing images and preparing GPU..."
                elif remaining < 10:
                    message = "Finalizing pixels and saving result..."
                else:
                    message = f"Generating textures (Step {step}/{total})..."
            
            yield f"data: {{\"status\": \"{current_status}\", \"message\": \"{message}\", \"remaining\": {remaining}}}\n\n"
            
            await asyncio.sleep(1) 

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
