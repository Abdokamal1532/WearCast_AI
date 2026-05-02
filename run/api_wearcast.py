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
import asyncio
import subprocess
import json
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
from utils_wearcast import smart_resize

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
    
    # Use fixed 20 steps as requested by user (previously was dynamic 15-20)
    total_steps = 20
    
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
    tasks[task_id]["last_step_time"] = None
    tasks[task_id]["avg_time_per_step"] = avg_step_est

    def progress_callback(step, t, latents):
        now = time.time()
        tasks[task_id]["current_step"] = step
        
        # 1. Handle end of preprocessing / start of UNet
        if tasks[task_id]["unet_start_time"] is None:
            tasks[task_id]["unet_start_time"] = now
            tasks[task_id]["last_step_time"] = now
            # Record actual preprocessing duration
            actual_preprocess = now - tasks[task_id]["preprocess_start_time"]
            GLOBAL_PERFORMANCE["preprocess_time"] = actual_preprocess
            return

        # 2. Update Step Timing (EMA) avoiding first step warmup spike
        step_duration = now - tasks[task_id]["last_step_time"]
        tasks[task_id]["last_step_time"] = now
        
        if step > 1:
            tasks[task_id]["avg_time_per_step"] = 0.7 * tasks[task_id]["avg_time_per_step"] + 0.3 * step_duration
            GLOBAL_PERFORMANCE["avg_step_time"] = tasks[task_id]["avg_time_per_step"]
            
        # 3. Dynamic Finish Time Recalculation
        remaining_steps = max(0, tasks[task_id]["total_steps"] - step)
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
            # Pre-process images using smart_resize
            vton_smart = smart_resize(vton_img)
            garm_smart = smart_resize(garm_img)
            
            # GPU memory snapshot
            if torch.cuda.is_available():
                m_alloc = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"[PROCESS] Task {task_id} VRAM: {m_alloc:.2f}GB used")

            with torch.no_grad():
                images = wearcast_model(
                    model_type='hd',
                    category='upperbody',
                    image_garm=garm_smart,
                    image_vton=vton_smart,
                    mask=None,
                    image_ori=vton_smart,
                    num_samples=1,
                    num_steps=total_steps,
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
from fastapi.openapi.docs import get_swagger_ui_html

@app.get("/")
async def root_dashboard():
    """Returns the premium WearCast AI Dashboard."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WearCast AI | Premium API Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(168, 85, 247, 0.15) 0%, transparent 50%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
        }

        .container {
            max-width: 1000px;
            width: 100%;
            padding: 2rem;
            margin-top: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 0.8s ease-out;
        }

        header h1 {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }

        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 1.5rem;
            padding: 1.5rem;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }

        .card:hover {
            border-color: rgba(99, 102, 241, 0.3);
        }

        .upload-zone {
            border: 2px dashed var(--glass-border);
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            height: 250px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: var(--glass-bg);
            overflow: hidden;
        }

        .upload-zone:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .upload-zone i {
            color: var(--primary);
            margin-bottom: 1rem;
        }

        .preview-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            padding: 0.5rem;
            background: var(--bg);
            z-index: 5;
        }

        .btn-container {
            grid-column: span 2;
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }

        @media (max-width: 768px) {
            .btn-container {
                grid-column: span 1;
            }
        }

        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 1rem 3rem;
            border-radius: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        }

        .btn:hover:not(:disabled) {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            filter: grayscale(1);
        }

        /* Progress Modal */
        .status-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(8px);
            z-index: 100;
            display: none;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }

        .status-card {
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 2rem;
            padding: 3rem;
            max-width: 500px;
            width: 100%;
            text-align: center;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .progress-circle {
            width: 120px;
            height: 120px;
            margin: 0 auto 2rem;
            position: relative;
        }

        .loader-ring {
            width: 100%;
            height: 100%;
            border: 4px solid var(--glass-border);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .time-badge {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.5rem;
            font-weight: 700;
        }

        .status-text {
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }

        .status-subtext {
            color: var(--text-muted);
            margin-bottom: 2rem;
        }

        /* Result View */
        .result-container {
            display: none;
            margin-top: 3rem;
            text-align: center;
            animation: zoomIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .result-img-wrapper {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 2rem;
            border: 1px solid var(--glass-border);
            display: inline-block;
            margin-bottom: 2rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .result-img {
            max-width: 100%;
            border-radius: 1.5rem;
            height: 500px;
        }

        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes zoomIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }

        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>WearCast AI</h1>
            <p>Premium Virtual Try-On Studio</p>
        </header>

        <div class="dashboard-grid" id="main-ui">
            <div class="card">
                <h3 style="margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <i data-lucide="user"></i> Model Image
                </h3>
                <div class="upload-zone" onclick="document.getElementById('person-input').click()">
                    <i data-lucide="image-plus" size="48"></i>
                    <p>Click to upload person image</p>
                    <input type="file" id="person-input" class="hidden" accept="image/*" onchange="preview(this, 'person-preview')">
                    <img id="person-preview" class="preview-img hidden">
                </div>
            </div>

            <div class="card">
                <h3 style="margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <i data-lucide="shirt"></i> Garment Image
                </h3>
                <div class="upload-zone" onclick="document.getElementById('garment-input').click()">
                    <i data-lucide="image-plus" size="48"></i>
                    <p>Click to upload garment image</p>
                    <input type="file" id="garment-input" class="hidden" accept="image/*" onchange="preview(this, 'garment-preview')">
                    <img id="garment-preview" class="preview-img hidden">
                </div>
            </div>

            <div class="btn-container">
                <button id="run-btn" class="btn" onclick="startTryOn()">
                    <i data-lucide="sparkles"></i> Generate Try-On
                </button>
            </div>
        </div>

        <div id="result-view" class="result-container">
            <div class="result-img-wrapper">
                <img id="final-result" class="result-img">
            </div>
            <div style="display: flex; justify-content: center; gap: 1rem;">
                <button class="btn" style="background: var(--glass-bg); border: 1px solid var(--glass-border);" onclick="resetUI()">
                    <i data-lucide="refresh-cw"></i> Try Another
                </button>
                <a id="download-link" download="wearcast_result.png" class="btn">
                    <i data-lucide="download"></i> Download Result
                </a>
            </div>
        </div>
    </div>

    <!-- Status Overlay -->
    <div id="status-overlay" class="status-overlay">
        <div class="status-card">
            <div class="progress-circle">
                <div class="loader-ring"></div>
                <div id="timer" class="time-badge">--s</div>
            </div>
            <div id="status-msg" class="status-text">Initializing Engine...</div>
            <div id="status-sub" class="status-subtext">Preparing GPU resources for inference.</div>
        </div>
    </div>

    <script>
        lucide.createIcons();
        
        let personFile = null;
        let garmentFile = null;

        function preview(input, imgId) {
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.getElementById(imgId);
                    img.src = e.target.result;
                    img.classList.remove('hidden');
                };
                reader.readAsDataURL(input.files[0]);
                if(imgId === 'person-preview') personFile = input.files[0];
                if(imgId === 'garment-preview') garmentFile = input.files[0];
            }
        }

        async function startTryOn() {
            if(!personFile || !garmentFile) {
                alert("Please upload both person and garment images.");
                return;
            }

            const formData = new FormData();
            formData.append('person', personFile);
            formData.append('garment', garmentFile);

            document.getElementById('run-btn').disabled = true;
            document.getElementById('status-overlay').style.display = 'flex';

            try {
                const response = await fetch('/tryon', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if(data.task_id) {
                    trackProgress(data.task_id);
                } else {
                    throw new Error("Failed to create task");
                }
            } catch (e) {
                alert("Error: " + e.message);
                resetUI();
            }
        }

        function trackProgress(taskId) {
            const eventSource = new EventSource(`/stream/${taskId}`);
            const timerEl = document.getElementById('timer');
            const statusEl = document.getElementById('status-msg');
            const subEl = document.getElementById('status-sub');

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if(data.status === 'completed') {
                    eventSource.close();
                    showResult(data.url);
                } else if(data.status === 'failed') {
                    eventSource.close();
                    alert("Processing failed: " + data.error);
                    resetUI();
                } else {
                    timerEl.innerText = data.remaining + 's';
                    statusEl.innerText = data.status.charAt(0).toUpperCase() + data.status.slice(1) + '...';
                    subEl.innerText = data.message;
                }
            };

            eventSource.onerror = () => {
                console.error("SSE Error - attempting to reconnect or handle disconnection");
                // Note: EventSource automatically tries to reconnect unless closed.
                // We don't close here to allow for temporary glitches.
            };
        }

        function showResult(url) {
            document.getElementById('status-overlay').style.display = 'none';
            document.getElementById('main-ui').classList.add('hidden');
            document.getElementById('result-view').style.display = 'block';
            const img = document.getElementById('final-result');
            img.src = url;
            document.getElementById('download-link').href = url;
        }

        function resetUI() {
            document.getElementById('status-overlay').style.display = 'none';
            document.getElementById('main-ui').classList.remove('hidden');
            document.getElementById('result-view').style.display = 'none';
            document.getElementById('run-btn').disabled = false;
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

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
    Includes robust headers to prevent premature disconnection.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        try:
            last_remaining = int(GLOBAL_PERFORMANCE["last_total_time"])
            while True:
                now = time.time()
                if task_id not in tasks:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Task vanished'})}\n\n"
                    break

                current_status = tasks[task_id]["status"]
                est_finish = tasks[task_id].get("est_finish_time", now + last_remaining)
                
                # Calculate remaining time dynamically
                raw_remaining = int(est_finish - now)
                
                # MONOTONIC LOGIC
                if current_status == "processing":
                    remaining = max(1, raw_remaining)
                elif current_status == "completed":
                    remaining = 0
                else:
                    remaining = max(1, raw_remaining)
                
                last_remaining = remaining
                
                if current_status == "completed":
                    data = {
                        "status": "completed",
                        "remaining": 0,
                        "url": f"/result/{task_id}"
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                elif current_status == "failed":
                    data = {
                        "status": "failed",
                        "error": tasks[task_id].get('error')
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                
                # Message mapping
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
                
                data = {
                    "status": current_status,
                    "message": message,
                    "remaining": remaining
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(1) 
        except Exception as e:
            error_data = {"status": "error", "message": f"Stream internal error: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Critical for Nginx/Proxy stability
            "Transfer-Encoding": "chunked",
            "Access-Control-Allow-Origin": "*"
        }
    )


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
