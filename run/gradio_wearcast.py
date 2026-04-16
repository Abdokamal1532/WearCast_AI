import os
import sys
from pathlib import Path

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

# Ensure PROJECT_ROOT is in path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch
from PIL import Image
from wearcast.inference_wearcast_hd import WearCastHD

# ============================================================
# STARTUP: System & Environment Diagnostics
# ============================================================
print("=" * 70)
print("[STARTUP] WearCast AI - System Diagnostics")
print("=" * 70)
print(f"[STARTUP] Python version   : {sys.version}")
print(f"[STARTUP] PROJECT_ROOT     : {PROJECT_ROOT}")
print(f"[STARTUP] sys.path[0]      : {sys.path[0]}")

# Torch / CUDA info
print(f"[STARTUP] PyTorch version  : {torch.__version__}")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"[STARTUP] CUDA Available   : YES  ({gpu_count} GPU(s))")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"[STARTUP]   GPU {i}          : {props.name} | Compute {props.major}.{props.minor} | VRAM {total_mem_gb:.1f} GB")
else:
    print("[STARTUP] CUDA Available   : NO  *** CRITICAL: No GPU found! ***")

# Check checkpoint directories
print("[STARTUP] Checking checkpoint directories...")
checkpoint_dirs = [
    "checkpoints/ootd",
    "checkpoints/ootd/vae",
    "checkpoints/ootd/unet_garm",
    "checkpoints/ootd/unet_vton",
    "checkpoints/ootd/tokenizer",
    "checkpoints/ootd/text_encoder",
    "checkpoints/ootd/scheduler",
    "checkpoints/clip-vit-large-patch14",
    "checkpoints/humanparsing",
    "checkpoints/openpose",
]
for rel_path in checkpoint_dirs:
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    exists = os.path.isdir(full_path)
    if exists:
        files = os.listdir(full_path)
        print(f"[STARTUP]   [OK]    {rel_path} ({len(files)} files)")
    else:
        print(f"[STARTUP]   [MISS]  {rel_path}  *** DOES NOT EXIST ***")

# Check example images
example_path = os.path.join(PROJECT_ROOT, 'run/examples')
print(f"[STARTUP] Example images path: {example_path}")
if os.path.exists(example_path):
    for sub in ['model', 'garment']:
        sub_path = os.path.join(example_path, sub)
        if os.path.isdir(sub_path):
            imgs = [f for f in os.listdir(sub_path) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
            print(f"[STARTUP]   {sub}: {len(imgs)} images found")
else:
    print("[STARTUP]   *** Example path does not exist! ***")

print("=" * 70)

# Initialize the model once (Memory Efficient)
print("[Gradio] Initializing WearCast Engine...")
wearcast_model = WearCastHD(0)
print("[Gradio] WearCast Engine ready.")
print("=" * 70)

def process_hd(vton_img_path, garm_img_path, n_samples, n_steps, image_scale, seed):
    print("=" * 60)
    print("[Gradio] *** NEW REQUEST RECEIVED ***")
    print(f"[Gradio]   Person image  : {vton_img_path}")
    print(f"[Gradio]   Garment image : {garm_img_path}")
    print(f"[Gradio]   Samples       : {n_samples}")
    print(f"[Gradio]   Steps         : {n_steps}")
    print(f"[Gradio]   Guidance Scale: {image_scale}")
    print(f"[Gradio]   Seed          : {seed}")

    if vton_img_path is None or garm_img_path is None:
        print("[Gradio] ERROR: One or both inputs are None. Aborting.")
        return None

    # Validate files exist
    if not os.path.isfile(vton_img_path):
        print(f"[Gradio] ERROR: Person image not found on disk: {vton_img_path}")
        return []
    if not os.path.isfile(garm_img_path):
        print(f"[Gradio] ERROR: Garment image not found on disk: {garm_img_path}")
        return []

    try:
        # Load images with full diagnostics
        print("[Gradio] Loading and resizing input images...")
        vton_img_raw = Image.open(vton_img_path).convert("RGB")
        garm_img_raw = Image.open(garm_img_path).convert("RGB")
        print(f"[Gradio]   Person  raw size : {vton_img_raw.size}  mode={vton_img_raw.mode}")
        print(f"[Gradio]   Garment raw size : {garm_img_raw.size}  mode={garm_img_raw.mode}")

        vton_img = vton_img_raw.resize((768, 1024))
        garm_img = garm_img_raw.resize((768, 1024))
        print(f"[Gradio]   Person  resized  : {vton_img.size}")
        print(f"[Gradio]   Garment resized  : {garm_img.size}")

        # GPU memory snapshot before inference
        if torch.cuda.is_available():
            alloc  = torch.cuda.memory_allocated(0) / (1024**3)
            reserv = torch.cuda.memory_reserved(0)  / (1024**3)
            print(f"[Gradio] GPU Mem before inference: {alloc:.2f} GB alloc / {reserv:.2f} GB reserved")

        print(f"[Gradio] Processing Request: Steps={n_steps}, Scale={image_scale}, Seed={seed}")

        # Run inference
        with torch.no_grad():
            images = wearcast_model(
                model_type='hd',
                category='upperbody',
                image_garm=garm_img,
                image_vton=vton_img,
                mask=None,   # Automated masking internally
                image_ori=vton_img,
                num_samples=n_samples,
                num_steps=n_steps,
                image_scale=image_scale,
                seed=seed,
            )

        # GPU memory snapshot after inference
        if torch.cuda.is_available():
            alloc  = torch.cuda.memory_allocated(0) / (1024**3)
            reserv = torch.cuda.memory_reserved(0)  / (1024**3)
            print(f"[Gradio] GPU Mem after  inference: {alloc:.2f} GB alloc / {reserv:.2f} GB reserved")

        if images:
            print(f"[Gradio] SUCCESS: {len(images)} image(s) returned. Size={images[0].size}")
        else:
            print("[Gradio] WARNING: Empty image list returned!")
        print("=" * 60)
        return images

    except Exception as e:
        import traceback
        print(f"[Gradio ERROR] Exception type : {type(e).__name__}")
        print(f"[Gradio ERROR] Message        : {str(e)}")
        print("[Gradio ERROR] Full traceback:")
        traceback.print_exc()
        print("=" * 60)
        return []

# --- UI DEFINITION ---
with gr.Blocks(title="WearCast AI: Premium Virtual Try-On", theme=gr.themes.Soft()) as block:
    gr.Markdown("""
    # 👕 WearCast AI: Professional Virtual Try-On
    ### Men's Half-Body High-Definition Pipeline
    *Upload a photo of a person and a garment to see the magic. Currently optimized for upper-body garments.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Model / Person")
            vton_input = gr.Image(label="Person Image", type="filepath", height=512)
            gr.Examples(
                examples=[
                    os.path.join(example_path, 'model/model_1.png'),
                    os.path.join(example_path, 'model/model_2.png'),
                    os.path.join(example_path, 'model/model_3.png'),
                    os.path.join(example_path, 'model/model_4.png'),
                    os.path.join(example_path, 'model/model_5.png'),
                ],
                inputs=vton_input,
                label="Common Models"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 2. Garment")
            garm_input = gr.Image(label="Garment Image", type="filepath", height=512)
            gr.Examples(
                examples=[
                    os.path.join(example_path, 'garment/03244_00.jpg'),
                    os.path.join(example_path, 'garment/00126_00.jpg'),
                    os.path.join(example_path, 'garment/03032_00.jpg'),
                    os.path.join(example_path, 'garment/06123_00.jpg'),
                    os.path.join(example_path, 'garment/02305_00.jpg'),
                ],
                inputs=garm_input,
                label="Garment Library"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 3. Result")
            result_gallery = gr.Gallery(label='Virtual Try-On Output', columns=1, height=512)
            run_btn = gr.Button("✨ Generate Try-On", variant="primary", scale=1)

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            n_samples = gr.Slider(label="Number of Samples", minimum=1, maximum=4, value=1, step=1)
            n_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=20, step=5)
            image_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
            seed = gr.Number(label="Seed (-1 for Random)", value=-1)

    run_btn.click(
        fn=process_hd,
        inputs=[vton_input, garm_input, n_samples, n_steps, image_scale, seed],
        outputs=result_gallery
    )

# Launch with share=True for Kaggle access
block.queue().launch(share=True, server_port=7865)
