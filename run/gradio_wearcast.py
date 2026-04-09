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

# Initialize the model once (Memory Efficient)
print("[Gradio] Initializing WearCast Engine...")
wearcast_model = WearCastHD(0)

def process_hd(vton_img_path, garm_img_path, n_samples, n_steps, image_scale, seed):
    if vton_img_path is None or garm_img_path is None:
        return None
    
    try:
        # Load images
        vton_img = Image.open(vton_img_path).convert("RGB").resize((768, 1024))
        garm_img = Image.open(garm_img_path).convert("RGB").resize((768, 1024))
        
        print(f"[Gradio] Processing Request: Steps={n_steps}, Scale={image_scale}, Seed={seed}")
        
        # Run inference using the stabilized pipeline
        # WearCastHD now handles all preprocessing (OpenPose, Parsing, Masking) internally
        with torch.no_grad():
            images = wearcast_model(
                model_type='hd',
                category='upperbody',
                image_garm=garm_img,
                image_vton=vton_img,
                mask=None, # Automated masking internally
                image_ori=vton_img,
                num_samples=n_samples,
                num_steps=n_steps,
                image_scale=image_scale,
                seed=seed,
            )
        
        return images
    except Exception as e:
        print(f"[Gradio ERROR] {str(e)}")
        # Return an error indicator or a placeholder
        return []

# --- UI DEFINITION ---
example_path = os.path.join(PROJECT_ROOT, 'run/examples')

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
            n_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=40, step=5)
            image_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=7.5, value=3.5, step=0.1)
            seed = gr.Number(label="Seed (-1 for Random)", value=-1)

    run_btn.click(
        fn=process_hd,
        inputs=[vton_input, garm_input, n_samples, n_steps, image_scale, seed],
        outputs=result_gallery
    )

# Launch with share=True for Kaggle access
block.queue().launch(share=True, server_port=7865)
