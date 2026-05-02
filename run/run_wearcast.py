import os
import sys

# --- Absolute Path Fix ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Kaggle Compatibility Monkey-Patch (ULTIMATE DEEP INJECTION v5) ---
try:
    # 1. Force-patch transformers before ANYTHING else
    import transformers
    if not hasattr(transformers, "EncoderDecoderCache"):
        # We create a dummy class and inject it into the module
        class DummyCache: pass
        transformers.EncoderDecoderCache = DummyCache
        # Also ensure it's in the actual module dictionary for 'from transformers import ...'
        sys.modules['transformers'].EncoderDecoderCache = DummyCache
        print("Successfully deep-patched transformers.EncoderDecoderCache")

    # 2. Patch huggingface_hub.errors for PEFT/Diffusers
    import huggingface_hub.errors
    base_err = getattr(huggingface_hub.errors, "HFValidationError", Exception)
    class HFEntryNotFoundError(base_err): pass
    
    for name in ["EntryNotFoundError", "LocalEntryNotFoundError"]:
        if not hasattr(huggingface_hub.errors, name):
            setattr(huggingface_hub.errors, name, HFEntryNotFoundError)
            # Inject into sys.modules to be safe
            sys.modules['huggingface_hub.errors'].__dict__[name] = HFEntryNotFoundError
    
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception as e:
    print(f"Warning: Monkey-patch failed: {e}")
# ---------------------------------------------------------------------

import argparse
import torch
from PIL import Image

def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"Detected GPU: {gpu_name} (Capability {capability[0]}.{capability[1]})")
        if capability[0] < 7:
            print("\n" + "!"*60)
            print("CRITICAL WARNING: YOU ARE ON A P100 GPU (Capability 6.0).")
            print("THIS MODEL REQUIRES A T4 GPU (Capability 7.0+) TO RUN.")
            print("Please switch Accelerator to 'GPU T4 x2' in the Kaggle sidebar.")
            print("!"*60 + "\n")
    else:
        print("CRITICAL: No GPU detected. Please enable GPU in Kaggle settings.")

from wearcast.inference_wearcast_hd import WearCastHD
from utils_wearcast import smart_resize

if __name__ == '__main__':
    check_gpu()
    
    parser = argparse.ArgumentParser(description='WearCast AI - Virtual Try-on')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the person image')
    parser.add_argument('--cloth_path', type=str, required=True, help='Path to the garment image')
    parser.add_argument('--category', type=int, default=0, help='0: upperbody')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()

    # HD only supports upperbody (0)
    category_map = {0: 'upperbody'}
    category_str = category_map.get(args.category, 'upperbody')

    model = WearCastHD(args.gpu_id)
    
    model_img = Image.open(args.model_path).convert('RGB')
    cloth_img = Image.open(args.cloth_path).convert('RGB')
    
    model_img_smart = smart_resize(model_img)
    cloth_img_smart = smart_resize(cloth_img)
    
    # Run inference
    result = model(
        model_type='hd',
        category=category_str,
        image_garm=cloth_img_smart,
        image_vton=model_img_smart,
        mask=None,
        image_ori=model_img_smart,
        num_samples=1,
        num_steps=20,
        image_scale=2.0,
        seed=-1
    )
    
    os.makedirs('images_output', exist_ok=True)
    result[0].save('images_output/output.png')
    print("Inference completed! Result saved to images_output/output.png")
