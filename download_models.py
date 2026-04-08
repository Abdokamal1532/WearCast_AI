import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_wearcast_models():
    # Establish Project Root absolutely
    PROJECT_ROOT = Path(__file__).absolute().parent
    repo_id = "levihsu/OOTDiffusion"
    
    # EXACT MAPPING from the HuggingFace repository to local paths (PROJECT-RELATIVE)
    models = [
        # Human Parsing (ONNX)
        ("checkpoints/humanparsing/parsing_atr.onnx", "checkpoints/humanparsing/parsing_atr.onnx"),
        ("checkpoints/humanparsing/parsing_lip.onnx", "checkpoints/humanparsing/parsing_lip.onnx"),
        
        # OpenPose
        ("checkpoints/openpose/ckpts/body_pose_model.pth", "checkpoints/openpose/ckpts/body_pose_model.pth"),
        
        # VAE
        ("checkpoints/ootd/vae/config.json", "checkpoints/ootd/vae/config.json"),
        ("checkpoints/ootd/vae/diffusion_pytorch_model.bin", "checkpoints/ootd/vae/diffusion_pytorch_model.bin"),
        
        # Text Encoder
        ("checkpoints/ootd/text_encoder/config.json", "checkpoints/ootd/text_encoder/config.json"),
        ("checkpoints/ootd/text_encoder/pytorch_model.bin", "checkpoints/ootd/text_encoder/pytorch_model.bin"),
        
        # UNet Garm (Checkpoint 36000)
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/config.json", "checkpoints/ootd/unet_garm/config.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors", "checkpoints/ootd/unet_garm/diffusion_pytorch_model.safetensors"),
        
        # UNet Vton (Checkpoint 36000)
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/config.json", "checkpoints/ootd/unet_vton/config.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors", "checkpoints/ootd/unet_vton/diffusion_pytorch_model.safetensors"),
    ]

    for hf_path, rel_local_path in models:
        # Use Absolute Path to prevent double-nesting if script is run from subfolders
        local_path = PROJECT_ROOT / rel_local_path
        
        print(f"Syncing {rel_local_path}...")
        os.makedirs(local_path.parent, exist_ok=True)
        if local_path.exists():
            print(f"Skipping {rel_local_path} (exists).")
            continue
            
        try:
            # We download to project root, keeping the repo's folder structure
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=str(PROJECT_ROOT),
                local_dir_use_symlinks=False
            )
            
            # If hf_hub_download didn't place it exactly where we wanted, move it
            # (Though local_dir usually handles this well now)
            actual_download_path = PROJECT_ROOT / hf_path
            if actual_download_path.exists() and actual_download_path != local_path:
                os.makedirs(local_path.parent, exist_ok=True)
                os.rename(actual_download_path, local_path)
            
        except Exception as e:
            print(f"FAILED to download {hf_path}: {e}")

    # CLIP Vision separately
    clip_repo = "openai/clip-vit-large-patch14"
    clip_files = ["pytorch_model.bin", "config.json", "preprocessor_config.json"]
    for fname in clip_files:
        local = PROJECT_ROOT / f"checkpoints/clip-vit-large-patch14/{fname}"
        if not local.exists():
            print(f"Syncing CLIP: {fname}...")
            os.makedirs(local.parent, exist_ok=True)
            hf_hub_download(repo_id=clip_repo, filename=fname, local_dir=str(PROJECT_ROOT), local_dir_use_symlinks=False)
            
            actual_dl = PROJECT_ROOT / fname # Sometimes hf_hub_download saves to root
            if actual_dl.exists() and actual_dl != local:
                os.rename(actual_dl, local)

if __name__ == "__main__":
    download_wearcast_models()
