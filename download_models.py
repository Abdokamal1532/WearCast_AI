import os
from huggingface_hub import hf_hub_download

def download_wearcast_models():
    repo_id = "levihsu/OOTDiffusion"
    
    # EXACT MAPPING from the HuggingFace repository to local paths
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

    for hf_path, local_path in models:
        print(f"Syncing {local_path}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            print(f"Skipping {local_path} (exists).")
            continue
            
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=".",
                local_dir_use_symlinks=False
            )
            
            # If hf_hub_download saved it to the hf_path instead of local_path, move it
            if hf_path != local_path and os.path.exists(hf_path):
                os.rename(hf_path, local_path)
            
        except Exception as e:
            print(f"FAILED to download {hf_path}: {e}")

    # CLIP separately since it's a different repo
    clip_files = [
        ("openai/clip-vit-large-patch14", "pytorch_model.bin", "checkpoints/clip-vit-large-patch14/pytorch_model.bin"),
        ("openai/clip-vit-large-patch14", "config.json", "checkpoints/clip-vit-large-patch14/config.json"),
        ("openai/clip-vit-large-patch14", "preprocessor_config.json", "checkpoints/clip-vit-large-patch14/preprocessor_config.json"),
    ]
    for repo, fname, local in clip_files:
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            hf_hub_download(repo_id=repo, filename=fname, local_dir=".", local_dir_use_symlinks=False)
            if fname != local and os.path.exists(fname):
                os.rename(fname, local)

if __name__ == "__main__":
    download_wearcast_models()
