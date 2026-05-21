import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_wearcast_models():
    PROJECT_ROOT = Path(__file__).absolute().parent
    repo_id = "levihsu/OOTDiffusion"
    
    # ADDED: model_index.json and scheduler/extractor configs
    models = [
        # Pipeline Root Configs
        ("checkpoints/ootd/model_index.json", "checkpoints/ootd/model_index.json"),
        ("checkpoints/ootd/scheduler/scheduler_config.json", "checkpoints/ootd/scheduler/scheduler_config.json"),
        ("checkpoints/ootd/feature_extractor/preprocessor_config.json", "checkpoints/ootd/feature_extractor/preprocessor_config.json"),

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
        
        # Tokenizer
        ("checkpoints/ootd/tokenizer/vocab.json", "checkpoints/ootd/tokenizer/vocab.json"),
        ("checkpoints/ootd/tokenizer/merges.txt", "checkpoints/ootd/tokenizer/merges.txt"),
        ("checkpoints/ootd/tokenizer/special_tokens_map.json", "checkpoints/ootd/tokenizer/special_tokens_map.json"),
        ("checkpoints/ootd/tokenizer/tokenizer_config.json", "checkpoints/ootd/tokenizer/tokenizer_config.json"),

        # UNet Garm (Checkpoint 36000)
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/config.json", "checkpoints/ootd/unet_garm/config.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors", "checkpoints/ootd/unet_garm/diffusion_pytorch_model.safetensors"),
        
        # UNet Vton (Checkpoint 36000)
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/config.json", "checkpoints/ootd/unet_vton/config.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors", "checkpoints/ootd/unet_vton/diffusion_pytorch_model.safetensors"),
    ]

    for hf_path, rel_local_path in models:
        local_path = PROJECT_ROOT / rel_local_path
        print(f"Syncing {rel_local_path}...")
        os.makedirs(local_path.parent, exist_ok=True)
        if local_path.exists():
            print(f"Skipping {rel_local_path} (exists).")
            continue
            
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=str(PROJECT_ROOT),
                local_dir_use_symlinks=False
            )
            
            actual_download_path = PROJECT_ROOT / hf_path
            if actual_download_path.exists() and actual_download_path != local_path:
                os.makedirs(local_path.parent, exist_ok=True)
                os.rename(actual_download_path, local_path)
            
        except Exception as e:
            print(f"FAILED to download {hf_path}: {e}")

    # 3. Download CLIP Vision model files
    VIT_PATH = PROJECT_ROOT / "checkpoints/clip-vit-large-patch14"
    clip_repo = "openai/clip-vit-large-patch14"
    clip_files = [
        "preprocessor_config.json", 
        "config.json", 
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",  # Added this critical file
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]
    
    for fname in clip_files:
        local = VIT_PATH / fname
        if not local.exists():
            print(f"Syncing {local}...")
            os.makedirs(local.parent, exist_ok=True)
            # We download directly to the target folder to stay clean
            hf_hub_download(
                repo_id=clip_repo, 
                filename=fname, 
                local_dir=str(VIT_PATH), 
                local_dir_use_symlinks=False
            )
            
    # Fix model_index.json to point to our new namespace
    index_path = PROJECT_ROOT / "checkpoints/ootd/model_index.json"
    if index_path.exists():
        import json
        with open(index_path, 'r') as f:
            data = json.load(f)
        
        # Change the module path from 'pipelines_ootd' to 'wearcast.pipelines_wearcast'
        if data.get("unet_garm", [None])[0] == "pipelines_ootd":
            data["unet_garm"][0] = "wearcast.pipelines_wearcast"
        if data.get("unet_vton", [None])[0] == "pipelines_ootd":
            data["unet_vton"][0] = "wearcast.pipelines_wearcast"
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
        print("Updated model_index.json with wearcast namespace.")

if __name__ == "__main__":
    download_wearcast_models()
