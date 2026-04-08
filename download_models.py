import os
from huggingface_hub import hf_hub_download, list_repo_files

def download_wearcast_models():
    repo_id = "levihsu/OOTDiffusion"
    
    # Files to download mapping: (HF Path Variant 1, HF Path Variant 2, Local Name)
    # We will try Variant 1 first, then Variant 2.
    models = [
        # Human Parsing (ONNX)
        ("checkpoints/humanparsing/parsing_atr.onnx", None, "checkpoints/humanparsing/parsing_atr.onnx"),
        ("checkpoints/humanparsing/parsing_lip.onnx", None, "checkpoints/humanparsing/parsing_lip.onnx"),
        
        # OpenPose
        ("checkpoints/openpose/ckpts/body_pose_model.pth", None, "checkpoints/openpose/ckpts/body_pose_model.pth"),
        
        # OOTD HD Core - Try with and without checkpoint-36000
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/pytorch_model.bin", "checkpoints/ootd/ootd_hd/pytorch_model.bin", "checkpoints/ootd/pytorch_model.bin"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/config.json", "checkpoints/ootd/ootd_hd/config.json", "checkpoints/ootd/config.json"),
        
        # VAE
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/vae/config.json", "checkpoints/ootd/ootd_hd/vae/config.json", "checkpoints/ootd/vae/config.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/vae/pytorch_model.bin", "checkpoints/ootd/ootd_hd/vae/pytorch_model.bin", "checkpoints/ootd/vae/pytorch_model.bin"),
        
        # Text Encoder
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/text_encoder/config.json", "checkpoints/ootd/ootd_hd/text_encoder/config.json", "checkpoints/ootd/text_encoder/config.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/text_encoder/pytorch_model.bin", "checkpoints/ootd/ootd_hd/text_encoder/pytorch_model.bin", "checkpoints/ootd/text_encoder/pytorch_model.bin"),
        
        # Tokenizer
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/tokenizer/vocab.json", "checkpoints/ootd/ootd_hd/tokenizer/vocab.json", "checkpoints/ootd/tokenizer/vocab.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/tokenizer/merges.txt", "checkpoints/ootd/ootd_hd/tokenizer/merges.txt", "checkpoints/ootd/tokenizer/merges.txt"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/tokenizer/special_tokens_map.json", "checkpoints/ootd/ootd_hd/tokenizer/special_tokens_map.json", "checkpoints/ootd/tokenizer/special_tokens_map.json"),
        ("checkpoints/ootd/ootd_hd/checkpoint-36000/tokenizer/tokenizer_config.json", "checkpoints/ootd/ootd_hd/tokenizer/tokenizer_config.json", "checkpoints/ootd/tokenizer/tokenizer_config.json"),
    ]

    for v1, v2, local_path in models:
        print(f"Syncing {local_path}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            print(f"Skipping {local_path} (exists).")
            continue
            
        success = False
        for variant in [v1, v2]:
            if variant is None: continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=variant,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                
                # Check if it downloaded to the named path, if not rename
                if variant != local_path and os.path.exists(variant):
                    os.rename(variant, local_path)
                
                success = True
                break
            except Exception as e:
                continue
        
        if not success:
            print(f"FAILED to download {local_path}")

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
