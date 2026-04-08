import os
from huggingface_hub import hf_hub_download

def download_wearcast_models():
    # Define models to download (Repo ID, Filename, Local Path)
    models = [
        ("levihsu/OOTDiffusion", "checkpoints/humanparsing/parsing_atr.onnx", "checkpoints/humanparsing/parsing_atr.onnx"),
        ("levihsu/OOTDiffusion", "checkpoints/humanparsing/parsing_lip.onnx", "checkpoints/humanparsing/parsing_lip.onnx"),
        ("levihsu/OOTDiffusion", "checkpoints/openpose/ckpts/body_pose_model.pth", "checkpoints/openpose/ckpts/body_pose_model.pth"),
        # UNet and other core models (HD version)
        ("levihsu/OOTDiffusion", "checkpoints/ootd/ootd_hd/pytorch_model.bin", "checkpoints/ootd/ootd_hd/pytorch_model.bin"),
        ("levihsu/OOTDiffusion", "checkpoints/ootd/ootd_hd/config.json", "checkpoints/ootd/ootd_hd/config.json"),
    ]

    for repo_id, filename, local_path in models:
        print(f"Downloading {filename}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=".",
                local_dir_use_symlinks=False
            )
        else:
            print(f"{filename} already exists, skipping.")

if __name__ == "__main__":
    download_wearcast_models()
