import os
import shutil
from pathlib import Path

def cleanup():
    project_root = Path(__file__).absolute().parent
    print(f"Cleaning up in {project_root}...")

    # 1. Remove the double-nested checkpoints folder (Huge waste of space)
    double_nested = project_root / "checkpoints" / "checkpoints"
    if double_nested.exists():
        print(f"Removing redundant nested folder: {double_nested}")
        shutil.rmtree(double_nested)

    # 2. Remove .huggingface cache folders
    for d in project_root.rglob(".huggingface"):
        if d.is_dir():
            print(f"Removing cache: {d}")
            shutil.rmtree(d)

    # 3. Remove unnecessary CLIP variants (We only need PyTorch or Safetensors)
    clip_dir = project_root / "checkpoints" / "clip-vit-large-patch14"
    if clip_dir.exists():
        for junk in ["flax_model.msgpack", "tf_model.h5"]:
            junk_path = clip_dir / junk
            if junk_path.exists():
                print(f"Removing junk CLIP format: {junk}")
                junk_path.unlink()

    # 4. Remove redundant humanparsing .pth files (we use .onnx)
    hp_dir = project_root / "checkpoints" / "humanparsing"
    if hp_dir.exists():
        for pth in hp_dir.glob("*.pth"):
            print(f"Removing old weights: {pth}")
            pth.unlink()

    # 5. Remove all __pycache__ folders
    for pyc in project_root.rglob("__pycache__"):
        if pyc.is_dir():
            shutil.rmtree(pyc)

    print("Cleanup complete! You should have significantly more space now.")

if __name__ == "__main__":
    cleanup()
