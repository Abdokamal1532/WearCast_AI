import pdb
import time

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import onnxruntime as ort
from parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)

        checkpoints_root = Path(__file__).absolute().parents[2].absolute()
        atr_path = os.path.join(checkpoints_root, 'checkpoints/humanparsing/parsing_atr.onnx')
        lip_path = os.path.join(checkpoints_root, 'checkpoints/humanparsing/parsing_lip.onnx')

        print(f"   [Parsing.__init__] Initialising ONNX sessions on GPU id={gpu_id}")
        print(f"   [Parsing.__init__] ATR model path: {atr_path}")
        print(f"   [Parsing.__init__]   -> Exists: {os.path.isfile(atr_path)}")
        print(f"   [Parsing.__init__] LIP model path: {lip_path}")
        print(f"   [Parsing.__init__]   -> Exists: {os.path.isfile(lip_path)}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))

        print(f"   [Parsing.__init__] Loading ATR ONNX session (CPUExecutionProvider)...")
        t0 = time.time()
        self.session = ort.InferenceSession(
            atr_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        print(f"   [Parsing.__init__] ATR session loaded in {time.time()-t0:.2f}s")
        print(f"   [Parsing.__init__]   Input  name: {self.session.get_inputs()[0].name}  shape={self.session.get_inputs()[0].shape}")
        print(f"   [Parsing.__init__]   Output names: {[o.name for o in self.session.get_outputs()]}")

        print(f"   [Parsing.__init__] Loading LIP ONNX session (CPUExecutionProvider)...")
        t0 = time.time()
        self.lip_session = ort.InferenceSession(
            lip_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        print(f"   [Parsing.__init__] LIP session loaded in {time.time()-t0:.2f}s")
        print(f"   [Parsing.__init__]   Input  name: {self.lip_session.get_inputs()[0].name}  shape={self.lip_session.get_inputs()[0].shape}")
        print(f"   [Parsing.__init__]   Output names: {[o.name for o in self.lip_session.get_outputs()]}")
        print(f"   [Parsing.__init__] Both ONNX sessions ready.")

    def __call__(self, input_image):
        torch.cuda.set_device(self.gpu_id)
        print(f"   [Parsing.__call__] Running ONNX inference on input image...")
        if hasattr(input_image, 'size'):
            print(f"   [Parsing.__call__] Input image size: {input_image.size}  mode={input_image.mode}")
        t_start = time.time()
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        elapsed = time.time() - t_start
        print(f"   [Parsing.__call__] ONNX inference complete in {elapsed:.2f}s")
        if hasattr(parsed_image, 'size'):
            import numpy as np
            arr = np.array(parsed_image)
            unique = np.unique(arr)
            print(f"   [Parsing.__call__] Parsed image size={parsed_image.size}  unique_labels={unique.tolist()}")
        print(f"   [Parsing.__call__] face_mask type={type(face_mask)}  shape={face_mask.shape if hasattr(face_mask, 'shape') else 'N/A'}")
        return parsed_image, face_mask
