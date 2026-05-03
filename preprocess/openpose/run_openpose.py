import pdb
import time

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os

import cv2
import einops
import numpy as np
import random
import json

from preprocess.openpose.annotator.util import resize_image, HWC3
from preprocess.openpose.annotator.openpose import OpenposeDetector

import argparse
from PIL import Image
import torch


class OpenPose:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        print(f"   [OpenPose.__init__] Initialising OpenposeDetector on GPU id={gpu_id}")
        t0 = time.time()
        self.preprocessor = OpenposeDetector()
        print(f"   [OpenPose.__init__] OpenposeDetector ready in {time.time()-t0:.2f}s")

    def __call__(self, input_image, resolution=384):
        torch.cuda.set_device(self.gpu_id)
        print(f"   [OpenPose.__call__] Running inference  resolution={resolution}")

        if isinstance(input_image, Image.Image):
            print(f"   [OpenPose.__call__] Input: PIL Image  size={input_image.size}  mode={input_image.mode}")
            input_image = np.asarray(input_image)
        elif type(input_image) == str:
            print(f"   [OpenPose.__call__] Input: file path  '{input_image}'")
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError(f"[OpenPose.__call__] Unsupported input type: {type(input_image)}")

        print(f"   [OpenPose.__call__] numpy image: shape={input_image.shape}  dtype={input_image.dtype}  min={input_image.min()}  max={input_image.max()}")

        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            print(f"   [OpenPose.__call__] After resize: H={H}  W={W}  C={C}")
            assert (H == 512 and W == 384), f'Incorrect input image shape: expected (512,384) got ({H},{W})'

            t0 = time.time()
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)
            elapsed = time.time() - t0
            print(f"   [OpenPose.__call__] OpenposeDetector forward pass: {elapsed:.2f}s")

            candidate = pose['bodies']['candidate']
            subset    = pose['bodies']['subset']
            if len(subset) == 0:
                print("   [OpenPose.__call__] WARNING: No person detected in the image.")
                # Return zeroed keypoints for all 18 joints
                candidate = [[0, 0] for _ in range(18)]
                keypoints = {"pose_keypoints_2d": candidate}
                return keypoints

            subset = subset[0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if subset[j] != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if subset[j] != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            # Scale to pixel coordinates
            for i in range(18):
                candidate[i][0] *= 384   # x (width)
                candidate[i][1] *= 512   # y (height)

            print(f"   [OpenPose.__call__] Final 18 keypoints (pixel coords):")
            kp_names = ['Nose','Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
                        'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye','REar','LEar']
            for i, (name, pt) in enumerate(zip(kp_names, candidate)):
                flagged = " *** MISSING (0,0)" if pt[0] == 0 and pt[1] == 0 else ""
                print(f"      KP[{i:02d}] {name:12s}: ({pt[0]:.1f}, {pt[1]:.1f}){flagged}")

            keypoints = {"pose_keypoints_2d": candidate}

        print(f"   [OpenPose.__call__] Done. keypoints dict ready.")
        return keypoints


if __name__ == '__main__':
    model = OpenPose(0)
    model('./images/bad_model.jpg')
