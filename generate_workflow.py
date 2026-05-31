from PIL import Image
import cv2
import numpy as np
import os

# Paths
base_dir = r'c:\Users\pc\PycharmProjects\WearCast_AI'
workflow_path = os.path.join(base_dir, 'images', 'workflow.png')
debug_folder = os.path.join(base_dir, 'debug_e5d7ed10-c6c5-4dd7-8f64-218a55ed642e')

comp_path = os.path.join(debug_folder, 'debug_phase4_comparison.jpg')
masked_path = os.path.join(debug_folder, 'debug_phase3_masked_person.jpg')
output_path = os.path.join(debug_folder, 'debug_final_output.jpg')

# Load original workflow image
workflow_img = Image.open(workflow_path).convert('RGB')

# Load and extract new images
comp_img = Image.open(comp_path)
W, H = comp_img.size
w = W // 3

garment = comp_img.crop((0, 0, w, H))
target = Image.open(os.path.join(base_dir, 'Abdo.png'))

masked = Image.open(masked_path)
output = Image.open(output_path)

# Resize all to 240x320
from PIL import ImageOps
new_size = (240, 320)
garment = garment.resize(new_size, Image.Resampling.LANCZOS)
masked = masked.resize(new_size, Image.Resampling.LANCZOS)
output = output.resize(new_size, Image.Resampling.LANCZOS)

# Preserve aspect ratio for target image so it's not distorted
target_fitted = ImageOps.contain(target, new_size, Image.Resampling.LANCZOS)
target_block = Image.new('RGB', new_size, (255, 255, 255))
offset_x = (new_size[0] - target_fitted.width) // 2
offset_y = (new_size[1] - target_fitted.height) // 2
target_block.paste(target_fitted, (offset_x, offset_y))
target = target_block

# Paste into workflow image
workflow_img.paste(garment, (0, 376))
workflow_img.paste(masked, (1500, 376))
workflow_img.paste(target, (1500, 856))
workflow_img.paste(output, (1500, 1296))

# Save the result
output_file = os.path.join(base_dir, 'generated_workflow.png')
workflow_img.save(output_file)
print('New workflow image created successfully at', output_file)
