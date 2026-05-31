from PIL import Image, ImageOps
import os

folders = [
    'debug_e5d7ed10-c6c5-4dd7-8f64-218a55ed642e',
    'debug_c626c9bb-c094-4627-a356-65b021cc5712',
    'debug_71f21125-0072-483d-bb49-0bb7153ba24f',
]

target_images = [
    'Abdo.png',
    'test.jpg',
    'user.jpg'
]

blocks = []

for i, folder in enumerate(folders):
    comp_path = os.path.join(folder, 'debug_phase4_comparison.jpg')
    if not os.path.exists(comp_path): continue
    comp_img = Image.open(comp_path)
    W, H = comp_img.size
    w = W // 3
    
    garment = comp_img.crop((0, 0, w, H))
    output = comp_img.crop((2*w, 0, 3*w, H))
    
    person = Image.open(target_images[i])
    
    garment = garment.resize((240, 320), Image.Resampling.LANCZOS)
    output = output.resize((480, 640), Image.Resampling.LANCZOS)
    
    # Preserve aspect ratio for person image
    person_fitted = ImageOps.contain(person, (240, 320), Image.Resampling.LANCZOS)
    person_block = Image.new('RGB', (240, 320), (255, 255, 255))
    offset_x = (240 - person_fitted.width) // 2
    offset_y = (320 - person_fitted.height) // 2
    person_block.paste(person_fitted, (offset_x, offset_y))
    person = person_block
    
    # Create block of 720x640 (photos only)
    block = Image.new('RGB', (720, 640), (255, 255, 255))
    
    block.paste(garment, (0, 0))
    block.paste(person, (0, 320))
    block.paste(output, (240, 0))
    
    blocks.append(block)

# Create the final canvas exactly fitting the blocks
final_img = Image.new('RGB', (720 * len(blocks), 640), (255, 255, 255))

for i, b in enumerate(blocks):
    x = i * 720
    y = 0
    final_img.paste(b, (x, y))

final_img.save('generated_demo.png')
print(f'generated_demo.png created successfully with size {720 * len(blocks)}x640')
