import cv2
import base64
import os

files = [
    'debug_904dc8f7-5d95-472c-825d-c69b8821041c/debug_phase4_comparison.jpg',
    'debug_216838ce-99ad-4bfa-881e-258706d07d69/debug_phase4_comparison.jpg',
    'debug_fae745d5-57ec-47fc-9883-edac7e7d2c52/debug_phase4_comparison.jpg'
]

html = '<html><body>'
for f in files:
    if os.path.exists(f):
        with open(f, 'rb') as img_f:
            b64 = base64.b64encode(img_f.read()).decode('utf-8')
            html += f'<h3>{f}</h3><img src="data:image/jpeg;base64,{b64}" width="800"><br>'
html += '</body></html>'
with open('debug_images.html', 'w') as f:
    f.write(html)
