import os
folders=['debug_7ff2fbd5-8e15-47a0-b025-ccb1e795ca92', 'debug_1673a2dc-71c3-415d-96a5-fe871f5c3207', 'debug_d3cd663b-f433-43ae-9b83-b86b20ab34e2', 'debug_f294d9e1-6870-4d4a-a16a-b9da56c2637c']
html="<html><body style='background:black;color:white;'><h1>WearCast Debug Images</h1>\n"
for f in folders:
    html += f"<h2>{f}</h2>\n"
    for img in ['debug_phase1_hard_mask.jpg', 'debug_phase4_comparison.jpg', 'debug_final_output.jpg']:
        html += f"<h3>{img}</h3><img src='{f}/{img}' width='600'/><br>\n"
html+="</body></html>"
with open('view_results2.html', 'w') as out:
    out.write(html)
print("Done")
