import os
folders=['debug_7d521908-b675-42f5-9616-c722df42c841', 'debug_91af092a-1d69-4534-bd47-8664361192c4', 'debug_82111d81-801e-4444-b559-fae088a84785', 'debug_e5e8324a-1016-4077-b808-1e10304ec5e2']
html="<html><body style='background:black;color:white;'><h1>WearCast Debug Images</h1>\n"
for f in folders:
    html += f"<h2>{f}</h2>\n"
    for img in ['debug_phase1_hard_mask.jpg', 'debug_phase4_comparison.jpg', 'debug_final_output.jpg']:
        html += f"<h3>{img}</h3><img src='{f}/{img}' width='600'/><br>\n"
html+="</body></html>"
with open('view_results2.html', 'w') as out:
    out.write(html)
print("Done")
