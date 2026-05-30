"""
WearCast AI — HTML Report Generator
=====================================
Run AFTER run_present.py to build a self-contained HTML report from the
run_output_present/ folder.

Usage:
    python run/generate_present_report.py
"""

import os, sys, json, base64, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
OUT = PROJECT_ROOT / "run_output_present"
REPORT_PATH = OUT / "PRESENTATION_REPORT.html"


def b64(path: Path) -> str:
    """Return base64-encoded image string for embedding in HTML."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_tag(path: Path, caption: str = "", width: str = "100%") -> str:
    data = b64(path)
    if not data:
        return f'<div class="missing">⚠ {path.name} not found</div>'
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg","jpeg") else ext
    alt = caption or path.name
    return (
        f'<figure>'
        f'<img src="data:image/{mime};base64,{data}" '
        f'alt="{alt}" style="width:{width};border-radius:8px;">'
        f'{"<figcaption>" + caption + "</figcaption>" if caption else ""}'
        f'</figure>'
    )


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def json_table(data: dict) -> str:
    if not data:
        return "<p><em>No data</em></p>"
    rows = ""
    for k, v in data.items():
        if isinstance(v, dict):
            v_str = "<pre>" + json.dumps(v, indent=2, default=str) + "</pre>"
        else:
            v_str = str(v)
        rows += f"<tr><td class='key'>{k}</td><td>{v_str}</td></tr>"
    return f"<table class='meta'><tbody>{rows}</tbody></table>"


def json_rows(rows: list, max_rows=40) -> str:
    if not rows:
        return "<p><em>No data</em></p>"
    shown = rows[:max_rows]
    html = "<table class='steps'><thead><tr>"
    keys = list(shown[0].keys()) if shown else []
    for k in keys:
        html += f"<th>{k}</th>"
    html += "</tr></thead><tbody>"
    for row in shown:
        html += "<tr>"
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, dict):
                v = json.dumps(v)
            html += f"<td>{v}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    if len(rows) > max_rows:
        html += f"<p><em>... showing {max_rows} of {len(rows)} rows</em></p>"
    return html


# ─────────────────────────────────────────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:    #0d0f1a;
  --card:  #151929;
  --card2: #1a1f35;
  --accent:#7c5cfc;
  --accent2:#4ea8de;
  --green: #3ecf8e;
  --text:  #d0d8f0;
  --muted: #6b7a99;
  --border:#2a3050;
  --radius:12px;
}
body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height:1.6; }
header { background: linear-gradient(135deg,#1a1060 0%,#0d1533 60%,#0a1a40 100%);
         padding: 48px 32px 36px; text-align:center; border-bottom: 1px solid var(--border); }
header h1 { font-size:2.6rem; font-weight:800; letter-spacing:-.5px;
            background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
header p  { color:var(--muted); margin-top:8px; font-size:.95rem; }
.badge { display:inline-block; background:var(--accent); color:#fff;
         border-radius:20px; padding:3px 14px; font-size:.78rem; margin:4px; }
main { max-width: 1400px; margin: 0 auto; padding: 32px 20px; }
section { margin-bottom:48px; }
.phase-header { display:flex; align-items:center; gap:12px; margin-bottom:20px;
                padding-bottom:12px; border-bottom: 2px solid var(--border); }
.phase-num  { width:44px; height:44px; border-radius:50%; display:flex; align-items:center;
              justify-content:center; font-weight:800; font-size:1.1rem; flex-shrink:0; }
.p0 { background:linear-gradient(135deg,#7c5cfc,#4ea8de); }
.p1 { background:linear-gradient(135deg,#f59e0b,#ef4444); }
.p2 { background:linear-gradient(135deg,#10b981,#3b82f6); }
.p3 { background:linear-gradient(135deg,#8b5cf6,#ec4899); }
.p4 { background:linear-gradient(135deg,#3ecf8e,#60a5fa); }
.ps { background:linear-gradient(135deg,#f59e0b,#3ecf8e); }
.phase-header h2 { font-size:1.4rem; font-weight:700; }
.phase-header p  { color:var(--muted); font-size:.9rem; }
.grid { display:grid; gap:16px; }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
@media(max-width:900px) { .grid-3,.grid-4 { grid-template-columns:1fr 1fr; } }
@media(max-width:600px) { .grid-2,.grid-3,.grid-4 { grid-template-columns:1fr; } }
.card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
        padding:20px; transition:transform .2s,box-shadow .2s; }
.card:hover { transform:translateY(-2px); box-shadow:0 8px 32px rgba(124,92,252,.15); }
.card h3 { font-size:1rem; font-weight:600; color:var(--accent2); margin-bottom:12px; }
figure { text-align:center; }
figure img { max-width:100%; border:1px solid var(--border); border-radius:8px; }
figcaption { font-size:.78rem; color:var(--muted); margin-top:6px; }
.full-width { grid-column:1/-1; }
table.meta { width:100%; border-collapse:collapse; font-size:.82rem; }
table.meta td { padding:6px 10px; border-bottom:1px solid var(--border); vertical-align:top; }
table.meta td.key { color:var(--accent2); font-weight:600; white-space:nowrap; width:40%; }
table.steps { width:100%; border-collapse:collapse; font-size:.75rem; }
table.steps th { background:var(--card2); padding:6px 8px; text-align:left;
                 color:var(--accent); border-bottom:2px solid var(--border); }
table.steps td { padding:5px 8px; border-bottom:1px solid var(--border); }
table.steps tr:hover td { background:var(--card2); }
pre { font-size:.75rem; background:var(--card2); padding:10px; border-radius:6px;
      overflow-x:auto; color:#a0c4ff; }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; }
.stat { background:var(--card2); border-radius:10px; padding:14px 16px; text-align:center; }
.stat .val { font-size:1.6rem; font-weight:800; color:var(--green); }
.stat .lbl { font-size:.75rem; color:var(--muted); margin-top:4px; }
.missing { background:#2a1010; color:#f87171; padding:10px; border-radius:6px; font-size:.8rem; }
.timeline { list-style:none; border-left:2px solid var(--border); padding-left:20px; }
.timeline li { margin-bottom:12px; position:relative; }
.timeline li::before { content:''; position:absolute; left:-26px; top:5px;
                        width:10px; height:10px; border-radius:50%;
                        background:var(--accent); border:2px solid var(--bg); }
.timeline .ev { font-weight:600; }
.timeline .ts { font-size:.78rem; color:var(--muted); margin-left:8px; }
footer { text-align:center; padding:32px; color:var(--muted); font-size:.82rem;
         border-top:1px solid var(--border); margin-top:48px; }
"""

# ─────────────────────────────────────────────────────────────────────────────
def build():
    hw   = load_json(OUT / "phase0_model_loading" / "hardware_info.json")
    inp  = load_json(OUT / "phase0_model_loading" / "input_metadata.json")
    mat  = load_json(OUT / "phase1_human_parsing_pose" / "p1a_matting_metadata.json")
    pose = load_json(OUT / "phase1_human_parsing_pose" / "p1b_pose_keypoints.json")
    par  = load_json(OUT / "phase1_human_parsing_pose" / "p1c_parse_metadata.json")
    msk  = load_json(OUT / "phase1_human_parsing_pose" / "p1d_mask_metadata.json")
    clip = load_json(OUT / "phase2_latent_processing_fusion" / "p2a_clip_embedding_stats.json")
    pe   = load_json(OUT / "phase2_latent_processing_fusion" / "p2b_prompt_embedding_stats.json")
    vae  = load_json(OUT / "phase2_latent_processing_fusion" / "p2c_vae_encoding_stats.json")
    gen  = load_json(OUT / "phase3_generation_diffusion" / "p3_generation_metadata.json")
    steps= load_json(OUT / "phase3_generation_diffusion" / "p3_per_step_latent_stats.json")
    q    = load_json(OUT / "phase4_postprocessing" / "p4_quality_metrics.json")
    tl   = load_json(OUT / "summary" / "full_timeline.json")

    total_s = q.get("total_pipeline_time_s", "—")
    gen_s   = q.get("generation_time_s", "—")
    gpu_name= hw.get("gpu_name", "Unknown GPU")
    gpu_ram = hw.get("gpu_vram_gb", "—")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WearCast AI — Presentation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<header>
  <h1>🧥 WearCast AI — Full Pipeline Report</h1>
  <p>Generated: {now_str} &nbsp;|&nbsp; GPU: {gpu_name} ({gpu_ram} GB VRAM)</p>
  <p>
    <span class="badge">Phase 0: Model Loading</span>
    <span class="badge">Phase 1: Parsing & Pose</span>
    <span class="badge">Phase 2: Latent Fusion</span>
    <span class="badge">Phase 3: Diffusion</span>
    <span class="badge">Phase 4: Post-processing</span>
  </p>
</header>

<main>

<!-- ──────────────────────────── QUICK STATS ──────────────────────────── -->
<section>
  <div class="stat-grid">
    <div class="stat"><div class="val">{total_s}s</div><div class="lbl">Total Pipeline Time</div></div>
    <div class="stat"><div class="val">{gen_s}s</div><div class="lbl">Diffusion Generation</div></div>
    <div class="stat"><div class="val">{gen.get("num_steps","30")}</div><div class="lbl">Denoising Steps</div></div>
    <div class="stat"><div class="val">{vae.get("garment",{{}}).get("roundtrip_psnr_dB","—")} dB</div><div class="lbl">Garment VAE PSNR</div></div>
    <div class="stat"><div class="val">{msk.get("mask_coverage_percent","—")}%</div><div class="lbl">Mask Coverage</div></div>
    <div class="stat"><div class="val">{pose.get("keypoint_count","18")}</div><div class="lbl">Pose Keypoints</div></div>
  </div>
</section>

<!-- ──────────────────────────── MASTER COLLAGE ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num ps">★</div>
    <div><h2>Master Pipeline Collage</h2>
         <p>Full before → after across all stages</p></div>
  </div>
  <div class="card">
    {img_tag(OUT/"summary"/"MASTER_COLLAGE.jpg", "Full pipeline: Input → Pose → Parse → UNet → Final")}
  </div>
</section>

<!-- ──────────────────────────── PHASE 0 ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num p0">0</div>
    <div><h2>Phase 0 — Hardware & Model Loading</h2>
         <p>Environment detection, WearCastHD initialisation, input preprocessing</p></div>
  </div>
  <div class="grid grid-2">
    <div class="card">
      <h3>Hardware Info</h3>
      {json_table(hw)}
    </div>
    <div class="card">
      <h3>Input Images</h3>
      {json_table(inp)}
    </div>
    <div class="card">
      <h3>Person Input (768×1024)</h3>
      {img_tag(OUT/"phase0_model_loading"/"input_person_resized_768x1024.jpg")}
    </div>
    <div class="card">
      <h3>Garment Input (768×1024)</h3>
      {img_tag(OUT/"phase0_model_loading"/"input_garment_resized_768x1024.jpg")}
    </div>
  </div>
</section>

<!-- ──────────────────────────── PHASE 1 ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num p1">1</div>
    <div><h2>Phase 1 — Human Parsing &amp; Pose Estimation</h2>
         <p>rembg garment matting · OpenPose skeleton · SCHP semantic parsing · inpainting mask</p></div>
  </div>

  <!-- Summary strip -->
  <div class="card" style="margin-bottom:16px">
    <h3>Phase 1 — Full Summary Strip</h3>
    {img_tag(OUT/"phase1_human_parsing_pose"/"PHASE1_SUMMARY.jpg")}
  </div>

  <div class="grid grid-3">
    <!-- Matting -->
    <div class="card">
      <h3>1a — Garment Matting (rembg U2-Net)</h3>
      {img_tag(OUT/"phase1_human_parsing_pose"/"p1a_garment_matting_strip.jpg")}
      {json_table(mat)}
    </div>
    <!-- OpenPose -->
    <div class="card">
      <h3>1b — OpenPose Skeleton</h3>
      {img_tag(OUT/"phase1_human_parsing_pose"/"p1b_openpose_strip.jpg")}
      {json_table({"inference_time_s": pose.get("inference_time_s","—"),
                   "keypoints_detected": pose.get("keypoint_count","—")})}
    </div>
    <!-- Parse -->
    <div class="card">
      <h3>1c — SCHP Semantic Parsing</h3>
      {img_tag(OUT/"phase1_human_parsing_pose"/"p1c_parse_strip.jpg")}
      {img_tag(OUT/"phase1_human_parsing_pose"/"p1c_parse_legend.png", "Legend")}
    </div>
    <!-- Mask -->
    <div class="card full-width">
      <h3>1d — Inpainting Mask Generation</h3>
      {img_tag(OUT/"phase1_human_parsing_pose"/"p1d_mask_strip.jpg")}
      {json_table(msk)}
    </div>
  </div>

  <!-- Pose keypoints table -->
  <div class="card" style="margin-top:16px">
    <h3>Keypoint Coordinates (18 joints)</h3>
    <div style="overflow-x:auto">
    <table class="steps"><thead><tr><th>ID</th><th>Name</th><th>X</th><th>Y</th></tr></thead><tbody>
"""
    kp_data = pose.get("keypoints", {})
    kp_names_list = ['Nose','Neck','RShoulder','RElbow','RWrist',
                     'LShoulder','LElbow','LWrist',
                     'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle',
                     'REye','LEye','REar','LEar']
    for i, name in enumerate(kp_names_list):
        pt = kp_data.get(name, {})
        x, y = pt.get("x","—"), pt.get("y","—")
        html += f"<tr><td>{i}</td><td>{name}</td><td>{x}</td><td>{y}</td></tr>"

    html += f"""</tbody></table>
    </div>
  </div>

  <!-- Parse labels table -->
  <div class="card" style="margin-top:16px">
    <h3>Parsing Labels Detected</h3>
    <div style="overflow-x:auto">
    <table class="steps"><thead><tr><th>Label ID</th><th>Name</th><th>Pixels</th><th>%</th></tr></thead><tbody>
"""
    for lid, info in par.get("labels", {}).items():
        html += f"<tr><td>{lid}</td><td>{info.get('name','?')}</td><td>{info.get('pixels',0):,}</td><td>{info.get('percent','—')}%</td></tr>"

    html += f"""</tbody></table>
    </div>
  </div>
</section>

<!-- ──────────────────────────── PHASE 2 ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num p2">2</div>
    <div><h2>Phase 2 — Latent Processing &amp; Fusion</h2>
         <p>CLIP Vision encoding · prompt embedding · VAE encoding · latent channel visualisation</p></div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <h3>Phase 2 — Full Summary Strip</h3>
    {img_tag(OUT/"phase2_latent_processing_fusion"/"PHASE2_SUMMARY.jpg")}
  </div>

  <div class="grid grid-3">
    <div class="card">
      <h3>2a — CLIP Input (224×224 crop)</h3>
      {img_tag(OUT/"phase2_latent_processing_fusion"/"p2a_clip_input_224x224.jpg")}
      {json_table({"projection_dim": clip.get("projection_dim","768"),
                   "hidden_size": clip.get("hidden_size","1024")})}
    </div>
    <div class="card">
      <h3>2a — CLIP Embedding (first 64 dims)</h3>
      {img_tag(OUT/"phase2_latent_processing_fusion"/"p2a_clip_embedding_barplot.jpg")}
      {json_table(clip.get("image_embeds",{{}}))}
    </div>
    <div class="card">
      <h3>2b — Prompt Embedding Stats</h3>
      {json_table(pe)}
    </div>
    <div class="card full-width">
      <h3>2c — VAE Round-trip Fidelity</h3>
      {img_tag(OUT/"phase2_latent_processing_fusion"/"p2c_vae_roundtrip_strip.jpg")}
      {json_table({
          "scaling_factor"   : vae.get("vae_scaling_factor","—"),
          "latent_channels"  : vae.get("vae_latent_channels","4"),
          "garment_psnr_dB"  : vae.get("garment",{}).get("roundtrip_psnr_dB","—"),
          "quality_rating"   : vae.get("garment",{}).get("psnr_quality","—"),
      })}
    </div>
    <div class="card">
      <h3>2c — Garment Latent (4 channels)</h3>
      {img_tag(OUT/"phase2_latent_processing_fusion"/"p2c_garment_latent_4ch.jpg")}
    </div>
    <div class="card">
      <h3>2c — Person Latent (4 channels)</h3>
      {img_tag(OUT/"phase2_latent_processing_fusion"/"p2c_person_latent_4ch.jpg")}
    </div>
  </div>
</section>

<!-- ──────────────────────────── PHASE 3 ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num p3">3</div>
    <div><h2>Phase 3 — Generation &amp; Reconstruction (Denoising Diffusion)</h2>
         <p>UNet-Garm × UNet-Vton dual-UNet · UniPC scheduler · {gen.get("num_steps","30")} denoising steps</p></div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <h3>Denoising Evolution Strip (every 5 steps)</h3>
    {img_tag(OUT/"phase3_generation_diffusion"/"p3_denoising_evolution_strip.jpg")}
  </div>

  <div class="grid grid-2">
    <div class="card">
      <h3>Generation Metadata</h3>
      {json_table(gen)}
    </div>
    <div class="card">
      <h3>Raw UNet Output</h3>
      {img_tag(OUT/"phase3_generation_diffusion"/"debug_phase4_raw_unet_output.jpg")}
    </div>
    <div class="card full-width">
      <h3>Per-Step Latent Statistics (first {min(40,len(steps))} of {len(steps)} steps)</h3>
      {json_rows(steps, max_rows=40)}
    </div>
  </div>
</section>

<!-- ──────────────────────────── PHASE 4 ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num p4">4</div>
    <div><h2>Phase 4 — Post-processing &amp; Final Result</h2>
         <p>Dynamic re-parsing · semantic alpha blending · feather mask · final compositing</p></div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <h3>Phase 4 — Full Summary Strip (5-panel)</h3>
    {img_tag(OUT/"phase4_postprocessing"/"PHASE4_SUMMARY.jpg")}
  </div>

  <div class="grid grid-4">
    <div class="card">
      <h3>Masked Person (UNet input)</h3>
      {img_tag(OUT/"phase4_postprocessing"/"debug_phase3_masked_person.jpg")}
    </div>
    <div class="card">
      <h3>Feather Mask</h3>
      {img_tag(OUT/"phase4_postprocessing"/"debug_phase4_feather_mask.jpg")}
    </div>
    <div class="card">
      <h3>Raw UNet Output</h3>
      {img_tag(OUT/"phase4_postprocessing"/"debug_phase4_raw_unet_output.jpg")}
    </div>
    <div class="card">
      <h3>✅ Final Try-on Result</h3>
      {img_tag(OUT/"phase4_postprocessing"/"p4_final_result.jpg")}
    </div>
    <div class="card full-width">
      <h3>Side-by-side Comparison (Garment | Raw UNet | Final)</h3>
      {img_tag(OUT/"phase4_postprocessing"/"debug_phase4_comparison.jpg")}
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Quality Metrics</h3>
    {json_table(q)}
  </div>
</section>

<!-- ──────────────────────────── TIMELINE ──────────────────────────── -->
<section>
  <div class="phase-header">
    <div class="phase-num ps">T</div>
    <div><h2>Full Pipeline Timeline</h2><p>Event-by-event elapsed times</p></div>
  </div>
  <div class="card">
    <ul class="timeline">
"""
    for ev in tl:
        html += f"""<li>
      <span class="ev">[{ev.get("phase","")}]  {ev.get("event","")}</span>
      <span class="ts">+{ev.get("elapsed_s","?")}s</span>
    </li>"""

    html += f"""
    </ul>
  </div>
</section>

</main>

<footer>
  WearCast AI Presentation Report · Generated {now_str} · GPU: {gpu_name}
</footer>

</body>
</html>
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML report saved to:\n   {REPORT_PATH}")
    print(f"   Open it in any browser — all images are embedded (no external dependencies).")
    return str(REPORT_PATH)


if __name__ == "__main__":
    if not OUT.exists():
        print(f"[ERROR] run_output_present/ not found at {OUT}")
        print("  Run  `python run/run_present.py --person X --garment Y`  first.")
        sys.exit(1)
    build()
