import requests
import time
import json
import urllib3
import os
import argparse
import glob

# Suppress insecure request warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. SET YOUR LIVE URL HERE
# Make sure this matches the "Public URL" printed in your Kaggle console!
BASE_URL = "https://typically-wheylike-magen.ngrok-free.dev" 

# Add headers to bypass ngrok's warning page
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "WearCastTestClient/1.0"
}

def find_fallback_images():
    """Search for Gemini-generated images if defaults are missing."""
    patterns = [
        "Gemini_Generated_Image_*.png",
        "Gemini_Generated_Image_*.jpg",
        "run/examples/model/*.png",
        "run/examples/garment/*.jpg"
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return sorted(found)

def test_tryon(person_path=None, garment_path=None):
    print(f"\n--- WearCast AI Professional Test Suite ---")
    print(f"Endpoint: {BASE_URL}")

    # Resolve paths robustly
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not person_path:
        # Default person: model_5.png
        person_path = os.path.join(root, "run", "examples", "model", "model_5.png")
    
    if not garment_path:
        # Default garment: 06123_00.jpg
        garment_path = os.path.join(root, "run", "examples", "garment", "06123_00.jpg")

    # Automatic Discovery Fallback
    if not os.path.exists(person_path) or not os.path.exists(garment_path):
        print("⚠️  Default images not found. Searching for Gemini alternatives...")
        fallbacks = find_fallback_images()
        if len(fallbacks) >= 2:
            # Simple heuristic: largest is often the model, smallest is often garment
            fallbacks.sort(key=os.path.getsize, reverse=True)
            if not os.path.exists(person_path): person_path = fallbacks[0]
            if not os.path.exists(garment_path): garment_path = fallbacks[1]
            print(f"✨ Auto-discovered: person={os.path.basename(person_path)}, garment={os.path.basename(garment_path)}")

    if not os.path.exists(person_path) or not os.path.exists(garment_path):
        print(f"❌ Error: Could not locate input images.")
        print(f"   Checked: {person_path}")
        print(f"   Checked: {garment_path}")
        return

    print(f"👤 Person  : {os.path.abspath(person_path)}")
    print(f"👕 Garment : {os.path.abspath(garment_path)}")

    # Create a session for better connection pooling
    session = requests.Session()
    session.verify = False
    session.trust_env = False  # Ignore system proxies

    try:
        # Determine MIME types
        p_ext = os.path.splitext(person_path)[1].lower()
        g_ext = os.path.splitext(garment_path)[1].lower()
        p_mime = "image/png" if p_ext == ".png" else "image/jpeg"
        g_mime = "image/png" if g_ext == ".png" else "image/jpeg"

        files = {
            'person': (os.path.basename(person_path), open(person_path, 'rb'), p_mime),
            'garment': (os.path.basename(garment_path), open(garment_path, 'rb'), g_mime)
        }
        
        print(f"\n[1/4] Uploading to API...")
        response = session.post(
            f"{BASE_URL}/tryon", 
            files=files, 
            headers=HEADERS,
            timeout=30
        )
        
        # Close file handles
        for key in files:
            files[key][1].close()
            
        if response.status_code != 200:
            print(f"❌ API Error ({response.status_code}): {response.text}")
            return

        data = response.json()
        task_id = data.get('task_id')
        estimate = data.get('estimated_time_seconds', 'unknown')
        
        print(f"✅ Task Created! ID: {task_id}")
        print(f"⏱️ Estimated Time: {estimate} seconds")

        # 3. Listen to the Professional Progress Stream
        print(f"\n[2/4] Opening Progress Stream...")
        print("-" * 50)
        
        result_url = None
        with session.get(f"{BASE_URL}/stream/{task_id}", stream=True, headers=HEADERS, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            event_data = json.loads(decoded_line[6:])
                            status = event_data.get('status', 'processing')
                            msg = event_data.get('message', '')
                            rem = event_data.get('remaining', 0)
                            
                            if status == "completed":
                                print(f"\n✨ [SUCCESS] Virtual Try-On Finished!")
                                result_url = event_data.get('url')
                                break
                            elif status == "failed":
                                print(f"\n❌ [FAILED] Task failed: {event_data.get('error')}")
                                return
                            else:
                                print(f"🚀 [{status.upper()}] {msg} | Remaining: {rem}s")

                        except json.JSONDecodeError:
                            pass

        if not result_url:
            print("❌ Error: Stream ended without a result URL.")
            return

        # 4. Download Result
        print(f"\n[3/4] Downloading final result...")
        img_response = session.get(f"{BASE_URL}{result_url}", headers=HEADERS)
        
        output_file = "tryon_result.png"
        with open(output_file, "wb") as f:
            f.write(img_response.content)
        print(f"🎉 Result saved to: {os.path.abspath(output_file)}")

        # 5. Download Debug Pipeline Images
        print(f"\n[4/4] Syncing Debug Assets...")
        # Extract task_id from result_url (/result/TASK_ID)
        task_id = result_url.split("/")[-1]
        debug_dir = os.path.join(os.getcwd(), f"debug_{task_id}")
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_files = [
            "debug_phase1_hard_mask.jpg",
            "debug_phase1_soft_mask.jpg",
            "debug_phase3_masked_person.jpg",
            "debug_phase4_comparison.jpg",
            "debug_final_output.jpg"
        ]
        
        for fname in debug_files:
            try:
                debug_url = f"{BASE_URL}/debug/{task_id}/{fname}"
                d_res = session.get(debug_url, stream=True)
                if d_res.status_code == 200:
                    local_path = os.path.join(debug_dir, fname)
                    with open(local_path, 'wb') as f:
                        for chunk in d_res.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  📥 Cached: {fname}")
            except:
                pass
        
        print(f"\n✨ [ALL DONE] Analysis Complete. Inspect results in: {debug_dir}")

    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WearCast AI API Professional Test Client")
    parser.add_argument("--person", type=str, help="Path to person image")
    parser.add_argument("--garment", type=str, help="Path to garment image")
    args = parser.parse_args()
    
    test_tryon(args.person, args.garment)


