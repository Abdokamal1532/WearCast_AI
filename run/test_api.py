import requests
import time
import json
import urllib3
import os

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

def test_tryon():
    print(f"\n--- Testing WearCast AI API ---")
    print(f"Target: {BASE_URL}")

    # Create a session for better connection pooling
    session = requests.Session()
    session.verify = False
    session.trust_env = False  # Ignore system proxies

    # 2. Start the Try-On Task
    print("\n[1/3] Sending images to /tryon...")
    
    person_path = 'run/examples/model/model_1.png'
    garment_path = 'run/examples/garment/03244_00.jpg'
    
    if not os.path.exists(person_path) or not os.path.exists(garment_path):
        print(f"❌ Error: Example images not found at {person_path} or {garment_path}")
        return

    try:
        files = {
            'person': (os.path.basename(person_path), open(person_path, 'rb'), 'image/png'),
            'garment': (os.path.basename(garment_path), open(garment_path, 'rb'), 'image/jpeg')
        }
        
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
        # Debug: print(f"DEBUG: Received data: {data}")
        
        task_id = data.get('task_id')
        estimate = data.get('estimated_time_seconds', data.get('estimate', 'unknown'))
        
        if not task_id:
            print(f"❌ Error: No task_id in response: {data}")
            return
            
        print(f"✅ Task Created! ID: {task_id}")
        print(f"⏱️ Estimated Time: {estimate} seconds")

        # 3. Listen to the Professional Progress Stream
        print(f"\n[2/3] Opening Professional Progress Stream...")
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
                            
                            if status == "initializing":
                                print(f"🕒 [INITIALIZING] {msg} (Est: {rem}s)")
                            elif status == "completed":
                                print(f"\n✨ [SUCCESS] Virtual Try-On Finished!")
                                result_url = event_data.get('url')
                                break
                            elif status == "failed":
                                print(f"\n❌ [FAILED] Task failed: {event_data.get('error')}")
                                return
                            elif status == "finalizing":
                                print(f"⏳ [FINALIZING] {msg}")
                            else:
                                display_msg = msg if msg else f"Status: {status}"
                                print(f"🚀 [PROCESSING] {display_msg} | Remaining: {rem}s")

                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: Could not parse SSE data: {decoded_line}")

        if not result_url:
            print("❌ Error: Stream ended without a result URL.")
            return

        # 4. Download Result
        print(f"\n[3/3] Downloading result...")
        img_response = session.get(f"{BASE_URL}{result_url}", headers=HEADERS)
        
        output_file = "tryon_result.png"
        with open(output_file, "wb") as f:
            f.write(img_response.content)
        print(f"🎉 Result saved to: {os.path.abspath(output_file)}")

    except requests.exceptions.SSLError as e:
        print(f"💥 SSL Error: {e}")
        print("Tip: This often happens with ngrok if the tunnel is unstable. Try restarting the Kaggle server.")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tryon()

