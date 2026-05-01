import requests
import time
import json

# 1. SET YOUR LIVE URL HERE
BASE_URL = "https://typically-wheylike-magen.ngrok-free.dev" 

def test_tryon():
    print(f"\n--- Testing WearCast AI API ---")
    print(f"Target: {BASE_URL}")

    # 2. Start the Try-On Task
    print("\n[1/3] Sending images to /tryon...")
    # Using your existing example images
    files = {
        'vton_img': open('run/examples/model/model_1.png', 'rb'),
        'garm_img': open('run/examples/garment/03244_00.jpg', 'rb')
    }
    
    response = requests.post(f"{BASE_URL}/tryon", files=files)
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    data = response.json()
    task_id = data['task_id']
    print(f"✅ Task Created! ID: {task_id}")
    print(f"⏱️ Estimated Time: {data['estimate']} seconds")

    # 3. Listen to the Professional Progress Stream
    print(f"\n[2/3] Opening Professional Progress Stream...")
    print("-" * 50)
    
    # Use 'stream=True' to listen to the SSE updates
    with requests.get(f"{BASE_URL}/stream/{task_id}", stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    event_data = json.loads(decoded_line[6:])
                    
                    status = event_data.get('status', 'processing')
                    remaining = event_data.get('remaining', 0)
                    msg = event_data.get('message', '')
                    
                    if status == "initializing":
                        print(f"🕒 [INITIALIZING] {msg} (Est: {remaining}s)")
                    elif status == "completed":
                        print(f"\n✨ [SUCCESS] Virtual Try-On Finished!")
                        result_url = event_data['url']
                        break
                    else:
                        print(f"🚀 [PROCESSING] Status: {status} | Remaining: {remaining}s")

    # 4. Download the final image
    print(f"\n[3/3] Downloading result from {BASE_URL}{result_url}")
    img_response = requests.get(f"{BASE_URL}{result_url}")
    with open("tryon_result.png", "wb") as f:
        f.write(img_response.content)
    print(f"🎉 Result saved to: tryon_result.png")

if __name__ == "__main__":
    test_tryon()
