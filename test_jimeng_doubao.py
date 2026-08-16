
import sys
import os
import importlib.util

# Load testV1.0.py dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "testV1.0.py")
spec = importlib.util.spec_from_file_location("testV1_0", file_path)
testV1_0 = importlib.util.module_from_spec(spec)
sys.modules["testV1_0"] = testV1_0
spec.loader.exec_module(testV1_0)

from testV1_0 import (
    edit_with_jimeng_image,
    edit_with_doubao_seedream,
    UPLOAD_FOLDER
)

def find_latest_image():
    if not os.path.exists(UPLOAD_FOLDER):
        return None
    files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def test_all():
    image_path = find_latest_image()
    if not image_path:
        print("No images found in uploads folder.")
        return

    print(f"Using image: {image_path}")
    prompt = "cyberpunk city, neon lights"

    print("\n=== Testing Jimeng ===")
    try:
        url = edit_with_jimeng_image(image_path, prompt)
        print(f"Jimeng Success: {url}")
    except Exception as e:
        print(f"Jimeng Failed: {e}")

    print("\n=== Testing Doubao ===")
    try:
        url = edit_with_doubao_seedream(image_path, prompt)
        print(f"Doubao Success: {url}")
    except Exception as e:
        print(f"Doubao Failed: {e}")

if __name__ == "__main__":
    test_all()
