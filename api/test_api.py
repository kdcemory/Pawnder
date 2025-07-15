# test_api.py
import requests
import base64
import json

def test_enhanced_api():
    # Simple 1x1 pixel image for testing
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    response = requests.post(
        "http://localhost:8000/predict-json",
        json={"image": test_image_b64, "filename": "test.png"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API Test Successful!")
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Safety Level: {result['report_card']['safety_assessment']['level']}")
        print(f"Dog Thoughts: {result['report_card']['dog_thoughts']}")
        print("\n📊 Full Report Card Available!")
    else:
        print(f"❌ API Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_enhanced_api()