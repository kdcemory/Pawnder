# test_debug_api.py - Test the debug API to see what's going wrong
import base64
import requests
import json
from pathlib import Path

def test_debug_api(image_path, api_url="http://localhost:8001"):
    """Test the debug API to see model outputs"""
    
    print(f"🐛 Testing Debug API with: {image_path}")
    
    # Convert image to base64
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return
    
    # Test model info first
    try:
        print("🔍 Getting model info...")
        response = requests.get(f"{api_url}/model-info")
        if response.status_code == 200:
            model_info = response.json()
            print("📊 Model Info:")
            print(json.dumps(model_info, indent=2))
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info request failed: {e}")
    
    print("\\n" + "="*50)
    
    # Test debug prediction
    payload = {
        "image": base64_image,
        "filename": Path(image_path).name
    }
    
    try:
        print("🚀 Running debug prediction...")
        response = requests.post(
            f"{api_url}/debug-predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\\n🎉 Debug Prediction Successful!")
            print(f"📊 Result: {result['emotion']} (confidence: {result['confidence']:.3f})")
            
            # Show debug info
            if 'debug_info' in result:
                print("\\n🐛 Debug Info:")
                debug_info = result['debug_info']
                for key, value in debug_info.items():
                    print(f"  {key}: {value}")
            
            # Save full result
            with open('debug_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("\\n💾 Full debug result saved to: debug_result.json")
            
        else:
            print(f"❌ Debug API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Debug request failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for test.png or any image
        if Path("test.png").exists():
            image_path = "test.png"
        else:
            # Find any image
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            current_dir = Path('.')
            found_images = []
            for ext in image_extensions:
                found_images.extend(current_dir.glob(f'*{ext}'))
                found_images.extend(current_dir.glob(f'*{ext.upper()}'))
            
            if found_images:
                image_path = str(found_images[0])
            else:
                print("❌ No image found. Usage: python test_debug_api.py your_image.jpg")
                exit(1)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        exit(1)
    
    test_debug_api(image_path)
