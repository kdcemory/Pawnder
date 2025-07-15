p# image_to_base64_test.py - Convert image to base64 and test your API
import base64
import requests
import json
from pathlib import Path

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            # Read the image file
            image_data = image_file.read()
            # Encode to base64
            base64_string = base64.b64encode(image_data).decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def test_api_with_image(image_path, api_url="http://localhost:8000"):
    """Test your API with an image file"""
    
    print(f"🔍 Testing API with image: {image_path}")
    
    # Convert image to base64
    base64_image = image_to_base64(image_path)
    
    if not base64_image:
        print("❌ Failed to convert image to base64")
        return
    
    print(f"✅ Image converted to base64 ({len(base64_image)} characters)")
    
    # Prepare the request payload
    payload = {
        "image": base64_image,
        "filename": Path(image_path).name
    }
    
    # Test the /predict-json endpoint
    try:
        print("🚀 Sending request to API...")
        response = requests.post(
            f"{api_url}/predict-json",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 API Response Successful!")
            print(f"📊 Predicted Emotion: {result['emotion']}")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"📈 Emotion Score: {result['score']:.2f}")
            
            # Show top 3 emotions
            print("\n📋 Top 3 Emotions:")
            all_emotions = result['all_emotions']
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
            for i, (emotion, score) in enumerate(sorted_emotions[:3], 1):
                print(f"  {i}. {emotion}: {score:.3f}")
            
            # Show safety info
            if 'report_card' in result:
                safety = result['report_card']['safety_assessment']
                print(f"\n🛡️ Safety Level: {safety['level']}")
                print(f"💡 Advice: {safety['advice']}")
                
                # Show what the dog might be thinking
                thoughts = result['report_card']['dog_thoughts']
                print(f"🐕 Dog Thoughts: \"{thoughts}\"")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (API might be processing)")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API - make sure it's running on localhost:8000")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def create_curl_command(image_path):
    """Create a curl command for testing"""
    base64_image = image_to_base64(image_path)
    if base64_image:
        # Truncate base64 for display
        display_base64 = base64_image[:50] + "..." if len(base64_image) > 50 else base64_image
        
        curl_command = f'''curl -X POST "http://localhost:8000/predict-json" \\
  -H "Content-Type: application/json" \\
  -d '{{"image": "{base64_image}", "filename": "{Path(image_path).name}"}}\''''
        
        print(f"\n📋 Curl command (for reference):")
        print(f"curl -X POST \"http://localhost:8000/predict-json\" \\")
        print(f"  -H \"Content-Type: application/json\" \\")
        print(f"  -d '{{\"image\": \"{display_base64}\", \"filename\": \"{Path(image_path).name}\"}}'")

def main():
    print("🐕 Pawnder API Tester")
    print("=" * 50)
    
    # Look for common image files in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    current_dir = Path('.')
    
    found_images = []
    for ext in image_extensions:
        found_images.extend(current_dir.glob(f'*{ext}'))
        found_images.extend(current_dir.glob(f'*{ext.upper()}'))
    
    if found_images:
        print(f"🖼️ Found {len(found_images)} image(s) in current directory:")
        for i, img in enumerate(found_images, 1):
            print(f"  {i}. {img.name}")
        
        # Test with first image found
        test_image = found_images[0]
        print(f"\n🎯 Testing with: {test_image.name}")
        
        result = test_api_with_image(test_image)
        
        if result:
            # Save the full result for inspection
            with open('api_test_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full result saved to: api_test_result.json")
        
        # Create curl command
        create_curl_command(test_image)
        
    else:
        print("❌ No image files found in current directory")
        print("📁 Please add a dog image (jpg, png, etc.) to test with")
        print("🔍 Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        
        # Show how to use manually
        print(f"\n📝 Manual usage:")
        print(f"python {Path(__file__).name} your_dog_image.jpg")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Image path provided as argument
        image_path = sys.argv[1]
        if Path(image_path).exists():
            test_api_with_image(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    else:
        # Auto-detect images in current directory
        main()
