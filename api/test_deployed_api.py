# test_deployed_api.py
# Test script for your deployed Pawnder ML API

import requests
import base64
import json
from pathlib import Path

# Replace with your actual service URL
SERVICE_URL = "https://pawnder-emotion-api-316099560158.us-east4.run.app"

def test_health():
    """Test the health endpoint"""
    print("🔧 Testing health endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n📋 Testing model info endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/model-info", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved successfully:")
            print(f"   Classes: {len(data['model_info']['class_names'])}")
            print(f"   Behavior input size: {data['preprocessing']['behavior_input_size']}")
            print(f"   Image size: {data['preprocessing']['image_size']}")
            return True
        else:
            print(f"❌ Model info failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_prediction_with_sample():
    """Test prediction with a sample base64 image"""
    print("\n🧪 Testing prediction endpoint...")
    
    # Create a small test image (1x1 pixel) as base64
    # In a real test, you'd use an actual dog image
    import io
    from PIL import Image
    
    # Create a tiny test image
    test_image = Image.new('RGB', (224, 224), color='red')
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    print("   Using test image (red 224x224 pixel)")
    
    try:
        data = {
            "image": image_base64
        }
        
        response = requests.post(
            f"{SERVICE_URL}/predict", 
            json=data, 
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Predicted emotion: {result['emotion']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Emotion score: {result['emotion_score']:.2f}")
            print(f"   Top 3 emotions:")
            
            # Sort emotions by probability
            emotions_sorted = sorted(
                result['all_emotions'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for i, (emotion, prob) in enumerate(emotions_sorted[:3]):
                print(f"     {i+1}. {emotion}: {prob:.3f}")
            
            return True
        else:
            print(f"❌ Prediction failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_with_real_image(image_path):
    """Test with a real dog image file"""
    print(f"\n📸 Testing with real image: {image_path}")
    
    try:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        data = {
            "image": encoded_string
        }
        
        response = requests.post(
            f"{SERVICE_URL}/predict", 
            json=data, 
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Real image prediction successful:")
            print(f"   Predicted emotion: {result['emotion']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Emotion score: {result['emotion_score']:.2f}")
            return True
        else:
            print(f"❌ Real image prediction failed: {response.status_code}")
            return False
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Real image prediction error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Pawnder ML API Deployment\n")
    
    # Update this with your actual service URL
    global SERVICE_URL
    if SERVICE_URL == "https://your-service-url-here":
        print("❌ Please update SERVICE_URL with your actual Cloud Run service URL")
        print("   You can find it with: gcloud run services list")
        return
    
    print(f"Testing service at: {SERVICE_URL}")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health():
        tests_passed += 1
    
    if test_model_info():
        tests_passed += 1
        
    if test_prediction_with_sample():
        tests_passed += 1
    
    # Test with real image if available
    sample_image = "corgi.jpg"  # Replace with actual image path
    if Path(sample_image).exists():
        print(f"\n🐕 Found sample image, testing...")
        test_with_real_image(sample_image)
    else:
        print(f"\n💡 To test with a real dog image:")
        print(f"   1. Place a dog image file in this directory")
        print(f"   2. Update the 'sample_image' variable with the filename")
        print(f"   3. Run the script again")
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your ML API is working correctly!")
        print("\n📋 Next steps:")
        print("   1. Save your service URL for FlutterFlow integration")
        print("   2. Set up Firebase for your mobile app")
        print("   3. Configure FlutterFlow custom actions")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
