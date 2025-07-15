# tf_version_detective.py - Find out what TF version created your models
import os
import json
import tensorflow as tf
from pathlib import Path

def check_savedmodel_version():
    """Check what TensorFlow version created the SavedModel"""
    print("🔍 Investigating SavedModel version...")
    
    saved_model_path = "saved_model"
    if not os.path.exists(saved_model_path):
        print("  ❌ No saved_model directory found")
        return None
    
    # Check saved_model.pb for version info
    pb_file = os.path.join(saved_model_path, "saved_model.pb")
    if os.path.exists(pb_file):
        print(f"  📄 Found saved_model.pb ({os.path.getsize(pb_file)} bytes)")
        
        # Try to read some metadata
        try:
            # Read the raw protobuf to look for version strings
            with open(pb_file, 'rb') as f:
                content = f.read()
                
            # Look for TensorFlow version strings in the binary
            content_str = str(content)
            
            # Common TF version patterns
            version_patterns = [
                '2.12.', '2.13.', '2.14.', '2.15.', '2.16.', '2.17.', '2.18.', '2.19.'
            ]
            
            found_versions = []
            for pattern in version_patterns:
                if pattern in content_str:
                    # Try to extract full version
                    start = content_str.find(pattern)
                    if start != -1:
                        # Extract a reasonable substring around the version
                        version_area = content_str[start:start+20]
                        found_versions.append(f"Possible: {pattern}x")
            
            if found_versions:
                print(f"  🎯 Potential TF versions found in model:")
                for version in set(found_versions):
                    print(f"    - {version}")
            else:
                print("  ❓ No clear version strings found in protobuf")
                
        except Exception as e:
            print(f"  ⚠️ Could not read protobuf: {e}")
    
    # Check variables folder for additional clues
    variables_path = os.path.join(saved_model_path, "variables")
    if os.path.exists(variables_path):
        print(f"  📁 Variables folder exists")
        var_files = os.listdir(variables_path)
        print(f"    Files: {var_files}")

def check_project_for_tf_version():
    """Look in the project directory for clues about TF version"""
    print("\n🔍 Checking project directory for TensorFlow version clues...")
    
    # Check for requirements files
    req_files = [
        "requirements.txt",
        "../requirements.txt", 
        "../../requirements.txt",
        "environment.yml",
        "../environment.yml"
    ]
    
    for req_file in req_files:
        if os.path.exists(req_file):
            print(f"  📄 Found: {req_file}")
            try:
                with open(req_file, 'r') as f:
                    content = f.read()
                    
                if 'tensorflow' in content.lower():
                    lines = content.split('\n')
                    tf_lines = [line for line in lines if 'tensorflow' in line.lower()]
                    for line in tf_lines:
                        print(f"    🎯 {line.strip()}")
                else:
                    print(f"    ❌ No tensorflow found in {req_file}")
            except Exception as e:
                print(f"    ⚠️ Could not read {req_file}: {e}")
    
    # Check for conda environment files
    conda_files = ["environment.yaml", "../environment.yaml"]
    for conda_file in conda_files:
        if os.path.exists(conda_file):
            print(f"  📄 Found conda file: {conda_file}")
    
    # Check for pip freeze output or logs
    log_patterns = ["*.log", "../*.log", "training_*.txt", "../training_*.txt"]
    for pattern in log_patterns:
        matches = list(Path('.').glob(pattern))
        for match in matches:
            print(f"  📄 Found log file: {match}")

def test_available_tf_versions():
    """Test which TensorFlow versions are actually available"""
    print("\n🧪 Testing available TensorFlow versions...")
    
    # The versions you saw as available
    available_versions = [
        "2.16.1", "2.16.2", "2.17.0", "2.17.1", 
        "2.18.0", "2.18.1", "2.19.0"
    ]
    
    print(f"  📋 Available versions: {', '.join(available_versions)}")
    print(f"  🎯 Currently installed: {tf.__version__}")
    
    # Recommend versions to try
    print("\n💡 Recommended versions to try (in order):")
    print("  1. tensorflow==2.16.1 (stable, likely compatible)")
    print("  2. tensorflow==2.17.1 (newer stable)")
    print("  3. tensorflow==2.18.1 (recent stable)")
    
    return available_versions

def create_version_test_script(versions):
    """Create a script to test loading with different TF versions"""
    script_content = f'''# test_tf_versions.py - Test model loading with available TF versions
# Run this script after installing different TF versions

import tensorflow as tf
import os

print(f"Testing with TensorFlow {{tf.__version__}}")

def test_savedmodel():
    """Test SavedModel loading"""
    if os.path.exists("saved_model"):
        try:
            print("🎯 Testing SavedModel...")
            model = tf.saved_model.load("saved_model")
            print("✅ SavedModel loaded successfully!")
            
            if hasattr(model, 'signatures'):
                signatures = list(model.signatures.keys())
                print(f"  Signatures: {{signatures}}")
                
                if 'serving_default' in signatures:
                    infer = model.signatures['serving_default']
                    input_spec = infer.structured_input_signature[1]
                    print(f"  Inputs: {{list(input_spec.keys())}}")
            return True
        except Exception as e:
            print(f"❌ SavedModel failed: {{e}}")
            return False
    else:
        print("❌ No saved_model directory found")
        return False

def test_keras():
    """Test Keras model loading"""
    keras_files = ["best_model.keras", "enhanced_dog_emotion_20250525-134150/best_model.keras"]
    
    for keras_file in keras_files:
        if os.path.exists(keras_file):
            try:
                print(f"🎯 Testing Keras: {{keras_file}}")
                model = tf.keras.models.load_model(keras_file, compile=False)
                print(f"✅ Keras model loaded successfully!")
                print(f"  Input shapes: {{[inp.shape for inp in model.inputs]}}")
                return True
            except Exception as e:
                print(f"❌ Keras failed: {{e}}")
    
    print("❌ No working Keras models found")
    return False

if __name__ == "__main__":
    print("="*50)
    
    savedmodel_works = test_savedmodel()
    keras_works = test_keras()
    
    if savedmodel_works or keras_works:
        print("\\n🎉 SUCCESS! At least one model type works with this TF version")
        print("You can now run your API with this TensorFlow version")
    else:
        print("\\n❌ No models work with this TF version")
        print("Try a different TensorFlow version")
    
    print("="*50)
'''
    
    with open("test_tf_versions.py", "w") as f:
        f.write(script_content)
    
    print(f"\n✅ Created test_tf_versions.py")
    print("Use this to test each TensorFlow version:")
    
    for version in versions[:3]:  # Show first 3 versions
        print(f"  pip install tensorflow=={version}")
        print(f"  python test_tf_versions.py")
        print()

def main():
    print("🕵️ TensorFlow Version Detective")
    print("=" * 50)
    print(f"Current TensorFlow: {tf.__version__}")
    
    # Check SavedModel for version clues
    check_savedmodel_version()
    
    # Check project directory for requirements
    check_project_for_tf_version()
    
    # Show available versions and recommendations
    available_versions = test_available_tf_versions()
    
    # Create test script
    create_version_test_script(available_versions)
    
    print("\n🎯 Next Steps:")
    print("1. Try the most likely compatible version first:")
    print("   pip install tensorflow==2.16.1")
    print("2. Run the test script:")
    print("   python test_tf_versions.py")
    print("3. If that works, use your API!")
    print("4. If not, try tensorflow==2.17.1, then 2.18.1")

if __name__ == "__main__":
    main()
