# model_diagnostic.py - Check what models are available
import os
import tensorflow as tf
from pathlib import Path

def check_directory_structure():
    """Check the current directory and find all model files"""
    current_dir = Path.cwd()
    print(f"🔍 Current directory: {current_dir}")
    print(f"📁 Directory contents:")
    
    # List all files and directories
    for item in current_dir.iterdir():
        if item.is_file():
            print(f"  📄 {item.name} ({item.suffix})")
        elif item.is_dir():
            print(f"  📁 {item.name}/")
            # Check if it's a savedmodel directory
            if (item / "saved_model.pb").exists():
                print(f"    ✅ Contains saved_model.pb (SavedModel format)")
            # List some contents of subdirectories
            try:
                contents = list(item.iterdir())[:5]  # First 5 items
                for subitem in contents:
                    print(f"    - {subitem.name}")
                if len(list(item.iterdir())) > 5:
                    print(f"    ... and {len(list(item.iterdir())) - 5} more items")
            except PermissionError:
                print(f"    ❌ Permission denied")

def find_model_files():
    """Find all potential model files"""
    print(f"\n🔍 Searching for model files...")
    
    # Search patterns
    patterns = [
        "**/*.keras",
        "**/*.h5", 
        "**/saved_model.pb",
        "**/best_model.*"
    ]
    
    current_dir = Path.cwd()
    found_models = []
    
    for pattern in patterns:
        matches = list(current_dir.glob(pattern))
        for match in matches:
            found_models.append(match)
            print(f"  ✅ Found: {match}")
    
    if not found_models:
        print("  ❌ No model files found in current directory tree")
    
    return found_models

def test_model_loading(model_path):
    """Test loading a specific model"""
    print(f"\n🧪 Testing model: {model_path}")
    
    try:
        if model_path.name == "saved_model.pb":
            # It's a SavedModel
            model_dir = model_path.parent
            print(f"  📁 Loading SavedModel from: {model_dir}")
            model = tf.saved_model.load(str(model_dir))
            print(f"  ✅ SavedModel loaded successfully!")
            
            # Check signatures
            if hasattr(model, 'signatures'):
                signatures = list(model.signatures.keys())
                print(f"  📋 Available signatures: {signatures}")
                
                if 'serving_default' in signatures:
                    infer = model.signatures['serving_default']
                    input_spec = infer.structured_input_signature[1]
                    print(f"  📥 Input specification:")
                    for name, spec in input_spec.items():
                        print(f"    - {name}: {spec.shape} ({spec.dtype})")
            
        elif model_path.suffix in ['.keras', '.h5']:
            # It's a Keras model
            print(f"  📄 Loading Keras model: {model_path}")
            model = tf.keras.models.load_model(model_path)
            print(f"  ✅ Keras model loaded successfully!")
            print(f"  📥 Input shapes: {[inp.shape for inp in model.inputs]}")
            print(f"  📤 Output shapes: {[out.shape for out in model.outputs]}")
            
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return False
    
    return True

def check_ports():
    """Check what's running on common ports"""
    import socket
    
    print(f"\n🔌 Checking ports...")
    ports_to_check = [8000, 8001, 8080, 5000]
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"  ❌ Port {port} is in use")
        else:
            print(f"  ✅ Port {port} is available")

def main():
    print("🚀 Pawnder Model Diagnostic Tool")
    print("=" * 50)
    
    # Check directory structure
    check_directory_structure()
    
    # Find model files
    model_files = find_model_files()
    
    # Test loading each model
    if model_files:
        print(f"\n🧪 Testing model loading...")
        for model_file in model_files:
            test_model_loading(model_file)
    
    # Check ports
    check_ports()
    
    print(f"\n📋 Summary:")
    print(f"  - Current directory: {Path.cwd()}")
    print(f"  - Models found: {len(model_files)}")
    print(f"  - TensorFlow version: {tf.__version__}")

if __name__ == "__main__":
    main()