# export_model_interactive.py
# Interactive version that lets you specify the model path

import os
import sys
from pathlib import Path

# Import the ModelExporter from the safe version
# (You'll need to copy the ModelExporter class from export_model_safe.py)

def find_available_models():
    """Find all available models in the Models directory"""
    models_dir = Path("C:/Users/kelly/Documents/GitHub/Pawnder/Models")
    model_files = []
    
    if models_dir.exists():
        # Find .keras files
        for model_file in models_dir.rglob("*.keras"):
            model_files.append(model_file)
        
        # Find .h5 files
        for model_file in models_dir.rglob("*.h5"):
            model_files.append(model_file)
    
    return sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)

def main():
    """Interactive main function"""
    print("🔍 Pawnder Model Export Tool\n")
    
    # Default paths
    PROJECT_PATH = "C:/Users/kelly/Documents/GitHub/Pawnder"
    DEFAULT_MODEL = "C:/Users/kelly/Documents/GitHub/Pawnder/Models/enhanced_dog_emotion_20250525-134150/best_model.keras"
    
    # Check if default model exists
    if os.path.exists(DEFAULT_MODEL):
        print(f"✅ Found default model: {DEFAULT_MODEL}")
        use_default = input("Use this model? (y/n): ").lower().strip()
        
        if use_default == 'y' or use_default == '':
            model_path = DEFAULT_MODEL
        else:
            # Show available models
            print("\n📁 Available models:")
            available_models = find_available_models()
            
            if available_models:
                for i, model in enumerate(available_models[:10]):  # Show top 10
                    print(f"  {i+1}. {model}")
                
                try:
                    choice = int(input(f"\nSelect model (1-{len(available_models[:10])}): ")) - 1
                    if 0 <= choice < len(available_models[:10]):
                        model_path = str(available_models[choice])
                    else:
                        print("Invalid choice, using default model")
                        model_path = DEFAULT_MODEL
                except ValueError:
                    print("Invalid input, using default model")
                    model_path = DEFAULT_MODEL
            else:
                print("No models found, please check your Models directory")
                return
    else:
        print(f"❌ Default model not found: {DEFAULT_MODEL}")
        
        # Show available models
        available_models = find_available_models()
        if available_models:
            print("\n📁 Available models:")
            for i, model in enumerate(available_models[:10]):
                print(f"  {i+1}. {model}")
            
            try:
                choice = int(input(f"Select model (1-{len(available_models[:10])}): ")) - 1
                if 0 <= choice < len(available_models[:10]):
                    model_path = str(available_models[choice])
                else:
                    print("Invalid choice")
                    return
            except ValueError:
                print("Invalid input")
                return
        else:
            print("No models found in Models directory")
            return
    
    print(f"\n🎯 Using model: {model_path}")
    
    # Import and use the ModelExporter
    # For this to work, you need to have the ModelExporter class available
    # You can either copy it here or import from export_model_safe.py
    
    try:
        # You would need to import or copy the ModelExporter class here
        # For now, just show what would happen
        print("🚀 Starting export process...")
        print(f"   Project path: {PROJECT_PATH}")
        print(f"   Model path: {model_path}")
        print("\n⚠️ To complete the export, run export_model_safe.py")
        print("   Or copy the ModelExporter class into this file")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
