"""
Pawnder Colab Startup Script
----------------------------
Run this at the beginning of each Colab notebook
"""

from google.colab import drive
import os
import sys
import subprocess

# Mount Google Drive
drive.mount('/content/drive')

# Define main project paths
PROJECT_PATH = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
GITHUB_PATH = "/content/pawnder"  # Where GitHub repo gets cloned
DATA_PATH = os.path.join(PROJECT_PATH, "Data")
MODELS_PATH = os.path.join(PROJECT_PATH, "Models")

# Add paths to Python's path
sys.path.append(PROJECT_PATH)
sys.path.append(GITHUB_PATH)

# Clone/update GitHub repository
subprocess.run(["git", "clone", "https://github.com/kdcemory/pawnder.git", "/content/pawnder"], 
              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
subprocess.run(["git", "-C", "/content/pawnder", "pull"], 
              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

# Ensure key directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(os.path.join(DATA_PATH, "Raw"), exist_ok=True)
os.makedirs(os.path.join(DATA_PATH, "processed"), exist_ok=True)

# Print configuration information
print("\n===== Pawnder Environment Ready =====")
print(f"Project Root: {PROJECT_PATH}")
print(f"Data Directory: {DATA_PATH}")
print(f"Models Directory: {MODELS_PATH}")
print(f"GitHub Code: {GITHUB_PATH}")

# Import helper function
def show_project_structure():
    """Display current project structure"""
    print("\n===== Project Structure =====")
    print("Main directories:")
    for item in os.listdir(PROJECT_PATH):
        if os.path.isdir(os.path.join(PROJECT_PATH, item)):
            print(f"  - {item}/")
    
    print("\nData directory contents:")
    try:
        for item in os.listdir(DATA_PATH):
            item_path = os.path.join(DATA_PATH, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/ ({len(os.listdir(item_path))} items)")
            else:
                print(f"  - {item}")
    except:
        print("  Unable to read Data directory")
        
    print("\nAvailable Python modules:")
    for item in os.listdir(PROJECT_PATH):
        if item.endswith('.py'):
            print(f"  - {item}")

# Show project structure
show_project_structure()

print("\n===== Ready to Start =====")
print("You can now import your project modules and start working")
print("Use 'show_project_structure()' to see your folders again")