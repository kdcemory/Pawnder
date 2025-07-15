# fix_firebase_project.py
# Fix Firebase project configuration

import json
import subprocess
from pathlib import Path

def get_firebase_projects():
    """Get list of available Firebase projects"""
    try:
        result = subprocess.run(['firebase', 'projects:list'], 
                              capture_output=True, text=True, check=True)
        print("Available Firebase projects:")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting projects: {e}")
        return None

def update_firebase_config(project_id):
    """Update Firebase configuration with correct project ID"""
    firebase_dir = Path(r"C:\Users\kelly\Documents\GitHub\Pawnder\firebase")
    
    # Update .firebaserc
    firebaserc_path = firebase_dir / ".firebaserc"
    if firebaserc_path.exists():
        firebaserc = {
            "projects": {
                "default": project_id
            }
        }
        
        with open(firebaserc_path, 'w', encoding='utf-8') as f:
            json.dump(firebaserc, f, indent=2)
        
        print(f"Updated .firebaserc with project ID: {project_id}")
    
    # Update app config
    app_dir = Path(r"C:\Users\kelly\Documents\GitHub\pawnder-app\flutterflow_integration")
    config_path = app_dir / "app_config.json"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        config["pawnder_app_config"]["firebase_project_id"] = project_id
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated app config with project ID: {project_id}")

def main():
    print("Firebase Project Configuration Fixer")
    print("=" * 40)
    
    # Get available projects
    projects = get_firebase_projects()
    
    if projects:
        print("\nEnter the correct Project ID from the list above:")
        project_id = input("Project ID: ").strip()
        
        if project_id:
            # Set the project
            try:
                subprocess.run(['firebase', 'use', project_id], check=True)
                print(f"Set Firebase project to: {project_id}")
                
                # Update configuration files
                update_firebase_config(project_id)
                
                print(f"\nNext steps:")
                print(f"1. Go to Firebase Console: https://console.firebase.google.com/project/{project_id}")
                print(f"2. Create Firestore Database (choose us-east4 region)")
                print(f"3. Create Storage bucket")
                print(f"4. Run: firebase deploy --only firestore:rules,storage")
                
            except subprocess.CalledProcessError as e:
                print(f"Error setting project: {e}")
        else:
            print("No project ID entered")
    else:
        print("\nNo Firebase projects found or error occurred.")
        print("You may need to:")
        print("1. Create a new Firebase project at https://console.firebase.google.com")
        print("2. Or add Firebase to your existing Google Cloud project")

if __name__ == "__main__":
    main()
