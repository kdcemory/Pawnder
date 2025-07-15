#!/usr/bin/env python3
# deploy_firebase.py
# Deploy Firebase configuration using Python

import subprocess
import sys

def run_command(command):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def deploy_firebase():
    """Deploy Firebase configuration"""
    project_id = "pawnder-457917"
    
    print("# Deploying Firebase configuration for Pawnder")
    print(f"Project ID: {project_id}")
    
    # Set project
    if not run_command(f"firebase use {project_id}"):
        print("# Failed to set Firebase project. Make sure you're logged in:")
        print("firebase login")
        return False
    
    # Deploy components
    components = [
        ("firestore:rules", "# Deploying Firestore rules..."),
        ("firestore:indexes", "# Deploying Firestore indexes..."),
        ("storage", "# Deploying Storage rules...")
    ]
    
    for component, message in components:
        print(message)
        if not run_command(f"firebase deploy --only {component}"):
            print(f"# Failed to deploy {component}")
            return False
    
    print("# Firebase deployment complete!")
    print(f"# Firebase Console: https://console.firebase.google.com/project/{project_id}")
    return True

if __name__ == "__main__":
    deploy_firebase()
