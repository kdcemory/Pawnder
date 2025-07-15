# setup_firebase_complete.py
# Complete Firebase setup for Pawnder mobile app

import json
import os
from pathlib import Path
from datetime import datetime

class PawnderFirebaseSetup:
    def __init__(self):
        self.pawnder_dir = Path(r"C:\Users\kelly\Documents\GitHub\Pawnder")
        self.app_dir = Path(r"C:\Users\kelly\Documents\GitHub\pawnder-app")
        self.firebase_dir = self.pawnder_dir / "firebase"
        
    def create_firebase_project_structure(self):
        """Create Firebase project configuration files"""
        print("🔧 Creating Firebase project structure...")
        
        # Create firebase directory
        self.firebase_dir.mkdir(parents=True, exist_ok=True)
        
        # Main Firebase configuration
        firebase_config = {
            "firestore": {
                "rules": "firestore.rules",
                "indexes": "firestore.indexes.json"
            },
            "storage": {
                "rules": "storage.rules"
            },
            "functions": {
                "source": "functions",
                "runtime": "python39"
            },
            "hosting": {
                "public": "public",
                "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
                "rewrites": [
                    {
                        "source": "**",
                        "destination": "/index.html"
                    }
                ]
            }
        }
        
        with open(self.firebase_dir / "firebase.json", 'w') as f:
            json.dump(firebase_config, f, indent=2)
        
        # Project reference file
        firebaserc = {
            "projects": {
                "default": "pawnder-457917"
            },
            "targets": {},
            "etags": {}
        }
        
        with open(self.firebase_dir / ".firebaserc", 'w') as f:
            json.dump(firebaserc, f, indent=2)
        
        print(f"# Firebase project files created in: {self.firebase_dir}")
        
    def create_firestore_rules(self):
        """Create Firestore security rules"""
        print("🔒 Creating Firestore security rules...")
        
        firestore_rules = '''rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own user document
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      allow create: if request.auth != null && request.auth.uid == userId;
      
      // User profile data validation
      allow write: if request.auth != null 
        && request.auth.uid == userId
        && validateUserData(request.resource.data);
    }
    
    // Predictions belong to specific users
    match /predictions/{predictionId} {
      allow read, write: if request.auth != null 
        && resource.data.userId == request.auth.uid;
      allow create: if request.auth != null 
        && request.resource.data.userId == request.auth.uid
        && validatePredictionData(request.resource.data);
      allow list: if request.auth != null 
        && resource.data.userId == request.auth.uid;
    }
    
    // User analytics - private to each user
    match /user_analytics/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      allow create: if request.auth != null && request.auth.uid == userId;
    }
    
    // Dog profiles for users
    match /dog_profiles/{dogId} {
      allow read, write: if request.auth != null 
        && resource.data.ownerId == request.auth.uid;
      allow create: if request.auth != null 
        && request.resource.data.ownerId == request.auth.uid;
    }
    
    // Public data (read-only for authenticated users)
    match /public/{document=**} {
      allow read: if request.auth != null;
      allow write: if false; // Only admins can write public data
    }
    
    // App metadata and configuration
    match /app_config/{document=**} {
      allow read: if request.auth != null;
      allow write: if false; // Only admins
    }
    
    // Helper functions for data validation
    function validateUserData(data) {
      return data.keys().hasAll(['email', 'displayName', 'createdAt'])
        && data.email is string
        && data.displayName is string;
    }
    
    function validatePredictionData(data) {
      return data.keys().hasAll(['userId', 'emotion', 'confidence', 'timestamp'])
        && data.userId is string
        && data.emotion is string
        && data.confidence is number
        && data.confidence >= 0 && data.confidence <= 1;
    }
  }
}'''
        
        with open(self.firebase_dir / "firestore.rules", 'w') as f:
            f.write(firestore_rules)
        
        print("# Firestore security rules created")
        
    def create_storage_rules(self):
        """Create Firebase Storage security rules"""
        print("# Creating Firebase Storage rules...")
        
        storage_rules = '''rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Users can only access their own uploaded images
    match /user_uploads/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      allow create: if request.auth != null 
        && request.auth.uid == userId
        && isValidImageUpload();
    }
    
    // Processed prediction results
    match /processed_results/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // User profile images
    match /profile_images/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      allow create: if request.auth != null 
        && request.auth.uid == userId
        && isValidImageUpload();
    }
    
    // Dog profile images
    match /dog_images/{userId}/{dogId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      allow create: if request.auth != null 
        && request.auth.uid == userId
        && isValidImageUpload();
    }
    
    // Public files (read-only for authenticated users)
    match /public/{allPaths=**} {
      allow read: if request.auth != null;
      allow write: if false; // Only admins
    }
    
    // Helper function for image validation
    function isValidImageUpload() {
      return request.resource.size < 10 * 1024 * 1024 // 10MB limit
        && request.resource.contentType.matches('image/.*');
    }
  }
}'''
        
        with open(self.firebase_dir / "storage.rules", 'w') as f:
            f.write(storage_rules)
        
        print("# Firebase Storage rules created")
        
    def create_firestore_indexes(self):
        """Create Firestore database indexes for efficient queries"""
        print("# Creating Firestore indexes...")
        
        indexes = {
            "indexes": [
                {
                    "collectionGroup": "predictions",
                    "queryScope": "COLLECTION",
                    "fields": [
                        {"fieldPath": "userId", "order": "ASCENDING"},
                        {"fieldPath": "timestamp", "order": "DESCENDING"}
                    ]
                },
                {
                    "collectionGroup": "predictions", 
                    "queryScope": "COLLECTION",
                    "fields": [
                        {"fieldPath": "userId", "order": "ASCENDING"},
                        {"fieldPath": "emotion", "order": "ASCENDING"},
                        {"fieldPath": "timestamp", "order": "DESCENDING"}
                    ]
                },
                {
                    "collectionGroup": "predictions",
                    "queryScope": "COLLECTION", 
                    "fields": [
                        {"fieldPath": "userId", "order": "ASCENDING"},
                        {"fieldPath": "confidence", "order": "DESCENDING"},
                        {"fieldPath": "timestamp", "order": "DESCENDING"}
                    ]
                },
                {
                    "collectionGroup": "dog_profiles",
                    "queryScope": "COLLECTION",
                    "fields": [
                        {"fieldPath": "ownerId", "order": "ASCENDING"},
                        {"fieldPath": "createdAt", "order": "DESCENDING"}
                    ]
                }
            ],
            "fieldOverrides": [
                {
                    "collectionGroup": "predictions",
                    "fieldPath": "allEmotions",
                    "indexes": [
                        {"queryScope": "COLLECTION", "fields": [{"fieldPath": "allEmotions", "order": "ASCENDING"}]}
                    ]
                }
            ]
        }
        
        with open(self.firebase_dir / "firestore.indexes.json", 'w') as f:
            json.dump(indexes, f, indent=2)
            
        print("# Firestore indexes created")
        
    def create_database_schema(self):
        """Create documentation for the Firestore database schema"""
        print("# Creating database schema documentation...")
        
        schema = {
            "database_schema": {
                "users": {
                    "document_id": "user_uid",
                    "fields": {
                        "email": "string",
                        "displayName": "string", 
                        "photoURL": "string (optional)",
                        "createdAt": "timestamp",
                        "lastLoginAt": "timestamp",
                        "preferences": {
                            "notifications": "boolean",
                            "dataSharing": "boolean",
                            "theme": "string (light/dark)"
                        },
                        "subscription": {
                            "plan": "string (free/premium)",
                            "expiresAt": "timestamp (optional)"
                        }
                    }
                },
                "predictions": {
                    "document_id": "auto_generated",
                    "fields": {
                        "userId": "string (reference to users)",
                        "emotion": "string",
                        "confidence": "number (0-1)",
                        "emotionScore": "number (0-1)",
                        "allEmotions": "map<string, number>",
                        "imageUrl": "string",
                        "imageMetadata": {
                            "size": "number",
                            "dimensions": "map",
                            "uploadedAt": "timestamp"
                        },
                        "timestamp": "timestamp",
                        "modelVersion": "string",
                        "processingTimeMs": "number",
                        "dogProfile": "string (optional reference)",
                        "notes": "string (optional user notes)"
                    }
                },
                "user_analytics": {
                    "document_id": "user_uid",
                    "fields": {
                        "totalPredictions": "number",
                        "emotionDistribution": "map<string, number>",
                        "lastActivity": "timestamp",
                        "createdAt": "timestamp",
                        "favoriteEmotions": "array<string>",
                        "streakDays": "number",
                        "monthlyStats": "map<string, map>"
                    }
                },
                "dog_profiles": {
                    "document_id": "auto_generated",
                    "fields": {
                        "ownerId": "string (reference to users)",
                        "name": "string",
                        "breed": "string",
                        "age": "number",
                        "profileImageUrl": "string",
                        "description": "string",
                        "createdAt": "timestamp",
                        "updatedAt": "timestamp",
                        "predictionCount": "number",
                        "dominantEmotion": "string"
                    }
                },
                "app_config": {
                    "document_id": "config_name",
                    "fields": {
                        "mlApiUrl": "string",
                        "supportedEmotions": "array<string>",
                        "appVersion": "string",
                        "maintenanceMode": "boolean",
                        "features": "map<string, boolean>"
                    }
                }
            }
        }
        
        with open(self.firebase_dir / "database_schema.json", 'w') as f:
            json.dump(schema, f, indent=2)
        
        print("# Database schema documentation created")
        
    def create_flutterflow_integration(self):
        """Create FlutterFlow integration files"""
        print("📱 Creating FlutterFlow integration files...")
        
        # Create app directory if it doesn't exist
        self.app_dir.mkdir(parents=True, exist_ok=True)
        flutterflow_dir = self.app_dir / "flutterflow_integration"
        flutterflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom actions for FlutterFlow
        custom_actions = '''// Custom Actions for Pawnder FlutterFlow App
// Copy these into FlutterFlow Custom Code > Actions

// PREDICT EMOTION ACTION
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> predictEmotion(
  String imageBase64,
  String apiUrl,
) async {
  try {
    final response = await http.post(
      Uri.parse('$apiUrl/predict'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'image': imageBase64,
      }),
    );

    if (response.statusCode == 200) {
      final Map<String, dynamic> result = jsonDecode(response.body);
      result['processed_at'] = DateTime.now().toIso8601String();
      return result;
    } else {
      throw Exception('Prediction failed: ${response.statusCode}');
    }
  } catch (e) {
    throw Exception('Network error: $e');
  }
}

// SAVE PREDICTION TO FIRESTORE
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

Future<String> savePrediction(
  Map<String, dynamic> predictionData,
  String imageUrl,
) async {
  final user = FirebaseAuth.instance.currentUser;
  if (user == null) throw Exception('User not authenticated');

  final prediction = {
    'userId': user.uid,
    'emotion': predictionData['emotion'],
    'confidence': predictionData['confidence'],
    'emotionScore': predictionData['emotion_score'],
    'allEmotions': predictionData['all_emotions'],
    'imageUrl': imageUrl,
    'timestamp': FieldValue.serverTimestamp(),
    'modelVersion': predictionData['model_version'] ?? '1.0',
    'processingTimeMs': predictionData['processing_time_ms'] ?? 0,
  };

  final docRef = await FirebaseFirestore.instance
      .collection('predictions')
      .add(prediction);

  // Update user analytics
  await updateUserAnalytics(user.uid, predictionData['emotion']);

  return docRef.id;
}

// UPDATE USER ANALYTICS
Future<void> updateUserAnalytics(String userId, String emotion) async {
  final analyticsRef = FirebaseFirestore.instance
      .collection('user_analytics')
      .doc(userId);

  await FirebaseFirestore.instance.runTransaction((transaction) async {
    final analyticsDoc = await transaction.get(analyticsRef);
    
    if (analyticsDoc.exists) {
      final currentData = analyticsDoc.data()!;
      final totalPredictions = (currentData['totalPredictions'] ?? 0) + 1;
      final emotionDist = Map<String, int>.from(
          currentData['emotionDistribution'] ?? {});
      
      emotionDist[emotion] = (emotionDist[emotion] ?? 0) + 1;
      
      transaction.update(analyticsRef, {
        'totalPredictions': totalPredictions,
        'emotionDistribution': emotionDist,
        'lastActivity': FieldValue.serverTimestamp(),
      });
    } else {
      transaction.set(analyticsRef, {
        'totalPredictions': 1,
        'emotionDistribution': {emotion: 1},
        'lastActivity': FieldValue.serverTimestamp(),
        'createdAt': FieldValue.serverTimestamp(),
        'streakDays': 1,
      });
    }
  });
}

// UPLOAD IMAGE TO FIREBASE STORAGE
import 'package:firebase_storage/firebase_storage.dart';
import 'dart:typed_data';

Future<String> uploadDogImage(Uint8List imageData) async {
  final user = FirebaseAuth.instance.currentUser;
  if (user == null) throw Exception('User not authenticated');

  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final fileName = 'dog_image_$timestamp.jpg';
  final path = 'user_uploads/${user.uid}/images/$fileName';

  final ref = FirebaseStorage.instance.ref().child(path);
  
  final uploadTask = ref.putData(
    imageData,
    SettableMetadata(
      contentType: 'image/jpeg',
      customMetadata: {
        'uploadedBy': user.uid,
        'uploadedAt': DateTime.now().toIso8601String(),
      },
    ),
  );

  final snapshot = await uploadTask;
  return await snapshot.ref.getDownloadURL();
}

// GET USER PREDICTIONS
Future<List<Map<String, dynamic>>> getUserPredictions(int limit) async {
  final user = FirebaseAuth.instance.currentUser;
  if (user == null) throw Exception('User not authenticated');

  final querySnapshot = await FirebaseFirestore.instance
      .collection('predictions')
      .where('userId', isEqualTo: user.uid)
      .orderBy('timestamp', descending: true)
      .limit(limit)
      .get();

  return querySnapshot.docs
      .map((doc) => {...doc.data(), 'id': doc.id})
      .toList();
}

// GET USER ANALYTICS
Future<Map<String, dynamic>> getUserAnalytics() async {
  final user = FirebaseAuth.instance.currentUser;
  if (user == null) throw Exception('User not authenticated');

  final doc = await FirebaseFirestore.instance
      .collection('user_analytics')
      .doc(user.uid)
      .get();

  if (doc.exists) {
    return doc.data()!;
  } else {
    return {
      'totalPredictions': 0,
      'emotionDistribution': {},
      'streakDays': 0,
    };
  }
}
'''
        
        with open(flutterflow_dir / "custom_actions.dart", 'w') as f:
            f.write(custom_actions)
        
        # Data types for FlutterFlow
        data_types = {
            "PredictionResult": {
                "emotion": "String",
                "confidence": "double",
                "emotionScore": "double", 
                "allEmotions": "JSON",
                "processedAt": "String",
                "modelVersion": "String"
            },
            "UserAnalytics": {
                "totalPredictions": "int",
                "emotionDistribution": "JSON",
                "lastActivity": "DateTime",
                "streakDays": "int"
            },
            "DogProfile": {
                "name": "String",
                "breed": "String",
                "age": "int",
                "profileImageUrl": "String",
                "description": "String"
            },
            "PredictionHistory": {
                "id": "String",
                "emotion": "String",
                "confidence": "double",
                "imageUrl": "String",
                "timestamp": "DateTime"
            }
        }
        
        with open(flutterflow_dir / "data_types.json", 'w') as f:
            json.dump(data_types, f, indent=2)
        
        # App configuration
        app_config = {
            "pawnder_app_config": {
                "ml_api_url": "https://pawnder-emotion-api-981944193835.us-east4.run.app",
                "firebase_project_id": "pawnder-457917",
                "supported_emotions": [
                    "Aggressive/Threatening",
                    "Curiosity/Alertness", 
                    "Fearful/Anxious",
                    "Happy/Playful",
                    "Relaxed",
                    "Stressed",
                    "Submissive/Appeasement"
                ],
                "features": {
                    "dog_profiles": True,
                    "prediction_history": True,
                    "analytics_dashboard": True,
                    "social_sharing": False,
                    "premium_features": False
                },
                "ui_config": {
                    "primary_color": "#FF6B35",
                    "secondary_color": "#004E89", 
                    "theme": "pet_friendly"
                }
            }
        }
        
        with open(flutterflow_dir / "app_config.json", 'w') as f:
            json.dump(app_config, f, indent=2)
        
        print(f"# FlutterFlow integration files created in: {flutterflow_dir}")
        
    def create_deployment_scripts(self):
        """Create Firebase deployment scripts"""
        print("🚀 Creating Firebase deployment scripts...")
        
        # PowerShell deployment script
        deploy_ps1 = '''# deploy_firebase.ps1
# Deploy Firebase configuration for Pawnder

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "pawnder-457917"
)

Write-Host "# Deploying Firebase configuration for Pawnder" -ForegroundColor Green
Write-Host "Project ID: $ProjectId" -ForegroundColor Yellow

# Set Firebase project
firebase use $ProjectId

if ($LASTEXITCODE -ne 0) {
    Write-Host "# Failed to set Firebase project. Make sure you're logged in:" -ForegroundColor Red
    Write-Host "firebase login" -ForegroundColor Yellow
    exit 1
}

Write-Host "# Deploying Firestore rules..." -ForegroundColor Cyan
firebase deploy --only firestore:rules

Write-Host "# Deploying Firestore indexes..." -ForegroundColor Cyan  
firebase deploy --only firestore:indexes

Write-Host "# Deploying Storage rules..." -ForegroundColor Cyan
firebase deploy --only storage

Write-Host "# Firebase deployment complete!" -ForegroundColor Green
Write-Host "# Firebase Console: https://console.firebase.google.com/project/$ProjectId" -ForegroundColor Green
'''
        
        with open(self.firebase_dir / "deploy_firebase.ps1", 'w') as f:
            f.write(deploy_ps1)
        
        # Python deployment script
        deploy_py = '''#!/usr/bin/env python3
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
'''
        
        with open(self.firebase_dir / "deploy_firebase.py", 'w') as f:
            f.write(deploy_py)
        
        print("# Deployment scripts created")
        
    def create_readme(self):
        """Create comprehensive README for Firebase setup"""
        print("📖 Creating setup documentation...")
        
        readme_content = '''# Pawnder Firebase Setup

## Overview
This directory contains all Firebase configuration for the Pawnder dog emotion recognition app.

## Files Structure
```
firebase/
├── firebase.json          # Main Firebase configuration
├── .firebaserc            # Project reference
├── firestore.rules        # Database security rules
├── firestore.indexes.json # Database indexes
├── storage.rules          # Storage security rules
├── database_schema.json   # Database schema documentation
├── deploy_firebase.ps1    # PowerShell deployment script
├── deploy_firebase.py     # Python deployment script
└── README.md              # This file
```

## Setup Instructions

### 1. Install Firebase CLI
```bash
npm install -g firebase-tools
```

### 2. Login to Firebase
```bash
firebase login
```

### 3. Initialize Firebase (if needed)
```bash
firebase init
```

### 4. Deploy Firebase Configuration
**Option A: Use PowerShell script**
```powershell
./deploy_firebase.ps1
```

**Option B: Use Python script**
```python
python deploy_firebase.py
```

**Option C: Manual deployment**
```bash
firebase deploy --only firestore:rules,firestore:indexes,storage
```

## Database Schema

### Collections

#### users/{userId}
- User profile and preferences
- Authentication data
- App settings

#### predictions/{predictionId} 
- Dog emotion predictions
- Image metadata
- User annotations

#### user_analytics/{userId}
- Prediction statistics
- Emotion distribution
- Usage analytics

#### dog_profiles/{dogId}
- Individual dog information
- Profile images
- Prediction history

## Security Rules

### Firestore
- Users can only access their own data
- Predictions are private to the user who created them
- Analytics are user-specific
- Public data is read-only

### Storage
- User uploads are private to each user
- 10MB file size limit for images
- Only image files allowed
- Automatic metadata tagging

## FlutterFlow Integration

Integration files are in: `../pawnder-app/flutterflow_integration/`

### Required Custom Actions
1. `predictEmotion` - Call ML API
2. `savePrediction` - Save to Firestore
3. `uploadDogImage` - Upload to Storage
4. `getUserPredictions` - Fetch history
5. `getUserAnalytics` - Get stats

### Data Types
- PredictionResult
- UserAnalytics  
- DogProfile
- PredictionHistory

## Next Steps

1. # Deploy Firebase configuration
2. 📱 Set up FlutterFlow project
3. # Connect FlutterFlow to Firebase
4. 🧪 Test authentication flow
5. 📸 Test image upload and prediction
6. # Test analytics dashboard

## Troubleshooting

### Common Issues

**Permission denied errors:**
- Make sure you're logged in: `firebase login`
- Check project permissions in Firebase Console

**Deploy failures:**
- Verify project ID in .firebaserc
- Check Firebase CLI version: `firebase --version`

**Rules validation errors:**
- Test rules in Firebase Console
- Check syntax in .rules files

## Links

- [Firebase Console](https://console.firebase.google.com/project/pawnder-457917)
- [ML API](https://pawnder-emotion-api-316099560158.us-east4.run.app)
- [FlutterFlow Integration Files](../pawnder-app/flutterflow_integration/)
'''
        
        with open(self.firebase_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("# Documentation created")
        
    def run_setup(self):
        """Run the complete Firebase setup"""
        print("# Starting Firebase setup for Pawnder...\n")
        
        # Create all components
        self.create_firebase_project_structure()
        self.create_firestore_rules() 
        self.create_storage_rules()
        self.create_firestore_indexes()
        self.create_database_schema()
        self.create_flutterflow_integration()
        self.create_deployment_scripts()
        self.create_readme()
        
        print(f"\n🎉 Firebase setup complete!")
        print(f"\n# Files created:")
        print(f"   Firebase config: {self.firebase_dir}")
        print(f"   FlutterFlow integration: {self.app_dir}/flutterflow_integration")
        
        print(f"\n🚀 Next steps:")
        print(f"1. Deploy Firebase configuration:")
        print(f"   cd {self.firebase_dir}")
        print(f"   firebase login")
        print(f"   ./deploy_firebase.ps1")
        print(f"2. Set up FlutterFlow project")
        print(f"3. Import custom actions from flutterflow_integration/")
        print(f"4. Connect FlutterFlow to Firebase")
        
        return self.firebase_dir, self.app_dir

def main():
    """Main setup function"""
    setup = PawnderFirebaseSetup()
    firebase_dir, app_dir = setup.run_setup()
    
    print(f"\n✨ Firebase backend ready for Pawnder mobile app!")

if __name__ == "__main__":
    main()
