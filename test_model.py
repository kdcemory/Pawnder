import os
import sys

def test_model_imports():
    """Test that the model can be imported"""
    print("Attempting to import DogEmotionWithBehaviors...")
    
    try:
        # Import the model class
        from dog_emotion_with_behaviors_fixed import DogEmotionWithBehaviors
        print("✓ Successfully imported DogEmotionWithBehaviors")
        
        # Try to create an instance
        classifier = DogEmotionWithBehaviors()
        print("✓ Successfully created classifier instance")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Error creating classifier: {str(e)}")
        return False

def test_data_loading():
    """Test that data can be loaded"""
    print("\nTesting data loading...")
    
    try:
        from dog_emotion_with_behaviors_fixed import DogEmotionWithBehaviors
        classifier = DogEmotionWithBehaviors()
        
        # Try to load annotations
        print("Attempting to load train annotations...")
        train_annotations = classifier.load_annotations('train')
        
        if train_annotations:
            print(f"✓ Successfully loaded {len(train_annotations)} train annotations")
            return True
        else:
            print("✗ Failed to load train annotations")
            return False
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Dog Emotion Classifier")
    print("=" * 80)
    
    # Test imports
    import_ok = test_model_imports()
    
    # Test data loading if imports succeeded
    if import_ok:
        data_ok = test_data_loading()
    else:
        data_ok = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Results:")
    print(f"  Model imports: {'✓ Success' if import_ok else '✗ Failed'}")
    print(f"  Data loading:  {'✓ Success' if data_ok else '✗ Failed'}")
    print("=" * 80)