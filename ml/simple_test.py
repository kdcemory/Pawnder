from dog_emotion_with_behaviors_fixed import DogEmotionWithBehaviors

try:
    # Create classifier
    print("Creating classifier...")
    classifier = DogEmotionWithBehaviors()
    
    # Call load_annotations directly
    print("\nCalling load_annotations('train')...")
    result = classifier.load_annotations('train')
    
    # Check result
    if result:
        print(f"Success! Loaded {len(result)} annotations")
    else:
        print("Failed to load annotations")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()