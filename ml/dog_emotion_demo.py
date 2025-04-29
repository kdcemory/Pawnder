"""
Dog Emotion Analyzer Demo

This script demonstrates how to use the improved Dog Emotion Classifier
with behavioral feature integration.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import cv2

# Import the improved model code
from improved_dog_emotion_model import DogEmotionWithBehaviors, find_directory

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Dog Emotion Analyzer Demo')
    parser.add_argument('--image', type=str, help='Path to an image file to analyze')
    parser.add_argument('--model', type=str, help='Path to a saved model file')
    parser.add_argument('--metadata', type=str, help='Path to model metadata file (optional)')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--fine_tune', action='store_true', help='Perform fine-tuning')
    parser.add_argument('--test_dir', type=str, help='Directory with test images')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    
    return parser.parse_args()

def analyze_image(classifier, image_path, output_dir=None):
    """Analyze a single image and visualize results"""
    # Make sure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing image: {image_path}")
    
    # Get prediction
    result = classifier.predict_image(image_path)
    
    if result is None:
        print(f"Failed to analyze image: {image_path}")
        return
    
    # Print results
    print(f"Predicted emotion: {result['emotion']} (Score: {result['emotion_score']:.2f})")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\nTop 3 emotions:")
    for emotion, score in sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {emotion}: {score:.2f}")
    
    # Create output path if output directory is provided
    output_path = None
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_analyzed_{timestamp}.png")
    
    # Visualize results
    classifier.visualize_prediction(image_path, result, output_path)
    
    return result

def analyze_directory(classifier, test_dir, output_dir=None):
    """Analyze all images in a directory"""
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files in {test_dir}")
    
    # Analyze each image
    results = {}
    for image_path in image_files:
        result = analyze_image(classifier, image_path, output_dir)
        if result:
            results[image_path] = result
    
    return results

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create dog emotion classifier
    classifier = DogEmotionWithBehaviors()
    
    # Explicitly set paths to correct directories
    
    # Train a new model if requested
    if args.train:
        print("\nTraining a new model...")
        history, model_dir = classifier.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            fine_tune=args.fine_tune
        )
        print(f"\nModel training completed! Model saved to: {model_dir}")
        
        # Set model path to the newly trained model
        args.model = os.path.join(model_dir, 'final_model.h5')
        args.metadata = os.path.join(model_dir, 'model_metadata.json')
    
    # Load existing model if specified
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            return
        
        print(f"\nLoading model from {args.model}")
        classifier.load_model(args.model, args.metadata)
    
    # Make sure we have a model
    if classifier.model is None:
        print("Error: No model available. Please specify a model to load or train a new one.")
        return
    
    # Create output directory if not specified
    if not args.output_dir:
        args.output_dir = os.path.join('results', datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"Using default output directory: {args.output_dir}")
    
    # Analyze a single image if specified
    if args.image:
        analyze_image(classifier, args.image, args.output_dir)
    
    # Analyze a directory of images if specified
    elif args.test_dir:
        analyze_directory(classifier, args.test_dir, args.output_dir)
    
    # If neither image nor test_dir specified, prompt user
    else:
        # First check if we're in a Jupyter notebook
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                # In notebook - create interactive demo
                from IPython.display import display, HTML
                display(HTML("<h3>Dog Emotion Analyzer Interactive Demo</h3>"))
                
                # Use IPython widgets for interactive demo
                try:
                    import ipywidgets as widgets
                    from IPython.display import clear_output
                    
                    # Create file upload widget
                    upload = widgets.FileUpload(
                        accept='.jpg,.jpeg,.png',
                        multiple=False,
                        description='Upload Image'
                    )
                    
                    # Create analyze button
                    analyze_button = widgets.Button(
                        description='Analyze Image',
                        button_style='success',
                        disabled=True
                    )
                    
                    # Create output area
                    output = widgets.Output()
                    
                    # Update button state when file is uploaded
                    def on_upload_change(change):
                        analyze_button.disabled = len(upload.value) == 0
                    
                    upload.observe(on_upload_change, names='value')
                    
                    # Handle button click
                    def on_button_click(b):
                        # Clear previous output
                        output.clear_output()
                        
                        with output:
                            if len(upload.value) == 0:
                                print("Please upload an image first")
                                return
                            
                            # Get the uploaded file
                            filename = list(upload.value.keys())[0]
                            content = upload.value[filename]['content']
                            
                            # Convert to numpy array
                            import io
                            image = plt.imread(io.BytesIO(content))
                            
                            # Save image to temporary file
                            temp_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                            plt.imsave(temp_path, image)
                            
                            # Analyze image
                            result = analyze_image(classifier, temp_path)
                            
                            # Clean up
                            os.remove(temp_path)
                    
                    analyze_button.on_click(on_button_click)
                    
                    # Display widgets
                    display(widgets.VBox([
                        widgets.HBox([upload, analyze_button]),
                        output
                    ]))
                    
                except ImportError:
                    print("ipywidgets not available. Please install with: pip install ipywidgets")
                    print("For now, please specify an image file using the --image option")
            else:
                # Not in notebook - show instructions
                print("\nPlease specify an image file to analyze using the --image option")
                print("Example: python dog_emotion_demo.py --image /path/to/dog_image.jpg --model /path/to/model.h5")
        except ImportError:
            # Not in IPython environment
            print("\nPlease specify an image file to analyze using the --image option")
            print("Example: python dog_emotion_demo.py --image /path/to/dog_image.jpg --model /path/to/model.h5")

if __name__ == "__main__":
    main()