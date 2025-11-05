"""
Cat Breed Classification - Inference Script

This script provides easy-to-use inference for cat breed classification.
Supports single image and batch prediction.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import argparse

# Import config
import config


class CatBreedClassifier:
    """Cat Breed Classification Model Wrapper"""

    def __init__(self, model_path=None, class_indices_path=None):
        """
        Initialize classifier

        Args:
            model_path: Path to trained Keras model (.keras file)
            class_indices_path: Path to class indices JSON file
        """
        # Default paths
        if model_path is None:
            model_path = config.MODELS_DIR / 'cat_breed_classifier_final.keras'
        if class_indices_path is None:
            class_indices_path = config.MODELS_DIR / 'class_indices.json'

        # Load model
        print(f"Loading model from: {model_path}")
        self.model = load_model(str(model_path))
        print("✓ Model loaded successfully")

        # Load class indices
        print(f"Loading class indices from: {class_indices_path}")
        with open(str(class_indices_path), 'r') as f:
            self.class_indices = json.load(f)

        # Create reverse mapping (index -> class name)
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        self.num_classes = len(self.class_indices)
        print(f"✓ Loaded {self.num_classes} cat breed classes")

        self.img_size = (config.IMG_WIDTH, config.IMG_HEIGHT)

    def preprocess_image(self, img_path):
        """
        Preprocess image for model input

        Args:
            img_path: Path to image file

        Returns:
            Preprocessed image array
        """
        # Load image
        img = image.load_img(img_path, target_size=self.img_size)

        # Convert to array
        img_array = image.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess (ResNet50V2 preprocessing)
        img_array = preprocess_input(img_array)

        return img_array

    def predict_single(self, img_path, top_k=5):
        """
        Predict cat breed for a single image

        Args:
            img_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        img_array = self.preprocess_image(img_path)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = []

        for idx in top_indices:
            breed_name = self.idx_to_class[idx]
            confidence = float(predictions[idx])
            top_predictions.append({
                'breed': breed_name,
                'confidence': confidence,
                'confidence_percent': confidence * 100
            })

        return {
            'image_path': str(img_path),
            'top_prediction': top_predictions[0],
            'top_k_predictions': top_predictions,
            'all_probabilities': {self.idx_to_class[i]: float(predictions[i])
                                  for i in range(len(predictions))}
        }

    def predict_batch(self, img_paths, top_k=5):
        """
        Predict cat breeds for multiple images

        Args:
            img_paths: List of image paths
            top_k: Number of top predictions to return

        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in img_paths:
            try:
                result = self.predict_single(img_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })

        return results

    def print_prediction(self, result):
        """Pretty print prediction result"""
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return

        print("\n" + "="*80)
        print(f"🐱 Image: {Path(result['image_path']).name}")
        print("="*80)

        print(f"\n🏆 Top Prediction:")
        top = result['top_prediction']
        print(f"   Breed: {top['breed'].replace('_', ' ').title()}")
        print(f"   Confidence: {top['confidence_percent']:.2f}%")

        print(f"\n📊 Top {len(result['top_k_predictions'])} Predictions:")
        for i, pred in enumerate(result['top_k_predictions'], 1):
            breed_name = pred['breed'].replace('_', ' ').title()
            bar_length = int(pred['confidence_percent'] / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {i}. {breed_name:30s} {bar} {pred['confidence_percent']:6.2f}%")

        print("="*80)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Cat Breed Classification Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python inference.py --image path/to/cat.jpg

  # Top 10 predictions
  python inference.py --image path/to/cat.jpg --top-k 10

  # Batch prediction
  python inference.py --batch path/to/images/*.jpg

  # Save results to JSON
  python inference.py --image path/to/cat.jpg --output results.json

  # Use custom model
  python inference.py --image cat.jpg --model models/my_model.keras
        """
    )

    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple images')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--class-indices', type=str, help='Path to class indices JSON')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions (default: 5)')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress output (only with --output)')

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.batch:
        parser.error("Must provide either --image or --batch")

    # Initialize classifier
    try:
        classifier = CatBreedClassifier(
            model_path=args.model,
            class_indices_path=args.class_indices
        )
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        sys.exit(1)

    # Perform inference
    if args.image:
        # Single image
        if not Path(args.image).exists():
            print(f"❌ Error: Image not found: {args.image}")
            sys.exit(1)

        result = classifier.predict_single(args.image, top_k=args.top_k)

        if not args.quiet:
            classifier.print_prediction(result)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\n✓ Results saved to {args.output}")

    elif args.batch:
        # Batch prediction
        img_paths = []
        for pattern in args.batch:
            from glob import glob
            img_paths.extend(glob(pattern))

        if not img_paths:
            print(f"❌ Error: No images found")
            sys.exit(1)

        print(f"\n📂 Processing {len(img_paths)} images...")
        results = classifier.predict_batch(img_paths, top_k=args.top_k)

        if not args.quiet:
            for result in results:
                classifier.print_prediction(result)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n✓ Results saved to {args.output}")

        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        print(f"\n📊 Summary: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
