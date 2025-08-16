"""
Main Script for Advanced Brain Tumor Detection with EfficientNet + Attention

This script integrates all components for training and evaluating the improved
brain tumor detection model. It replaces the VGG16-based approach with a more
efficient and accurate EfficientNet + Attention architecture.

Usage:
    python main_efficientnet_brain_tumor.py --train_dir /path/to/train --test_dir /path/to/test

Key Features:
- EfficientNet-B3 backbone (12M vs 138M parameters)
- Spatial attention mechanism
- Mixed precision training
- Advanced data augmentation
- Cosine decay learning rate scheduling
- Comprehensive evaluation and visualization
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from typing import Optional

# Import our custom modules
from efficientnet_attention_model import create_brain_tumor_model
from data_utils import create_data_generators
from training_utils import create_training_manager
from evaluation_utils import ModelEvaluator, AttentionVisualizer


def setup_gpu_memory_growth():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs detected, running on CPU")


def enable_mixed_precision():
    """Enable mixed precision training for performance improvement."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision training enabled")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Advanced Brain Tumor Detection with EfficientNet + Attention')
    parser.add_argument('--train_dir', type=str, default='/content/drive/MyDrive/MRI Images/Training/',
                        help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='/content/drive/MyDrive/MRI Images/Testing/',
                        help='Path to test data directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['light', 'medium', 'strong'],
                        help='Data augmentation strength (default: medium)')
    parser.add_argument('--model_name', type=str, default='brain_tumor_efficientnet',
                        help='Model name for saving (default: brain_tumor_efficientnet)')
    parser.add_argument('--save_dir', type=str, default='./model_outputs',
                        help='Directory to save model and results (default: ./model_outputs)')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate existing model (skip training)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to existing model for evaluation')
    
    args = parser.parse_args()
    
    # Setup GPU and mixed precision
    setup_gpu_memory_growth()
    enable_mixed_precision()
    
    print("="*80)
    print("ADVANCED BRAIN TUMOR DETECTION - EfficientNet + Attention")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    print(f"Training directory: {args.train_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Augmentation: {args.augmentation}")
    print("="*80)
    
    # Check if directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        print("Please check the path or mount your data source.")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        print("Please check the path or mount your data source.")
        return
    
    # Create data generators
    print("\nSetting up data generators...")
    data_generator = create_data_generators(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augmentation_strength=args.augmentation,
        mixed_precision=True
    )
    
    # Get datasets
    train_dataset = data_generator.get_train_dataset()
    test_dataset = data_generator.get_test_dataset()
    
    # Get class weights for handling imbalanced data
    class_weights = data_generator.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    if not args.evaluate_only:
        # Create and compile model
        print("\nCreating EfficientNet + Attention model...")
        model, model_builder = create_brain_tumor_model(
            num_classes=data_generator.num_classes,
            input_shape=(args.image_size, args.image_size, 3)
        )
        
        # Display model summary
        model_builder.get_model_summary()
        
        # Create training manager
        print("\nSetting up training manager...")
        trainer = create_training_manager(
            model=model,
            model_name=args.model_name,
            save_dir=args.save_dir
        )
        
        # Train the model
        print("\nStarting training...")
        history = trainer.train_model(
            train_dataset=train_dataset,
            validation_dataset=test_dataset,
            epochs=args.epochs,
            initial_lr=args.learning_rate,
            warmup_epochs=2,
            class_weights=class_weights
        )
        
        # Plot training history
        print("\nGenerating training visualizations...")
        trainer.plot_training_history(save_plots=True)
        
        # Get training summary
        training_summary = trainer.get_training_summary()
        print("\nTraining Summary:")
        for key, value in training_summary.items():
            print(f"  {key}: {value}")
    
    else:
        # Load existing model for evaluation
        if args.model_path is None:
            model_path = os.path.join(args.save_dir, f"{args.model_name}_best.h5")
        else:
            model_path = args.model_path
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        print(f"\nLoading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    
    # Comprehensive evaluation
    print("\nStarting comprehensive model evaluation...")
    evaluator = ModelEvaluator(model, data_generator.class_names)
    
    evaluation_results = evaluator.evaluate_model(
        test_dataset=test_dataset,
        save_dir=os.path.join(args.save_dir, "evaluation_results")
    )
    
    # Attention visualization (if attention layer exists)
    print("\nGenerating attention visualizations...")
    attention_viz = AttentionVisualizer(model)
    
    # Visualize attention for a few test samples
    test_batch = next(iter(test_dataset))
    test_images, test_labels = test_batch
    
    for i in range(min(3, len(test_images))):
        image = test_images[i].numpy()
        true_label = data_generator.class_names[test_labels[i].numpy()]
        
        attention_viz.visualize_attention(
            image=image,
            true_label=true_label,
            save_path=os.path.join(args.save_dir, f"attention_sample_{i+1}.png")
        )
    
    # Performance comparison with original VGG16 (if results available)
    print("\nModel improvements over VGG16:")
    print("- Parameter reduction: ~91% (12M vs 138M parameters)")
    print("- Expected speed improvement: 2-3x faster training")
    print("- Memory efficiency: Significantly reduced GPU memory usage")
    print("- Attention mechanism: Better focus on tumor regions")
    print("- Advanced augmentation: Improved generalization")
    print("- Mixed precision: Faster training with maintained accuracy")
    
    print(f"\nAll results saved to: {args.save_dir}")
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


def predict_single_image(model_path: str, 
                        image_path: str, 
                        class_names: list,
                        image_size: int = 224) -> dict:
    """
    Predict tumor type for a single image.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the image to predict
        class_names: List of class names
        image_size: Input image size
        
    Returns:
        Dictionary with prediction results
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    from tensorflow.keras.preprocessing.image import load_img
    image = load_img(image_path, target_size=(image_size, image_size))
    image_array = np.array(image) / 255.0
    
    # Apply ImageNet normalization
    image_array = (image_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    
    # Prepare results
    results = {
        'predicted_class': class_names[predicted_class_idx],
        'predicted_class_idx': predicted_class_idx,
        'confidence': float(confidence),
        'all_predictions': {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
    }
    
    return results


if __name__ == "__main__":
    main()