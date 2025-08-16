"""
Advanced Data Processing and Augmentation for Brain Tumor Detection

This module provides enhanced data loading, preprocessing, and augmentation
capabilities for the brain tumor detection model.

Key improvements over the original implementation:
- Advanced augmentation pipeline with medical image considerations
- Efficient data generators with prefetching
- Mixed precision support
- Better normalization and preprocessing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from sklearn.utils import shuffle
import random
from PIL import Image, ImageEnhance
from typing import Tuple, List, Generator


class AdvancedDataAugmentation:
    """
    Advanced data augmentation pipeline optimized for medical images using TensorFlow.
    
    This class provides sophisticated augmentation techniques that preserve
    important medical features while improving model generalization.
    """
    
    def __init__(self, image_size: int = 224, augmentation_strength: str = 'medium'):
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.augmentation_layers = self._create_augmentation_layers()
    
    def _create_augmentation_layers(self):
        """Create TensorFlow augmentation layers."""
        
        layers = []
        
        if self.augmentation_strength == 'light':
            layers = [
                tf.keras.layers.RandomBrightness(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),  # ~10 degrees
            ]
        elif self.augmentation_strength == 'medium':
            layers = [
                tf.keras.layers.RandomBrightness(0.2),
                tf.keras.layers.RandomContrast(0.2),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomFlip("vertical"),
                tf.keras.layers.RandomRotation(0.08),  # ~15 degrees
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomZoom(0.1),
            ]
        else:  # strong
            layers = [
                tf.keras.layers.RandomBrightness(0.3),
                tf.keras.layers.RandomContrast(0.3),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomFlip("vertical"),
                tf.keras.layers.RandomRotation(0.1),  # ~20 degrees
                tf.keras.layers.RandomTranslation(0.15, 0.15),
                tf.keras.layers.RandomZoom(0.15),
            ]
        
        return layers
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image (np.ndarray): Input image as numpy array
            
        Returns:
            np.ndarray: Augmented and normalized image
        """
        if isinstance(image, str):
            # If image is a path, load it
            image = np.array(Image.open(image))
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        # Apply augmentations
        for layer in self.augmentation_layers:
            image_tensor = layer(image_tensor, training=True)
        
        # Apply ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.numpy()


class BrainTumorDataGenerator:
    """
    Efficient data generator for brain tumor detection with advanced preprocessing.
    
    This generator provides optimized data loading with prefetching and mixed precision support.
    """
    
    def __init__(self, 
                 train_dir: str,
                 test_dir: str = None,
                 image_size: int = 224,
                 batch_size: int = 32,
                 augmentation_strength: str = 'medium',
                 mixed_precision: bool = True):
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        
        # Initialize augmentation
        self.augmentor = AdvancedDataAugmentation(image_size, augmentation_strength)
        
        # Load and prepare data
        self.train_paths, self.train_labels = self._load_data(train_dir)
        if test_dir:
            self.test_paths, self.test_labels = self._load_data(test_dir)
        
        # Get class information
        self.class_names = sorted(os.listdir(train_dir))
        self.num_classes = len(self.class_names)
        
        print(f"Data loaded successfully:")
        print(f"  Classes: {self.class_names}")
        print(f"  Training samples: {len(self.train_paths)}")
        if test_dir:
            print(f"  Test samples: {len(self.test_paths)}")
    
    def _load_data(self, data_dir: str) -> Tuple[List[str], List[str]]:
        """Load image paths and labels from directory structure."""
        paths = []
        labels = []
        
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for image_file in os.listdir(label_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        paths.append(os.path.join(label_dir, image_file))
                        labels.append(label)
        
        # Shuffle the data
        paths, labels = shuffle(paths, labels, random_state=42)
        return paths, labels
    
    def _encode_labels(self, labels: List[str]) -> np.ndarray:
        """Convert string labels to integer encoding."""
        return np.array([self.class_names.index(label) for label in labels])
    
    def _load_and_preprocess_image(self, image_path: str, apply_augmentation: bool = True) -> np.ndarray:
        """Load and preprocess a single image."""
        try:
            # Load image
            image = load_img(image_path, target_size=(self.image_size, self.image_size))
            image = np.array(image)
            
            # Apply augmentation if requested
            if apply_augmentation:
                image = self.augmentor.augment_image(image)
            else:
                # Just normalize without augmentation (for validation/test)
                image = image / 255.0
                # Apply ImageNet normalization
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to appropriate dtype for mixed precision
            if self.mixed_precision:
                image = image.astype(np.float16)
            else:
                image = image.astype(np.float32)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            blank_image = np.zeros((self.image_size, self.image_size, 3))
            if self.mixed_precision:
                return blank_image.astype(np.float16)
            else:
                return blank_image.astype(np.float32)
    
    def create_tf_dataset(self, 
                         paths: List[str], 
                         labels: List[str], 
                         apply_augmentation: bool = True,
                         shuffle_buffer_size: int = 1000) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset with optimizations.
        
        Args:
            paths: List of image paths
            labels: List of corresponding labels
            apply_augmentation: Whether to apply augmentation
            shuffle_buffer_size: Buffer size for shuffling
            
        Returns:
            tf.data.Dataset: Optimized TensorFlow dataset
        """
        
        def load_image_and_label(path, label):
            """Load and preprocess image and label."""
            # Convert path to string if it's a tensor
            path = path.numpy().decode('utf-8') if hasattr(path, 'numpy') else path
            
            # Load and preprocess image
            image = self._load_and_preprocess_image(path, apply_augmentation)
            
            # Encode label
            label_encoded = self.class_names.index(label.numpy().decode('utf-8') if hasattr(label, 'numpy') else label)
            
            return image, label_encoded
        
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        # Shuffle if requested
        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        
        # Map the loading function
        dataset = dataset.map(
            lambda path, label: tf.py_function(
                load_image_and_label,
                [path, label],
                [tf.float16 if self.mixed_precision else tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch the dataset
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_train_dataset(self) -> tf.data.Dataset:
        """Get training dataset with augmentation."""
        return self.create_tf_dataset(
            self.train_paths, 
            self.train_labels, 
            apply_augmentation=True
        )
    
    def get_test_dataset(self) -> tf.data.Dataset:
        """Get test dataset without augmentation."""
        if not self.test_paths:
            raise ValueError("Test directory not provided during initialization")
        
        return self.create_tf_dataset(
            self.test_paths, 
            self.test_labels, 
            apply_augmentation=False,
            shuffle_buffer_size=0  # Don't shuffle test data
        )
    
    def get_class_weights(self) -> dict:
        """Calculate class weights for handling class imbalance."""
        from sklearn.utils.class_weight import compute_class_weight
        
        labels_encoded = self._encode_labels(self.train_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels_encoded),
            y=labels_encoded
        )
        
        return {i: weight for i, weight in enumerate(class_weights)}


def create_data_generators(train_dir: str, 
                          test_dir: str = None,
                          image_size: int = 224,
                          batch_size: int = 32,
                          augmentation_strength: str = 'medium',
                          mixed_precision: bool = True) -> BrainTumorDataGenerator:
    """
    Factory function to create data generator with optimal settings.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory (optional)
        image_size: Size to resize images to
        batch_size: Batch size for training
        augmentation_strength: 'light', 'medium', or 'strong'
        mixed_precision: Whether to use mixed precision
        
    Returns:
        BrainTumorDataGenerator: Configured data generator
    """
    return BrainTumorDataGenerator(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
        augmentation_strength=augmentation_strength,
        mixed_precision=mixed_precision
    )


if __name__ == "__main__":
    # Example usage
    print("Testing Advanced Data Augmentation...")
    
    # Test augmentation pipeline
    augmentor = AdvancedDataAugmentation(image_size=224, augmentation_strength='medium')
    print(f"Augmentation pipeline created with {len(augmentor.augmentation_pipeline.transforms)} transforms")
    
    print("Data utilities ready for use!")