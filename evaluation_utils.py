"""
Advanced Evaluation and Visualization for Brain Tumor Detection

This module provides comprehensive model evaluation and visualization capabilities
including confusion matrices, ROC curves, attention visualizations, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import cv2
from typing import List, Dict, Tuple, Optional
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation with advanced visualizations.
    
    This class provides detailed evaluation metrics and visualizations
    for the brain tumor detection model.
    """
    
    def __init__(self, model: tf.keras.Model, class_names: List[str]):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate_model(self, 
                      test_dataset: tf.data.Dataset,
                      save_dir: str = "./evaluation_results") -> Dict:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            test_dataset: Test dataset for evaluation
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting model evaluation...")
        
        # Get predictions and true labels
        y_true, y_pred, y_pred_proba = self._get_predictions(test_dataset)
        
        # Calculate basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Create visualizations
        self._plot_confusion_matrix(y_true, y_pred, save_dir)
        self._plot_roc_curves(y_true, y_pred_proba, save_dir)
        self._plot_precision_recall_curves(y_true, y_pred_proba, save_dir)
        self._plot_prediction_distribution(y_pred_proba, save_dir)
        
        # Compile results
        results = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "num_samples": len(y_true),
            "class_distribution": np.bincount(y_true)
        }
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _get_predictions(self, test_dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and true labels from test dataset."""
        y_true_list = []
        y_pred_proba_list = []
        
        print("Generating predictions...")
        for batch_idx, (images, labels) in enumerate(test_dataset):
            # Get predictions
            predictions = self.model.predict(images, verbose=0)
            y_pred_proba_list.append(predictions)
            y_true_list.append(labels.numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
        
        # Concatenate all predictions and labels
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_true, y_pred, y_pred_proba
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot absolute numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Labels')
        ax1.set_ylabel('True Labels')
        
        # Plot percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Labels')
        ax2.set_ylabel('True Labels')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_dir: str):
        """Plot ROC curves for all classes."""
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        # Calculate ROC curve and AUC for each class
        fpr, tpr, roc_auc = {}, {}, {}
        
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], linewidth=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Classes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def _plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_dir: str):
        """Plot Precision-Recall curves for all classes."""
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{self.class_names[i]} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for All Classes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_distribution(self, y_pred_proba: np.ndarray, save_dir: str):
        """Plot distribution of prediction confidences."""
        # Get maximum probability for each prediction
        max_probs = np.max(y_pred_proba, axis=1)
        predicted_classes = np.argmax(y_pred_proba, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot overall confidence distribution
        ax1.hist(max_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Maximum Prediction Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Confidences')
        ax1.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence by class
        for i, class_name in enumerate(self.class_names):
            class_mask = predicted_classes == i
            if np.any(class_mask):
                class_confidences = max_probs[class_mask]
                ax2.hist(class_confidences, bins=20, alpha=0.6, 
                        label=f'{class_name} (μ={np.mean(class_confidences):.3f})')
        
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution by Predicted Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_evaluation_summary(self, results: Dict):
        """Print a comprehensive evaluation summary."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Total Test Samples: {results['num_samples']}")
        
        print("\nClass Distribution:")
        for i, count in enumerate(results['class_distribution']):
            percentage = count / results['num_samples'] * 100
            print(f"  {self.class_names[i]}: {count} samples ({percentage:.1f}%)")
        
        print("\nPer-Class Performance:")
        class_report = results['classification_report']
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-Score: {metrics['f1-score']:.4f}")
                print(f"    Support: {int(metrics['support'])}")
        
        print(f"\nMacro Average F1-Score: {class_report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Average F1-Score: {class_report['weighted avg']['f1-score']:.4f}")
        print("="*60)


class AttentionVisualizer:
    """
    Visualize attention maps from the spatial attention mechanism.
    
    This class provides tools to visualize what regions the model
    is focusing on when making predictions.
    """
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.attention_model = self._create_attention_model()
    
    def _create_attention_model(self):
        """Create a model that outputs attention weights."""
        # Find the spatial attention layer
        attention_layer = None
        for layer in self.model.layers:
            if 'spatial_attention' in layer.name.lower():
                attention_layer = layer
                break
        
        if attention_layer is None:
            print("Warning: No spatial attention layer found in the model.")
            return None
        
        # Create model that outputs attention weights
        attention_output = attention_layer.output
        return tf.keras.Model(inputs=self.model.input, outputs=attention_output)
    
    def visualize_attention(self, 
                           image: np.ndarray, 
                           true_label: str = None,
                           save_path: str = None) -> None:
        """
        Visualize attention maps for a given image.
        
        Args:
            image: Input image (preprocessed)
            true_label: True label for the image
            save_path: Path to save the visualization
        """
        if self.attention_model is None:
            print("Attention visualization not available.")
            return
        
        # Get prediction
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Get attention weights
        attention_weights = self.attention_model.predict(np.expand_dims(image, axis=0), verbose=0)
        attention_map = attention_weights[0, :, :, 0]  # First channel of attention
        
        # Resize attention map to match input image size
        attention_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        im = axes[1].imshow(attention_resized, cmap='jet', alpha=0.8)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(attention_resized, cmap='jet', alpha=0.4)
        title = f'Attention Overlay\nPredicted: Class {predicted_class} ({confidence:.3f})'
        if true_label:
            title += f'\nTrue: {true_label}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def compare_model_performance(old_results: Dict, new_results: Dict, class_names: List[str]):
    """
    Compare performance between old and new models.
    
    Args:
        old_results: Results from VGG16 model
        new_results: Results from EfficientNet + Attention model
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Overall accuracy comparison
    old_acc = old_results.get('accuracy', 0)
    new_acc = new_results.get('accuracy', 0)
    improvement = new_acc - old_acc
    
    print(f"Overall Accuracy:")
    print(f"  VGG16 Model: {old_acc:.4f} ({old_acc*100:.2f}%)")
    print(f"  EfficientNet + Attention: {new_acc:.4f} ({new_acc*100:.2f}%)")
    print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Per-class comparison
    print(f"\nPer-Class F1-Score Comparison:")
    old_report = old_results.get('classification_report', {})
    new_report = new_results.get('classification_report', {})
    
    for class_name in class_names:
        old_f1 = old_report.get(class_name, {}).get('f1-score', 0)
        new_f1 = new_report.get(class_name, {}).get('f1-score', 0)
        f1_improvement = new_f1 - old_f1
        
        print(f"  {class_name}:")
        print(f"    VGG16: {old_f1:.4f}")
        print(f"    EfficientNet: {new_f1:.4f}")
        print(f"    Improvement: {f1_improvement:+.4f}")
    
    print("="*80)


if __name__ == "__main__":
    print("Evaluation and visualization utilities ready for use!")
    
    # Test with dummy data
    class_names = ['pituitary', 'glioma', 'notumor', 'meningioma']
    print(f"Configured for {len(class_names)} classes: {class_names}")