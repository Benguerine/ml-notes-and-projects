"""
Advanced Training Utilities for Brain Tumor Detection

This module provides comprehensive training utilities including:
- Cosine decay learning rate scheduling
- Advanced callbacks for monitoring and optimization
- Mixed precision training support
- Model checkpointing and early stopping
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import math


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay learning rate schedule with optional warmup.
    
    This scheduler provides smooth learning rate decay that often leads to
    better convergence and final model performance.
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 decay_steps: int,
                 alpha: float = 0.0,
                 warmup_steps: int = 0,
                 warmup_learning_rate: float = 0.0,
                 name: str = None):
        
        super(CosineDecayWithWarmup, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.name = name
    
    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecayWithWarmup"):
            step = tf.cast(step, tf.float32)
            
            if self.warmup_steps > 0:
                # Linear warmup
                warmup_percent = step / self.warmup_steps
                warmup_lr = self.warmup_learning_rate + (
                    self.initial_learning_rate - self.warmup_learning_rate
                ) * warmup_percent
                
                # Cosine decay after warmup
                cosine_step = tf.maximum(0.0, step - self.warmup_steps)
                cosine_decay = 0.5 * (1 + tf.cos(
                    math.pi * cosine_step / (self.decay_steps - self.warmup_steps)
                ))
                cosine_lr = self.alpha + (self.initial_learning_rate - self.alpha) * cosine_decay
                
                return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
            else:
                # Standard cosine decay
                cosine_decay = 0.5 * (1 + tf.cos(math.pi * step / self.decay_steps))
                return self.alpha + (self.initial_learning_rate - self.alpha) * cosine_decay
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_steps": self.warmup_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "name": self.name
        }


class TrainingManager:
    """
    Comprehensive training manager for the brain tumor detection model.
    
    This class handles all aspects of model training including callbacks,
    monitoring, and optimization strategies.
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 model_name: str = "brain_tumor_efficientnet",
                 save_dir: str = "./model_outputs"):
        
        self.model = model
        self.model_name = model_name
        self.save_dir = save_dir
        self.history = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize mixed precision if not already done
        if tf.keras.mixed_precision.global_policy().name != 'mixed_float16':
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled for training.")
    
    def create_callbacks(self, 
                        monitor_metric: str = 'val_sparse_categorical_accuracy',
                        patience: int = 10,
                        reduce_lr_patience: int = 5,
                        min_lr: float = 1e-7) -> List[tf.keras.callbacks.Callback]:
        """
        Create a comprehensive set of callbacks for training.
        
        Args:
            monitor_metric: Metric to monitor for callbacks
            patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            min_lr: Minimum learning rate
            
        Returns:
            List of configured callbacks
        """
        callbacks = []
        
        # Model checkpointing - save best model
        checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best.h5")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor_metric,
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode='max',
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard_dir = os.path.join(self.save_dir, "tensorboard_logs")
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # CSV logging
        csv_logger = CSVLogger(
            filename=os.path.join(self.save_dir, f"{self.model_name}_training_log.csv"),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        # Custom learning rate logging
        lr_logger = LearningRateLogger()
        callbacks.append(lr_logger)
        
        return callbacks
    
    def create_lr_schedule(self, 
                          steps_per_epoch: int,
                          epochs: int,
                          initial_lr: float = 0.001,
                          warmup_epochs: int = 2) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create cosine decay learning rate schedule with warmup.
        
        Args:
            steps_per_epoch: Number of training steps per epoch
            epochs: Total number of training epochs
            initial_lr: Initial learning rate
            warmup_epochs: Number of warmup epochs
            
        Returns:
            Learning rate schedule
        """
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        
        return CosineDecayWithWarmup(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=0.01 * initial_lr,  # Final LR is 1% of initial
            warmup_steps=warmup_steps,
            warmup_learning_rate=0.1 * initial_lr  # Start at 10% of initial
        )
    
    def train_model(self,
                   train_dataset: tf.data.Dataset,
                   validation_dataset: tf.data.Dataset,
                   epochs: int = 30,
                   initial_lr: float = 0.001,
                   warmup_epochs: int = 2,
                   class_weights: Dict[int, float] = None) -> tf.keras.callbacks.History:
        """
        Train the model with advanced configurations.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of training epochs
            initial_lr: Initial learning rate
            warmup_epochs: Number of warmup epochs
            class_weights: Dictionary of class weights for imbalanced data
            
        Returns:
            Training history
        """
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_dataset)
        
        # Create learning rate schedule
        lr_schedule = self.create_lr_schedule(
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            initial_lr=initial_lr,
            warmup_epochs=warmup_epochs
        )
        
        # Update optimizer with new learning rate schedule
        self.model.optimizer.learning_rate = lr_schedule
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Total epochs: {epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Initial learning rate: {initial_lr}")
        print(f"Warmup epochs: {warmup_epochs}")
        print(f"Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
        print(f"Class weights: {'Yes' if class_weights else 'No'}")
        print("=" * 60)
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, f"{self.model_name}_final.h5")
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_history(self, save_plots: bool = True):
        """
        Plot training history with comprehensive metrics.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training and Validation Accuracy
        axes[0, 1].plot(epochs, history['sparse_categorical_accuracy'], 'b-', 
                       label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_sparse_categorical_accuracy'], 'r-', 
                       label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Schedule
        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Top-2 Accuracy (if available)
        if 'top_2_accuracy' in history:
            axes[1, 1].plot(epochs, history['top_2_accuracy'], 'b-', 
                           label='Training Top-2', linewidth=2)
            axes[1, 1].plot(epochs, history['val_top_2_accuracy'], 'r-', 
                           label='Validation Top-2', linewidth=2)
            axes[1, 1].set_title('Top-2 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-2 Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.save_dir, f"{self.model_name}_training_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {plot_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of training results.
        
        Returns:
            Dictionary containing training metrics and statistics
        """
        if self.history is None:
            return {"error": "No training history available"}
        
        history = self.history.history
        
        # Find best epoch
        best_epoch = np.argmax(history['val_sparse_categorical_accuracy']) + 1
        
        summary = {
            "total_epochs": len(history['loss']),
            "best_epoch": best_epoch,
            "best_val_accuracy": max(history['val_sparse_categorical_accuracy']),
            "best_val_loss": min(history['val_loss']),
            "final_train_accuracy": history['sparse_categorical_accuracy'][-1],
            "final_val_accuracy": history['val_sparse_categorical_accuracy'][-1],
            "accuracy_improvement": (
                max(history['val_sparse_categorical_accuracy']) - 
                history['val_sparse_categorical_accuracy'][0]
            ),
            "training_stable": (
                abs(history['sparse_categorical_accuracy'][-1] - 
                    history['val_sparse_categorical_accuracy'][-1]) < 0.1
            )
        }
        
        return summary


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Custom callback to log learning rate during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)


def create_training_manager(model: tf.keras.Model, 
                           model_name: str = "brain_tumor_efficientnet",
                           save_dir: str = "./model_outputs") -> TrainingManager:
    """
    Factory function to create a training manager.
    
    Args:
        model: Compiled Keras model
        model_name: Name for saving model files
        save_dir: Directory to save model outputs
        
    Returns:
        Configured training manager
    """
    return TrainingManager(model, model_name, save_dir)


if __name__ == "__main__":
    # Example usage
    print("Testing Training Utilities...")
    
    # Test learning rate schedule
    schedule = CosineDecayWithWarmup(
        initial_learning_rate=0.001,
        decay_steps=1000,
        warmup_steps=100
    )
    
    # Test learning rates for first 200 steps
    steps = list(range(0, 200, 10))
    lrs = [schedule(step).numpy() for step in steps]
    
    print(f"Learning rate schedule test:")
    for step, lr in zip(steps[:5], lrs[:5]):
        print(f"  Step {step}: LR = {lr:.6f}")
    
    print("Training utilities ready for use!")