"""
Production-Grade Training Pipeline for Plant Disease Classification.
Features:
- Full reproducibility: fixed seeds across random, numpy, tensorflow.
- Flexible model selection: baseline_cnn, optimized_custom_cnn, mobilenet_v3, efficientnet_v2, resnet50_v2.
- Advanced optimization: AdamW, Adam, or SGD with learning rate scheduling.
- Regularization: Label smoothing cross-entropy loss, dropout, weight decay.
- Robust callbacks: ModelCheckpoint (best val_loss), EarlyStopping with weight restoration, ReduceLROnPlateau, CSVLogger, TensorBoard.
"""

import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset_loader import load_dataset
from models.custom_cnn import build_baseline_cnn, build_optimized_custom_cnn
from models.transfer_learning import build_mobilenet_v3, build_efficientnet_v2, build_resnet50_v2

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"Random seed set to {seed}")

def get_optimizer(optimizer_name: str, lr: float, weight_decay: float = 1e-4) -> tf.keras.optimizers.Optimizer:
    """Instantiates the requested optimizer."""
    name = optimizer_name.lower()
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    elif name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from 'adamw', 'adam', 'sgd'.")

def get_model(model_name: str) -> tf.keras.Model:
    """Instantiates the selected model architecture."""
    name = model_name.lower()
    if name == "baseline_cnn":
        return build_baseline_cnn()
    elif name == "optimized_custom_cnn":
        return build_optimized_custom_cnn()
    elif name == "mobilenet_v3":
        return build_mobilenet_v3(freeze_base=True)
    elif name == "efficientnet_v2":
        return build_efficientnet_v2(freeze_base=True)
    elif name == "resnet50_v2":
        return build_resnet50_v2(freeze_base=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_pipeline(
    train_dir: str,
    val_dir: str,
    model_name: str = "optimized_custom_cnn",
    optimizer_name: str = "adamw",
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.INITIAL_LR,
    weight_decay: float = config.WEIGHT_DECAY,
    label_smoothing: float = config.LABEL_SMOOTHING,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    output_dir: str = config.CHECKPOINT_DIR
) -> tf.keras.callbacks.History:
    """
    Executes the training and validation pipeline.
    """
    set_seed(config.RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    print("Loading datasets...")
    train_ds = load_dataset(train_dir, batch_size=batch_size, shuffle=True, augment=True)
    val_ds = load_dataset(val_dir, batch_size=batch_size, shuffle=False, augment=False)

    print(f"Building model: {model_name}...")
    model = get_model(model_name)
    model.summary()

    optimizer = get_optimizer(optimizer_name, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )

    checkpoint_path = os.path.join(output_dir, f"best_{model_name}.keras")
    csv_log_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_log.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f"tb_{model_name}"),
            histogram_freq=1
        )
    ]

    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"Training completed. Best model saved to: {checkpoint_path}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant disease classification model")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--model", type=str, default="optimized_custom_cnn", help="Model architecture")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer (adamw, adam, sgd)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    args = parser.parse_args()

    train_pipeline(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_name=args.model,
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
