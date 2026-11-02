"""
Production tf.data Dataset Pipeline for Plant Disease Classification.
Includes:
- Robust online data augmentation layers
- High-throughput tf.data pipeline (caching, prefetching with AUTOTUNE)
- Proper label formatting and batching
"""

import tensorflow as tf
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_data_augmentation_layers() -> tf.keras.Sequential:
    """
    Builds data augmentation pipeline for training.
    Robust against leaf orientation, scaling, lighting, and cropping variations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

def load_dataset(
    directory: str,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False,
    cache: bool = True,
    seed: int = config.RANDOM_SEED
) -> tf.data.Dataset:
    """
    Loads images from directory using image_dataset_from_directory and builds an optimized tf.data pipeline.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
        interpolation="bilinear"
    )

    if augment:
        augmentation_pipeline = get_data_augmentation_layers()
        ds = ds.map(lambda x, y: (augmentation_pipeline(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=seed)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
