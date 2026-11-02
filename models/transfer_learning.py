"""
Transfer Learning Architectures for Plant Disease Classification.
Provides pre-trained backbones fine-tuned for high-accuracy leaf disease detection:
1. MobileNetV3-Large: High accuracy, low-latency, edge-optimized (~4.2M params).
2. EfficientNetV2-B0: State-of-the-art parameter efficiency and generalization (~5.9M params).
3. ResNet50V2: Deep residual representations (~23.6M params).
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_mobilenet_v3(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    freeze_base: bool = True
) -> tf.keras.Model:
    """
    MobileNetV3-Large architecture with ImageNet weights.
    Applies squeeze-and-excitation attention and hard-swish activations.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    # MobileNetV3 expects [0, 255] or handles rescaling internally
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = not freeze_base

    x = base_model(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenet_v3_plant_disease")
    return model

def build_efficientnet_v2(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    freeze_base: bool = True
) -> tf.keras.Model:
    """
    EfficientNetV2-B0 architecture with ImageNet weights.
    Compound scaled depth, width, and resolution with fused MBConv blocks.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = not freeze_base

    x = base_model(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_v2_plant_disease")
    return model

def build_resnet50_v2(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    freeze_base: bool = True
) -> tf.keras.Model:
    """
    ResNet50V2 architecture with ImageNet weights and identity skip connections.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    # ResNetV2 uses caffe-style preprocessing: [-1, 1]
    rescaled = layers.Rescaling(1./127.5, offset=-1)(inputs)
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = not freeze_base

    x = base_model(rescaled, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_v2_plant_disease")
    return model

def unfreeze_for_finetuning(model: tf.keras.Model, unfreeze_last_n: int = 30) -> None:
    """
    Unfreezes the last N layers of the backbone for fine-tuning with a reduced learning rate.
    """
    # Find the backbone layer
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        print("No sub-model backbone found to unfreeze.")
        return

    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_last_n]:
        layer.trainable = False
    for layer in backbone.layers[-unfreeze_last_n:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Freeze BN stats during fine-tuning
        else:
            layer.trainable = True

    print(f"Unfrozen top {unfreeze_last_n} layers of backbone for fine-tuning.")
