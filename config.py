"""
Plant Disease Classification Project Configuration.
Centralized hyperparameters and pipeline paths for reproducibility.
"""

import os
from typing import Tuple, List

# Reproducibility
RANDOM_SEED: int = 42

# Image and Batch Dimensions
IMAGE_SIZE: Tuple[int, int] = (128, 128)
INPUT_SHAPE: Tuple[int, int, int] = (128, 128, 3)
BATCH_SIZE: int = 32

# Training Hyperparameters
INITIAL_LR: float = 1e-4
MIN_LR: float = 1e-6
WEIGHT_DECAY: float = 1e-4
EPOCHS: int = 20
EARLY_STOPPING_PATIENCE: int = 5
LABEL_SMOOTHING: float = 0.1

# Directories
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "train")
VALID_DIR: str = os.path.join(BASE_DIR, "valid")
TEST_DIR: str = os.path.join(BASE_DIR, "test")
CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR: str = os.path.join(BASE_DIR, "logs")

# Class Names (38 Classes from PlantVillage / New Plant Diseases Dataset)
CLASS_NAMES: List[str] = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

NUM_CLASSES: int = len(CLASS_NAMES)
