"""Model definitions for Plant Disease Classification (TensorFlow & PyTorch)."""

__all__ = [
    "build_optimized_custom_cnn",
    "build_baseline_cnn",
    "build_mobilenet_v3",
    "build_efficientnet_v2",
    "build_resnet50_v2",
    "OptimizedPlantCNN",
    "build_pytorch_mobilenet_v3"
]

# Optional TensorFlow models
try:
    from .custom_cnn import build_optimized_custom_cnn, build_baseline_cnn
    from .transfer_learning import build_mobilenet_v3, build_efficientnet_v2, build_resnet50_v2
except ImportError:
    pass

# Optional PyTorch models
try:
    from .pytorch_model import OptimizedPlantCNN, build_pytorch_mobilenet_v3
except ImportError:
    pass
