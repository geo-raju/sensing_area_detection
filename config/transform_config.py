"""Data transformation configuration"""

from dataclasses import dataclass, field
from typing import Tuple


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ImageConfig:
    """Image processing configuration."""
    size: int = 896


@dataclass
class AugmentationLimits:
    """Augmentation parameter limits."""
    brightness_contrast: float = 0.2
    rotation: int = 10
    blur: int = 3
    noise: Tuple[float, float] = (0.02, 0.1)
    compression_quality: Tuple[int, int] = (70, 100)
    shift: float = 0.0625
    scale: float = 0.1


@dataclass
class AugmentationProbabilities:
    """Augmentation probabilities for different transformations."""
    brightness_contrast: float = 0.8
    clahe: float = 0.2
    augmentation: float = 0.9
    shift_scale_rotate: float = 0.7
    blur: float = 0.3
    motion_blur: float = 0.3
    noise: float = 0.4
    blur_group: float = 0.7
    compression: float = 0.3
    shadow: float = 0.2
    horizontal_flip: float = 0.5


@dataclass
class TransformConfig:
    """Main configuration class for data transforms."""
    
    # Core configurations
    image: ImageConfig = field(default_factory=ImageConfig)
    limits: AugmentationLimits = field(default_factory=AugmentationLimits)
    probabilities: AugmentationProbabilities = field(default_factory=AugmentationProbabilities)
    
    # Normalization parameters
    mean: list = None
    std: list = None
    
    def __post_init__(self):
        """Set default normalization values if not provided."""
        if self.mean is None:
            self.mean = IMAGENET_MEAN
        if self.std is None:
            self.std = IMAGENET_STD
    
    def get_image_size(self) -> int:
        """Get configured image size."""
        return self.image.size
    
    def get_normalization_params(self) -> Tuple[list, list]:
        """Get normalization parameters."""
        return self.mean, self.std
    
    def get_augmentation_config(self) -> dict:
        """Get augmentation configuration as dictionary."""
        return {
            'limits': {
                'brightness_contrast': self.limits.brightness_contrast,
                'rotation': self.limits.rotation,
                'blur': self.limits.blur,
                'noise': self.limits.noise,
                'compression_quality': self.limits.compression_quality,
                'shift': self.limits.shift,
                'scale': self.limits.scale,
            },
            'probabilities': {
                'brightness_contrast': self.probabilities.brightness_contrast,
                'clahe': self.probabilities.clahe,
                'augmentation': self.probabilities.augmentation,
                'shift_scale_rotate': self.probabilities.shift_scale_rotate,
                'blur': self.probabilities.blur,
                'motion_blur': self.probabilities.motion_blur,
                'noise': self.probabilities.noise,
                'blur_group': self.probabilities.blur_group,
                'compression': self.probabilities.compression,
                'shadow': self.probabilities.shadow,
                'horizontal_flip': self.probabilities.horizontal_flip,
            }
        }