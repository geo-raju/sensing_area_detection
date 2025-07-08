"""Data transformation configuration."""

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformConfig:
    """Configuration class for data transforms."""
    
    # Image dimensions
    IMAGE_SIZE = 896
    
    # Augmentation parameters
    BRIGHTNESS_CONTRAST_LIMIT = 0.2
    ROTATION_LIMIT = 10
    BLUR_LIMIT = 3
    NOISE_LIMIT = (10.0, 50.0)
    COMPRESSION_QUALITY = (70, 100)
    
    # Augmentation probabilities
    BRIGHTNESS_CONTRAST_PROB = 0.8
    CLAHE_PROB = 0.2
    AUGMENTATION_PROB = 0.9
    ROTATION_PROB = 0.7
    BLUR_PROB = 0.3
    MOTION_BLUR_PROB = 0.3
    NOISE_PROB = 0.4
    BLUR_GROUP_PROB = 0.7
    COMPRESSION_PROB = 0.3
    SHADOW_PROB = 0.2