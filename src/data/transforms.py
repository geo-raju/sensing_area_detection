"""Data augmentation transforms."""

import logging
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.transform_config import TransformConfig, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

def get_train_transform(config: Optional[TransformConfig] = None) -> A.Compose:
    """
    Create training transforms with data augmentation.
    
    Args:
        config: Transform configuration. If None, uses default TransformConfig.
        
    Returns:
        Albumentations compose object for training transforms
    """
    if config is None:
        config = TransformConfig()
    
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=config.BRIGHTNESS_CONTRAST_LIMIT,
                contrast_limit=config.BRIGHTNESS_CONTRAST_LIMIT,
                p=config.BRIGHTNESS_CONTRAST_PROB
            ),
            A.CLAHE(p=config.CLAHE_PROB),
        ], p=config.AUGMENTATION_PROB),
        A.Rotate(limit=config.ROTATION_LIMIT, p=config.ROTATION_PROB),
        A.OneOf([
            A.GaussianBlur(blur_limit=config.BLUR_LIMIT, p=config.BLUR_PROB),
            A.MotionBlur(blur_limit=config.BLUR_LIMIT, p=config.MOTION_BLUR_PROB),
            A.GaussNoise(var_limit=config.NOISE_LIMIT, p=config.NOISE_PROB),
        ], p=config.BLUR_GROUP_PROB),
        A.ImageCompression(
            quality_lower=config.COMPRESSION_QUALITY[0],
            quality_upper=config.COMPRESSION_QUALITY[1],
            p=config.COMPRESSION_PROB
        ),
        A.RandomShadow(p=config.SHADOW_PROB),  # Simulates specular/reflection shadows
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transform(config: Optional[TransformConfig] = None) -> A.Compose:
    """
    Create validation transforms without augmentation.
    
    Args:
        config: Transform configuration. If None, uses default TransformConfig.
        
    Returns:
        Albumentations compose object for validation transforms
    """
    if config is None:
        config = TransformConfig()
    
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])