import logging
import cv2
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.transform_config import TransformConfig

logger = logging.getLogger(__name__)

class TransformPipeline:
    """Centralized transform pipeline for stereo vision with keypoints using ReplayCompose."""
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self._keypoint_params = A.KeypointParams(format='xy', remove_invisible=False)
    
    def _get_base_transforms(self) -> list:
        """Get the base geometric transforms applied to both train and validation."""
        return [
            A.LongestMaxSize(max_size=self.config.get_image_size()),
            A.PadIfNeeded(
                min_height=self.config.get_image_size(),
                min_width=self.config.get_image_size(),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
        ]
    
    def _get_augmentation_transforms(self) -> list:
        """Get training-specific augmentation transforms."""
        return [
            A.Affine(
                translate_percent={'x': self.config.limits.shift, 'y': self.config.limits.shift},
                scale=(1.0 - self.config.limits.scale, 1.0 + self.config.limits.scale),
                rotate=(-self.config.limits.rotation, self.config.limits.rotation),
                p=self.config.probabilities.shift_scale_rotate,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
            A.HorizontalFlip(p=self.config.probabilities.horizontal_flip),
            self._get_photometric_augmentations(),
            self._get_noise_blur_augmentations(),
            A.ImageCompression(
                quality_range=self.config.limits.compression_quality,
                p=self.config.probabilities.compression
            ),
            A.RandomShadow(p=self.config.probabilities.shadow),
        ]
    
    def _get_photometric_augmentations(self) -> A.OneOf:
        """Get photometric augmentations (brightness, contrast, CLAHE)."""
        return A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=self.config.limits.brightness_contrast,
                contrast_limit=self.config.limits.brightness_contrast,
                p=self.config.probabilities.brightness_contrast
            ),
            A.CLAHE(p=self.config.probabilities.clahe),
        ], p=self.config.probabilities.augmentation)
    
    def _get_noise_blur_augmentations(self) -> A.OneOf:
        """Get noise and blur augmentations."""
        return A.OneOf([
            A.GaussianBlur(
                blur_limit=self.config.limits.blur, 
                p=self.config.probabilities.blur
            ),
            A.MotionBlur(
                blur_limit=self.config.limits.blur, 
                p=self.config.probabilities.motion_blur
            ),
            A.GaussNoise(
                std_range=self.config.limits.noise, 
                p=self.config.probabilities.noise
            ),
        ], p=self.config.probabilities.blur_group)
    
    def _get_normalization_transforms(self) -> list:
        """Get normalization and tensor conversion transforms."""
        mean, std = self.config.get_normalization_params()
        return [
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    
    def get_train_transform(self) -> A.ReplayCompose:
        """Get training transform pipeline with augmentations using ReplayCompose."""
        transforms = (
            self._get_base_transforms() +
            self._get_augmentation_transforms() +
            self._get_normalization_transforms()
        )
        
        return A.ReplayCompose(
            transforms,
            keypoint_params=self._keypoint_params
        )
    
    def get_val_transform(self) -> A.ReplayCompose:
        """Get validation transform pipeline without augmentations using ReplayCompose."""
        transforms = (
            self._get_base_transforms() +
            self._get_normalization_transforms()
        )
        
        return A.ReplayCompose(
            transforms,
            keypoint_params=self._keypoint_params
        )
    
    def get_test_transform(self) -> A.ReplayCompose:
        """Get test transform pipeline (same as validation)."""
        return self.get_val_transform()