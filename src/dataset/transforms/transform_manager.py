from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import albumentations as A
import logging

logger = logging.getLogger(__name__)


class TransformManager:
    """Handles data transformations and keypoint processing with stereo consistency."""

    def __init__(self, transform: Optional[A.ReplayCompose] = None):
        self.transform = transform

    def apply_transform(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        left_center: Tuple[float, float],
        right_center: Tuple[float, float],
        left_axis: np.ndarray,
        right_axis: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float], Tuple[float, float], np.ndarray, np.ndarray]:
        """Apply transformations to images and keypoints with stereo consistency."""

        if self.transform is None:
            return (
                torch.from_numpy(left_img).permute(2, 0, 1).float(),
                torch.from_numpy(right_img).permute(2, 0, 1).float(),
                left_center,
                right_center,
                left_axis,
                right_axis
            )

        # Prepare keypoints for transformation
        left_keypoints = [list(left_center)] + [list(pt) for pt in left_axis]
        right_keypoints = [list(right_center)] + [list(pt) for pt in right_axis]

        try:
            # Apply transform to left image and keypoints first
            transformed_left = self.transform(
                image=left_img,
                keypoints=left_keypoints
            )
            
            # Get the replay data from the first transformation
            replay_data = transformed_left.get('replay', {})
            
            # Apply the SAME transformation to right image using replay
            transformed_right = A.ReplayCompose.replay(
                replay_data,
                image=right_img,
                keypoints=right_keypoints
            )

            # Extract transformed data
            left_img_t = transformed_left['image']
            right_img_t = transformed_right['image']
            left_kpts_t = transformed_left.get('keypoints', left_keypoints)
            right_kpts_t = transformed_right.get('keypoints', right_keypoints)

            # Validate keypoints are within image boundaries
            if isinstance(left_img_t, torch.Tensor):
                img_height, img_width = left_img_t.shape[-2:]
            else:
                img_height, img_width = left_img_t.shape[:2]
            
            left_kpts_t = self._validate_keypoints(left_kpts_t, img_height, img_width)
            right_kpts_t = self._validate_keypoints(right_kpts_t, img_height, img_width)

            # Separate center points and axis points
            left_center_t = tuple(left_kpts_t[0]) if left_kpts_t else left_center
            right_center_t = tuple(right_kpts_t[0]) if right_kpts_t else right_center
            left_axis_t = np.array(left_kpts_t[1:], dtype=np.float32) if len(left_kpts_t) > 1 else np.empty((0, 2), dtype=np.float32)
            right_axis_t = np.array(right_kpts_t[1:], dtype=np.float32) if len(right_kpts_t) > 1 else np.empty((0, 2), dtype=np.float32)

            # Convert images to tensors
            left_img_t = self._convert_to_tensor(left_img_t)
            right_img_t = self._convert_to_tensor(right_img_t)

            return left_img_t, right_img_t, left_center_t, right_center_t, left_axis_t, right_axis_t

        except Exception as e:
            logger.warning(f"Transform failed, using original data: {e}")
            return (
                torch.from_numpy(left_img).permute(2, 0, 1).float(),
                torch.from_numpy(right_img).permute(2, 0, 1).float(),
                left_center,
                right_center,
                left_axis,
                right_axis
            )

    def _validate_keypoints(self, keypoints: List[List[float]], img_height: int, img_width: int) -> List[List[float]]:
        """Validate and clamp keypoints to image boundaries."""
        validated = []
        for kpt in keypoints:
            if len(kpt) >= 2:
                x, y = kpt[0], kpt[1]
                # Clamp to image boundaries
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                validated.append([x, y])
            else:
                validated.append(kpt)  # Keep as-is if invalid format
        return validated

    def _convert_to_tensor(self, img):
        """Convert image to tensor with proper format."""
        if isinstance(img, np.ndarray):
            return torch.from_numpy(img).permute(2, 0, 1).float()
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] != 3:
                return img.permute(2, 0, 1).float()
            else:
                return img.float()
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def get_transform_info(self) -> Dict[str, Any]:
        """Get information about the current transform."""
        if self.transform is None:
            return {"transform": None, "type": "none"}

        transform_info = {
            "transform": str(type(self.transform).__name__),
            "type": "albumentations_replay"
        }

        # Try to get configuration if available
        if hasattr(self.transform, 'transforms'):
            transform_info["transforms"] = [str(type(t).__name__) for t in self.transform.transforms]

        return transform_info