from pathlib import Path
import numpy as np
import logging
import cv2

from src.dataset.core.data_structures import DatasetError

logger = logging.getLogger(__name__)


class ImageLoader:
    """Handles image loading operations."""

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """Load image and convert to RGB."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise DatasetError(f"Could not load image: {image_path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)