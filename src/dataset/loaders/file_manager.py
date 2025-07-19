import os
from pathlib import Path
from itertools import product
from typing import List
import logging

from config.data_config import (
    CAMERA_CONFIG, 
    DATA_TYPE_CONFIG, 
    LABEL_PROC_DIR, 
    IMG_PROC_DIR, 
    PROBE_PROC_DIR
)
from src.dataset.core.data_structures import DatasetError

logger = logging.getLogger(__name__)


class FileManager:
    """Handles file operations and directory management."""

    def __init__(self, root: Path, subset: str):
        self.root = root
        self.subset = subset
        self.subset_dir = root / subset
        self._directories = {}
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Set up directory paths based on configuration."""
        # Ensure LABEL_PROC_DIR is included for file management
        data_types = list(DATA_TYPE_CONFIG.values()) + [LABEL_PROC_DIR, PROBE_PROC_DIR]

        for camera, data_type in product(CAMERA_CONFIG.values(), data_types):
            key = f"{camera}_{data_type}"
            self._directories[key] = self.subset_dir / camera / data_type

    def get_directory(self, camera: str, data_type: str) -> Path:
        """Get directory path for camera and data type."""
        key = f"{camera}_{data_type}"
        if key not in self._directories:
            raise DatasetError(f"Directory key '{key}' not found")
        return self._directories[key]

    def validate_structure(self) -> None:
        """Validate directory structure exists."""
        missing = [str(path) for path in self._directories.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing directories: {missing}. "
                "Run data organization pipeline first."
            )
        logger.info(f"Directory structure validated for {self.subset_dir}")

    def get_image_filenames(self) -> List[str]:
        """Get sorted list of image filenames."""
        left_img_dir = self.get_directory(list(CAMERA_CONFIG.values())[0], IMG_PROC_DIR)

        if not left_img_dir.exists():
            # If the directory doesn't exist, it will be caught by validate_structure,
            # but this specific check is for getting filenames from an existing path.
            raise DatasetError(f"Left image directory not found: {left_img_dir}")

        filenames = sorted([
            f for f in os.listdir(left_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not filenames:
            raise DatasetError(f"No images found in {left_img_dir}")

        return filenames