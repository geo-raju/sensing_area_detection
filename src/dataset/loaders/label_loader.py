from pathlib import Path
import logging
from functools import cached_property
from typing import Tuple, Dict
import numpy as np

from src.dataset.core.data_structures import DatasetError
from src.dataset.loaders.file_manager import FileManager
from config.data_config import (
    CAMERA_CONFIG, 
    LABEL_PROC_DIR, 
    LABEL_FILE
)

logger = logging.getLogger(__name__)


class LabelLoader:
    """Handles loading and parsing of label files."""

    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self._center_points = None

    @cached_property
    def center_points(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load center points for both cameras (cached)."""
        if self._center_points is None:
            self._center_points = self._load_center_points()
        return self._center_points

    def _load_center_points(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load center points from label files."""
        points = {}

        for camera in CAMERA_CONFIG.values():
            label_dir = self.file_manager.get_directory(camera, LABEL_PROC_DIR)
            label_file = label_dir / LABEL_FILE

            try:
                points[camera] = self._parse_label_file(label_file) if label_file.exists() else {}
            except Exception as e:
                logger.warning(f"Error loading labels for {camera}: {e}")
                points[camera] = {}

        left_count = len(points.get('left', {}))
        right_count = len(points.get('right', {}))

        if left_count != right_count:
            logger.warning(f"Mismatch in label counts: left={left_count}, right={right_count}")

        logger.info(f"Loaded {left_count} left, {right_count} right center points")

        return points

    def _parse_label_file(self, file_path: Path) -> Dict[str, Tuple[float, float]]:
        """Parse label file with flexible format support."""
        points = {}

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Try different separators
                for sep in [', ', ' ', '\t']:
                    parts = line.split(sep)
                    if len(parts) >= 3:
                        try:
                            filename = parts[0]
                            x, y = float(parts[1]), float(parts[2])

                            if np.isnan(x) or np.isnan(y):
                                logger.warning(f"Label contains NaN for {filename} in {file_path}:{line_num}. Skipping line: {line}")
                                continue

                            points[filename] = (x, y)
                            break
                        except ValueError:
                            continue
                else:
                    logger.warning(f"Invalid format in {file_path}:{line_num}: {line}")

        return points

    def get_label(self, filename: str, camera: str) -> Tuple[float, float]:
        """Get label coordinates for filename and camera."""
        if camera not in self.center_points:
            raise DatasetError(f"Invalid camera '{camera}'")

        # Access center_points directly as it's a cached property
        if filename not in self.center_points[camera]:
            raise DatasetError(f"Label not found for {camera} camera: {filename}")

        return self.center_points[camera][filename]