from pathlib import Path
import numpy as np
import logging

from src.dataset.core.data_structures import DatasetError

logger = logging.getLogger(__name__)


class AxisLoader:
    """Handles loading of axis point files."""

    @staticmethod
    def load_axis_points(axis_path: Path) -> np.ndarray:
        """Load probe axis points from text file."""
        if not axis_path.exists():
            return np.empty((0, 2), dtype=np.float32)

        try:
            points = np.loadtxt(axis_path, dtype=np.float32)

            # Handle empty file
            if points.size == 0:
                return np.empty((0, 2), dtype=np.float32)

            # Ensure proper shape
            if points.ndim == 1:
                if points.size == 2:
                    points = points.reshape(1, -1)
                elif points.size % 2 == 0:
                    points = points.reshape(-1, 2)
                else:
                    raise ValueError(f"Cannot reshape {points.size} points to (N, 2)")

            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(f"Invalid shape: {points.shape}, expected (N, 2)")

            return points

        except (ValueError, OSError) as e:
            raise DatasetError(f"Error reading axis points from {axis_path}: {e}")
