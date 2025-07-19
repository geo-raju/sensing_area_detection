import os
from typing import Dict, Any, List
import logging

from config.data_config import (
    CAMERA_CONFIG,
    IMG_PROC_DIR,
    PROBE_PROC_DIR
)
from src.dataset.loaders.file_manager import FileManager
from src.dataset.loaders.label_loader import LabelLoader

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Handles validation operations for dataset samples."""

    def __init__(self, file_manager: FileManager, label_loader: LabelLoader, filenames: List[str]):
        """
        Initialize validator.
        
        Args:
            file_manager: FileManager instance for directory operations
            label_loader: LabelLoader instance for accessing labels
            filenames: List of filenames in the dataset
        """
        self.file_manager = file_manager
        self.label_loader = label_loader
        self.filenames = filenames

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample information without loading data."""
        if not 0 <= idx < len(self.filenames):
            raise IndexError(f"Index {idx} out of range")

        filename = self.filenames[idx]
        filename_no_ext = os.path.splitext(filename)[0]
        cameras = list(CAMERA_CONFIG.values())

        return {
            "filename": filename,
            "filename_no_ext": filename_no_ext,
            "left_img_path": str(self.file_manager.get_directory(cameras[0], IMG_PROC_DIR) / filename),
            "right_img_path": str(self.file_manager.get_directory(cameras[1], IMG_PROC_DIR) / filename),
            "left_axis_path": str(self.file_manager.get_directory(cameras[0], PROBE_PROC_DIR) / f"{filename_no_ext}.txt"),
            "right_axis_path": str(self.file_manager.get_directory(cameras[1], PROBE_PROC_DIR) / f"{filename_no_ext}.txt")
        }

    def validate_sample_integrity(self, idx: int) -> Dict[str, bool]:
        """Validate sample file integrity."""
        if not 0 <= idx < len(self.filenames):
            raise IndexError(f"Index {idx} out of range")

        filename = self.filenames[idx]
        filename_no_ext = os.path.splitext(filename)[0]
        cameras = list(CAMERA_CONFIG.values())
        center_points = self.label_loader.center_points

        return {
            'left_image': (self.file_manager.get_directory(cameras[0], IMG_PROC_DIR) / filename).exists(),
            'right_image': (self.file_manager.get_directory(cameras[1], IMG_PROC_DIR) / filename).exists(),
            'left_label': filename in center_points.get(cameras[0], {}),
            'right_label': filename in center_points.get(cameras[1], {}),
            'left_axis': (self.file_manager.get_directory(cameras[0], PROBE_PROC_DIR) / f"{filename_no_ext}.txt").exists(),
            'right_axis': (self.file_manager.get_directory(cameras[1], PROBE_PROC_DIR) / f"{filename_no_ext}.txt").exists(),
        }

    def validate_all_samples(self) -> Dict[str, List[int]]:
        """Validate all samples and return problematic indices."""
        issues = {
            'missing_left_image': [],
            'missing_right_image': [],
            'missing_left_label': [],
            'missing_right_label': [],
            'missing_left_axis': [],
            'missing_right_axis': []
        }

        for idx in range(len(self.filenames)):
            validation = self.validate_sample_integrity(idx)

            for key, is_valid in validation.items():
                if not is_valid:
                    issues[f'missing_{key}'].append(idx)

        return issues

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        issues = self.validate_all_samples()
        
        total_samples = len(self.filenames)
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        return {
            'total_samples': total_samples,
            'total_issues': total_issues,
            'issues_by_type': {k: len(v) for k, v in issues.items()},
            'problematic_samples': issues,
            'validation_passed': total_issues == 0
        }