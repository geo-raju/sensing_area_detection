import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

from config.data_config import (
    PROC_DIR_PATH, CAMERA_CONFIG, SPLIT_NAMES,
    LABEL_PROC_DIR, IMG_PROC_DIR, PROBE_PROC_DIR, DEPTH_PROC_DIR
)
from src.dataset.core.data_structures import DatasetError
from src.dataset.core.data_structures import SampleData
from src.dataset.loaders.file_manager import FileManager
from src.dataset.loaders.image_loader import ImageLoader
from src.dataset.loaders.label_loader import LabelLoader
from src.dataset.loaders.axis_loader import AxisLoader
from src.dataset.transforms.transform_manager import TransformManager
from src.dataset.utils import DatasetValidator

logger = logging.getLogger(__name__)


class SensingAreaDataset(Dataset):
    """Optimized dataset for stereo vision sensing area detection."""

    def __init__(
        self,
        root: str = PROC_DIR_PATH,
        subset: str = 'train',
        transform: Optional[Any] = None,
        validate_structure: bool = True,
        lazy_load: bool = True,
        seed: Optional[int] = None
    ) -> None:
        """Initialize dataset."""
        if subset not in SPLIT_NAMES:
            raise ValueError(f"Invalid subset '{subset}'. Must be: {SPLIT_NAMES}")

        self.root = Path(root)
        self.subset = subset
        self.lazy_load = lazy_load
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Initialize components
        self.file_manager = FileManager(self.root, subset)
        self.label_loader = LabelLoader(self.file_manager)
        self.image_loader = ImageLoader()
        self.axis_loader = AxisLoader()
        self.transform_manager = TransformManager(transform)

        if validate_structure:
            try:
                self.file_manager.validate_structure()
            except FileNotFoundError as e:
                logger.error(f"Dataset initialization failed: {e}")
                raise # Re-raise the exception to stop initialization

        # Load filenames
        try:
            self.filenames = self.file_manager.get_image_filenames()
        except DatasetError as e:
            logger.error(f"Failed to get image filenames for {subset} subset: {e}")
            self.filenames = [] # Initialize as empty to prevent further errors

        # Initialize validator
        self.validator = DatasetValidator(self.file_manager, self.label_loader, self.filenames)

        # Pre-load labels if not lazy loading
        if not lazy_load and self.filenames: # Only try to load if there are filenames
            _ = self.label_loader.center_points

        logger.info(f"Initialized {subset} dataset with {len(self)} samples")

        # Log transform info
        transform_info = self.transform_manager.get_transform_info()
        logger.info(f"Transform info: {transform_info}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> SampleData:
        """Get dataset item."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        if self.seed is not None:
            np.random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)

        filename = self.filenames[idx]

        try:
            return self._load_sample(filename)
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({filename}): {e}")
            # Re-raise as DatasetError for consistent error handling outside
            raise DatasetError(f"Failed to load sample {idx}: {e}") from e

    def _load_sample(self, filename: str) -> SampleData:
        """Load a single sample."""
        cameras = list(CAMERA_CONFIG.values())

        # Load images
        left_img_path = self.file_manager.get_directory(cameras[0], IMG_PROC_DIR) / filename
        right_img_path = self.file_manager.get_directory(cameras[1], IMG_PROC_DIR) / filename

        left_img = self.image_loader.load_image(left_img_path)
        right_img = self.image_loader.load_image(right_img_path)

        # Load labels
        left_center = self.label_loader.get_label(filename, cameras[0])
        right_center = self.label_loader.get_label(filename, cameras[1])

        # --- ADD THESE DEBUG LOGS ---
        if left_center is None or right_center is None:
            logger.error(f"Missing center point for {filename}: Left={left_center}, Right={right_center}")
        elif not (isinstance(left_center, tuple) and len(left_center) == 2 and
                isinstance(right_center, tuple) and len(right_center) == 2):
            logger.error(f"Invalid center point format for {filename}: Left={left_center}, Right={right_center}")
        # --- END DEBUG LOGS ---

        # Load axis points
        filename_no_ext = os.path.splitext(filename)[0]
        left_axis_path = self.file_manager.get_directory(cameras[0], PROBE_PROC_DIR) / f"{filename_no_ext}.txt"
        right_axis_path = self.file_manager.get_directory(cameras[1], PROBE_PROC_DIR) / f"{filename_no_ext}.txt"

        left_axis = self.axis_loader.load_axis_points(left_axis_path)
        right_axis = self.axis_loader.load_axis_points(right_axis_path)

        # Apply transformations
        left_img_t, right_img_t, left_center_t, right_center_t, left_axis_t, right_axis_t = \
            self.transform_manager.apply_transform(
                left_img, right_img, left_center, right_center, left_axis, right_axis
            )

        return SampleData(
            left_image=left_img_t,
            right_image=right_img_t,
            left_label=torch.tensor(left_center_t, dtype=torch.float32),
            right_label=torch.tensor(right_center_t, dtype=torch.float32),
            left_axis=torch.tensor(left_axis_t, dtype=torch.float32),
            right_axis=torch.tensor(right_axis_t, dtype=torch.float32),
            filename=filename
        )

    def get_subset_stats(self) -> Dict[str, Any]:
        """Get subset statistics."""
        center_points = self.label_loader.center_points
        cameras = list(CAMERA_CONFIG.values())

        return {
            "subset": self.subset,
            "total_samples": len(self),
            "left_labels_count": len(center_points.get(cameras[0], {})),
            "right_labels_count": len(center_points.get(cameras[1], {})),
            "root_path": str(self.root),
            "subset_path": str(self.file_manager.subset_dir),
            "transform_info": self.transform_manager.get_transform_info()
        }

    def get_camera_names(self) -> List[str]:
        """Get camera names from configuration."""
        return list(CAMERA_CONFIG.values())

    def get_data_types(self) -> List[str]:
        """Get data types from configuration."""
        return [IMG_PROC_DIR, LABEL_PROC_DIR, PROBE_PROC_DIR]

    def update_transform(self, new_transform: Optional[Any]) -> None:
        """Update the transform for this dataset."""
        self.transform_manager = TransformManager(new_transform)
        transform_info = self.transform_manager.get_transform_info()
        logger.info(f"Updated transform: {transform_info}")

    # Validation methods - delegate to validator
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample information without loading data."""
        sample_info = self.validator.get_sample_info(idx)
        # Add subset and transform info that are specific to this dataset instance
        sample_info.update({
            "subset": self.subset,
            "transform_info": self.transform_manager.get_transform_info()
        })
        return sample_info

    def validate_sample_integrity(self, idx: int) -> Dict[str, bool]:
        """Validate sample file integrity."""
        return self.validator.validate_sample_integrity(idx)

    def validate_all_samples(self) -> Dict[str, List[int]]:
        """Validate all samples and return problematic indices."""
        return self.validator.validate_all_samples()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return self.validator.get_validation_summary()
    
if __name__ == "__main__":
    def create_sensing_area_dataset(
        root: str = './data/processed',  # Default processed data directory
        subset: str = 'train',
        transform: Optional[Any] = None,
        validate_structure: bool = True,
        lazy_load: bool = True,
        seed: Optional[int] = None
    ) -> SensingAreaDataset:
        """
        Factory function to create an instance of SensingAreaDataset.

        Args:
            root: The root directory path where the processed dataset is located.
                Defaults to './data/processed'.
            subset: The dataset subset to load ('train', 'val', or 'test').
                    Defaults to 'train'.
            transform: Optional Albumentations transform to apply to the data.
                    Defaults to None.
            validate_structure: If True, validates the dataset directory structure upon initialization.
                                Defaults to True.
            lazy_load: If True, labels are loaded lazily when accessed. If False, labels are pre-loaded.
                    Defaults to True.
            seed: Optional integer for setting random seeds for reproducibility.
                Defaults to None.

        Returns:
            An initialized SensingAreaDataset object.

        Raises:
            ValueError: If an invalid subset is provided.
            FileNotFoundError: If validate_structure is True and the directory structure is invalid.
            DatasetError: If there's an issue loading image filenames.
        """
        if subset not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid subset '{subset}'. Must be one of 'train', 'val', or 'test'.")

        try:
            dataset = SensingAreaDataset(
                root=Path(root).resolve(),  # Ensure root path is absolute
                subset=subset,
                transform=transform,
                validate_structure=validate_structure,
                lazy_load=lazy_load,
                seed=seed
            )
            return dataset
        except Exception as e:
            print(f"Error creating SensingAreaDataset for subset '{subset}': {e}")
            raise # Re-raise the exception after logging
    
    # 1. Create a training dataset with default settings
    print("Creating training dataset...")
    try:
        train_dataset = create_sensing_area_dataset(subset='train')
        print(f"Training dataset created with {len(train_dataset)} samples.")
        print(train_dataset.get_subset_stats())
        # Example of accessing a sample (will only work if actual data exists)
        # sample = train_dataset[0]
        # print(f"Sample 0 filename: {sample.filename}")
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to create training dataset: {e}")

    print("\n" + "="*50 + "\n")

    # 2. Create a validation dataset with a specific root and no structure validation
    print("Creating validation dataset with custom root and no structure validation...")
    try:
        # Assuming 'data/processed' exists relative to where you run this script
        val_dataset = create_sensing_area_dataset(
            root='./data/processed',
            subset='val',
            validate_structure=False, # Skip initial structure validation for testing
            lazy_load=True,
            seed=42
        )
        print(f"Validation dataset created with {len(val_dataset)} samples.")
        print(val_dataset.get_subset_stats())
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to create validation dataset: {e}")

    print("\n" + "="*50 + "\n")

    # 3. Demonstrate error handling for an invalid subset
    print("Attempting to create dataset with invalid subset...")
    try:
        invalid_dataset = create_sensing_area_dataset(subset='development')
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    print("\n" + "="*50 + "\n")

    # 4. Create a dataset with a dummy transform (you'd replace this with a real Albumentations transform)
    # This requires `albumentations` to be installed.
    # pip install albumentations
    try:
        import albumentations as A
        dummy_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize()
        ])
        print("Creating test dataset with a dummy transform...")
        test_dataset = create_sensing_area_dataset(
            subset='test',
            transform=dummy_transform,
            lazy_load=False # Pre-load labels
        )
        print(f"Test dataset created with {len(test_dataset)} samples.")
        print(test_dataset.get_subset_stats())
    except ImportError:
        print("Albumentations not installed. Skipping transform example.")
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to create test dataset with transform: {e}")