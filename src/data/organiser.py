"""
Data organisation module for processing and organising image datasets.
Handles train/val/test splits and file organisation with configurable structure.
"""

import shutil
import logging
from itertools import product
from typing import List, Dict, Tuple
from glob import glob
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DataSplitConfig:
    """Configuration for data splitting ratios."""
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int = 42
    
    def __post_init__(self):
        """Validate that ratios sum to 1.0."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")


@dataclass
class DirectoryConfig:
    """Configuration for directory structure."""
    img_raw_dir: str
    img_proc_dir: str
    label_raw_dir: str
    label_proc_dir: str
    probe_raw_dir: str
    probe_proc_dir: str
    label_file: str


class DataOrganiser:
    """
    Class to handle data organisation and splitting for machine learning datasets.
    """
    
    def __init__(
        self, 
        raw_dir: str, 
        output_dir: str, 
        camera_config: Dict[str, str], 
        data_type_config: Dict[str, str],
        directory_config: DirectoryConfig,
        split_config: DataSplitConfig
    ):
        """
        Initialise the DataOrganiser.
        
        Args:
            raw_dir: Path to raw data directory
            output_dir: Path to output directory
            camera_config: Mapping of camera names to output names
            data_type_config: Mapping of data types to output names
            directory_config: Configuration for directory structure
            split_config: Configuration for data splitting
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.camera_config = camera_config
        self.data_type_config = data_type_config
        self.dir_config = directory_config
        self.split_config = split_config
        
        # Validate configurations
        self._validate_configs()

    def _validate_configs(self) -> None:
        """Validate input configurations."""
        if not self.camera_config:
            raise ValueError("Camera configuration cannot be empty")
        if not self.data_type_config:
            raise ValueError("Data type configuration cannot be empty")
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {self.raw_dir}")

    def validate_raw_structure(self) -> bool:
        """
        Validate that the raw directory has the expected structure.
        
        Returns:
            True if structure is valid
            
        Raises:
            FileNotFoundError: If required directories are missing
        """
        missing_paths = []
        required_dirs = list(self.data_type_config.keys()) + [self.dir_config.label_raw_dir]
        
        for camera in self.camera_config.keys():
            for data_type in required_dirs:
                path = self.raw_dir / camera / data_type
                if not path.exists():
                    missing_paths.append(str(path))
        
        if missing_paths:
            error_msg = f"Required directories missing: {missing_paths}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("Raw directory structure validation passed")
        return True
        
    def _create_directory_structure(self, subset_name: str, cameras: List[str], data_types: List[str]) -> None:
        """
        Create all necessary directories using combinations of cameras and data types.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            cameras: List of camera names
            data_types: List of data types
        """
        for camera, data_type in product(cameras, data_types):
            directory = self.output_dir / subset_name / camera / data_type
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _get_file_paths(self, subset_name: str, camera: str, data_type: str, idx: str) -> Tuple[Path, Path]:
        """
        Get source and destination file paths for a given configuration.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            camera: Camera name
            data_type: Data type
            idx: File index
            
        Returns:
            Tuple of (source_path, destination_path)
        """
        is_image = data_type == self.dir_config.img_raw_dir
        filename = f"{idx}.jpg" if is_image else f"{idx}.txt"
        
        source_file = self.raw_dir / camera / data_type / filename
        dest_file = (self.output_dir / subset_name / 
                    self.camera_config[camera] / 
                    self.data_type_config[data_type] / filename)
        
        return source_file, dest_file
    
    def _process_files(self, subset_name: str, file_list: List[str]) -> None:
        """
        Process and copy files based on configuration mappings.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            file_list: List of file paths to process
        """
        for path in file_list:
            fname = Path(path).name
            idx = fname.split('.')[0]
            
            for camera in self.camera_config.keys():
                for data_type in self.data_type_config.keys():
                    source_file, dest_file = self._get_file_paths(subset_name, camera, data_type, idx)
                    
                    if source_file.exists():
                        shutil.copy2(source_file, dest_file)
                        logger.debug(f"Copied {self.camera_config[camera]}/{self.data_type_config[data_type]}: {source_file.name}")
                    else:
                        logger.warning(f"File not found: {source_file}")
    
    def _process_labels(self, subset_name: str, file_list: List[str]) -> None:
        """
        Process label files with special handling for center points.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            file_list: List of file paths to process
        """
        # Get all indices from the file list
        indices = {Path(path).stem for path in file_list}
        
        for camera in self.camera_config.keys():
            source_file = self.raw_dir / camera / self.dir_config.label_raw_dir / self.dir_config.label_file
            dest_file = (self.output_dir / subset_name / 
                        self.camera_config[camera] / 
                        self.dir_config.label_proc_dir / 
                        self.dir_config.label_file)
            
            if source_file.exists():
                matching_lines = self._filter_label_file(source_file, dest_file, indices)
                logger.info(f"Processed {self.camera_config[camera]} labels: {matching_lines} entries")
            else:
                logger.warning(f"Label source not found: {source_file}")
    
    def _filter_label_file(self, source_file: Path, dest_file: Path, indices: set) -> int:
        """
        Filter label file to include only lines matching the given indices.
        
        Args:
            source_file: Source label file path
            dest_file: Destination label file path
            indices: Set of indices to include
            
        Returns:
            Number of matching lines processed
        """
        matching_lines = 0
        with open(source_file, 'r') as input_file, open(dest_file, 'w') as output_file:
            for line in input_file:
                line = line.strip()
                if line and line.split('.')[0] in indices:
                    output_file.write(line + '\n')
                    matching_lines += 1
        return matching_lines
    
    def organise_subset(self, subset_name: str, file_list: List[str]) -> None:
        """
        Organises files into train/val/test subsets with configurable directory structure.
        
        Args:
            subset_name: 'train', 'val', or 'test'
            file_list: List of file paths to process
        """
        logger.info(f"Organising {subset_name} subset with {len(file_list)} files")
        
        # Create all directory combinations
        data_types = list(self.data_type_config.values()) + [self.dir_config.label_proc_dir]
        self._create_directory_structure(subset_name, list(self.camera_config.values()), data_types)
        
        # Process files for each camera and data type
        self._process_files(subset_name, file_list)
        
        # Process labels separately (special handling)
        self._process_labels(subset_name, file_list)
        
        logger.info(f"Completed organising {subset_name} subset")
    
    def split_data(self, file_pattern: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            file_pattern: Glob pattern to find files
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        # Get all files
        all_files = sorted(glob(file_pattern))
        if not all_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        logger.info(f"Found {len(all_files)} files matching pattern")
        
        # First split: train vs (val + test)
        train_files, temp_files = train_test_split(
            all_files, 
            test_size=(self.split_config.val_ratio + self.split_config.test_ratio),
            random_state=self.split_config.random_state
        )
        
        # Second split: val vs test
        if self.split_config.val_ratio > 0 and self.split_config.test_ratio > 0:
            val_files, test_files = train_test_split(
                temp_files,
                test_size=(self.split_config.test_ratio / (self.split_config.val_ratio + self.split_config.test_ratio)),
                random_state=self.split_config.random_state
            )
        elif self.split_config.val_ratio > 0:
            val_files, test_files = temp_files, []
        else:
            val_files, test_files = [], temp_files
        
        logger.info(f"Data split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        return train_files, val_files, test_files
    
    def organise_all_data(self, file_pattern: str) -> None:
        """
        Complete data organisation pipeline.
        
        Args:
            file_pattern: Glob pattern to find files
        """
        logger.info("Starting data organisation pipeline")

        # Validate raw directory structure
        self.validate_raw_structure()
        
        # Split the data
        train_files, val_files, test_files = self.split_data(file_pattern)
        
        # Organise each subset
        if train_files:
            self.organise_subset('train', train_files)
        if val_files:
            self.organise_subset('val', val_files)
        if test_files:
            self.organise_subset('test', test_files)
        
        logger.info("Data organisation pipeline completed successfully")