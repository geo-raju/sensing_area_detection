"""
Data organisation module for processing and organising image datasets.
Handles train/val/test splits and file organisation with configurable structure.
"""

import os
import shutil
import logging
from itertools import product
from typing import List, Dict, Tuple
from glob import glob
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataOrganiser:
    """
    Class to handle data organisation and splitting for machine learning datasets.
    """
    
    def __init__(self, raw_dir: str, output_dir: str, camera_config: Dict, data_type_config: Dict):
        """
        Initialise the DataOrganiser.
        
        Args:
            raw_dir: Path to raw data directory
            output_dir: Path to output directory
            camera_config: Mapping of camera names to output names
            data_type_config: Mapping of data types to output names
        """
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.camera_config = camera_config
        self.data_type_config = data_type_config
        
    def validate_raw_structure(self) -> bool:
        """
        Validate that the raw directory has the expected structure.
        
        Returns:
            True if structure is valid, False otherwise
            
        Raises:
            FileNotFoundError: If required directories are missing
        """
        missing_paths = []
        
        for camera in self.camera_config.keys():
            for data_type in self.data_type_config.keys():
                path = os.path.join(self.raw_dir, camera, data_type)
                if not os.path.exists(path):
                    missing_paths.append(path)
            
            # Check for label directory
            label_path = os.path.join(self.raw_dir, camera, 'laserptGT')
            if not os.path.exists(label_path):
                missing_paths.append(label_path)
        
        if missing_paths:
            logger.error(f"Missing required directories: {missing_paths}")
            logger.error("Cannot proceed with data organization")
            raise FileNotFoundError(f"Required directories missing: {missing_paths}")
        
        logger.info("Raw directory structure validation passed")
        return True
        
    def create_directory_structure(self, subset_name: str, cameras: List[str], data_types: List[str]) -> None:
        """
        Create all necessary directories using combinations of cameras and data types.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            cameras: List of camera names
            data_types: List of data types
        """
        for camera, data_type in product(cameras, data_types):
            directory = os.path.join(self.output_dir, subset_name, camera, data_type)
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def process_files(self, subset_name: str, file_list: List[str]) -> None:
        """
        Process and copy files based on configuration mappings.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            file_list: List of file paths to process
        """
        for path in file_list:
            fname = os.path.basename(path)
            idx = fname.split('.')[0]
            
            for camera in self.camera_config.keys():
                for data_type in self.data_type_config.keys():
                    source_file = os.path.join(
                        self.raw_dir, camera, data_type, 
                        fname if data_type == 'laser_off' else f"{idx}.txt"
                    )
                    
                    dest_file = os.path.join(
                        self.output_dir, subset_name,
                        self.camera_config[camera],
                        self.data_type_config[data_type],
                        fname if data_type == 'laser_off' else f"{idx}.txt"
                    )
                    
                    if os.path.exists(source_file):
                        shutil.copy(source_file, dest_file)
                        logger.info(f"Copied {self.camera_config[camera]}/{self.data_type_config[data_type]}: {os.path.basename(source_file)}")
                    else:
                        logger.warning(f"File not found: {source_file}")
    
    def process_labels(self, subset_name: str, file_list: List[str]) -> None:
        """
        Process label files with special handling for center points.
        
        Args:
            subset_name: Name of the subset (train/val/test)
            file_list: List of file paths to process
        """
        # Get all indices from the file list
        indices = {os.path.basename(path).split('.')[0] for path in file_list}
        
        for camera in self.camera_config.keys():
            source_file = os.path.join(self.raw_dir, camera, 'laserptGT', 'CenterPt.txt')
            dest_file = os.path.join(
                self.output_dir, subset_name, 
                self.camera_config[camera], 'labels', 'CenterPt.txt'
            )
            
            if os.path.exists(source_file):
                with open(source_file, 'r') as i_f:
                    with open(dest_file, 'w') as o_f:
                        matching_lines = 0
                        for line in i_f:
                            line = line.strip()
                            if line and line.split('.')[0] in indices:
                                o_f.write(line + '\n')
                                matching_lines += 1
                logger.info(f"Processed {self.camera_config[camera]} labels: {matching_lines} entries")
            else:
                logger.warning(f"Label source not found: {source_file}")
    
    def organise_subset(self, subset_name: str, file_list: List[str]) -> None:
        """
        Organises files into train/val/test subsets with configurable directory structure.
        
        Args:
            subset_name: 'train', 'val', or 'test'
            file_list: List of file paths to process
        """
        logger.info(f"Organising {subset_name} subset with {len(file_list)} files")
        
        # Create all directory combinations
        data_types = list(self.data_type_config.values()) + ['labels']
        self.create_directory_structure(subset_name, self.camera_config.values(), data_types)
        
        # Process files for each camera and data type
        self.process_files(subset_name, file_list)
        
        # Process labels separately (special handling)
        self.process_labels(subset_name, file_list)
        
        logger.info(f"Completed organising {subset_name} subset")
    
    def split_data(self, file_pattern: str, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, test_ratio: float = 0.15, 
                   random_state: int = 42) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            file_pattern: Glob pattern to find files
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Get all files
        all_files = sorted(glob(file_pattern))
        if not all_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        logger.info(f"Found {len(all_files)} files matching pattern")
        
        # First split: train vs (val + test)
        train_files, temp_files = train_test_split(
            all_files, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Second split: val vs test
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_state
        )
        
        logger.info(f"Data split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        return train_files, val_files, test_files
    
    def organise_all_data(self, file_pattern: str, train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, test_ratio: float = 0.15,
                         random_state: int = 42) -> None:
        """
        Complete data organisation pipeline.
        
        Args:
            file_pattern: Glob pattern to find files
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
        """
        logger.info("Starting data organisation pipeline")

        # Validate raw directory structure
        self.validate_raw_structure()
        
        # Split the data
        train_files, val_files, test_files = self.split_data(
            file_pattern, train_ratio, val_ratio, test_ratio, random_state
        )
        
        # Organise each subset
        self.organise_subset('train', train_files)
        self.organise_subset('val', val_files)
        self.organise_subset('test', test_files)
        
        logger.info("Data organisation pipeline completed successfully")