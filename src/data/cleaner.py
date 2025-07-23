import shutil
import logging
from pathlib import Path
from typing import Set, Dict
from dataclasses import dataclass
from enum import Enum

from config.data_config import (
    CAMERA_CONFIG, DATA_TYPE_CONFIG,
    LEFT_CAM_PROC_DIR, RIGHT_CAM_PROC_DIR,
    IMG_RAW_DIR, IMG_PROC_DIR,
    PROBE_RAW_DIR, PROBE_PROC_DIR,
    LABEL_RAW_DIR, LABEL_PROC_DIR,
    LABEL_FILE,
    DEPTH_RAW_DIR, DEPTH_PROC_DIR
)
from src.data.processor import FileProcessor

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Enumeration of different file types to process."""
    IMAGE = ('.jpg', IMG_RAW_DIR, IMG_PROC_DIR)
    PROBE = ('.txt', PROBE_RAW_DIR, PROBE_PROC_DIR)
    DEPTH = ('.npy', DEPTH_RAW_DIR, DEPTH_PROC_DIR)
    
    def __init__(self, extension: str, raw_dir: str, proc_dir: str):
        self.extension = extension
        self.raw_dir = raw_dir
        self.proc_dir = proc_dir


@dataclass
class ProcessingStats:
    """Container for processing statistics."""
    copied: int = 0
    skipped: int = 0
    removed: int = 0
    
    def add_copied(self, count: int = 1) -> None:
        """Add to copied count."""
        self.copied += count
        
    def add_skipped(self, count: int = 1) -> None:
        """Add to skipped count."""
        self.skipped += count
        
    def add_removed(self, count: int = 1) -> None:
        """Add to removed count."""
        self.removed += count
    
    def total_processed(self) -> int:
        """Return total number of files processed."""
        return self.copied + self.skipped + self.removed


@dataclass
class CopyResult:
    """Result of file copying operation."""
    stats: ProcessingStats
    valid_files: Set[str]


class DirectoryManager:
    """Handles directory operations for data cleaning."""
    
    def __init__(self, cleaned_dir: Path, cameras: Dict[str, str], data_types: Dict[str, str]):
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras
        self.data_types = data_types
    
    def create_directory_structure(self) -> None:
        """Create the cleaned data directory structure."""
        try:
            for camera_proc in self.cameras.values():
                subdirs = list(self.data_types.values()) + [LABEL_PROC_DIR]
                FileProcessor.create_directories(self.cleaned_dir / camera_proc, subdirs)
                
            logger.info("Created directory structure for data cleaning")
        except Exception as e:
            logger.error(f"Failed to create directory structure: {e}")
            raise


class LabelFileManager:
    """Handles operations on label files."""
    
    def __init__(self, raw_dir: Path, cleaned_dir: Path, cameras: Dict[str, str],
                 label_dir: str, label_file: str):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras
        self.label_dir = label_dir
        self.label_file = label_file
    
    def clean_label_files(self) -> Dict[str, Dict[str, str]]:
        """Clean label files for both cameras and return valid filenames."""
        camera_files = {}
        
        for camera_raw, camera_proc in self.cameras.items():
            source_file = self.raw_dir / camera_raw / self.label_dir / self.label_file
            
            try:
                valid_files_dict = FileProcessor.get_valid_label_files(source_file)
                camera_files[camera_proc] = valid_files_dict
                
                logger.info(f"Loaded {len(valid_files_dict)} entries for {camera_proc} camera labels")
            except Exception as e:
                logger.error(f"Failed to process label file for {camera_proc}: {e}")
                camera_files[camera_proc] = {}
        
        return camera_files
    
    def update_label_files(self, files_dict: Dict[str, Dict[str, str]], 
                          valid_items: Set[str]) -> None:
        """Update label files to keep only specified items."""
        for camera_proc in self.cameras.values():
            valid_lines = []
            label_path = self.cleaned_dir / camera_proc / LABEL_PROC_DIR / self.label_file
            
            # Collect valid lines for this camera
            for valid_item in valid_items:
                filename_key = f"{valid_item}.jpg"
                if filename_key in files_dict[camera_proc]:
                    valid_lines.append(files_dict[camera_proc][filename_key])
            
            try:
                FileProcessor.write_file(label_path, sorted(valid_lines))
                logger.info(f"Updated label file for {camera_proc} with {len(valid_lines)} entries")
            except Exception as e:
                logger.error(f"Failed to update label file for {camera_proc}: {e}")


class FileSetAnalyzer:
    """Analyzes and compares file sets between cameras."""
    
    @staticmethod
    def get_common_files(files_dict: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Get files common to both camera datasets."""
        if LEFT_CAM_PROC_DIR not in files_dict or RIGHT_CAM_PROC_DIR not in files_dict:
            logger.error("Missing camera data in files dictionary")
            return {}
        
        left_files = set(files_dict[LEFT_CAM_PROC_DIR].keys())
        right_files = set(files_dict[RIGHT_CAM_PROC_DIR].keys())
        common_filenames = left_files & right_files
        
        common_files = {}
        for camera_proc in [LEFT_CAM_PROC_DIR, RIGHT_CAM_PROC_DIR]:
            common_files[camera_proc] = {
                filename: files_dict[camera_proc][filename] 
                for filename in common_filenames
            }
        
        logger.info(f"Found {len(common_filenames)} common files between cameras")
        return common_files


class FileCopier:
    """Handles copying files with validation."""
    
    def __init__(self, raw_dir: Path, cleaned_dir: Path, cameras: Dict[str, str]):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras

    def _copy_files_for_camera(self, filenames: Set[str], source_dir: Path, 
                              dest_dir: Path, file_type: FileType) -> CopyResult:
        """Copy files for a single camera and return results."""
        stats = ProcessingStats()
        valid_files = set()
        
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return CopyResult(stats, valid_files)
        
        for filename in filenames:
            source_file = source_dir / f"{filename}{file_type.extension}"
            
            if not source_file.exists():
                logger.debug(f"Source file not found: {source_file}")
                stats.add_skipped()
                continue
        
            # Copy file
            try:
                dest_file = dest_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                stats.add_copied()
                valid_files.add(filename)
                logger.debug(f"Successfully copied: {source_file} -> {dest_file}")
            except Exception as e:
                logger.error(f"Failed to copy {source_file}: {e}")
                stats.add_skipped()
        
        return CopyResult(stats, valid_files)

    def copy_files_by_type(self, filenames: Set[str], file_type: FileType) -> Set[str]:
        """Copy files from source to destination and return successfully processed files."""
        successfully_processed = set()
        total_stats = ProcessingStats()
        
        for camera_raw, camera_proc in self.cameras.items():
            source_dir = self.raw_dir / camera_raw / file_type.raw_dir
            dest_dir = self.cleaned_dir / camera_proc / file_type.proc_dir
            
            result = self._copy_files_for_camera(filenames, source_dir, dest_dir, file_type)
            
            # For the first camera, initialize the set. For subsequent cameras,
            # only keep files that were successfully copied for ALL cameras
            if not successfully_processed:
                successfully_processed = result.valid_files.copy()
            else:
                successfully_processed &= result.valid_files
            
            total_stats.add_copied(result.stats.copied)
            total_stats.add_skipped(result.stats.skipped)
            
            self._log_copy_results(camera_raw, file_type, result.stats)
        
        logger.info(f"Successfully processed {len(successfully_processed)} {file_type.name.lower()} files across all cameras")
        return successfully_processed
    
    def _log_copy_results(self, camera_name: str, file_type: FileType, stats: ProcessingStats) -> None:
        """Log the results of file copying operation."""
        status_msg = f"Copied {stats.copied} {file_type.extension} files from {camera_name}/{file_type.raw_dir}"
        if stats.skipped > 0:
            status_msg += f" (skipped {stats.skipped} files)"
        logger.info(status_msg)


class ProbeFileValidator:
    """Specialized validator for probe files."""
    
    def __init__(self, raw_dir: Path, cameras: Dict[str, str]):
        self.raw_dir = raw_dir
        self.cameras = cameras
    
    def get_valid_probe_indices(self, file_indices: Set[str]) -> Set[str]:
        """Process probe axis files and return indices of files valid across all cameras."""
        valid_indices = set()
        
        for index in file_indices:
            # Check if ALL cameras have valid probe files for this index
            is_valid_for_all_cameras = True
            
            for camera in self.cameras.keys():
                probe_file = self.raw_dir / camera / PROBE_RAW_DIR / f"{index}.txt"
                
                # First check if file exists, then check for NaN values
                if not probe_file.exists():
                    logger.debug(f"Probe file not found: {probe_file}")
                    is_valid_for_all_cameras = False
                    break
                
                if not FileProcessor.check_file_for_nan(probe_file):
                    logger.debug(f"NaN values found in probe file: {probe_file}")
                    is_valid_for_all_cameras = False
                    break
            
            if is_valid_for_all_cameras:
                valid_indices.add(index)
            else:
                logger.debug(f"Index {index} excluded - invalid probe files in one or more cameras")
        
        logger.info(f"Valid probe files found for {len(valid_indices)} out of {len(file_indices)} indices")
        return valid_indices


class DataCleaner:
    """Main class to handle data cleaning operations."""
    
    def __init__(self, raw_dir: Path, cleaned_dir: Path):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cameras = CAMERA_CONFIG
        self.data_types = DATA_TYPE_CONFIG
        
        # Initialize managers
        self.directory_manager = DirectoryManager(cleaned_dir, self.cameras, self.data_types)
        self.label_manager = LabelFileManager(
            raw_dir, cleaned_dir, self.cameras, LABEL_RAW_DIR, LABEL_FILE
        )
        self.file_copier = FileCopier(raw_dir, cleaned_dir, self.cameras)
        self.probe_validator = ProbeFileValidator(raw_dir, self.cameras)
    
    def clean_data(self) -> None:
        """Execute the complete data cleaning pipeline."""
        logger.info("Starting data cleaning pipeline")
        
        try:
            # Step 1: Create directory structure
            logger.info("Step 1: Creating directory structure")
            self.directory_manager.create_directory_structure()
            
            # Step 2: Clean label files and find common files
            logger.info("Step 2: Processing label files")
            common_files = self._process_label_files()
            
            if not common_files or not any(common_files.values()):
                logger.error("No common files found between cameras!")
                return
            
            # Step 3: Validate probe files and get valid indices
            logger.info("Step 3: Validating probe files")
            file_indices = self._extract_file_indices(common_files)
            valid_file_indices = self.probe_validator.get_valid_probe_indices(file_indices)
            
            if not valid_file_indices:
                logger.error("No valid probe files found!")
                return
            
            # Step 4: Update label files with valid entries
            logger.info("Step 4: Updating label files")
            self.label_manager.update_label_files(common_files, valid_file_indices)
            
            # Step 5: Copy all file types
            logger.info("Step 5: Copying files")
            final_valid_files = self._copy_all_file_types(valid_file_indices)
            
            logger.info(f"Data cleaning completed successfully! Final files processed: {len(final_valid_files)}")
            
        except Exception as e:
            logger.error(f"Data cleaning pipeline failed: {e}")
            raise
    
    def _process_label_files(self) -> Dict[str, Dict[str, str]]:
        """Process label files and return common files."""
        files_dict = self.label_manager.clean_label_files()
        common_files = FileSetAnalyzer.get_common_files(files_dict)
        return common_files
    
    def _extract_file_indices(self, common_files: Dict[str, Dict[str, str]]) -> Set[str]:
        """Extract file indices from common files."""
        # Get filenames from any camera (they should be the same due to common files)
        first_camera = next(iter(common_files.values()))
        filenames = set(first_camera.keys())
        file_indices = FileProcessor.get_file_indices(filenames)
        
        logger.info(f"Extracted {len(file_indices)} file indices from common files")
        return file_indices

    def _copy_all_file_types(self, valid_files: Set[str]) -> Set[str]:
        """Copy all file types and return final valid file set."""
        final_valid_files = valid_files.copy()
        
        file_types = [FileType.IMAGE, FileType.PROBE, FileType.DEPTH]
        
        for i, file_type in enumerate(file_types, 1):
            logger.info(f"Step 5.{i}: Copying {file_type.name.lower()} files")
            
            try:
                successfully_copied = self.file_copier.copy_files_by_type(valid_files, file_type)
                # Keep intersection to ensure all file types are present for each valid file
                final_valid_files &= successfully_copied
                
                if not final_valid_files:
                    logger.warning(f"No files remain after copying {file_type.name.lower()} files")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to copy {file_type.name.lower()} files: {e}")
                raise
        
        logger.info(f"Final dataset contains {len(final_valid_files)} complete file sets")
        return final_valid_files