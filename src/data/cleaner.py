import shutil
import logging
from pathlib import Path
from typing import Set, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from config.data_config import (
    CAMERA_CONFIG, DATA_TYPE_CONFIG,
    LEFT_CAM_RAW_DIR, LEFT_CAM_PROC_DIR,
    RIGHT_CAM_RAW_DIR, RIGHT_CAM_PROC_DIR,
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
    IMAGE = ('.jpg', IMG_RAW_DIR, IMG_PROC_DIR, False)
    PROBE = ('.txt', PROBE_RAW_DIR, PROBE_PROC_DIR, True)
    DEPTH = ('.npy', DEPTH_RAW_DIR, DEPTH_PROC_DIR, False)
    
    def __init__(self, extension: str, raw_dir: str, proc_dir: str, check_nan: bool):
        self.extension = extension
        self.raw_dir = raw_dir
        self.proc_dir = proc_dir
        self.check_nan = check_nan


@dataclass
class ProcessingStats:
    """Container for processing statistics."""
    copied: int = 0
    skipped: int = 0
    removed: int = 0
    
    def add_copied(self, count: int) -> None:
        self.copied += count
        
    def add_skipped(self, count: int) -> None:
        self.skipped += count
        
    def add_removed(self, count: int) -> None:
        self.removed += count


@dataclass
class CleaningResult:
    """Container for cleaning operation results."""
    valid_files: Set[str]
    stats: ProcessingStats
    
    
class DirectoryManager:
    """Handles directory operations for data cleaning."""
    
    def __init__(self, cleaned_dir: Path, cameras: Dict[str, str], data_types: Dict[str, str]):
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras
        self.data_types = data_types
    
    def create_directory_structure(self) -> None:
        """Create the cleaned data directory structure."""
        for camera_proc in self.cameras.values():
            subdirs = list(self.data_types.values()) + [LABEL_PROC_DIR]
            FileProcessor.create_directories(self.cleaned_dir / camera_proc, subdirs)
            
        logger.info("Created directory structure for data cleaning")


class LabelFileManager:
    """Handles operations on label files."""
    
    def __init__(self, raw_dir: Path, cleaned_dir: Path, cameras: Dict[str, str],
                 label_dir: str, label_file: str):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras
        self.label_dir = label_dir
        self.label_file = label_file
    
    def clean_label_files(self) -> Tuple[Set[str], Set[str]]:
        """Clean label files for both cameras and return valid filenames."""
        camera_files = {}
        
        camera_mappings = [
            (LEFT_CAM_RAW_DIR, LEFT_CAM_PROC_DIR, 'left'),
            (RIGHT_CAM_RAW_DIR, RIGHT_CAM_PROC_DIR, 'right')
        ]
        
        for camera_raw, camera_proc, camera_name in camera_mappings:
            source_file = self.raw_dir / camera_raw / self.label_dir / self.label_file
            dest_file = self.cleaned_dir / camera_proc / LABEL_PROC_DIR / self.label_file
            
            valid_files, count = FileProcessor.filter_label_file(source_file, dest_file, set())
            camera_files[camera_name] = valid_files
            
            logger.info(f"Cleaned {count} entries for {camera_name} camera labels")
        
        return camera_files.get('left', set()), camera_files.get('right', set())
    
    def update_label_files(self, valid_items: Set[str], is_filename_filter: bool = True) -> None:
        """Update label files to keep only specified items."""
        for camera_proc in self.cameras.values():
            label_path = self.cleaned_dir / camera_proc / LABEL_PROC_DIR / self.label_file
            temp_path = label_path.with_suffix('.tmp')
            
            try:
                FileProcessor.filter_label_file(label_path, temp_path, valid_items, is_filename_filter)
                temp_path.replace(label_path)
                logger.debug(f"Updated label file for {camera_proc}")
            except Exception as e:
                logger.error(f"Failed to update label file for {camera_proc}: {e}")
                if temp_path.exists():
                    temp_path.unlink()


class FileSetAnalyzer:
    """Analyzes and compares file sets between cameras."""
    
    @staticmethod
    def get_common_files(left_files: Set[str], right_files: Set[str]) -> Set[str]:
        """Get files common to both camera datasets."""
        common_files = left_files.intersection(right_files)
        
        left_only = left_files - right_files
        right_only = right_files - left_files
        
        if left_only:
            logger.info(f"Files only in left camera: {len(left_only)}")
        if right_only:
            logger.info(f"Files only in right camera: {len(right_only)}")
        
        logger.info(f"Common files between cameras: {len(common_files)}")
        return common_files


class FileCopier:
    """Handles copying files with validation."""
    
    def __init__(self, raw_dir: Path, cleaned_dir: Path, cameras: Dict[str, str]):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras
    
    def copy_files_by_type(self, filenames: Set[str], file_type: FileType) -> CleaningResult:
        """Copy files from source to destination with optional validation."""
        successfully_processed = set()
        total_stats = ProcessingStats()
        
        for camera_raw, camera_proc in self.cameras.items():
            source_dir = self.raw_dir / camera_raw / file_type.raw_dir
            dest_dir = self.cleaned_dir / camera_proc / file_type.proc_dir
            
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue
            
            result = self._copy_camera_files(filenames, source_dir, dest_dir, file_type)
            successfully_processed.update(result.valid_files)
            total_stats.add_copied(result.stats.copied)
            total_stats.add_skipped(result.stats.skipped)
            
            self._log_copy_results(camera_raw, file_type, result.stats)
        
        return CleaningResult(successfully_processed, total_stats)
    
    def _copy_camera_files(self, filenames: Set[str], source_dir: Path, 
                          dest_dir: Path, file_type: FileType) -> CleaningResult:
        """Copy files for a single camera."""
        valid_files = set()
        stats = ProcessingStats()
        
        for filename in filenames:
            source_file, file_key = self._determine_source_file(filename, source_dir, file_type)
            
            if not source_file.exists():
                continue
            
            # Validate file if needed
            if file_type.check_nan and not FileProcessor.check_file_for_nan(source_file):
                stats.add_skipped(1)
                continue
            
            # Copy file
            try:
                dest_file = dest_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                stats.add_copied(1)
                valid_files.add(file_key)
            except Exception as e:
                logger.error(f"Failed to copy {source_file}: {e}")
                stats.add_skipped(1)
        
        return CleaningResult(valid_files, stats)
    
    def _determine_source_file(self, filename: str, source_dir: Path, 
                              file_type: FileType) -> Tuple[Path, str]:
        """Determine the source file path and key based on file type."""
        if file_type == FileType.IMAGE:
            return source_dir / filename, filename
        else:
            # For non-image files, use index-based naming
            index = filename.split('.')[0] if '.' in filename else filename
            source_file = source_dir / f"{index}{file_type.extension}"
            return source_file, index
    
    def _log_copy_results(self, camera_name: str, file_type: FileType, stats: ProcessingStats) -> None:
        """Log the results of file copying operation."""
        status_msg = f"Copied {stats.copied} {file_type.extension} files from {camera_name}/{file_type.raw_dir}"
        if stats.skipped > 0:
            status_msg += f" (skipped {stats.skipped} invalid files)"
        logger.info(status_msg)


class FileCleanupManager:
    """Handles cleanup of invalid files."""
    
    def __init__(self, cleaned_dir: Path, cameras: Dict[str, str]):
        self.cleaned_dir = cleaned_dir
        self.cameras = cameras
    
    def cleanup_invalid_files(self, valid_indices: Set[str]) -> ProcessingStats:
        """Remove image files without valid probe axis files."""
        valid_filenames = {f"{idx}.jpg" for idx in valid_indices}
        total_stats = ProcessingStats()
        
        for camera_proc in self.cameras.values():
            images_dir = self.cleaned_dir / camera_proc / IMG_PROC_DIR
            if not images_dir.exists():
                continue
            
            removed_count = self._remove_invalid_images(images_dir, valid_filenames)
            total_stats.add_removed(removed_count)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} invalid image files from {camera_proc}")
        
        return total_stats
    
    def _remove_invalid_images(self, images_dir: Path, valid_filenames: Set[str]) -> int:
        """Remove invalid image files from a directory."""
        removed_count = 0
        
        try:
            for image_file in images_dir.glob('*.jpg'):
                if image_file.name not in valid_filenames:
                    image_file.unlink()
                    removed_count += 1
        except Exception as e:
            logger.error(f"Error during cleanup in {images_dir}: {e}")
        
        return removed_count


class DataCleaner:
    """Main class to handle data cleaning operations."""
    
    def __init__(self, raw_dir: Path, cleaned_dir: Path):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cameras = CAMERA_CONFIG
        self.data_types = DATA_TYPE_CONFIG
        self.label_dir = LABEL_RAW_DIR
        self.label_file = LABEL_FILE
        
        # Initialize managers
        self.directory_manager = DirectoryManager(cleaned_dir, self.cameras, self.data_types)
        self.label_manager = LabelFileManager(
            raw_dir, cleaned_dir, self.cameras, self.label_dir, self.label_file
        )
        self.file_copier = FileCopier(raw_dir, cleaned_dir, self.cameras)
        self.cleanup_manager = FileCleanupManager(cleaned_dir, self.cameras)
    
    def clean_data(self) -> None:
        """Execute the complete data cleaning pipeline."""
        logger.info("Starting data cleaning pipeline")
        
        try:
            # Step 1: Create directory structure
            self.directory_manager.create_directory_structure()
            
            # Step 2: Clean label files and find common files
            logger.info("Step 1: Cleaning label files")
            common_files = self._process_label_files()
            
            if not common_files:
                logger.error("No common files found between cameras!")
                return
            
            # Step 3: Process different file types
            logger.info("Step 2: Filtering label files for common files")
            self.label_manager.update_label_files(common_files)
            
            # Step 4: Copy files and handle validation
            final_valid_files = self._copy_all_file_types(common_files)
            
            logger.info(f"Data cleaning completed! Total files processed: {len(final_valid_files)}")
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise
    
    def _process_label_files(self) -> Set[str]:
        """Process label files and return common files."""
        left_files, right_files = self.label_manager.clean_label_files()
        common_files = FileSetAnalyzer.get_common_files(left_files, right_files)
        return common_files
    
    def _copy_all_file_types(self, common_files: Set[str]) -> Set[str]:
        """Copy all file types and return final valid file set."""
        # Copy image files
        logger.info("Step 3: Copying image files")
        self.file_copier.copy_files_by_type(common_files, FileType.IMAGE)
        
        # Handle probe files with NaN checking
        logger.info("Step 4: Copying probe axis files (checking for NaN)")
        file_indices = FileProcessor.get_file_indices(common_files)
        probe_result = self.file_copier.copy_files_by_type(file_indices, FileType.PROBE)
        valid_probe_indices = probe_result.valid_files
        
        # Update files based on valid probe indices
        if len(valid_probe_indices) < len(file_indices):
            logger.info("Step 5: Updating files after probe filtering")
            self.label_manager.update_label_files(valid_probe_indices, is_filename_filter=False)
            self.cleanup_manager.cleanup_invalid_files(valid_probe_indices)
        
        # Copy depth files
        logger.info("Step 6: Copying depth label files")
        self.file_copier.copy_files_by_type(valid_probe_indices, FileType.DEPTH)
        
        return valid_probe_indices