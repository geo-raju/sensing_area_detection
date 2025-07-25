import shutil
import logging
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager
import json
import time

from config.data_config import (
    RAW_DIR_PATH, CLEAN_DIR_PATH,
    CAMERA_CONFIG, DATA_TYPE_CONFIG,
    LEFT_CAM_PROC_DIR, RIGHT_CAM_PROC_DIR,
    IMG_RAW_DIR, IMG_PROC_DIR,
    PROBE_RAW_DIR, PROBE_PROC_DIR,
    LABEL_RAW_DIR, LABEL_PROC_DIR, LABEL_FILE,
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
    """Container for processing statistics with thread-safe operations."""
    copied: int = 0
    skipped: int = 0
    removed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_copied(self, count: int = 1) -> None:
        """Thread-safe add to copied count."""
        with self._lock:
            self.copied += count
        
    def add_skipped(self, count: int = 1) -> None:
        """Thread-safe add to skipped count."""
        with self._lock:
            self.skipped += count
        
    def add_removed(self, count: int = 1) -> None:
        """Thread-safe add to removed count."""
        with self._lock:
            self.removed += count
    
    def total_processed(self) -> int:
        """Return total number of files processed."""
        with self._lock:
            return self.copied + self.skipped + self.removed
    
    def merge(self, other: 'ProcessingStats') -> None:
        """Merge another stats object into this one."""
        with self._lock:
            self.copied += other.copied
            self.skipped += other.skipped
            self.removed += other.removed


@dataclass
class CopyResult:
    """Result of file copying operation."""
    stats: ProcessingStats
    valid_files: Set[str]
    errors: List[str] = field(default_factory=list)


@dataclass
class CleaningProgress:
    """Tracks progress of the cleaning operation."""
    total_files: int = 0
    processed_files: int = 0
    current_stage: str = ""
    start_time: float = field(default_factory=time.time)
    
    def update_progress(self, processed: int, stage: str = None) -> None:
        """Update progress information."""
        self.processed_files = processed
        if stage:
            self.current_stage = stage
    
    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


class DirectoryManager:
    """Handles directory operations for data cleaning."""
    
    def __init__(self, base_dir: Path, cameras: Dict[str, str], 
                 data_types: Dict[str, str], label_dir: Path):
        self.base_dir = base_dir
        self.cameras = cameras
        self.data_types = data_types
        self.label_dir = label_dir
        self._creation_lock = threading.Lock()
    
    def create_directory_structure(self) -> None:
        """Create the directory structure with atomic operations."""
        with self._creation_lock:
            try:
                for camera_proc in self.cameras.values():
                    subdirs = list(self.data_types.values()) + [self.label_dir]
                    FileProcessor.create_directories(self.base_dir / camera_proc, subdirs)
                    
                logger.info(f"Created directory structure in {self.base_dir}")
                
            except Exception as e:
                logger.error(f"Failed to create directory structure: {e}")
                raise


class LabelFileManager:
    """Handles operations on label files with improved memory management."""
    
    def __init__(self, source_dir: Path, dest_dir: Path, cameras: Dict[str, str],
                 label_raw_dir: Path, label_dir: Path, label_file: Path):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.cameras = cameras
        self.label_raw_dir = label_raw_dir
        self.label_dir = label_dir
        self.label_file = label_file
    
    def clean_label_files(self) -> Dict[str, Dict[str, str]]:
        """Clean label files for both cameras with parallel processing."""
        camera_files = {}
        
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            # Submit tasks for each camera
            future_to_camera = {
                executor.submit(self._process_camera_labels, camera_raw, camera_proc): camera_proc
                for camera_raw, camera_proc in self.cameras.items()
            }
            
            # Collect results
            for future in as_completed(future_to_camera):
                camera_proc = future_to_camera[future]
                try:
                    valid_files_dict = future.result()
                    camera_files[camera_proc] = valid_files_dict
                    logger.info(f"Loaded {len(valid_files_dict)} entries for {camera_proc} camera labels")
                except Exception as e:
                    logger.error(f"Failed to process label file for {camera_proc}: {e}")
                    camera_files[camera_proc] = {}
        
        return camera_files
    
    def _process_camera_labels(self, camera_raw: str, camera_proc: str) -> Dict[str, str]:
        """Process labels for a single camera."""
        source_file = self.source_dir / camera_raw / self.label_raw_dir / self.label_file
        
        try:
            return FileProcessor.get_valid_label_files(source_file)
        except Exception as e:
            logger.error(f"Failed to process label file for {camera_proc}: {e}")
            return {}
    
    def update_label_files(self, files_dict: Dict[str, Dict[str, str]], 
                          valid_items: Set[str]) -> None:
        """Update label files to keep only specified items."""
        FileProcessor.process_label_files(
            indices=valid_items,
            cameras=self.cameras,
            source_dir=self.dest_dir,
            dest_dir=self.dest_dir,
            label_dir=self.label_dir,
            label_file=self.label_file,
            files_dict=files_dict,
            action_name="Updated"
        )


class FileSetAnalyzer:
    """Analyzes and compares file sets between cameras with optimization."""
    
    @staticmethod
    def get_common_files(files_dict: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Get files common to both camera datasets using set operations."""
        if LEFT_CAM_PROC_DIR not in files_dict or RIGHT_CAM_PROC_DIR not in files_dict:
            logger.error("Missing camera data in files dictionary")
            return {}
        
        # Use set intersection for better performance
        left_files = set(files_dict[LEFT_CAM_PROC_DIR].keys())
        right_files = set(files_dict[RIGHT_CAM_PROC_DIR].keys())
        common_filenames = left_files & right_files
        
        if not common_filenames:
            logger.warning("No common files found between cameras")
            return {}
        
        # Build result dictionary efficiently
        common_files = {
            camera_proc: {
                filename: files_dict[camera_proc][filename] 
                for filename in common_filenames
            }
            for camera_proc in [LEFT_CAM_PROC_DIR, RIGHT_CAM_PROC_DIR]
        }
        
        logger.info(f"Found {len(common_filenames)} common files between cameras")
        return common_files


class FileCopier:
    """Handles copying files with validation and parallel processing."""
    
    def __init__(self, source_dir: Path, dest_dir: Path, cameras: Dict[str, str], 
                 max_workers: int = 4):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.cameras = cameras
        self.max_workers = max_workers

    def _copy_single_file(self, filename: str, source_file: Path, 
                         dest_file: Path) -> Tuple[bool, Optional[str]]:
        """Copy a single file and return success status and any error."""
        try:
            if not source_file.exists():
                return False, f"Source file not found: {source_file}"
            
            # Ensure destination directory exists
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file with metadata preservation
            shutil.copy2(source_file, dest_file)
            logger.debug(f"Successfully copied: {source_file} -> {dest_file}")
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to copy {source_file}: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _copy_files_for_camera_parallel(self, filenames: Set[str], source_subdir: Path, 
                                      dest_subdir: Path, file_type: FileType) -> CopyResult:
        """Copy files for a single camera with parallel processing."""
        stats = ProcessingStats()
        valid_files = set()
        errors = []
        
        if not source_subdir.exists():
            error_msg = f"Source directory not found: {source_subdir}"
            logger.warning(error_msg)
            return CopyResult(stats, valid_files, [error_msg])
        
        # Prepare file operations
        file_operations = []
        for filename in filenames:
            source_file = source_subdir / f"{filename}{file_type.extension}"
            dest_file = dest_subdir / f"{filename}{file_type.extension}"
            file_operations.append((filename, source_file, dest_file))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_filename = {
                executor.submit(self._copy_single_file, filename, source_file, dest_file): filename
                for filename, source_file, dest_file in file_operations
            }
            
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    success, error = future.result()
                    if success:
                        stats.add_copied()
                        valid_files.add(filename)
                    else:
                        stats.add_skipped()
                        if error:
                            errors.append(error)
                except Exception as e:
                    stats.add_skipped()
                    error_msg = f"Unexpected error processing {filename}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        return CopyResult(stats, valid_files, errors)

    def copy_files_by_type(self, filenames: Set[str], file_type: FileType) -> Set[str]:
        """Copy files from source to destination with parallel processing."""
        successfully_processed = set()
        total_stats = ProcessingStats()
        all_errors = []
        
        # Process each camera
        for camera_raw, camera_proc in self.cameras.items():
            source_subdir = self._get_source_path(camera_raw, file_type)
            dest_subdir = self.dest_dir / camera_proc / file_type.proc_dir
            
            result = self._copy_files_for_camera_parallel(
                filenames, source_subdir, dest_subdir, file_type
            )
            
            # For the first camera, initialize the set. For subsequent cameras,
            # only keep files that were successfully copied for ALL cameras
            if not successfully_processed:
                successfully_processed = result.valid_files.copy()
            else:
                successfully_processed &= result.valid_files
            
            total_stats.merge(result.stats)
            all_errors.extend(result.errors)
            
            self._log_copy_results(camera_raw, file_type, result.stats)
        
        if all_errors:
            logger.warning(f"Encountered {len(all_errors)} errors during file copying")
            # Log first few errors as examples
            for error in all_errors[:5]:
                logger.debug(error)
        
        logger.info(f"Successfully processed {len(successfully_processed)} {file_type.name.lower()} files across all cameras")
        return successfully_processed
    
    def _get_source_path(self, camera_key: str, file_type: FileType) -> Path:
        """Determine the source path based on directory structure."""
        if camera_key in self.cameras.values():
            # For cleaned data (camera_key is already processed camera name)
            return self.source_dir / camera_key / file_type.proc_dir
        else:
            # For raw data (camera_key is raw camera name)
            return self.source_dir / camera_key / file_type.raw_dir
    
    def _log_copy_results(self, camera_name: str, file_type: FileType, stats: ProcessingStats) -> None:
        """Log the results of file copying operation."""
        status_msg = f"Copied {stats.copied} {file_type.extension} files from {camera_name}"
        if stats.skipped > 0:
            status_msg += f" (skipped {stats.skipped} files)"
        logger.info(status_msg)


class ProbeFileValidator:
    """Specialized validator for probe files with batch processing."""
    
    def __init__(self, raw_dir: Path, cameras: Dict[str, str], probe_raw_dir: Path, 
                 max_workers: int = 4):
        self.raw_dir = raw_dir
        self.cameras = cameras
        self.probe_raw_dir = probe_raw_dir
        self.max_workers = max_workers
    
    def get_valid_probe_indices(self, file_indices: Set[str]) -> Set[str]:
        """Process probe axis files and return indices of files valid across all cameras."""
        valid_indices = set()
        
        # Process indices in batches for better performance
        indices_list = list(file_indices)
        batch_size = max(100, len(indices_list) // self.max_workers)
        batches = [indices_list[i:i + batch_size] for i in range(0, len(indices_list), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._validate_batch, batch): batch
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_valid = future.result()
                    valid_indices.update(batch_valid)
                except Exception as e:
                    logger.error(f"Error validating probe files batch: {e}")
        
        logger.info(f"Valid probe files found for {len(valid_indices)} out of {len(file_indices)} indices")
        return valid_indices
    
    def _validate_batch(self, indices_batch: List[str]) -> Set[str]:
        """Validate a batch of indices."""
        valid_indices = set()
        
        for index in indices_batch:
            if self._validate_single_index(index):
                valid_indices.add(index)
        
        return valid_indices
    
    def _validate_single_index(self, index: str) -> bool:
        """Validate probe files for a single index across all cameras."""
        for camera in self.cameras.keys():
            probe_file = self.raw_dir / camera / self.probe_raw_dir / f"{index}.txt"
            
            if not probe_file.exists():
                logger.debug(f"Probe file not found: {probe_file}")
                return False
            
            if not FileProcessor.check_file_for_nan(probe_file):
                logger.debug(f"NaN values found in probe file: {probe_file}")
                return False
        
        return True


class ProgressTracker:
    """Tracks and reports progress of the cleaning operation."""
    
    def __init__(self):
        self.progress = CleaningProgress()
        self._lock = threading.Lock()
    
    def set_total_files(self, total: int) -> None:
        """Set the total number of files to process."""
        with self._lock:
            self.progress.total_files = total
    
    def update_progress(self, processed: int, stage: str = None) -> None:
        """Update progress information."""
        with self._lock:
            self.progress.update_progress(processed, stage)
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        percentage = self.progress.get_progress_percentage()
        elapsed = self.progress.get_elapsed_time()
        
        logger.info(
            f"Progress: {percentage:.1f}% ({self.progress.processed_files}/"
            f"{self.progress.total_files}) - {self.progress.current_stage} - "
            f"Elapsed: {elapsed:.1f}s"
        )


class StateManager:
    """Manages persistent state for resume capability."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
    
    def save_state(self, stage: str, data: Dict) -> None:
        """Save current processing state."""
        try:
            state = {
                'stage': stage,
                'timestamp': time.time(),
                'data': data
            }
            
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.debug(f"Saved state at stage: {stage}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def load_state(self) -> Optional[Dict]:
        """Load previous processing state."""
        try:
            if not self.state_file.exists():
                return None
                
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            logger.info(f"Loaded state from stage: {state.get('stage', 'unknown')}")
            return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return None
    
    def clear_state(self) -> None:
        """Clear saved state."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.debug("Cleared saved state")
        except Exception as e:
            logger.warning(f"Failed to clear state: {e}")


class DataCleaner:
    """Main class to handle data cleaning operations with improved performance and reliability."""
    
    def __init__(self, raw_dir: Path = RAW_DIR_PATH, cleaned_dir: Path = CLEAN_DIR_PATH,
                 max_workers: int = 4, enable_resume: bool = True):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.max_workers = max_workers
        self.enable_resume = enable_resume
        
        # Configuration
        self.cameras = CAMERA_CONFIG
        self.data_types = DATA_TYPE_CONFIG
        self.label_raw_dir = LABEL_RAW_DIR
        self.label_dir = LABEL_PROC_DIR
        self.label_file = LABEL_FILE
        self.probe_raw_dir = PROBE_RAW_DIR
        
        # Initialize managers
        self.directory_manager = DirectoryManager(
            self.cleaned_dir, self.cameras, self.data_types, self.label_dir
        )
        self.label_manager = LabelFileManager(
            self.raw_dir, self.cleaned_dir, self.cameras, 
            self.label_raw_dir, self.label_dir, self.label_file
        )
        self.file_copier = FileCopier(
            self.raw_dir, self.cleaned_dir, self.cameras, self.max_workers
        )
        self.probe_validator = ProbeFileValidator(
            self.raw_dir, self.cameras, self.probe_raw_dir, self.max_workers
        )
        self.progress_tracker = ProgressTracker()
        
        # State management
        if self.enable_resume:
            state_file = self.cleaned_dir / '.cleaning_state.json'
            self.state_manager = StateManager(state_file)
        else:
            self.state_manager = None
    
    @contextmanager
    def _stage_context(self, stage_name: str, stage_data: Dict = None):
        """Context manager for handling stages with state saving."""
        try:
            logger.info(f"Starting: {stage_name}")
            self.progress_tracker.update_progress(
                self.progress_tracker.progress.processed_files, stage_name
            )
            
            if self.state_manager and stage_data:
                self.state_manager.save_state(stage_name, stage_data)
            
            yield
            
            logger.info(f"Completed: {stage_name}")
            
        except Exception as e:
            logger.error(f"Failed during {stage_name}: {e}")
            raise
    
    def clean_data(self) -> None:
        """Execute the complete data cleaning pipeline with improved error handling and performance."""
        logger.info("Starting enhanced data cleaning pipeline")
        
        try:
            # Check for resume capability
            if self.enable_resume and self.state_manager:
                saved_state = self.state_manager.load_state()
                if saved_state:
                    logger.info(f"Resume capability available from stage: {saved_state.get('stage')}")
                    # Note: Full resume implementation would require more complex state tracking
            
            # Step 1: Create directory structure
            with self._stage_context("Creating directory structure"):
                self.directory_manager.create_directory_structure()
            
            # Step 2: Clean label files and find common files
            with self._stage_context("Processing label files") as _:
                common_files = self._process_label_files()
                
                if not common_files or not any(common_files.values()):
                    raise ValueError("No common files found between cameras!")
            
            # Step 3: Extract file indices and estimate total work
            file_indices = self._extract_file_indices(common_files)
            total_estimated_files = len(file_indices) * len(self.cameras) * len(FileType)
            self.progress_tracker.set_total_files(total_estimated_files)
            
            # Step 4: Validate probe files
            with self._stage_context("Validating probe files", {'file_count': len(file_indices)}):
                valid_file_indices = self.probe_validator.get_valid_probe_indices(file_indices)
                
                if not valid_file_indices:
                    raise ValueError("No valid probe files found!")
            
            # Step 5: Update label files
            with self._stage_context("Updating label files"):
                self.label_manager.update_label_files(common_files, valid_file_indices)
            
            # Step 6: Copy all file types
            with self._stage_context("Copying files"):
                final_valid_files = self._copy_all_file_types(valid_file_indices)
            
            # Clear state on successful completion
            if self.state_manager:
                self.state_manager.clear_state()
            
            logger.info(
                f"Data cleaning completed successfully! "
                f"Final files processed: {len(final_valid_files)} "
                f"(Total time: {self.progress_tracker.progress.get_elapsed_time():.1f}s)"
            )
            
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
        if not common_files:
            return set()
            
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
            stage_name = f"Copying {file_type.name.lower()} files"
            
            with self._stage_context(stage_name):
                try:
                    successfully_copied = self.file_copier.copy_files_by_type(
                        valid_files, file_type
                    )
                    
                    # Keep intersection to ensure all file types are present
                    final_valid_files &= successfully_copied
                    
                    if not final_valid_files:
                        logger.warning(f"No files remain after copying {file_type.name.lower()} files")
                        break
                        
                    # Update progress
                    files_processed = len(valid_files) * len(self.cameras) * i
                    self.progress_tracker.update_progress(files_processed)
                        
                except Exception as e:
                    logger.error(f"Failed to copy {file_type.name.lower()} files: {e}")
                    raise
        
        logger.info(f"Final dataset contains {len(final_valid_files)} complete file sets")
        return final_valid_files