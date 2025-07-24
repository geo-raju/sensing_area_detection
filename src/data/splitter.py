import logging
from pathlib import Path
from typing import Set, Dict, List
from dataclasses import dataclass
import random
import time

from config.data_config import (
    DATA_TYPE_CONFIG, RIGHT_CAM_PROC_DIR,
    CLEAN_DIR_PATH, PROC_DIR_PATH,
    LEFT_CAM_PROC_DIR, LABEL_PROC_DIR, LABEL_FILE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
)
from src.data.processor import FileProcessor
from src.data.cleaner import (
    FileType, FileCopier, 
    ProcessingStats, ProgressTracker, StateManager
)

logger = logging.getLogger(__name__)


class DataSplitConfig:
    """Configuration for data splitting with validation."""
    
    SPLIT_NAMES = ['train', 'val', 'test']
    SPLIT_RATIOS = {
        'train': TRAIN_RATIO,
        'val': VAL_RATIO, 
        'test': TEST_RATIO
    }
    
    @classmethod
    def validate_ratios(cls) -> None:
        """Validate that split ratios sum to 1.0."""
        total_ratio = sum(cls.SPLIT_RATIOS.values())
        if not abs(total_ratio - 1.0) < 1e-10:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")


@dataclass
class SplitResult:
    """Result of data splitting operation with enhanced metrics."""
    split_name: str
    file_indices: Set[str]
    stats: ProcessingStats
    success: bool = True
    error_message: str = ""
    
    def get_file_count(self) -> int:
        """Get the number of files in this split."""
        return len(self.file_indices)
    
    def get_percentage(self, total_files: int) -> float:
        """Get percentage of total files this split represents."""
        if total_files == 0:
            return 0.0
        return (self.get_file_count() / total_files) * 100


class IndexSplitter:
    """Handles the logic for splitting file indices into train/val/test sets."""
    
    @staticmethod
    def split_indices(indices: Set[str], ratios: Dict[str, float], 
                     random_state: int = None) -> Dict[str, Set[str]]:
        """
        Split file indices into train/val/test sets with proper validation.
        
        Args:
            indices: Set of file indices to split
            ratios: Dictionary of split names to ratios
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping split names to sets of indices
        """
        if not indices:
            logger.warning("No indices provided for splitting")
            return {name: set() for name in ratios.keys()}
        
        # Validate ratios
        total_ratio = sum(ratios.values())
        if not abs(total_ratio - 1.0) < 1e-10:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Prepare for splitting
        indices_list = sorted(list(indices))
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices_list)
        
        total_files = len(indices_list)
        splits = {}
        start_idx = 0
        
        # Calculate splits (all but last use ratio, last gets remainder)
        split_names = list(ratios.keys())
        for i, split_name in enumerate(split_names):
            if i == len(split_names) - 1:
                # Last split gets remaining files
                split_indices = set(indices_list[start_idx:])
            else:
                split_size = int(total_files * ratios[split_name])
                split_indices = set(indices_list[start_idx:start_idx + split_size])
                start_idx += split_size
            
            splits[split_name] = split_indices
            
            # Log split information
            percentage = len(split_indices) / total_files * 100 if total_files > 0 else 0
            logger.info(f"{split_name.capitalize()} set: {len(split_indices)} files ({percentage:.1f}%)")
        
        return splits


class SplitDirectoryManager:
    """Manages directory creation for data splits, extending DirectoryManager."""
    
    def __init__(self, base_dir: Path, cameras: Dict[str, str], 
                 data_types: Dict[str, str], label_dir: str):
        self.base_dir = base_dir
        self.cameras = cameras
        self.data_types = data_types
        self.label_dir = label_dir
    
    def create_split_directories(self, split_names: List[str]) -> None:
        """Create directory structure for all splits."""
        try:
            for split_name in split_names:
                split_dir = self.base_dir / split_name
                
                # Create camera subdirectories with all data types
                for camera_proc in self.cameras.values():
                    subdirs = list(self.data_types.values()) + [self.label_dir]
                    FileProcessor.create_directories(split_dir / camera_proc, subdirs)
            
            logger.info(f"Created directory structure for splits: {', '.join(split_names)}")
            
        except Exception as e:
            logger.error(f"Failed to create split directories: {e}")
            raise


class SplitLabelManager:
    """Manages label file operations for splits, extending LabelFileManager functionality."""
    
    def __init__(self, cameras: Dict[str, str], label_dir: str, label_file: str):
        self.cameras = cameras
        self.label_dir = label_dir
        self.label_file = label_file
    
    def process_split_labels(self, split_name: str, indices: Set[str], 
                           source_dir: Path, dest_dir: Path, action: str = "Processed") -> None:
        """
        Process label files for a split using the existing processor functionality.
        
        This method reuses FileProcessor.process_label_files with proper parameters.
        """
        try:
            FileProcessor.process_label_files(
                indices=indices,
                cameras=self.cameras,
                source_dir=source_dir,
                dest_dir=dest_dir,
                label_dir=self.label_dir,
                label_file=self.label_file,
                action_name=f"{action} for {split_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to process label files for {split_name}: {e}")
            raise


class DataSplitter:
    """
    Handles splitting cleaned data into train/val/test sets.
    
    This class maximizes reuse of existing components from cleaner.py and processor.py
    while providing specialized functionality for data splitting.
    """
    
    def __init__(self, cleaned_dir: Path = CLEAN_DIR_PATH, 
                 processed_dir: Path = PROC_DIR_PATH, 
                 max_workers: int = 4, enable_resume: bool = True):
        self.cleaned_dir = cleaned_dir
        self.processed_dir = processed_dir
        self.max_workers = max_workers
        self.enable_resume = enable_resume
        
        # Configuration
        self.cameras = {LEFT_CAM_PROC_DIR: LEFT_CAM_PROC_DIR,
                        RIGHT_CAM_PROC_DIR: RIGHT_CAM_PROC_DIR}
        self.data_types = DATA_TYPE_CONFIG
        
        # Validate configuration
        DataSplitConfig.validate_ratios()
        
        # Initialize specialized managers
        self.directory_manager = SplitDirectoryManager(
            processed_dir, self.cameras, self.data_types, LABEL_PROC_DIR
        )
        self.label_manager = SplitLabelManager(
            self.cameras, LABEL_PROC_DIR, LABEL_FILE
        )
        self.index_splitter = IndexSplitter()
        
        # Reuse existing managers from cleaner.py
        self.file_copier = FileCopier(
            self.cleaned_dir, self.processed_dir, self.cameras, self.max_workers
        )
        self.progress_tracker = ProgressTracker()
        
        # State management for resume capability
        if self.enable_resume:
            state_file = self.processed_dir / '.splitting_state.json'
            self.state_manager = StateManager(state_file)
        else:
            self.state_manager = None
    
    def split_data(self) -> Dict[str, SplitResult]:
        """Execute the complete data splitting pipeline with progress tracking."""
        logger.info("Starting enhanced data splitting pipeline")
        start_time = time.time()
        
        try:
            # Check for resume capability
            if self.enable_resume and self.state_manager:
                saved_state = self.state_manager.load_state()
                if saved_state:
                    logger.info(f"Resume capability available from stage: {saved_state.get('stage')}")
            
            # Step 1: Get all available file indices from cleaned data
            self.progress_tracker.update_progress(0, "Gathering available file indices")
            available_indices = self._get_available_indices()
            
            if not available_indices:
                raise ValueError("No files found in cleaned data!")
            
            # Step 2: Split indices into train/val/test
            self.progress_tracker.update_progress(10, "Splitting indices into sets")
            split_indices = self.index_splitter.split_indices(
                available_indices, DataSplitConfig.SPLIT_RATIOS, RANDOM_STATE
            )
            
            # Set up progress tracking
            total_operations = sum(len(indices) for indices in split_indices.values()) * len(FileType)
            self.progress_tracker.set_total_files(total_operations)
            
            # Step 3: Create directory structure for each split
            self.progress_tracker.update_progress(20, "Creating directory structure")
            self.directory_manager.create_split_directories(DataSplitConfig.SPLIT_NAMES)
            
            # Step 4: Copy files for each split
            self.progress_tracker.update_progress(30, "Copying files to split directories")
            results = self._copy_files_to_splits(split_indices)
            
            # Save completion state
            if self.state_manager:
                self.state_manager.save_state("completed", {"results": len(results)})
                self.state_manager.clear_state()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Data splitting completed successfully in {elapsed_time:.1f}s!")
            self._log_split_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Data splitting pipeline failed: {e}")
            raise
    
    def _get_available_indices(self) -> Set[str]:
        """
        Get all available file indices from the cleaned data.
        
        Reuses FileProcessor functionality for consistency with cleaner.py.
        """
        label_path = self.cleaned_dir / LEFT_CAM_PROC_DIR / LABEL_PROC_DIR / LABEL_FILE
        
        if not FileProcessor.check_file_exists(label_path):
            logger.error(f"Label file not found: {label_path}")
            return set()
        
        try:
            files_dict = FileProcessor.get_valid_label_files(label_path)
            filenames = set(files_dict.keys())
            indices = FileProcessor.get_file_indices(filenames)
            
            logger.info(f"Found {len(indices)} available file indices")
            return indices
            
        except Exception as e:
            logger.error(f"Failed to get available indices: {e}")
            return set()
    
    def _copy_files_to_splits(self, split_indices: Dict[str, Set[str]]) -> Dict[str, SplitResult]:
        """Copy files to their respective split directories with progress tracking."""
        results = {}
        
        for i, (split_name, indices) in enumerate(split_indices.items(), 1):
            stage_name = f"Processing {split_name} split"
            logger.info(f"{stage_name} ({len(indices)} files)")
            
            try:
                # Save state for resume capability
                if self.state_manager:
                    self.state_manager.save_state(
                        f"processing_{split_name}", 
                        {"split": split_name, "file_count": len(indices)}
                    )
                
                split_dir = self.processed_dir / split_name
                result = self._copy_split_files(split_name, indices, split_dir)
                results[split_name] = result
                
                # Update progress
                progress = 30 + (i / len(split_indices)) * 60
                self.progress_tracker.update_progress(int(progress), f"Completed {split_name}")
                
            except Exception as e:
                error_msg = f"Failed to process {split_name} split: {e}"
                logger.error(error_msg)
                results[split_name] = SplitResult(
                    split_name, set(), ProcessingStats(), False, error_msg
                )
        
        return results
    
    def _copy_split_files(self, split_name: str, indices: Set[str], split_dir: Path) -> SplitResult:
        """
        Copy all file types for a single split.
        
        Maximizes reuse of FileCopier and other existing components.
        """
        total_stats = ProcessingStats()
        final_valid_indices = indices.copy()
        
        try:
            # Step 1: Copy label files first using the specialized manager
            self.label_manager.process_split_labels(
                split_name, indices, self.cleaned_dir, split_dir, "Copied"
            )
            
            # Step 2: Copy other file types using existing FileCopier
            file_types = [FileType.IMAGE, FileType.PROBE, FileType.DEPTH]
            split_file_copier = FileCopier(self.cleaned_dir, split_dir, self.cameras, self.max_workers)
            
            for file_type in file_types:
                logger.info(f"Copying {file_type.name.lower()} files for {split_name}")
                
                try:
                    successfully_copied = split_file_copier.copy_files_by_type(indices, file_type)
                    
                    # Keep intersection to ensure all file types are present
                    final_valid_indices &= successfully_copied
                    
                    if not final_valid_indices:
                        logger.warning(
                            f"No files remain after copying {file_type.name.lower()} files for {split_name}"
                        )
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to copy {file_type.name.lower()} files for {split_name}: {e}")
                    raise
            
            # Step 3: Update label files to only include final valid indices
            if final_valid_indices != indices:
                logger.info(f"Updating label files for {split_name} to match available files")
                self.label_manager.process_split_labels(
                    split_name, final_valid_indices, split_dir, split_dir, "Updated"
                )
            
            return SplitResult(split_name, final_valid_indices, total_stats)
            
        except Exception as e:
            logger.error(f"Error copying files for {split_name}: {e}")
            return SplitResult(split_name, set(), total_stats, False, str(e))
    
    def _log_split_summary(self, results: Dict[str, SplitResult]) -> None:
        """Log comprehensive summary of splitting results."""
        logger.info("=" * 60)
        logger.info("DATA SPLITTING SUMMARY")
        logger.info("=" * 60)
        
        successful_results = {k: v for k, v in results.items() if v.success}
        failed_results = {k: v for k, v in results.items() if not v.success}
        
        total_files = sum(result.get_file_count() for result in successful_results.values())
        
        # Log successful splits
        if successful_results:
            logger.info("SUCCESSFUL SPLITS:")
            for split_name in DataSplitConfig.SPLIT_NAMES:
                if split_name in successful_results:
                    result = successful_results[split_name]
                    percentage = result.get_percentage(total_files)
                    logger.info(f"  {split_name.upper()}: {result.get_file_count()} files ({percentage:.1f}%)")
        
        # Log failed splits
        if failed_results:
            logger.warning("FAILED SPLITS:")
            for split_name, result in failed_results.items():
                logger.warning(f"  {split_name.upper()}: {result.error_message}")
        
        logger.info(f"TOTAL SUCCESSFUL: {total_files} files processed")
        logger.info("=" * 60)
    
    def validate_splits(self, results: Dict[str, SplitResult]) -> bool:
        """Validate that splits were created correctly."""
        try:
            for split_name, result in results.items():
                if not result.success:
                    logger.error(f"Split {split_name} failed: {result.error_message}")
                    return False
                
                split_dir = self.processed_dir / split_name
                if not split_dir.exists():
                    logger.error(f"Split directory not found: {split_dir}")
                    return False
                
                # Check that all cameras and data types exist
                for camera_proc in self.cameras.values():
                    camera_dir = split_dir / camera_proc
                    if not camera_dir.exists():
                        logger.error(f"Camera directory not found: {camera_dir}")
                        return False
                    
                    expected_dirs = list(self.data_types.values()) + [LABEL_PROC_DIR]
                    for data_type_dir in expected_dirs:
                        type_dir = camera_dir / data_type_dir
                        if not type_dir.exists():
                            logger.error(f"Data type directory not found: {type_dir}")
                            return False
            
            logger.info("Split validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Split validation failed: {e}")
            return False


def main():
    """Main function to run the data splitting with enhanced error handling."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize data splitter with configuration
        splitter = DataSplitter(
            cleaned_dir=CLEAN_DIR_PATH,
            processed_dir=PROC_DIR_PATH,
            max_workers=4,
            enable_resume=True
        )
        
        # Run splitting pipeline
        results = splitter.split_data()
        
        # Validate results
        if splitter.validate_splits(results):
            logger.info("Data splitting and validation completed successfully!")
        else:
            logger.error("Data splitting validation failed!")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())