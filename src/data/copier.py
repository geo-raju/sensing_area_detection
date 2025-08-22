import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.data.directory_manager import DirectoryManager
from src.data.file_loader import FileLoader
from src.data.file_processor import FileProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration class for data organisation parameters."""
    base_dir: str = str(Path(__file__).parent.parent.parent)
    raw_dir: str = base_dir + "data/raw"
    processed_dir: str = base_dir + "data/processed"
    cameras: Dict[str, str] = None
    splits: List[str] = None
    categories: Dict[str, Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.cameras is None:
            self.cameras = {"left": "camera0", "right": "camera1"}
        
        if self.splits is None:
            self.splits = ["train", "val", "test"]
        
        if self.categories is None:
            self.categories = {
                "depth_labels": ("depthGT", ".npy"),
                "probe_axis": ("line_annotation_sample", ".txt"),
                "images": ("rgb", ".jpg")
            }


class Dataorganiser(DirectoryManager, FileLoader, FileProcessor):
    """organises raw camera data into processed train/val/test splits."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialise the Dataorganiser.
        
        Args:
            config: Configuration object with directory paths and settings
        """
        self.config = config or DataConfig()
        self.labels_dict = {}
        self.stats = {
            'processed_files': 0,
            'missing_files': 0,
            'errors': 0
        }
    
    def copy_file(self, src_path: Path, dst_path: Path) -> bool:
        """
        Safely copy a file with error handling.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not src_path.exists():
                logger.debug(f"Source file not found: {src_path}")
                self.stats['missing_files'] += 1
                return False
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            self.stats['processed_files'] += 1
            return True
        
        except Exception as e:
            logger.error(f"Error copying {src_path} to {dst_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def copy_data_for_index(self, index: str, split: str) -> Dict[str, int]:
        """
        Copy all data files for a single index to the appropriate split directory.
        
        Args:
            index: Data index to copy
            split: Split name (train/val/test)
            
        Returns:
            Dictionary with copy statistics
        """
        stats = {'copied': 0, 'missing': 0, 'errors': 0}
        
        for side, cam in self.config.cameras.items():
            split_path = Path(self.config.processed_dir) / split / side
            
            for proc_subdir, (raw_subdir, extension) in self.config.categories.items():
                src_file = Path(self.config.raw_dir) / cam / raw_subdir / f"{index}{extension}"
                dst_file = split_path / proc_subdir / f"{index}{extension}"
                
                if self.copy_file(src_file, dst_file):
                    stats['copied'] += 1
                else:
                    stats['missing'] += 1
        
        return stats
    
    def filter_labels_by_split(self, df_labels: pd.DataFrame, df_split: pd.DataFrame) -> pd.DataFrame:
        """
        Filter labels DataFrame based on indices in split DataFrame.
        
        Args:
            df_labels: DataFrame with label data including 'index' column
            df_split: DataFrame with 'index' column from split file
            
        Returns:
            Filtered DataFrame containing only rows where index is in split
        """
        if df_labels is None or df_labels.empty:
            logger.warning("Labels DataFrame is empty or None")
            return pd.DataFrame()
        
        if df_split is None or df_split.empty:
            logger.warning("Split DataFrame is empty or None")
            return pd.DataFrame()
        
        # Merge to keep only labels that exist in the split
        df_filtered = df_labels.merge(df_split, on='index', how='inner')
        
        return df_filtered
    
    def copy_labels(self, indices: pd.DataFrame, split: str) -> bool:
        """
        Copy label files for a specific split.
        
        Args:
            indices: DataFrame containing indices for the split
            split: Split name (train/val/test)
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        for side, cam in self.config.cameras.items():
            split_path = Path(self.config.processed_dir) / split / side / "labels"
            split_path.mkdir(parents=True, exist_ok=True)
            
            label_file_path = split_path / "CenterPt.txt"
            source_label_path = Path(self.config.raw_dir) / cam / "laserptGT" / "CenterPt.txt"
            
            try:
                # Load and process labels
                df_label = self.load_labels(str(source_label_path))
                if df_label is None:
                    logger.error(f"Failed to load labels from {source_label_path}")
                    success = False
                    continue
                
                df_label_with_indices = self.extract_indices(df_label)
                df_label_split = self.filter_labels_by_split(df_label_with_indices, indices)
                
                # Write filtered labels to file
                if not self.write_df_to_file(
                    df_label_split, 
                    label_file_path, 
                    columns=["filename", "x", "y"]
                ):
                    logger.error(f"Failed to write labels to {label_file_path}")
                    success = False
                
            except Exception as e:
                logger.error(f"Error processing labels for {side} camera in {split} split: {e}")
                success = False
        
        return success

    def process_split(self, split: str, indices: pd.DataFrame) -> Dict[str, int]:
        """
        Process a single data split.
        
        Args:
            split: Split name (train/val/test)
            indices: DataFrame with indices to process
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {split} split: {len(indices)} indices")
        
        # Create directory structure
        base_path = Path(self.config.processed_dir) / split
        self.create_dirs(
            base_path, 
            list(self.config.cameras.keys()), 
            list(self.config.categories.keys()) + ["labels"]
        )
        
        # Process each index
        split_stats = {'copied': 0, 'missing': 0, 'errors': 0}
        indices_list = indices['index'].tolist()
        
        for i, idx in enumerate(indices_list, 1):
            if i % 100 == 0:  # Progress logging
                logger.info(f"Processing {split}: {i}/{len(indices_list)} completed")
            
            idx_stats = self.copy_data_for_index(idx, split)
            
            # Accumulate statistics
            for key in split_stats:
                if key in idx_stats:
                    split_stats[key] += idx_stats[key]
        
        # Copy labels for this split
        if not self.copy_labels(indices, split):
            logger.warning(f"Issues encountered while copying labels for {split} split")
        
        logger.info(f"Completed {split} split - Copied: {split_stats['copied']}, "
                   f"Missing: {split_stats['missing']}, Errors: {split_stats['errors']}")
        
        return split_stats
    
    def organise_data(self, split_dir: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        Main method to organise all data splits.
        
        Args:
            split_dir: Directory containing split files (train.txt, val.txt, test.txt)
            
        Returns:
            Dictionary with statistics for each split
        """
        logger.info("Starting data organisation process...")
        
        # Process each split
        all_stats = {}

        for split in self.config.splits:
            if split_dir:
                split_file = Path(split_dir) / f"{split}.txt" 
            else:
                split_file = Path(self.config.base_dir) / "data" / "indices" / "split" / f"{split}.txt"  
            
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue
            
            indices = self.load_split_file(str(split_file))
            
            if indices is None or indices.empty:
                logger.warning(f"Skipping {split} split due to missing or empty file")
                continue
            
            split_stats = self.process_split(split, indices)
            all_stats[split] = split_stats
            
            # Update global stats
            for key in ['missing_files', 'errors', 'processed_files']:
                if key == 'processed_files':
                    self.stats[key] += split_stats.get('copied', 0)
                else:
                    self.stats[key] += split_stats.get(key.split('_')[0], 0)
        
        self.print_summary(all_stats)
        logger.info("Data organisation complete!")
        
        return all_stats
    
    def print_summary(self, all_stats: Dict[str, Dict[str, int]]) -> None:
        """Print a summary of the organisation process."""
        print("\n" + "="*50)
        print("DATA organisation SUMMARY")
        print("="*50)
        
        total_copied = sum(stats.get('copied', 0) for stats in all_stats.values())
        total_missing = sum(stats.get('missing', 0) for stats in all_stats.values())
        total_errors = sum(stats.get('errors', 0) for stats in all_stats.values())
        
        for split, stats in all_stats.items():
            print(f"\n {split.upper()}:")
            print(f"   Files copied: {stats.get('copied', 0)}")
            print(f"   Files missing: {stats.get('missing', 0)}")
            print(f"   Errors: {stats.get('errors', 0)}")
        
        print(f"\n TOTALS:")
        print(f"   Total files copied: {total_copied}")
        print(f"   Total files missing: {total_missing}")
        print(f"   Total errors: {total_errors}")
        
        if total_errors == 0 and total_missing == 0:
            print("\n All files processed successfully!")
        elif total_errors == 0:
            print(f"\n  Completed with {total_missing} missing files")
        else:
            print(f"\n Completed with {total_errors} errors and {total_missing} missing files")
