from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
import logging
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.data.dataframe_file_processor import DataFrameFileProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration class for data organisation parameters."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42
    base_dir: Path = Path(__file__).parent.parent.parent
    
    def __post_init__(self):
        # Validate ratios
        if not abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0. Got: {self.train_ratio + self.val_ratio + self.test_ratio}")
        
        if any(ratio <= 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("All ratios must be positive")


class DatasetSplitter(DataFrameFileProcessor):
    """Handles splitting dataset indices into train/validation/test sets."""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialise the DatasetSplitter.
        
        Args:
            config: Configuration object with split ratios and validation
        """
        
        self.config = config or SplitConfig()
        # Fix the circular reference issues
        self.train_ratio = self.config.train_ratio
        self.val_ratio = self.config.val_ratio
        self.test_ratio = self.config.test_ratio
        self.random_state = self.config.random_state
        self.base_dir = self.config.base_dir
        
        # Calculate split ratios for sklearn
        self.temp_ratio = self.val_ratio + self.test_ratio  # Combined val+test ratio
        self.val_test_split_ratio = self.test_ratio / self.temp_ratio  # Test ratio within val+test
        
        logger.info(f"Initialised DatasetSplitter with ratios - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio}")
    
    def split_indices(self, indices: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Split indices into train, validation, and test sets.
        
        Args:
            indices: List of indices to split
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if not indices:
            raise ValueError("Cannot split empty indices list")
        
        if len(indices) < 3:
            raise ValueError(f"Need at least 3 indices to create 3 splits. Got: {len(indices)}")
        
        logger.info(f"Splitting {len(indices)} indices with ratios {self.train_ratio}/{self.val_ratio}/{self.test_ratio}")
        
        # First split: Train vs (Val + Test)
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=self.temp_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Second split: Validation vs Test (from remaining temp_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.val_test_split_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Log results
        logger.info(f"Split complete - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        self._log_split_statistics(len(indices), len(train_indices), len(val_indices), len(test_indices))
        
        return train_indices, val_indices, test_indices
    
    def _log_split_statistics(self, total: int, train_count: int, val_count: int, test_count: int) -> None:
        """Log detailed statistics about the split."""
        train_actual = train_count / total
        val_actual = val_count / total
        test_actual = test_count / total
        
        logger.info(f"Actual ratios - Train: {train_actual:.3f}, Val: {val_actual:.3f}, Test: {test_actual:.3f}")
    
    def save_split_files(self, train_indices: List[str], val_indices: List[str], test_indices: List[str],
                        output_dir: Optional[str] = None, filenames: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Save split indices to text files.
        
        Args:
            train_indices: Training indices
            val_indices: Validation indices  
            test_indices: Test indices
            output_dir: Directory to save files (default: "split" subdirectory)
            filenames: Custom filenames for splits (default: train.txt, val.txt, test.txt)
            
        Returns:
            Dictionary mapping split names to file paths
        """
        # Use the base_dir from config
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(self.config.base_dir) / "data" / "indices" / "split"
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        saved_files = {}
        
        for split_name, indices in splits.items():
            filename = filenames.get(split_name, f"{split_name}.txt") if filenames else f"{split_name}.txt"
            filepath = output_path / filename
            
            try:
                df_index = pd.DataFrame(sorted(indices), columns=["Index"])
                self.write_df_to_file(df_index, filepath)
                saved_files[split_name] = str(filepath)
                logger.info(f"Saved {len(indices)} {split_name} indices to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save {split_name} indices to {filepath}: {e}")
                raise
        
        return saved_files
    
    def split_and_save(self, indices: List[str], output_dir: Optional[str] = None, 
                      filenames: Optional[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Complete pipeline: split indices and save to files.
        
        Args:
            indices: List of indices to split
            output_dir: Directory to save files (default: "split")
            filenames: Custom filenames for splits
            
        Returns:
            Dictionary containing split indices and file paths
        """
        logger.info("Starting data splitting process...")
        # Split the indices
        train_indices, val_indices, test_indices = self.split_indices(indices)
        
        # Save to files
        saved_files = self.save_split_files(train_indices, val_indices, test_indices, output_dir, filenames)
        
        # Prepare summary
        result = {
            'splits': {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            },
            'files': saved_files,
            'summary': {
                'total': len(indices),
                'train_count': len(train_indices),
                'val_count': len(val_indices),
                'test_count': len(test_indices)
            }
        }
        
        self.print_summary(result['summary'])
        return result
    
    def print_summary(self, summary: Dict[str, int]) -> None:
        """Print a formatted summary of the split."""
        print("\nDataset split complete:")
        print(f"Total indices: {summary['total']}")
        print(f"Train: {summary['train_count']} ({summary['train_count']/summary['total']:.1%})")
        print(f"Validation: {summary['val_count']} ({summary['val_count']/summary['total']:.1%})")
        print(f"Test: {summary['test_count']} ({summary['test_count']/summary['total']:.1%})")