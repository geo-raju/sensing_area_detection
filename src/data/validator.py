import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional
import logging

from src.data.file_loader import FileLoader
from src.data.file_processor import FileProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataValidator(FileLoader, FileProcessor):
    """Validates camera data files and probe axis files for consistency."""
    
    def __init__(self, label_paths: Dict[str, str], probe_dirs: Dict[str, str]):
        """
        Initialise the DataValidator.
        
        Args:
            label_paths: Dictionary mapping camera names to label file paths
            probe_dirs: Dictionary mapping camera names to probe directory paths
        """
        self.label_paths = label_paths
        self.probe_dirs = probe_dirs
        self.valid_indices: List[str] = []
    
    def validate_data(self, df: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
        """
        Validate and clean data by converting coordinates to numeric and removing invalid entries.
        
        Args:
            df: Raw DataFrame with x, y coordinates
            source_path: Source file path for logging purposes
            
        Returns:
            Cleaned DataFrame with valid data
        """
        if df is None or df.empty:
            logger.warning(f"Empty or None DataFrame provided for validation from {source_path}")
            return pd.DataFrame()
        
        initial_count = len(df)
        
        # Convert coordinates to numeric, coercing errors to NaN
        df_copy = df.copy()
        df_copy["x"] = pd.to_numeric(df_copy["x"], errors="coerce")
        df_copy["y"] = pd.to_numeric(df_copy["y"], errors="coerce")
        
        # Remove rows with invalid coordinates
        df_clean = df_copy.dropna(subset=["x", "y"])
        dropped_count = initial_count - len(df_clean)
        
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} invalid rows from {source_path}")
        
        return df_clean
    
    def is_valid_probe_axis(self, path: str) -> bool:
        """
        Check if probe axis file is valid using the validate_data function.
        
        Args:
            path: Path to the probe axis file
            
        Returns:
            True if file is valid (has exactly 50 valid rows), False otherwise
        """
        try:
            # Load the probe axis file
            raw_df = self.load_probe_axis(path)
            if raw_df is None:
                return False
            
            # Use validate_data to clean the data
            df_clean = self.validate_data(raw_df, path)
            
            # File is valid if it has exactly 50 clean rows
            is_valid = len(df_clean) == 50
            if not is_valid:
                logger.debug(f"Probe axis file {path} has {len(df_clean)} rows (expected 50)")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating probe axis file {path}: {e}")
            return False
    
    def validate_probe_files(self, indices: Set[str]) -> List[str]:
        """
        Validate probe axis files for given indices.
        An index is only considered valid if ALL cameras have valid probe files for it.
        
        Args:
            indices: Set of indices to validate
            
        Returns:
            List of valid indices (only those valid across all cameras)
        """
        valid_indices_set = set()
        
        try:
            # For each index, check if it's valid across ALL cameras
            for idx in sorted(indices):
                is_valid_for_all_cameras = True
                
                for cam_name, probe_dir in self.probe_dirs.items():
                    probe_path = Path(probe_dir) / f"{idx}.txt"
                    
                    if not self.is_valid_probe_axis(str(probe_path)):
                        print("i am here")
                        logger.debug(f"Invalid probe axis file for {cam_name}, file: {idx}.txt")
                        is_valid_for_all_cameras = False
                        break  # No need to check other cameras for this index
                
                if is_valid_for_all_cameras:
                    valid_indices_set.add(idx)
                    logger.debug(f"Index {idx} is valid across all cameras")
            
            valid_indices = sorted(list(valid_indices_set))
            logger.info(f"Validated {len(valid_indices)} out of {len(indices)} probe indices")
            return valid_indices
            
        except Exception as e:
            logger.error(f"Error during probe file validation: {e}")
            return []
    
    def validate_label_file(self, path: str) -> Optional[pd.DataFrame]:
        """
        Complete pipeline to process a single label file.
        
        Args:
            path: Path to the label file
            
        Returns:
            Processed DataFrame with validated labels and indices
        """
        try:
            raw_df = self.load_labels(path)
            if raw_df is None:
                return None
            
            validated_df = self.validate_data(raw_df, path)
            if validated_df.empty:
                logger.error(f"No valid labels found in {path}")
                return None
            
            final_df = self.extract_indices(validated_df)
            return final_df
            
        except Exception as e:
            logger.error(f"Error processing label file {path}: {e}")
            return None
    
    def run_validation(self) -> List[str]:
        """
        Run the complete validation process.
        
        Returns:
            List of final valid indices
        """
        try:
            logger.info("Starting data validation process...")
            
            # Step 1: Process all label files
            logger.info("Validating label files...")
            label_dataframes = []
            
            for cam_name, label_path in self.label_paths.items():
                logger.info(f"Processing {cam_name} labels from {label_path}")
                processed_df = self.validate_label_file(label_path)
                
                if processed_df is not None:
                    label_dataframes.append(processed_df)
                    logger.info(f"Successfully processed {len(processed_df)} labels for {cam_name}")
                else:
                    logger.error(f"Failed to process labels for {cam_name}")
            
            if len(label_dataframes) != len(self.label_paths):
                logger.error("Not all label files could be loaded")
                return []
            
            # Step 2: Get common valid indices
            logger.info("Finding common indices across cameras...")
            common_indices = self.get_common_indices(*label_dataframes)
            logger.info(f"Found {len(common_indices)} common indices")
            
            if not common_indices:
                logger.warning("No common indices found")
                return []
            
            # Step 3: Validate probe axis files
            logger.info("Validating probe axis files...")
            self.valid_indices = self.validate_probe_files(common_indices)
            
            # Step 4: Report results
            logger.info(f"Final valid indices: {len(self.valid_indices)}")
            if self.valid_indices:
                logger.info(f"Sample indices: {self.valid_indices[:10]}")
            
            return self.valid_indices
            
        except Exception as e:
            logger.error(f"Error during validation process: {e}")
            return []