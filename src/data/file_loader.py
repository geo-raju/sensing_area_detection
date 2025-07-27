import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FileLoader:
    """Handles loading of CSV files for data processing."""
    
    def load_csv_file(self, path: str, column_names: List[str], sep: str = " ", dtype=str) -> Optional[pd.DataFrame]:
        """
        General function to load CSV files.
        
        Args:
            path: Path to the CSV file
            column_names: List of column names to assign
            
        Returns:
            Raw DataFrame or None if loading fails
        """
        try:
            if not Path(path).exists():
                logger.error(f"File not found: {path}")
                return None
            
            df = pd.read_csv(path, sep=sep, header=None, names=column_names, dtype=dtype)
            return df
            
        except Exception as e:
            logger.error(f"Error loading file from {path}: {e}")
            return None
    
    def load_probe_axis(self, path: str, sep: str = " ") -> Optional[pd.DataFrame]:
        """
        Load probe axis file from CSV.
        
        Args:
            path: Path to the probe axis file
            
        Returns:
            Raw DataFrame or None if loading fails
        """
        return self.load_csv_file(path, ["x", "y"], sep=sep)
    
    def load_labels(self, path: str, sep: str = ",") -> Optional[pd.DataFrame]:
        """
        Load label file from CSV.
        
        Args:
            path: Path to the label file
            
        Returns:
            Raw DataFrame or None if loading fails
        """
        return self.load_csv_file(path, ["filename", "x", "y"], sep=sep)
    
    def load_split_file(self, split_file: str) -> Optional[pd.DataFrame]:
        """
        Read split file and return as DataFrame.
        
        Args:
            split_file: Path to the split file (e.g., "train.txt")
            
        Returns:
            DataFrame with 'index' column or None if file doesn't exist
        """
        try:
            df = self.load_csv_file(split_file, ["index"], dtype=str)

            # Clean up any whitespace and remove empty rows
            df["index"] = df["index"].str.strip()
            df = df[df["index"] != ""]
            df = df[df["index"].notna()]
            
            logger.info(f"Loaded {len(df)} indices from {split_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading split file {split_file}: {e}")
            return None