import pandas as pd
from pathlib import Path
from typing import Set, Union, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FileProcessor:
    """Utilities for handling DataFrame operations, filename processing, and file I/O."""
    
    def extract_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract indices from filename column.
        
        Args:
            df: DataFrame with filename column
            
        Returns:
            DataFrame with added index column
        """
        if df is None or df.empty:
            return df
            
        # Extract index from filename (remove extension)
        df_with_indices = df.copy()
        df_with_indices["index"] = df_with_indices["filename"].apply(lambda x: Path(x).stem)
        return df_with_indices
    
    def get_common_indices(self, *dataframes: pd.DataFrame) -> Set[str]:
        """
        Get indices that are common across all provided DataFrames.
        
        Args:
            *dataframes: Variable number of DataFrames
            
        Returns:
            Set of common indices
        """
        if not dataframes:
            return set()
            
        # Filter out None or empty dataframes
        valid_dataframes = [
            df for df in dataframes 
            if df is not None and not df.empty and "index" in df.columns
        ]
        
        if not valid_dataframes:
            logger.warning("No valid dataframes with 'index' column found")
            return set()
            
        common_indices = set(valid_dataframes[0]["index"])
        for df in valid_dataframes[1:]:
            common_indices = common_indices.intersection(set(df["index"]))

        return common_indices
    
    def write_df_to_file(
        self, 
        df: pd.DataFrame, 
        output_path: Union[str, Path], 
        columns: Optional[List[str]] = None,
        separator: str = ",",
        header: bool = False,
        index: bool = False, 
        encoding: str = "utf-8"
    ) -> bool:
        """
        Write DataFrame to file in various formats.
        
        Args:
            df: DataFrame to write
            output_path: Path to output file
            columns: Specific columns to write (None = all columns)
            separator: Column separator for text formats
            encoding: File encoding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Select columns if specified
            df_to_write = df[columns] if columns else df
            df_to_write.to_csv(output_path, sep=separator, header=header, index=index, encoding=encoding)
            logger.info(f"Successfully wrote {len(df_to_write)} rows to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing DataFrame to {output_path}: {e}")
            return False