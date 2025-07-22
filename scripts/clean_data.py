"""
Data cleaning script for removing NaN values and ensuring consistency between cameras.
Processes label files, removes invalid entries, and copies corresponding data files.
"""

import sys
import logging
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.data_config import (
    RAW_DIR_PATH, CLEAN_DIR_PATH
)
from config.logging_config import setup_logging
from src.data.cleaner import DataCleaner

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Clean image dataset by removing NaN values and ensuring consistency',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--raw-dir',
        type=str,
        default=str(RAW_DIR_PATH),
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--cleaned-dir',
        type=str,
        default=str(CLEAN_DIR_PATH),
        help='Path to cleaned data output directory'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.log_level)
        
        # Create paths
        raw_dir = Path(args.raw_dir)
        cleaned_dir = Path(args.cleaned_dir)
        
        # Validate raw directory exists
        if not raw_dir.exists():
            logger.error(f"Raw directory does not exist: {raw_dir}")
            sys.exit(1)
        
        logger.info(f"Raw directory: {raw_dir}")
        logger.info(f"Cleaned directory: {cleaned_dir}")
        
        # Create and run cleaner
        cleaner = DataCleaner(raw_dir, cleaned_dir)
        cleaner.clean_data()
        
        logger.info("Script completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()