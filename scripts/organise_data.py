"""Script to organize raw data into train/val/test splits."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.organiser import DataOrganiser
from config.data_config import (
    RAW_DIR, PROCESSED_DIR, CAMERA_CONFIG, DATA_TYPE_CONFIG,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE
)
from config.logging_config import setup_logging

def validate_raw_directory(raw_dir: str) -> bool:
    """
    Validate that the raw data directory exists and contains expected structure.
    
    Args:
        raw_dir: Path to raw data directory
        
    Returns:
        True if validation passes, False otherwise
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        return False
    
    # Check if the expected camera directories exist
    camera0_path = raw_path / "camera0" / "laser_off"
    if not camera0_path.exists():
        logger.error(f"Expected camera0/laser_off directory not found: {camera0_path}")
        return False
    
    logger.info(f"Raw directory structure validated: {raw_dir}")
    return True

def main():
    """Main function to execute the data organisation pipeline."""
    
    setup_logging()
    global logger
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Organise image dataset into train/val/test splits')
    parser.add_argument('--raw-dir', type=str, default=str(RAW_DIR), 
                       help='Path to raw data directory')
    parser.add_argument('--output-dir', type=str, default=str(PROCESSED_DIR),
                       help='Path to output directory')
    parser.add_argument('--train-split', type=float, default=TRAIN_SPLIT,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=VAL_SPLIT,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=TEST_SPLIT,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                       help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without actually doing it')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set logging level')
    
    args = parser.parse_args()

    # Update logging level if specified
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.info(f"Logging level set to {args.log_level}")
    
    # Validate input arguments
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        logger.error("Train, validation, and test splits must sum to 1.0")
        sys.exit(1)
    
    # Validate directories
    if not validate_raw_directory(args.raw_dir):
        sys.exit(1)
    
    try:
        # Initialise data organiser
        organiser = DataOrganiser(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            camera_config=CAMERA_CONFIG,
            data_type_config=DATA_TYPE_CONFIG
        )
        
        # Create file pattern based on raw directory
        file_pattern = str(Path(args.raw_dir) / 'camera0/laser_off/*.jpg')
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be copied")
            # Just perform the split to show what would happen
            train_files, val_files, test_files = organiser.split_data(
                file_pattern, args.train_split, args.val_split, 
                args.test_split, args.random_state
            )
            logger.info(f"Would organise {len(train_files)} training files")
            logger.info(f"Would organise {len(val_files)} validation files")
            logger.info(f"Would organise {len(test_files)} test files")
        else:
            # Run the complete pipeline
            organiser.organise_all_data(
                file_pattern=file_pattern,
                train_ratio=args.train_split,
                val_ratio=args.val_split,
                test_ratio=args.test_split,
                random_state=args.random_state
            )
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()