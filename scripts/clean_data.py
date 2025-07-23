"""
Enhanced data cleaning script for removing NaN values and ensuring consistency between cameras.
Processes label files, removes invalid entries, and copies corresponding data files with 
improved performance, error handling, and progress tracking.
"""

import sys
import logging
import signal
from pathlib import Path
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.data_config import (
    RAW_DIR_PATH, CLEAN_DIR_PATH
)
from config.logging_config import setup_logging
from src.data.cleaner import DataCleaner
from src.data.processor import FileValidator

logger = logging.getLogger(__name__)

# Global variable for graceful shutdown
cleanup_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global cleanup_requested
    if not cleanup_requested:
        cleanup_requested = True
        logger.info("Interrupt received. Attempting graceful shutdown...")
        logger.info("Press Ctrl+C again to force exit.")
    else:
        logger.warning("Force exit requested.")
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description='Clean image dataset by removing NaN values and ensuring consistency between cameras',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default directories
  %(prog)s --raw-dir /path/to/raw --cleaned-dir /path/to/clean
  %(prog)s --max-workers 8 --disable-resume  # Use 8 threads, disable resume
  %(prog)s --validate-only                   # Only validate without cleaning
        """
    )
    
    # Directory arguments
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
    
    # Performance arguments
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of worker threads for parallel processing'
    )
    parser.add_argument(
        '--disable-resume',
        action='store_true',
        help='Disable resume capability (clears any existing state)'
    )
    
    # Operation mode arguments
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without performing cleaning operations'
    )
    parser.add_argument(
        '--force-clean',
        action='store_true',
        help='Force clean even if cleaned directory exists and contains data'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear validation cache before starting'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output (equivalent to --log-level WARNING)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (equivalent to --log-level DEBUG)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments and paths."""
    # Adjust log level based on quiet/verbose flags
    if args.quiet and args.verbose:
        raise ValueError("Cannot specify both --quiet and --verbose")
    
    if args.quiet:
        args.log_level = 'WARNING'
    elif args.verbose:
        args.log_level = 'DEBUG'
    
    # Validate worker count
    if args.max_workers < 1:
        raise ValueError("max-workers must be at least 1")
    
    if args.max_workers > 16:
        logger.warning(f"Using {args.max_workers} workers might be excessive. Consider using fewer workers.")
    
    # Validate paths
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw path is not a directory: {raw_dir}")
    
    # Check if cleaned directory exists and has content
    cleaned_dir = Path(args.cleaned_dir)
    if cleaned_dir.exists() and any(cleaned_dir.iterdir()) and not args.force_clean:
        raise ValueError(
            f"Cleaned directory already exists and contains data: {cleaned_dir}\n"
            "Use --force-clean to overwrite existing data"
        )


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def perform_pre_validation(raw_dir: Path) -> bool:
    """Perform basic validation of the raw data structure."""
    logger.info("Performing pre-validation of raw data structure...")
    
    validation_errors = []
    
    # Check if required directories exist
    required_dirs = ['left_cam', 'right_cam']  # Update based on your config
    for dir_name in required_dirs:
        dir_path = raw_dir / dir_name
        if not dir_path.exists():
            validation_errors.append(f"Missing required directory: {dir_path}")
        elif not dir_path.is_dir():
            validation_errors.append(f"Path is not a directory: {dir_path}")
    
    if validation_errors:
        logger.error("Pre-validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Pre-validation completed successfully")
    return True


def create_cleaner(args: argparse.Namespace) -> DataCleaner:
    """Create and configure the DataCleaner instance."""
    raw_dir = Path(args.raw_dir).resolve()
    cleaned_dir = Path(args.cleaned_dir).resolve()
    
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Cleaned directory: {cleaned_dir}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Resume enabled: {not args.disable_resume}")
    
    return DataCleaner(
        raw_dir=raw_dir,
        cleaned_dir=cleaned_dir,
        max_workers=args.max_workers,
        enable_resume=not args.disable_resume
    )


def perform_validation_only(cleaner: DataCleaner) -> bool:
    """Perform validation-only operation."""
    logger.info("Running validation-only mode...")
    
    try:
        # This would require adding a validation method to DataCleaner
        # For now, we'll do basic checks
        logger.info("Validating directory structure...")
        cleaner.directory_manager.create_directory_structure()
        
        logger.info("Validating label files...")
        common_files = cleaner._process_label_files()
        
        if not common_files or not any(common_files.values()):
            logger.error("Validation failed: No common files found between cameras")
            return False
        
        file_indices = cleaner._extract_file_indices(common_files)
        logger.info(f"Found {len(file_indices)} file indices")
        
        logger.info("Validating probe files...")
        valid_indices = cleaner.probe_validator.get_valid_probe_indices(file_indices)
        logger.info(f"Found {len(valid_indices)} valid probe files")
        
        if len(valid_indices) == 0:
            logger.error("Validation failed: No valid probe files found")
            return False
        
        success_rate = len(valid_indices) / len(file_indices) * 100
        logger.info(f"Validation completed. Success rate: {success_rate:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def monitor_progress(cleaner: DataCleaner) -> None:
    """Monitor and log cleaning progress periodically."""
    last_logged_progress = -1
    start_time = time.time()
    
    while True:
        try:
            current_progress = cleaner.progress_tracker.progress.get_progress_percentage()
            
            # Log progress every 10% or every 30 seconds
            if (current_progress - last_logged_progress >= 10 or 
                time.time() - start_time > 30):
                
                elapsed = cleaner.progress_tracker.progress.get_elapsed_time()
                
                if current_progress > 0:
                    estimated_total = elapsed / (current_progress / 100)
                    remaining = estimated_total - elapsed
                    logger.info(
                        f"Progress: {current_progress:.1f}% - "
                        f"Elapsed: {elapsed:.1f}s - "
                        f"Estimated remaining: {remaining:.1f}s"
                    )
                
                last_logged_progress = current_progress
                start_time = time.time()
            
            if current_progress >= 100:
                break
                
            time.sleep(1)
            
        except Exception as e:
            logger.debug(f"Error monitoring progress: {e}")
            break


def main() -> int:
    """
    Main entry point for the script.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    start_time = time.time()
    
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Set up logging
        setup_logging(args.log_level)
        
        # Set up signal handlers
        setup_signal_handlers()
        
        # Clear validation cache if requested
        if args.clear_cache:
            logger.info("Clearing validation cache...")
            FileValidator.clear_validation_cache()
        
        logger.info("=" * 60)
        logger.info("Starting enhanced data cleaning pipeline")
        logger.info("=" * 60)
        
        # Perform pre-validation
        raw_dir = Path(args.raw_dir)
        if not perform_pre_validation(raw_dir):
            logger.error("Pre-validation failed. Aborting.")
            return 1
        
        # Create cleaner instance
        cleaner = create_cleaner(args)
        
        # Check for interrupt before starting
        if cleanup_requested:
            logger.info("Cleanup requested before starting. Exiting.")
            return 1
        
        # Perform operation based on mode
        if args.validate_only:
            success = perform_validation_only(cleaner)
            return 0 if success else 1
        
        # Start the cleaning process
        logger.info("Starting data cleaning process...")
        cleaner.clean_data()
        
        # Calculate and log final statistics
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Data cleaning completed successfully!")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2
        
    except NotADirectoryError as e:
        logger.error(f"Directory error: {e}")
        return 2
        
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        return 2
        
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        return 13
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if args.log_level == 'DEBUG':
            logger.exception("Full traceback:")
        else:
            logger.info("Run with --verbose for full traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())