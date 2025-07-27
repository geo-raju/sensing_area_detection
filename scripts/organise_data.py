#!/usr/bin/env python3
"""
Simple pipeline runner with step-by-step execution and configuration.

Usage examples:
    python scripts/organise_data.py --validate-only
    python scripts/organise_data.py --split-only
    python scripts/organise_data.py --organise-only
    python scripts/organise_data.py --all
"""

import argparse
import logging
from pathlib import Path

from src.data.validator import DataValidator
from src.data.splitter import DatasetSplitter, SplitConfig
from src.data.copier import Dataorganiser, DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def setup_paths():
    """
    Setup and return default paths. Modify these to match your directory structure.
    """
    base_dir = Path(__file__).parent.parent  # Current working directory
    raw_data_dir = base_dir / "data" / "raw"
    processed_data_dir = base_dir / "data" / "processed"
    
    # Camera configuration - modify as needed
    cameras = {
        "left": "camera0",
        "right": "camera1"
    }
    
    return {
        'base_dir': base_dir,
        'raw_data_dir': raw_data_dir,
        'processed_data_dir': processed_data_dir,
        'cameras': cameras
    }


def run_validation(paths):
    """Run only the validation step."""
    
    # Setup label and probe paths
    label_paths = {}
    probe_dirs = {}
    
    for cam_name, cam_dir in paths['cameras'].items():
        label_paths[cam_name] = str(paths['raw_data_dir'] / cam_dir / "laserptGT" / "CenterPt.txt")
        probe_dirs[cam_name] = str(paths['raw_data_dir'] / cam_dir / "line_annotation_sample")
    
    # Run validation
    validator = DataValidator(label_paths, probe_dirs)
    valid_indices = validator.run_validation()
    
    if valid_indices:
        logger.info(f"Validation successful.")
    else:
        logger.error("Validation failed: No valid indices found")
        return None
        
    # Save valid indices to file for later use
    try:
        # Create directory
        indices_dir = paths['base_dir'] / "data" / "indices"
        indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the file path
        file_path = indices_dir / "valid_indices.txt"
        
        # Check if the path exists as a directory and remove it
        if file_path.exists() and file_path.is_dir():
            logger.warning(f"Found directory at {file_path}, removing it to create file")
            import shutil
            shutil.rmtree(file_path)
        
        # Write valid indices to file
        with open(file_path, 'w') as f:
            for idx in valid_indices:
                f.write(f"{idx}\n")
        
        logger.info(f"Valid indices saved to: {file_path}")
        return valid_indices
    except OSError as e:
        logger.error(f"OS error occurred (disk space, invalid path, etc.): {e}")
        return None
        

def run_splitting(paths, indices=None):
    """Run only the splitting step."""
    
    # Load indices if not provided
    if indices is None:
        indices_file = paths['base_dir'] / "data" / "indices" / "valid_indices.txt"
        if not indices_file.exists():
            logger.error(f"No indices file found at {indices_file}. Run validation first or provide indices.")
            return None
        
        with open(indices_file, 'r') as f:
            indices = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(indices)} indices from {indices_file}")
    
    # Setup splitter configuration
    split_config = SplitConfig()
    
    # Run splitting
    splitter = DatasetSplitter(split_config)
    results = splitter.split_and_save(indices)
    
    if results:
        logger.info("Splitting successful")
        return results
    else:
        logger.error("Splitting failed")
        return None


def run_Organisation(paths):
    """Run only the Organisation step."""
    
    # Check if split files exist
    split_dir = paths['base_dir'] / "data" / "indices"
    if not split_dir.exists():
        logger.error(f"Split directory not found at {split_dir}. Run splitting first.")
        return None
    
    # Setup organiser configuration
    data_config = DataConfig(
        base_dir=str(paths['base_dir']),
        raw_dir=str(paths['raw_data_dir']),
        processed_dir=str(paths['processed_data_dir']),
        cameras=paths['cameras']
    )
    
    # Run Organisation
    organiser = Dataorganiser(data_config)
    results = organiser.organise_data()
    
    if results:
        logger.info("Organisation successful")
        return results
    else:
        logger.error("Organisation failed")
        return None


def run_all_steps(paths):
    """Run all pipeline steps in sequence."""
    logger.info("Running complete pipeline...")
    
    # Step 1: Validation
    indices = run_validation(paths)
    if not indices:
        return False
    
    print("\n" + "="*50)
    
    # Step 2: Splitting
    split_results = run_splitting(paths, indices)
    if not split_results:
        return False
    
    print("\n" + "="*50)
    
    # Step 3: Organisation
    org_results = run_Organisation(paths)
    if not org_results:
        return False
    
    logger.info("Complete pipeline finished successfully!")
    return True


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Run data processing pipeline")
    parser.add_argument("--validate-only", action="store_true", help="Run only validation step")
    parser.add_argument("--split-only", action="store_true", help="Run only splitting step")
    parser.add_argument("--organise-only", action="store_true", help="Run only Organisation step")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--base-dir", type=str, help="Base directory path")
    
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_paths()
    
    # Override base directory if provided
    if args.base_dir:
        paths['base_dir'] = Path(args.base_dir)
        paths['raw_data_dir'] = paths['base_dir'] / "data" / "raw"
        paths['processed_data_dir'] = paths['base_dir'] / "data" / "processed"
    
    # Print configuration
    print("Pipeline Configuration:")
    print(f"  Raw data: {paths['raw_data_dir']}")
    print(f"  Processed data: {paths['processed_data_dir']}")
    print("")
    
    # Run requested steps
    try:
        if args.validate_only:
            run_validation(paths)
        elif args.split_only:
            run_splitting(paths)
        elif args.organise_only:
            run_Organisation(paths)
        elif args.all:
            run_all_steps(paths)
        else:
            # Default: run all steps
            print("No specific step requested. Running all steps...")
            run_all_steps(paths)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())