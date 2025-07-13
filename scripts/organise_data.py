#!/usr/bin/env python3
"""
Data organisation CLI script for processing and organising image datasets.
Handles train/val/test splits and file organisation with configurable structure.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.organiser import DataOrganiser, DataSplitConfig, DirectoryConfig
from config.data_config import (
    RAW_DIR_PATH, PROC_DIR_PATH,
    IMG_RAW_DIR, IMG_PROC_DIR, 
    LABEL_RAW_DIR, LABEL_PROC_DIR, LABEL_FILE,
    PROBE_RAW_DIR, PROBE_PROC_DIR,
    CAMERA_CONFIG, DATA_TYPE_CONFIG,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
)
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class ScriptConfig:
    """Configuration class for script arguments."""
    raw_dir: Path
    output_dir: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    dry_run: bool
    log_level: str
    camera: str = 'camera0'
    
    def __post_init__(self):
        """Validate configuration after initialisation."""
        self.raw_dir = Path(self.raw_dir)
        self.output_dir = Path(self.output_dir)
        
        # Validate ratios
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Validate paths
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {self.raw_dir}")


class DataOrganiserFactory:
    """Factory class for creating DataOrganiser instances."""
    
    @staticmethod
    def create_directory_config() -> DirectoryConfig:
        """
        Create a DirectoryConfig instance with project-specific settings.
        
        Returns:
            DirectoryConfig instance with all directory paths configured
        """
        return DirectoryConfig(
            img_raw_dir=IMG_RAW_DIR,
            img_proc_dir=IMG_PROC_DIR,
            label_raw_dir=LABEL_RAW_DIR,
            label_proc_dir=LABEL_PROC_DIR,
            probe_raw_dir=PROBE_RAW_DIR,
            probe_proc_dir=PROBE_PROC_DIR,
            label_file=LABEL_FILE
        )

    @staticmethod
    def create_split_config(
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_state: int = RANDOM_STATE
    ) -> DataSplitConfig:
        """
        Create a DataSplitConfig instance with project-specific settings.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed for reproducibility
        
        Returns:
            DataSplitConfig instance with split ratios configured
        """
        return DataSplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )

    @staticmethod
    def create_data_organiser(
        config: ScriptConfig,
        camera_config: Dict[str, str] = CAMERA_CONFIG,
        data_type_config: Dict[str, str] = DATA_TYPE_CONFIG
    ) -> DataOrganiser:
        """
        Factory function to create a DataOrganiser configured for this specific project.
        
        Args:
            config: Script configuration
            camera_config: Camera mapping configuration
            data_type_config: Data type mapping configuration
        
        Returns:
            Fully configured DataOrganiser instance
        """
        directory_config = DataOrganiserFactory.create_directory_config()
        split_config = DataOrganiserFactory.create_split_config(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            random_state=config.random_state
        )
        
        return DataOrganiser(
            raw_dir=str(config.raw_dir),
            output_dir=str(config.output_dir),
            camera_config=camera_config,
            data_type_config=data_type_config,
            directory_config=directory_config,
            split_config=split_config
        )


class DataOrganiserCLI:
    """Command Line Interface for the Data Organiser."""
    
    def __init__(self):
        """Initialise the CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description='Organise image dataset into train/val/test splits',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Directory arguments
        parser.add_argument(
            '--raw-dir', 
            type=str, 
            default=str(RAW_DIR_PATH),
            help='Path to raw data directory'
        )
        parser.add_argument(
            '--output-dir', 
            type=str, 
            default=str(PROC_DIR_PATH),
            help='Path to output directory'
        )
        
        # Split ratio arguments
        parser.add_argument(
            '--train-ratio', 
            type=float, 
            default=TRAIN_RATIO,
            help='Training set ratio'
        )
        parser.add_argument(
            '--val-ratio', 
            type=float, 
            default=VAL_RATIO,
            help='Validation set ratio'
        )
        parser.add_argument(
            '--test-ratio', 
            type=float, 
            default=TEST_RATIO,
            help='Test set ratio'
        )
        
        # Other configuration
        parser.add_argument(
            '--random-state', 
            type=int, 
            default=RANDOM_STATE,
            help='Random seed for reproducibility'
        )
        parser.add_argument(
            '--camera', 
            type=str, 
            default='camera0',
            choices=list(CAMERA_CONFIG.keys()),
            help='Camera to use for file pattern generation'
        )
        
        # Execution options
        parser.add_argument(
            '--dry-run', 
            action='store_true',
            help='Print what would be done without actually doing it'
        )
        parser.add_argument(
            '--log-level', 
            type=str, 
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Set logging level'
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> ScriptConfig:
        """
        Parse command line arguments and return configuration.
        
        Args:
            args: Optional list of arguments (for testing)
            
        Returns:
            ScriptConfig instance with parsed arguments
        """
        parsed_args = self.parser.parse_args(args)
        
        return ScriptConfig(
            raw_dir=parsed_args.raw_dir,
            output_dir=parsed_args.output_dir,
            train_ratio=parsed_args.train_ratio,
            val_ratio=parsed_args.val_ratio,
            test_ratio=parsed_args.test_ratio,
            random_state=parsed_args.random_state,
            dry_run=parsed_args.dry_run,
            log_level=parsed_args.log_level,
            camera=parsed_args.camera
        )


class DataOrganiserRunner:
    """Runner class for executing the data organisation process."""
    
    def __init__(self, config: ScriptConfig):
        """Initialise the runner with configuration."""
        self.config = config
        self.organiser = DataOrganiserFactory.create_data_organiser(config)
    
    def get_file_pattern(self) -> str:
        """
        Generate file pattern for data splitting based on project structure.
        
        Returns:
            Glob pattern string for finding files
        """
        return str(self.config.raw_dir / self.config.camera / IMG_RAW_DIR / '*.jpg')
    
    def run_dry_run(self) -> None:
        """Execute dry run mode - show what would be done without doing it."""
        logger.info("DRY RUN MODE - No files will be copied")
        
        file_pattern = self.get_file_pattern()
        
        try:
            # Validate structure first
            self.organiser.validate_raw_structure()
            
            # Perform the split to show what would happen
            train_files, val_files, test_files = self.organiser.split_data(file_pattern)
            
            logger.info(f"Would organise {len(train_files)} training files")
            logger.info(f"Would organise {len(val_files)} validation files")
            logger.info(f"Would organise {len(test_files)} test files")
            
            # Show some example files
            if train_files:
                logger.info(f"Example training files: {train_files[:3]}")
            if val_files:
                logger.info(f"Example validation files: {val_files[:3]}")
            if test_files:
                logger.info(f"Example test files: {test_files[:3]}")
                
        except Exception as e:
            logger.error(f"Dry run failed: {str(e)}")
            raise
    
    def run_full_pipeline(self) -> None:
        """Execute the complete data organisation pipeline."""
        logger.info("Starting full data organisation pipeline")
        
        file_pattern = self.get_file_pattern()
        
        try:
            # Ensure output directory exists
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run the complete pipeline
            self.organiser.organise_all_data(file_pattern)
            
            logger.info("Data organisation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def run(self) -> None:
        """Execute the appropriate run mode based on configuration."""
        if self.config.dry_run:
            self.run_dry_run()
        else:
            self.run_full_pipeline()


def setup_logging_level(log_level: str) -> None:
    """Set up logging with the specified level."""
    setup_logging()
    
    if log_level:
        numeric_level = getattr(logging, log_level.upper())
        logging.getLogger().setLevel(numeric_level)
        logger.info(f"Logging level set to {log_level}")


def main() -> None:
    """Main entry point for the script."""
    try:
        # Parse command line arguments
        cli = DataOrganiserCLI()
        config = cli.parse_args()
        
        # Set up logging
        setup_logging_level(config.log_level)
        
        # Log configuration
        logger.info(f"Raw directory: {config.raw_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Split ratios - Train: {config.train_ratio}, Val: {config.val_ratio}, Test: {config.test_ratio}")
        logger.info(f"Random state: {config.random_state}")
        logger.info(f"Camera: {config.camera}")
        
        # Create and run the organiser
        runner = DataOrganiserRunner(config)
        runner.run()
        
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