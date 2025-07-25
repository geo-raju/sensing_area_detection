#!/usr/bin/env python3
"""
Data splitting CLI script for processing and organizing image datasets.
Handles train/val/test splits with configurable structure and comprehensive validation.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.data_config import (
    CLEAN_DIR_PATH, PROC_DIR_PATH,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
)
from config.logging_config import setup_logging
from src.data.splitter import DataSplitter, SplitResult

logger = logging.getLogger(__name__)


@dataclass
class ScriptConfig:
    """Configuration class for script arguments with comprehensive validation."""
    cleaned_dir: Path
    processed_dir: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    max_workers: int
    enable_resume: bool
    validate_splits: bool
    export_summary: bool
    summary_file: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string paths to Path objects
        self.cleaned_dir = Path(self.cleaned_dir)
        self.processed_dir = Path(self.processed_dir)
        
        if self.summary_file:
            self.summary_file = Path(self.summary_file)
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Train, validation, and test ratios must sum to 1.0, got {total_ratio:.6f}"
            )
        
        # Validate individual ratios
        for ratio_name, ratio_value in [
            ("train", self.train_ratio), 
            ("validation", self.val_ratio), 
            ("test", self.test_ratio)
        ]:
            if not 0.0 <= ratio_value <= 1.0:
                raise ValueError(f"{ratio_name} ratio must be between 0.0 and 1.0, got {ratio_value}")
        
        # Validate paths
        if not self.cleaned_dir.exists():
            raise FileNotFoundError(f"Cleaned data directory does not exist: {self.cleaned_dir}")
        
        if not self.cleaned_dir.is_dir():
            raise NotADirectoryError(f"Cleaned data path is not a directory: {self.cleaned_dir}")
        
        # Validate worker count
        if self.max_workers < 1:
            raise ValueError(f"Max workers must be at least 1, got {self.max_workers}")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "cleaned_dir": str(self.cleaned_dir),
            "processed_dir": str(self.processed_dir),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_state": self.random_state,
            "max_workers": self.max_workers,
            "enable_resume": self.enable_resume,
            "validate_splits": self.validate_splits,
            "export_summary": self.export_summary,
            "summary_file": str(self.summary_file) if self.summary_file else None
        }


class DataSplitterCLI:
    """Command Line Interface for the Data Splitter with enhanced functionality."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser with comprehensive options."""
        parser = argparse.ArgumentParser(
            description='Split cleaned image dataset into train/val/test sets with validation',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog="""
Examples:
  %(prog)s --cleaned-dir ./data/cleaned --processed-dir ./data/processed
  %(prog)s --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
  %(prog)s --max-workers 8 --no-resume --export-summary
            """
        )
        
        # Directory arguments
        parser.add_argument(
            '--cleaned-dir', 
            type=str, 
            default=str(CLEAN_DIR_PATH),
            help='Path to cleaned data directory (input)'
        )
        parser.add_argument(
            '--processed-dir', 
            type=str, 
            default=str(PROC_DIR_PATH),
            help='Path to processed data directory (output)'
        )
        
        # Split ratio arguments
        parser.add_argument(
            '--train-ratio', 
            type=float, 
            default=TRAIN_RATIO,
            help='Training set ratio (0.0-1.0)'
        )
        parser.add_argument(
            '--val-ratio', 
            type=float, 
            default=VAL_RATIO,
            help='Validation set ratio (0.0-1.0)'
        )
        parser.add_argument(
            '--test-ratio', 
            type=float, 
            default=TEST_RATIO,
            help='Test set ratio (0.0-1.0)'
        )
        
        # Performance and behavior arguments
        parser.add_argument(
            '--random-state', 
            type=int, 
            default=RANDOM_STATE,
            help='Random seed for reproducible splits'
        )
        parser.add_argument(
            '--max-workers', 
            type=int, 
            default=4,
            help='Maximum number of worker threads for parallel processing'
        )
        
        # Feature flags
        parser.add_argument(
            '--no-resume', 
            action='store_true',
            help='Disable resume capability (start fresh)'
        )
        parser.add_argument(
            '--no-validation', 
            action='store_true',
            help='Skip split validation after completion'
        )
        parser.add_argument(
            '--export-summary', 
            action='store_true',
            help='Export split summary to JSON file'
        )
        parser.add_argument(
            '--summary-file', 
            type=str,
            help='Path for summary JSON file (default: processed_dir/split_summary.json)'
        )
        
        # Logging control
        parser.add_argument(
            '--verbose', '-v', 
            action='store_true',
            help='Enable verbose logging'
        )
        parser.add_argument(
            '--quiet', '-q', 
            action='store_true',
            help='Suppress non-error output'
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
        
        # Handle conflicting verbosity flags
        if parsed_args.verbose and parsed_args.quiet:
            self.parser.error("--verbose and --quiet cannot be used together")
        
        # Set default summary file if export is requested but no file specified
        summary_file = None
        if parsed_args.export_summary:
            if parsed_args.summary_file:
                summary_file = parsed_args.summary_file
            else:
                summary_file = Path(parsed_args.processed_dir) / "split_summary.json"
        
        return ScriptConfig(
            cleaned_dir=parsed_args.cleaned_dir,
            processed_dir=parsed_args.processed_dir,
            train_ratio=parsed_args.train_ratio,
            val_ratio=parsed_args.val_ratio,
            test_ratio=parsed_args.test_ratio,
            random_state=parsed_args.random_state,
            max_workers=parsed_args.max_workers,
            enable_resume=not parsed_args.no_resume,
            validate_splits=not parsed_args.no_validation,
            export_summary=parsed_args.export_summary,
            summary_file=summary_file
        )


class SplitSummaryExporter:
    """Handles exporting split summaries to various formats."""
    
    @staticmethod
    def export_to_json(results: Dict[str, SplitResult], config: ScriptConfig, 
                      output_file: Path) -> None:
        """Export split results to JSON format."""
        try:
            # Prepare summary data
            summary_data = {
                "configuration": config.to_dict(),
                "splits": {},
                "totals": {
                    "total_files": sum(len(result.file_indices) for result in results.values()),
                    "successful_splits": sum(1 for result in results.values() if result.success),
                    "failed_splits": sum(1 for result in results.values() if not result.success)
                }
            }
            
            # Add split details
            for split_name, result in results.items():
                summary_data["splits"][split_name] = {
                    "success": result.success,
                    "file_count": len(result.file_indices),
                    "percentage": result.get_percentage(summary_data["totals"]["total_files"]),
                    "error_message": result.error_message if not result.success else None,
                    "stats": {
                        "copied": result.stats.copied,
                        "skipped": result.stats.skipped,
                        "removed": result.stats.removed
                    }
                }
            
            # Write to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, sort_keys=True)
            
            logger.info(f"Split summary exported to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export summary to {output_file}: {e}")
            raise


class DataSplitterRunner:
    """Runner class for executing the data splitting process with comprehensive error handling."""
    
    def __init__(self, config: ScriptConfig):
        """Initialize the runner with configuration."""
        self.config = config
        self.summary_exporter = SplitSummaryExporter()
    
    def run_split(self) -> Dict[str, SplitResult]:
        """
        Execute the complete data splitting pipeline.
        
        Returns:
            Dictionary mapping split names to SplitResult objects
        """
        logger.info("Starting data splitting pipeline")
        
        try:
            # Log configuration
            self._log_configuration()
            
            # Ensure output directory exists
            self.config.processed_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory prepared: {self.config.processed_dir}")
            
            splitter = DataSplitter(
                cleaned_dir=self.config.cleaned_dir,
                processed_dir=self.config.processed_dir,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                random_state=self.config.random_state,
                max_workers=self.config.max_workers,
                enable_resume=self.config.enable_resume
            )
            
            # Execute the splitting process
            results = splitter.split_data()
            
            # Validate splits if requested
            if self.config.validate_splits:
                logger.info("Validating split results...")
                if not splitter.validate_splits(results):
                    raise RuntimeError("Split validation failed")
                logger.info("Split validation completed successfully")
            
            # Export summary if requested
            if self.config.export_summary and self.config.summary_file:
                logger.info("Exporting split summary...")
                self.summary_exporter.export_to_json(
                    results, self.config, self.config.summary_file
                )
            
            logger.info("Data splitting pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _log_configuration(self) -> None:
        """Log the current configuration."""
        logger.info("=" * 50)
        logger.info("DATA SPLITTING CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"Cleaned directory: {self.config.cleaned_dir}")
        logger.info(f"Processed directory: {self.config.processed_dir}")
        logger.info(f"Split ratios - Train: {self.config.train_ratio:.3f}, "
                   f"Val: {self.config.val_ratio:.3f}, Test: {self.config.test_ratio:.3f}")
        logger.info(f"Random state: {self.config.random_state}")
        logger.info(f"Max workers: {self.config.max_workers}")
        logger.info(f"Resume enabled: {self.config.enable_resume}")
        logger.info(f"Validation enabled: {self.config.validate_splits}")
        logger.info(f"Export summary: {self.config.export_summary}")
        if self.config.summary_file:
            logger.info(f"Summary file: {self.config.summary_file}")
        logger.info("=" * 50)

def setup_logging_level(verbose: bool, quiet: bool) -> None:
    """Setup logging level based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def main() -> None:
    """Main entry point for the script with comprehensive error handling."""
    exit_code = 0
    
    try:
        # Parse command line arguments
        cli = DataSplitterCLI()
        config = cli.parse_args()
        
        # Set up logging
        setup_logging()
        
        # Adjust logging level based on verbosity flags
        parsed_args = cli.parser.parse_args()
        setup_logging_level(parsed_args.verbose, parsed_args.quiet)
        
        # Create and run the splitter
        runner = DataSplitterRunner(config)
        results = runner.run_split()
        
        # Check if all splits were successful
        failed_splits = [name for name, result in results.items() if not result.success]
        if failed_splits:
            logger.error(f"Some splits failed: {', '.join(failed_splits)}")
            exit_code = 1
        else:
            logger.info("All splits completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        exit_code = 130  # Standard exit code for SIGINT
    except FileNotFoundError as e:
        logger.error(f"File or directory not found: {e}")
        exit_code = 2
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        exit_code = 3
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        exit_code = 4
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()