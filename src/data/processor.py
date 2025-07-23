import logging
from pathlib import Path
from typing import Set, List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedLine:
    """Container for parsed line data."""
    line_number: int
    original_text: str
    parts: List[str]


@dataclass
class LabelEntry:
    """Container for label file entry data."""
    filename: str
    x_coord: float
    y_coord: float


class FileSystemOperations:
    """Handles basic file system operations."""
    
    @staticmethod
    def create_directories(base_path: Path, subdirs: List[str]) -> None:
        """Create multiple subdirectories under a base path."""
        for subdir in subdirs:
            directory = base_path / subdir
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    @staticmethod
    def check_file_exists(file_path: Path) -> bool:
        """Check if a file exists."""
        return file_path.exists()

    @staticmethod
    def write_lines_to_file(file_path: Path, lines: List[str]) -> None:
        """Write lines to a file, creating parent directories if needed."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as output_file:
            for line in lines:
                output_file.write(line + '\n')


class FileParser:
    """Handles parsing of text files."""
    
    @staticmethod
    def parse_file(file_path: Path, separator: str = " ") -> List[ParsedLine]:
        """Parse a text file and return structured data."""
        if not FileSystemOperations.check_file_exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        parsed_lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as input_file:
                for line_num, line in enumerate(input_file, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = [part.strip() for part in line.split(sep=separator)]
                    parsed_lines.append(ParsedLine(line_num, line, parts))
                    
        except (IOError, OSError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
            
        return parsed_lines

    @staticmethod
    def contains_nan_values(parts: List[str]) -> bool:
        """Check if any part contains NaN values."""
        return any(str(part).lower() == 'nan' for part in parts)


class FileValidator:
    """Handles file validation operations."""
    
    @staticmethod
    def check_file_for_nan(file_path: Path, separator: str = " ") -> bool:
        """
        Check if a text file contains NaN values.
        
        Returns:
            True if file is valid (no NaN values), False otherwise
        """
        try:
            parsed_lines = FileParser.parse_file(file_path, separator)
            
            for parsed_line in parsed_lines:
                if FileParser.contains_nan_values(parsed_line.parts):
                    logger.warning(
                        f"NaN value found in {file_path} at line {parsed_line.line_number}: "
                        f"{parsed_line.original_text}"
                    )
                    return False
                    
            return True
            
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False


class FilenameUtils:
    """Utilities for working with filenames."""
    
    @staticmethod
    def extract_indices(filenames: Set[str]) -> Set[str]:
        """
        Extract indices from filenames (e.g., '01097' from '01097.jpg').
        
        Args:
            filenames: Set of filenames
            
        Returns:
            Set of indices (filename without extension)
        """
        return {filename.split('.')[0] for filename in filenames}


class LabelFileProcessor:
    """Specialized processor for label files."""
    
    @staticmethod
    def parse_label_entry(line: str, parts: List[str]) -> LabelEntry:
        """Parse a single line from label file into a LabelEntry."""
        if len(parts) != 3:
            raise ValueError(f"Invalid line format - expected 3 parts, got {len(parts)}: {line}")
        
        filename = parts[0]
        
        try:
            x_coord = float(parts[1])
            y_coord = float(parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid coordinates in line: {line}") from e
        
        return LabelEntry(filename, x_coord, y_coord)
    
    @staticmethod
    def get_valid_label_files(source_file: Path) -> Dict[str, str]:
        """
        Parse and validate label file entries.
        
        Args:
            source_file: Path to the label file
            
        Returns:
            Dictionary mapping filename to original line text
        """
        if not FileSystemOperations.check_file_exists(source_file):
            logger.error(f"Source label file not found: {source_file}")
            return {}

        valid_label_dict = {}

        try:
            parsed_lines = FileParser.parse_file(source_file, ',')
        except (IOError, OSError, FileNotFoundError):
            return {}

        for parsed_line in parsed_lines:
            try:
                # Check for NaN values
                if FileParser.contains_nan_values(parsed_line.parts):
                    raise ValueError(f"NaN values found in line: {parsed_line.original_text}")

                # Parse the label entry
                entry = LabelFileProcessor.parse_label_entry(
                    parsed_line.original_text, parsed_line.parts
                )

                valid_label_dict[entry.filename] = parsed_line.original_text
            
            except ValueError as e:
                logger.warning(f"Skipping line {parsed_line.line_number}: {e}")
                continue
        
        logger.info(f"Loaded {len(valid_label_dict)} valid label entries from {source_file}")
        return valid_label_dict


class FileProcessor:
    """
    Main file processor class that provides a unified interface for file operations.
    
    This class serves as a facade that delegates to specialized classes for different
    types of file operations, maintaining backward compatibility while providing
    better organization and maintainability.
    """
    
    # File system operations
    @staticmethod
    def create_directories(base_path: Path, subdirs: List[str]) -> None:
        """Create multiple subdirectories under a base path."""
        return FileSystemOperations.create_directories(base_path, subdirs)
    
    @staticmethod
    def check_file_exists(file_path: Path) -> bool:
        """Check if a file exists."""
        return FileSystemOperations.check_file_exists(file_path)
    
    @staticmethod
    def write_file(file_path: Path, lines: List[str]) -> None:
        """Write lines to a file, creating parent directories if needed."""
        return FileSystemOperations.write_lines_to_file(file_path, lines)
    
    # File parsing operations
    @staticmethod
    def parse_file(file_path: Path, separator: str = " ") -> List[ParsedLine]:
        """Parse a text file and return structured data."""
        return FileParser.parse_file(file_path, separator)
    
    @staticmethod
    def check_for_nan(parts: List[str]) -> bool:
        """Check if any part contains NaN values."""
        return FileParser.contains_nan_values(parts)
    
    # File validation operations
    @staticmethod
    def check_file_for_nan(file_path: Path, separator: str = " ") -> bool:
        """Check if a text file contains NaN values."""
        return FileValidator.check_file_for_nan(file_path, separator)
    
    # Label file operations
    @staticmethod
    def parse_label_line(line: str, parts: List[str]) -> LabelEntry:
        """Parse a single line from label file into a LabelEntry."""
        return LabelFileProcessor.parse_label_entry(line, parts)
    
    @staticmethod
    def get_valid_label_files(source_file: Path) -> Dict[str, str]:
        """Parse and validate label file entries."""
        return LabelFileProcessor.get_valid_label_files(source_file)
    
    # Filename utilities
    @staticmethod
    def get_file_indices(filenames: Set[str]) -> Set[str]:
        """Extract indices from filenames."""
        return FilenameUtils.extract_indices(filenames)