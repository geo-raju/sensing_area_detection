import logging
from pathlib import Path
from typing import Set, Tuple, List, Optional
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
    def _should_include_entry(entry: LabelEntry, valid_items: Optional[Set[str]], 
                            is_filename_filter: bool) -> bool:
        """Determine if an entry should be included based on filter criteria."""
        if not valid_items:
            return True
            
        if is_filename_filter:
            return entry.filename in valid_items
        else:
            # Filter by index (filename without extension)
            index = entry.filename.split('.')[0]
            return index in valid_items
    
    @staticmethod
    def filter_label_file(source_file: Path, dest_file: Path, 
                         valid_items: Optional[Set[str]] = None, 
                         is_filename_filter: bool = True) -> Tuple[Set[str], int]:
        """
        Filter label file and return valid filenames and count.
        
        Args:
            source_file: Source label file path
            dest_file: Destination label file path
            valid_items: Set of valid filenames or indices to keep
            is_filename_filter: If True, filter by filename; if False, filter by index
            
        Returns:
            Tuple of (valid_filenames_set, valid_entries_count)
        """
        if not source_file.exists():
            logger.error(f"Source label file not found: {source_file}")
            return set(), 0
        
        valid_filenames = set()
        valid_lines = []

        try:
            parsed_lines = FileParser.parse_file(source_file, ',')
        except (IOError, OSError):
            return set(), 0

        for parsed_line in parsed_lines:
            try:
                # Check for NaN values first
                if FileParser.contains_nan_values(parsed_line.parts):
                    raise ValueError(f"NaN values found in line: {parsed_line.original_text}")
                
                # Parse the label entry
                entry = LabelFileProcessor.parse_label_entry(
                    parsed_line.original_text, parsed_line.parts
                )
                
                # Check if this entry should be included
                if LabelFileProcessor._should_include_entry(entry, valid_items, is_filename_filter):
                    valid_filenames.add(entry.filename)
                    valid_lines.append(parsed_line.original_text)
                    
            except ValueError as e:
                logger.warning(f"Skipping line {parsed_line.line_number}: {e}")
                continue
        
        # Write filtered file
        FileSystemOperations.write_lines_to_file(dest_file, valid_lines)
        
        logger.info(f"Processed labels: {len(valid_lines)} valid entries from {source_file}")
        return valid_filenames, len(valid_lines)


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
            
        except (IOError, OSError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
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


class FileProcessor:
    """
    Main file processor class that combines all file processing operations.
    This maintains backward compatibility with the original interface.
    """
    
    # Delegate to specialized classes for better organization
    create_directories = FileSystemOperations.create_directories
    write_file = FileSystemOperations.write_lines_to_file
    parse_file = FileParser.parse_file
    check_for_nan = FileParser.contains_nan_values
    parse_label_line = LabelFileProcessor.parse_label_entry
    filter_label_file = LabelFileProcessor.filter_label_file
    check_file_for_nan = FileValidator.check_file_for_nan
    get_file_indices = FilenameUtils.extract_indices