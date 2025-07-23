import logging
from pathlib import Path
from typing import Set, List, Dict, Iterator
from dataclasses import dataclass
from contextlib import contextmanager
import mmap
import threading

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
    """Handles basic file system operations with improved error handling."""
    
    @staticmethod
    def create_directories(base_path: Path, subdirs: List[str]) -> None:
        """Create multiple subdirectories under a base path atomically."""
        try:
            directories_to_create = [base_path / subdir for subdir in subdirs]
            
            # Create all directories in one pass
            for directory in directories_to_create:
                directory.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Created {len(directories_to_create)} directories under {base_path}")
            
        except OSError as e:
            logger.error(f"Failed to create directories under {base_path}: {e}")
            raise

    @staticmethod
    def check_file_exists(file_path: Path) -> bool:
        """Check if a file exists with proper error handling."""
        try:
            return file_path.exists() and file_path.is_file()
        except OSError as e:
            logger.debug(f"Error checking file existence {file_path}: {e}")
            return False

    @staticmethod
    def write_lines_to_file(file_path: Path, lines: List[str]) -> None:
        """Write lines to a file atomically, creating parent directories if needed."""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first, then move to target (atomic operation)
            temp_file = file_path.with_suffix(file_path.suffix + '.tmp')
            
            with open(temp_file, 'w', encoding='utf-8', newline='\n') as output_file:
                for line in lines:
                    output_file.write(line + '\n')
            
            # Atomic move
            temp_file.replace(file_path)
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            # Clean up temp file if it exists
            if 'temp_file' in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except OSError as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return 0

    @staticmethod
    @contextmanager
    def memory_mapped_file(file_path: Path, mode: str = 'r'):
        """Context manager for memory-mapped file operations."""
        if not FileSystemOperations.check_file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_obj = None
        mmap_obj = None
        
        try:
            file_obj = open(file_path, 'rb' if 'b' in mode else 'r', encoding='utf-8' if 'b' not in mode else None)
            
            if file_obj.readable():
                mmap_obj = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
                yield mmap_obj
            else:
                yield file_obj
                
        except (IOError, OSError, mmap.error) as e:
            logger.error(f"Error with memory-mapped file {file_path}: {e}")
            raise
        finally:
            if mmap_obj:
                mmap_obj.close()
            if file_obj:
                file_obj.close()


class FileParser:
    """Handles parsing of text files with enhanced performance and memory efficiency."""
    
    @staticmethod
    def parse_file(file_path: Path, separator: str = " ", use_mmap: bool = False) -> List[ParsedLine]:
        """Parse a text file and return structured data with optional memory mapping."""
        if not FileSystemOperations.check_file_exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        parsed_lines = []
        
        try:
            if use_mmap and FileSystemOperations.get_file_size(file_path) > 1024 * 1024:  # Use mmap for files > 1MB
                with FileSystemOperations.memory_mapped_file(file_path) as mmap_file:
                    content = mmap_file.read().decode('utf-8')
                    lines = content.splitlines()
            else:
                with open(file_path, 'r', encoding='utf-8') as input_file:
                    lines = input_file.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = [part.strip() for part in line.split(sep=separator)]
                parsed_lines.append(ParsedLine(line_num, line, parts))
                    
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
            
        return parsed_lines

    @staticmethod
    def parse_file_lazy(file_path: Path, separator: str = " ") -> Iterator[ParsedLine]:
        """Lazy parse a text file to save memory for large files."""
        if not FileSystemOperations.check_file_exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as input_file:
                for line_num, line in enumerate(input_file, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = [part.strip() for part in line.split(sep=separator)]
                    yield ParsedLine(line_num, line, parts)
                    
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def contains_nan_values(parts: List[str]) -> bool:
        """Check if any part contains NaN values with improved detection."""
        nan_variants = {'nan', 'n/a', 'null', 'none', '', 'na'}
        return any(str(part).lower().strip() in nan_variants for part in parts)

    @staticmethod
    def validate_numeric_parts(parts: List[str], expected_count: int = None) -> bool:
        """Validate that parts contain valid numeric values."""
        if expected_count and len(parts) != expected_count:
            return False
        
        for part in parts[1:]:  # Skip filename, check coordinates
            try:
                float(part.strip())
            except (ValueError, AttributeError):
                return False
        
        return True


class FileValidator:
    """Handles file validation operations with improved performance."""
    
    _validation_cache = {}
    _cache_lock = threading.Lock()
    
    @staticmethod
    def check_file_for_nan(file_path: Path, separator: str = " ", use_cache: bool = True) -> bool:
        """
        Check if a text file contains NaN values with caching support.
        
        Args:
            file_path: Path to the file to validate
            separator: Separator to use for parsing
            use_cache: Whether to use validation cache
            
        Returns:
            True if file is valid (no NaN values), False otherwise
        """
        # Check cache first
        if use_cache:
            cache_key = (str(file_path), separator)
            with FileValidator._cache_lock:
                if cache_key in FileValidator._validation_cache:
                    return FileValidator._validation_cache[cache_key]
        
        try:
            is_valid = True
            
            # Use lazy parsing for better memory efficiency
            for parsed_line in FileParser.parse_file_lazy(file_path, separator):
                if FileParser.contains_nan_values(parsed_line.parts):
                    logger.warning(
                        f"NaN value found in {file_path} at line {parsed_line.line_number}: "
                        f"{parsed_line.original_text}"
                    )
                    is_valid = False
                    break
            
            # Cache the result
            if use_cache:
                with FileValidator._cache_lock:
                    FileValidator._validation_cache[cache_key] = is_valid
                    
            return is_valid
            
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    @staticmethod
    def batch_validate_files(file_paths: List[Path], separator: str = " ") -> Dict[Path, bool]:
        """Validate multiple files efficiently."""
        results = {}
        
        for file_path in file_paths:
            results[file_path] = FileValidator.check_file_for_nan(file_path, separator)
            
        return results

    @staticmethod
    def clear_validation_cache() -> None:
        """Clear the validation cache."""
        with FileValidator._cache_lock:
            FileValidator._validation_cache.clear()


class FilenameUtils:
    """Utilities for working with filenames with enhanced functionality."""
    
    @staticmethod
    def extract_indices(filenames: Set[str]) -> Set[str]:
        """
        Extract indices from filenames (e.g., '01097' from '01097.jpg').
        
        Args:
            filenames: Set of filenames
            
        Returns:
            Set of indices (filename without extension)
        """
        indices = set()
        for filename in filenames:
            # Handle multiple extensions and edge cases
            base_name = filename.split('.')[0]
            if base_name:  # Ensure we don't add empty strings
                indices.add(base_name)
        
        return indices

    @staticmethod
    def group_by_extension(filenames: Set[str]) -> Dict[str, Set[str]]:
        """Group filenames by their extensions."""
        grouped = {}
        
        for filename in filenames:
            extension = Path(filename).suffix.lower()
            if extension not in grouped:
                grouped[extension] = set()
            grouped[extension].add(filename)
        
        return grouped

    @staticmethod
    def validate_filename_format(filename: str, expected_pattern: str = None) -> bool:
        """Validate filename against expected pattern."""
        if not filename or not filename.strip():
            return False
        
        # Basic validation - contains valid characters
        invalid_chars = {'<', '>', ':', '"', '|', '?', '*'}
        if any(char in filename for char in invalid_chars):
            return False
        
        # Pattern validation if provided
        if expected_pattern:
            import re
            try:
                return bool(re.match(expected_pattern, filename))
            except re.error:
                logger.warning(f"Invalid regex pattern: {expected_pattern}")
                return True
        
        return True


class LabelFileProcessor:
    """Specialized processor for label files with enhanced validation and performance."""
    
    @staticmethod
    def parse_label_entry(line: str, parts: List[str]) -> LabelEntry:
        """Parse a single line from label file into a LabelEntry with enhanced validation."""
        if len(parts) != 3:
            raise ValueError(f"Invalid line format - expected 3 parts, got {len(parts)}: {line}")
        
        filename = parts[0].strip()
        
        # Validate filename
        if not FilenameUtils.validate_filename_format(filename):
            raise ValueError(f"Invalid filename format: {filename}")
        
        try:
            x_coord = float(parts[1].strip())
            y_coord = float(parts[2].strip())
            
            # Validate coordinate ranges (assuming normalized coordinates)
            if not (0.0 <= x_coord <= 1.0) or not (0.0 <= y_coord <= 1.0):
                logger.warning(f"Coordinates outside expected range [0,1]: x={x_coord}, y={y_coord}")
            
        except ValueError as e:
            raise ValueError(f"Invalid coordinates in line: {line}") from e
        
        return LabelEntry(filename, x_coord, y_coord)
    
    @staticmethod
    def get_valid_label_files(source_file: Path, use_lazy_parsing: bool = False) -> Dict[str, str]:
        """
        Parse and validate label file entries with memory-efficient options.
        
        Args:
            source_file: Path to the label file
            use_lazy_parsing: Use lazy parsing for large files
            
        Returns:
            Dictionary mapping filename to original line text
        """
        if not FileSystemOperations.check_file_exists(source_file):
            logger.error(f"Source label file not found: {source_file}")
            return {}

        valid_label_dict = {}
        parse_errors = 0
        max_errors = 100  # Limit error reporting

        try:
            # Choose parsing method based on file size and preference
            if use_lazy_parsing or FileSystemOperations.get_file_size(source_file) > 10 * 1024 * 1024:  # 10MB
                parsed_lines_iter = FileParser.parse_file_lazy(source_file, ',')
            else:
                parsed_lines_iter = FileParser.parse_file(source_file, ',')

            for parsed_line in parsed_lines_iter:
                try:
                    # Check for NaN values
                    if FileParser.contains_nan_values(parsed_line.parts):
                        raise ValueError(f"NaN values found in line: {parsed_line.original_text}")

                    # Validate numeric parts
                    if not FileParser.validate_numeric_parts(parsed_line.parts, 3):
                        raise ValueError(f"Invalid numeric values in line: {parsed_line.original_text}")

                    # Parse the label entry
                    entry = LabelFileProcessor.parse_label_entry(
                        parsed_line.original_text, parsed_line.parts
                    )

                    # Check for duplicate entries
                    if entry.filename in valid_label_dict:
                        logger.warning(f"Duplicate entry found for {entry.filename}, keeping first occurrence")
                        continue

                    valid_label_dict[entry.filename] = parsed_line.original_text
                
                except ValueError as e:
                    parse_errors += 1
                    if parse_errors <= max_errors:
                        logger.warning(f"Skipping line {parsed_line.line_number}: {e}")
                    elif parse_errors == max_errors + 1:
                        logger.warning(f"Too many parse errors, suppressing further error messages...")
                    continue
        
        except (IOError, OSError, FileNotFoundError):
            return {}
        
        if parse_errors > 0:
            logger.info(f"Skipped {parse_errors} invalid lines while processing {source_file}")
        
        logger.info(f"Loaded {len(valid_label_dict)} valid label entries from {source_file}")
        return valid_label_dict
    
    @staticmethod
    def process_label_files(indices: Set[str], cameras: Dict[str, str], 
                          source_dir: Path, dest_dir: Path, 
                          label_dir: str, label_file: str,
                          files_dict: Dict[str, Dict[str, str]] = None,
                          action_name: str = "Processed") -> None:
        """
        Process label files by filtering entries for given indices and writing to destination.
        
        Args:
            indices: Set of file indices to include
            cameras: Dictionary mapping camera names
            source_dir: Directory to read label files from (if files_dict not provided)
            dest_dir: Directory to write label files to
            label_dir: Sub directory of label file
            label_file: File containing labels
            files_dict: Pre-loaded files dictionary (optional, will load from source if None)
            action_name: Description of action for logging (e.g., "Copied", "Updated")
        """
        processed_cameras = []
        failed_cameras = []
        
        for camera_raw, camera_proc in cameras.items():
            source_label_path = source_dir / camera_proc / label_dir / label_file
            dest_label_path = dest_dir / camera_proc / label_dir / label_file
            
            try:
                # Use provided files_dict or load from source
                if files_dict and camera_proc in files_dict:
                    current_files_dict = files_dict[camera_proc]
                else:
                    if not source_label_path.exists():
                        logger.warning(f"Source label file not found: {source_label_path}")
                        failed_cameras.append(camera_proc)
                        continue
                    current_files_dict = LabelFileProcessor.get_valid_label_files(source_label_path)
                
                # Filter for specified indices with better performance
                valid_lines = []
                found_indices = set()
                
                for index in indices:
                    filename_key = f"{index}.jpg"
                    if filename_key in current_files_dict:
                        valid_lines.append(current_files_dict[filename_key])
                        found_indices.add(index)
                
                # Log missing indices if significant
                missing_indices = indices - found_indices
                if missing_indices:
                    logger.debug(f"Missing {len(missing_indices)} indices for {camera_proc}")
                
                # Write filtered labels with sorted output for consistency
                FileSystemOperations.write_lines_to_file(dest_label_path, sorted(valid_lines))
                logger.info(f"{action_name} {len(valid_lines)} label entries for {camera_proc}")
                processed_cameras.append(camera_proc)
                
            except Exception as e:
                logger.error(f"Failed to process label file for {camera_proc}: {e}")
                failed_cameras.append(camera_proc)
                raise
        
        # Summary logging
        if processed_cameras:
            logger.info(f"Successfully processed label files for: {', '.join(processed_cameras)}")
        if failed_cameras:
            logger.warning(f"Failed to process label files for: {', '.join(failed_cameras)}")


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
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes."""
        return FileSystemOperations.get_file_size(file_path)
    
    # File parsing operations
    @staticmethod
    def parse_file(file_path: Path, separator: str = " ") -> List[ParsedLine]:
        """Parse a text file and return structured data."""
        return FileParser.parse_file(file_path, separator)
    
    @staticmethod
    def parse_file_lazy(file_path: Path, separator: str = " ") -> Iterator[ParsedLine]:
        """Lazy parse a text file to save memory."""
        return FileParser.parse_file_lazy(file_path, separator)
    
    @staticmethod
    def check_for_nan(parts: List[str]) -> bool:
        """Check if any part contains NaN values."""
        return FileParser.contains_nan_values(parts)
    
    @staticmethod
    def validate_numeric_parts(parts: List[str], expected_count: int = None) -> bool:
        """Validate that parts contain valid numeric values."""
        return FileParser.validate_numeric_parts(parts, expected_count)
    
    # File validation operations
    @staticmethod
    def check_file_for_nan(file_path: Path, separator: str = " ") -> bool:
        """Check if a text file contains NaN values."""
        return FileValidator.check_file_for_nan(file_path, separator)
    
    @staticmethod
    def batch_validate_files(file_paths: List[Path], separator: str = " ") -> Dict[Path, bool]:
        """Validate multiple files efficiently."""
        return FileValidator.batch_validate_files(file_paths, separator)
    
    @staticmethod
    def clear_validation_cache() -> None:
        """Clear the validation cache."""
        return FileValidator.clear_validation_cache()
    
    # Label file operations
    @staticmethod
    def parse_label_line(line: str, parts: List[str]) -> LabelEntry:
        """Parse a single line from label file into a LabelEntry."""
        return LabelFileProcessor.parse_label_entry(line, parts)
    
    @staticmethod
    def get_valid_label_files(source_file: Path, use_lazy_parsing: bool = False) -> Dict[str, str]:
        """Parse and validate label file entries."""
        return LabelFileProcessor.get_valid_label_files(source_file, use_lazy_parsing)
    
    @staticmethod
    def process_label_files(indices: Set[str], cameras: Dict[str, str], 
                            source_dir: Path, dest_dir: Path, 
                            label_dir: str, label_file: str,
                            files_dict: Dict[str, Dict[str, str]] = None, 
                            action_name: str = "Processed") -> None:
        """Process label files with filtering and validation."""
        return LabelFileProcessor.process_label_files(
            indices, cameras, source_dir, dest_dir, label_dir, label_file, files_dict, action_name
        )
    
    # Filename utilities
    @staticmethod
    def get_file_indices(filenames: Set[str]) -> Set[str]:
        """Extract indices from filenames."""
        return FilenameUtils.extract_indices(filenames)
    
    @staticmethod
    def group_filenames_by_extension(filenames: Set[str]) -> Dict[str, Set[str]]:
        """Group filenames by their extensions."""
        return FilenameUtils.group_by_extension(filenames)
    
    @staticmethod
    def validate_filename_format(filename: str, pattern: str = None) -> bool:
        """Validate filename format."""
        return FilenameUtils.validate_filename_format(filename, pattern)