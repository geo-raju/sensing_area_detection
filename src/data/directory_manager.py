import logging
from pathlib import Path
from typing import List
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DirectoryManager:

    def create_dirs(self, base_path: Path, camdirs: List[str], subdirs: List[str]) -> None:
        """Create multiple subdirectories under a base path atomically."""
        logger.debug(f"Creating directory structure at {base_path}")
        try:
            for camdir in camdirs:
                for subdir in subdirs:
                    dir_path = base_path / camdir / subdir
                    dir_path.mkdir(parents=True, exist_ok=True)     
        except OSError as e:
            logger.error(f"Failed to create directories under {base_path}: {e}")
            raise

    def validate_dirs(self, base_path: Path, camdirs: List[str], subdirs: List[str], splitdirs: List[str] = None) -> None:
        with threading.lock():
            try:
                dirs_to_validate = DirectoryManager.get_dirs_path(base_path=base_path, camdirs=camdirs , 
                                                                subdirs=subdirs, splitdirs=splitdirs)
                missing_dirs = []
                # validate all directories
                for directory in dirs_to_validate:
                    if not DirectoryManager.check_dir_exists(directory):
                        missing_dirs.append(directory)
                if missing_dirs:
                    return True
                return False
            except OSError as e:
                logger.error(f"Failed to validate directories under {base_path}: {e}")
                raise

    def get_dirs_path(self, base_path: Path, camdirs: List[str], subdirs: List[str], splitdirs: List[str] = None) -> List[Path]:
        dirs = []
                
        # Use [""] as default to create directly under base_path when splitdirs is empty
        split_levels = splitdirs if splitdirs else [""]
        
        for splitdir in split_levels:
            for camdir in camdirs:
                for subdir in subdirs:
                    if splitdir:  # If splitdir exists, include it in path
                        dir_path = base_path / splitdir / camdir / subdir
                    else:  # If splitdir is empty, skip it in path
                        dir_path = base_path / camdir / subdir
                    dirs.append(dir_path)
        return dirs

    def check_dir_exists(self, dir_path: Path) -> bool:
        """Check if a dir exists"""
        try:
            return dir_path.exists() and dir_path.is_file()
        except OSError as e:
            logger.debug(f"Error checking dir existence {dir_path}: {e}")
            return False