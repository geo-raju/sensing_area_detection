"""
Configuration module for the Data Organisation system.
Contains all configuration constants and factory functions.
"""

from pathlib import Path
from typing import Dict, Final

# Directory paths
BASE_DIR: Final[Path] = Path(__file__).parent.parent
RAW_DIR_PATH: Final[Path] = BASE_DIR / 'dataset' / 'raw'
PROC_DIR_PATH: Final[Path] = BASE_DIR / 'dataset' / 'processed'

# Directory structure constants
IMG_RAW_DIR: Final[str] = 'laser_off'
IMG_PROC_DIR: Final[str] = 'images'
PROBE_RAW_DIR: Final[str] = 'line_annotation_sample'
PROBE_PROC_DIR: Final[str] = 'probe_axis'
LABEL_RAW_DIR: Final[str] = 'laserptGT'
LABEL_PROC_DIR: Final[str] = 'labels'
LABEL_FILE: Final[str] = 'CenterPt.txt'

# Data split configuration
TRAIN_RATIO: Final[float] = 0.7
VAL_RATIO: Final[float] = 0.15
TEST_RATIO: Final[float] = 0.15
RANDOM_STATE: Final[int] = 42

# Camera and data type mappings
CAMERA_CONFIG: Final[Dict[str, str]] = {
    'camera0': 'left',
    'camera1': 'right'
}

DATA_TYPE_CONFIG: Final[Dict[str, str]] = {
    'laser_off': 'images',
    'line_annotation_sample': 'probe_axis'
}