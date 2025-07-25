"""
Configuration module for the Data Organisation system.
Contains all configuration constants and factory functions.
"""

from pathlib import Path
from typing import Dict, Final, List

# Directory paths
BASE_DIR: Final[Path] = Path(__file__).parent.parent
RAW_DIR_PATH: Final[Path] = BASE_DIR / 'data' / 'raw'
CLEAN_DIR_PATH: Final[Path] = BASE_DIR / 'data' / 'clean'
PROC_DIR_PATH: Final[Path] = BASE_DIR / 'data' / 'processed'

# Directory structure constants
IMG_RAW_DIR: Final[str] = 'laser_off'
IMG_PROC_DIR: Final[str] = 'images'
PROBE_RAW_DIR: Final[str] = 'line_annotation_sample'
PROBE_PROC_DIR: Final[str] = 'probe_axis'
LABEL_RAW_DIR: Final[str] = 'laserptGT'
LABEL_PROC_DIR: Final[str] = 'labels'
LABEL_FILE: Final[str] = 'CenterPt.txt'
DEPTH_RAW_DIR: Final[str] = 'depthGT'
DEPTH_PROC_DIR: Final[str] = 'depth_labels'
LEFT_CAM_RAW_DIR: Final[str] = 'camera0'
LEFT_CAM_PROC_DIR: Final[str] = 'left'
RIGHT_CAM_RAW_DIR: Final[str] = 'camera1'
RIGHT_CAM_PROC_DIR: Final[str] = 'right'

# Data split configuration
SPLIT_NAMES:Final[List] = ['train', 'val', 'test']
TRAIN_RATIO: Final[float] = 0.7
VAL_RATIO: Final[float] = 0.15
TEST_RATIO: Final[float] = 0.15
RANDOM_STATE: Final[int] = 42

# Camera and data type mappings
CAMERA_CONFIG: Final[Dict[str, str]] = {
    LEFT_CAM_RAW_DIR: LEFT_CAM_PROC_DIR,
    RIGHT_CAM_RAW_DIR: RIGHT_CAM_PROC_DIR
}

DATA_TYPE_CONFIG: Final[Dict[str, str]] = {
    IMG_RAW_DIR: IMG_PROC_DIR,
    PROBE_RAW_DIR: PROBE_PROC_DIR,
    DEPTH_RAW_DIR: DEPTH_PROC_DIR
}