"""Data organisation configuration."""
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# Data split configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Camera and data type mappings
CAMERA_CONFIG = {
    'camera0': 'left',
    'camera1': 'right'
}

DATA_TYPE_CONFIG = {
    'laser_off': 'images',
    'line_annotation_sample': 'probe_axis'
}
