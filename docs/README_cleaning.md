# Data Cleaning Process

A robust, multi-threaded data cleaning pipeline for stereo camera datasets that removes NaN values, validates data consistency, and ensures synchronized files across cameras.

## Overview

The cleaning process handles three main data types:
- **Images** (.jpg files)
- **Probe data** (.txt files containing sensor measurements)
- **Depth maps** (.npy files)
- **Labels** (coordinate annotations in CSV format)

## Quick Start

```bash
# Basic cleaning with default settings
python clean_data.py

# Custom directories and parallel processing
python clean_data.py --raw-dir /path/to/raw --cleaned-dir /path/to/clean --max-workers 8

# Validate data without cleaning
python clean_data.py --validate-only
```

## Directory Structure

### Input (Raw Data)
```
raw_data/
├── left_cam/
│   ├── images/          # .jpg files
│   ├── probe_data/      # .txt files
│   ├── depth_maps/      # .npy files
│   └── labels/          # coordinate labels
└── right_cam/
    ├── images/
    ├── probe_data/
    ├── depth_maps/
    └── labels/
```

### Output (Cleaned Data)
```
cleaned_data/
├── left_cam_proc/       # Cleaned left camera data
└── right_cam_proc/      # Cleaned right camera data
```

## Cleaning Pipeline

1. **Label Processing**: Parse and validate coordinate labels, find common files between cameras
2. **Probe Validation**: Check probe files for NaN values and data integrity
3. **File Synchronization**: Ensure all file types exist for each valid index
4. **Data Copying**: Copy validated files with parallel processing
5. **Consistency Check**: Verify final dataset integrity

## Command Line Options

### Basic Options
- `--raw-dir PATH`: Input directory path
- `--cleaned-dir PATH`: Output directory path
- `--max-workers N`: Number of parallel threads (default: 4)

### Operation Modes
- `--validate-only`: Check data without cleaning
- `--force-clean`: Overwrite existing cleaned data
- `--disable-resume`: Disable checkpoint/resume capability

### Logging
- `--verbose`: Enable debug output
- `--quiet`: Suppress progress messages
- `--log-level LEVEL`: Set specific log level

### Maintenance
- `--clear-cache`: Clear validation cache before starting

## Features

### Performance
- **Parallel Processing**: Multi-threaded file operations
- **Memory Optimization**: Lazy loading for large files
- **Caching**: Validation results cached to avoid reprocessing

### Reliability
- **Resume Capability**: Automatically resume interrupted cleaning
- **Data Validation**: Comprehensive NaN detection and format checking
- **Error Recovery**: Graceful handling of corrupted or missing files
- **Progress Tracking**: Real-time progress monitoring with time estimates

### Safety
- **Atomic Operations**: Prevents partial file writes
- **Backup State**: Maintains processing state for recovery
- **Validation First**: Checks data integrity before cleaning

## Examples

```bash
# Standard cleaning
python clean_data.py --raw-dir ./raw_data --cleaned-dir ./clean_data

# High-performance cleaning with validation
python clean_data.py --max-workers 12 --verbose

# Test data integrity without cleaning
python clean_data.py --validate-only --raw-dir ./suspicious_data

# Clean existing directory (overwrite)
python clean_data.py --force-clean --clear-cache

# Quiet operation for automated scripts
python clean_data.py --quiet --disable-resume
```

## Output Statistics

The process logs detailed statistics:
- Files processed per camera and data type
- Validation success rates
- Processing time and throughput
- Error counts and types

## Error Handling

Common issues and solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| No common files | Mismatched camera data | Check file naming consistency |
| NaN values found | Corrupted probe data | Review data collection process |
| Permission denied | Insufficient file access | Check directory permissions |
| Out of memory | Large dataset | Reduce `--max-workers` or process in batches |

## Dependencies

- Python 3.7+
- Standard library modules (pathlib, threading, multiprocessing)
- NumPy (for .npy file handling)
- Custom modules: `config.data_config`, `src.data.cleaner`, `src.data.processor`

## Performance Tips

- Use `--max-workers 4-8` for optimal performance on most systems
- Enable `--clear-cache` if data has changed significantly
- Use `--validate-only` first for large datasets to estimate processing time
- Monitor disk space - cleaned data may be substantial

---

For technical details, see the source code documentation in `cleaner.py` and `processor.py`.