"""Centralised logging configuration."""
import logging

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(log_level=LOG_LEVEL, log_format=LOG_FORMAT):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        force=True
    )