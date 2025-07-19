import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SampleData:
    """Data structure for a single dataset sample."""
    left_image: torch.Tensor
    right_image: torch.Tensor
    left_label: torch.Tensor
    right_label: torch.Tensor
    left_axis: torch.Tensor
    right_axis: torch.Tensor
    filename: str

    def __getitem__(self, key):
        """Allow dictionary-style access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key):
        """Allow 'in' operator for backward compatibility."""
        return hasattr(self, key)

    def keys(self):
        """Return keys for iteration compatibility."""
        return self.__dataclass_fields__.keys()


class DatasetError(Exception):
    """Custom exception for dataset-related errors."""
    pass