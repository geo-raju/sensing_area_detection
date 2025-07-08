import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SensingAreaDataset(Dataset):
    """
    Dataset class for sensing area detection compatible with the data organisation pipeline.
    """
    
    def __init__(
        self, 
        root: str, 
        transform: Optional[A.Compose] = None,
        validate_structure: bool = True
    ):
        """
        Initialise the dataset.
        
        Args:
            root: Root directory containing the organised data
            transform: Albumentations transform to apply to images
            validate_structure: Whether to validate directory structure on init
        """
        self.root = Path(root)
        self.transform = transform
        
        # Define directory paths based on organised structure
        self.left_img_dir = self.root / 'left' / 'images'
        self.right_img_dir = self.root / 'right' / 'images'
        self.label_dir = self.root / 'left' / 'labels'
        self.axis_dir = self.root / 'left' / 'probe_axis'
        
        if validate_structure:
            self._validate_structure()
        
        # Get all image filenames
        self.filenames = sorted([
            f for f in os.listdir(self.left_img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not self.filenames:
            raise ValueError(f"No image files found in {self.left_img_dir}")
        
        logger.info(f"Initialised dataset with {len(self.filenames)} samples from {root}")
    
    def _validate_structure(self) -> None:
        """
        Validate that the expected directory structure exists.
        
        Raises:
            FileNotFoundError: If required directories are missing
        """
        required_dirs = [
            self.left_img_dir,
            self.right_img_dir,
            self.label_dir,
            self.axis_dir
        ]
        
        missing_dirs = [d for d in required_dirs if not d.exists()]
        
        if missing_dirs:
            missing_paths = [str(d) for d in missing_dirs]
            raise FileNotFoundError(
                f"Required directories missing: {missing_paths}. "
                "Please ensure data is organised using the data organisation pipeline."
            )
        
        logger.info(f"Directory structure validation passed for {self.root}")
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load an image using OpenCV and convert to RGB.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            RGB image as numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image using OpenCV (BGR format)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_label(self, label_path: Path) -> Tuple[float, float]:
        """
        Load label coordinates from text file.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            Tuple of (x, y) coordinates
            
        Raises:
            FileNotFoundError: If label file doesn't exist
            ValueError: If label format is invalid
        """
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        try:
            with open(label_path, 'r') as f:
                line = f.read().strip()
                if not line:
                    raise ValueError(f"Empty label file: {label_path}")
                
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid label format in {label_path}. Expected 2 values, got {len(parts)}")
                
                x, y = map(float, parts)
                return x, y
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Error reading label from {label_path}: {e}")
    
    def _load_axis_points(self, axis_path: Path) -> np.ndarray:
        """
        Load probe axis points from text file.
        
        Args:
            axis_path: Path to the axis points file
            
        Returns:
            Array of shape (N, 2) containing axis points
            
        Raises:
            FileNotFoundError: If axis file doesn't exist
            ValueError: If axis format is invalid
        """
        if not axis_path.exists():
            raise FileNotFoundError(f"Axis file not found: {axis_path}")
        
        try:
            axis_pts = np.loadtxt(axis_path)
            
            # Ensure 2D array
            if axis_pts.ndim == 1:
                axis_pts = axis_pts.reshape(1, -1)
            
            # Validate shape
            if axis_pts.shape[1] != 2:
                raise ValueError(f"Invalid axis points shape: {axis_pts.shape}. Expected (N, 2)")
            
            return axis_pts
        except (ValueError, OSError) as e:
            raise ValueError(f"Error reading axis points from {axis_path}: {e}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - 'left': Left image tensor
            - 'right': Right image tensor  
            - 'label': Label coordinates tensor
            - 'axis': Axis points tensor
        """
        if idx >= len(self.filenames):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.filenames)}")
        
        filename = self.filenames[idx]
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Load images
        left_img = self._load_image(self.left_img_dir / filename)
        right_img = self._load_image(self.right_img_dir / filename)
        
        # Load label
        label_path = self.label_dir / f"{filename_no_ext}.txt"
        x, y = self._load_label(label_path)
        
        # Load probe axis points
        axis_path = self.axis_dir / f"{filename_no_ext}.txt"
        axis_pts = self._load_axis_points(axis_path)
        
        # Apply transforms if specified
        if self.transform:
            augmented_left = self.transform(image=left_img)
            left_img = augmented_left['image']
            
            augmented_right = self.transform(image=right_img)
            right_img = augmented_right['image']
        
        return {
            "left": left_img,
            "right": right_img,
            "label": torch.tensor([x, y], dtype=torch.float32),
            "axis": torch.tensor(axis_pts, dtype=torch.float32),
            "filename": filename
        }
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.filenames)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a sample without loading the full data.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with sample information
        """
        if idx >= len(self.filenames):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.filenames)}")
        
        filename = self.filenames[idx]
        filename_no_ext = os.path.splitext(filename)[0]
        
        return {
            "filename": filename,
            "filename_no_ext": filename_no_ext,
            "left_img_path": str(self.left_img_dir / filename),
            "right_img_path": str(self.right_img_dir / filename),
            "label_path": str(self.label_dir / f"{filename_no_ext}.txt"),
            "axis_path": str(self.axis_dir / f"{filename_no_ext}.txt")
        }