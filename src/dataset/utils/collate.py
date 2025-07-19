import torch
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)

from src.dataset.core.data_structures import SampleData


def custom_collate_fn(batch: List['SampleData']) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for SampleData objects.
    
    Handles:
    - Custom dataclass collation
    - Variable-length axis points with padding
    - Proper tensor stacking
    - Error handling for malformed batches
    
    Args:
        batch: List of SampleData objects
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    if not batch:
        return {}
    
    try:
        # Extract each field from all samples
        left_images = torch.stack([sample.left_image for sample in batch])
        right_images = torch.stack([sample.right_image for sample in batch])
        left_labels = torch.stack([sample.left_label for sample in batch])
        right_labels = torch.stack([sample.right_label for sample in batch])
        filenames = [sample.filename for sample in batch]
        
        # Handle variable-length axis points by padding
        max_left_axis = max(sample.left_axis.shape[0] for sample in batch)
        max_right_axis = max(sample.right_axis.shape[0] for sample in batch)
        
        # Create masks to track valid points (not padding)
        left_axis_masks = []
        right_axis_masks = []
        left_axis_padded = []
        right_axis_padded = []
        
        for sample in batch:
            left_axis = sample.left_axis
            right_axis = sample.right_axis
            
            # Create masks for valid points
            left_mask = torch.ones(max_left_axis, dtype=torch.bool)
            right_mask = torch.ones(max_right_axis, dtype=torch.bool)
            
            # Pad left axis if necessary
            if left_axis.shape[0] < max_left_axis:
                padding = torch.zeros(max_left_axis - left_axis.shape[0], 2, dtype=left_axis.dtype)
                left_axis = torch.cat([left_axis, padding], dim=0)
                left_mask[sample.left_axis.shape[0]:] = False
            
            # Pad right axis if necessary  
            if right_axis.shape[0] < max_right_axis:
                padding = torch.zeros(max_right_axis - right_axis.shape[0], 2, dtype=right_axis.dtype)
                right_axis = torch.cat([right_axis, padding], dim=0)
                right_mask[sample.right_axis.shape[0]:] = False
                
            left_axis_padded.append(left_axis)
            right_axis_padded.append(right_axis)
            left_axis_masks.append(left_mask)
            right_axis_masks.append(right_mask)
        
        # Stack all tensors
        left_axis_batch = torch.stack(left_axis_padded)
        right_axis_batch = torch.stack(right_axis_padded)
        left_axis_masks_batch = torch.stack(left_axis_masks)
        right_axis_masks_batch = torch.stack(right_axis_masks)
        
        return {
            'left_image': left_images,
            'right_image': right_images,
            'left_label': left_labels,
            'right_label': right_labels,
            'left_axis': left_axis_batch,
            'right_axis': right_axis_batch,
            'left_axis_mask': left_axis_masks_batch,  # Mask for valid points
            'right_axis_mask': right_axis_masks_batch,  # Mask for valid points
            'filename': filenames
        }
        
    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {e}")
        logger.error(f"Batch size: {len(batch)}")
        for i, sample in enumerate(batch):
            logger.error(f"Sample {i}: left_axis shape={sample.left_axis.shape}, "
                        f"right_axis shape={sample.right_axis.shape}")
        raise


def flexible_collate_fn(batch: List['SampleData']) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    More flexible collate function that handles empty axis points gracefully.
    
    This version handles cases where some samples might have no axis points at all.
    """
    if not batch:
        return {}
    
    try:
        # Extract basic fields
        left_images = torch.stack([sample.left_image for sample in batch])
        right_images = torch.stack([sample.right_image for sample in batch])
        left_labels = torch.stack([sample.left_label for sample in batch])
        right_labels = torch.stack([sample.right_label for sample in batch])
        filenames = [sample.filename for sample in batch]
        
        # Handle axis points more flexibly
        left_axis_sizes = [sample.left_axis.shape[0] for sample in batch]
        right_axis_sizes = [sample.right_axis.shape[0] for sample in batch]
        
        # If all samples have empty axis points, return empty tensors
        if all(size == 0 for size in left_axis_sizes + right_axis_sizes):
            return {
                'left_image': left_images,
                'right_image': right_images,
                'left_label': left_labels,
                'right_label': right_labels,
                'left_axis': torch.empty(len(batch), 0, 2, dtype=torch.float32),
                'right_axis': torch.empty(len(batch), 0, 2, dtype=torch.float32),
                'left_axis_mask': torch.empty(len(batch), 0, dtype=torch.bool),
                'right_axis_mask': torch.empty(len(batch), 0, dtype=torch.bool),
                'filename': filenames
            }
        
        # Otherwise, pad to maximum size
        max_left_axis = max(left_axis_sizes) if left_axis_sizes else 0
        max_right_axis = max(right_axis_sizes) if right_axis_sizes else 0
        
        left_axis_padded = []
        right_axis_padded = []
        left_axis_masks = []
        right_axis_masks = []
        
        for sample in batch:
            left_axis = sample.left_axis
            right_axis = sample.right_axis
            
            # Handle left axis
            if max_left_axis > 0:
                left_mask = torch.zeros(max_left_axis, dtype=torch.bool)
                if left_axis.shape[0] > 0:
                    left_mask[:left_axis.shape[0]] = True
                    if left_axis.shape[0] < max_left_axis:
                        padding = torch.zeros(max_left_axis - left_axis.shape[0], 2, dtype=left_axis.dtype)
                        left_axis = torch.cat([left_axis, padding], dim=0)
                else:
                    left_axis = torch.zeros(max_left_axis, 2, dtype=torch.float32)
            else:
                left_axis = torch.empty(0, 2, dtype=torch.float32)
                left_mask = torch.empty(0, dtype=torch.bool)
            
            # Handle right axis
            if max_right_axis > 0:
                right_mask = torch.zeros(max_right_axis, dtype=torch.bool)
                if right_axis.shape[0] > 0:
                    right_mask[:right_axis.shape[0]] = True
                    if right_axis.shape[0] < max_right_axis:
                        padding = torch.zeros(max_right_axis - right_axis.shape[0], 2, dtype=right_axis.dtype)
                        right_axis = torch.cat([right_axis, padding], dim=0)
                else:
                    right_axis = torch.zeros(max_right_axis, 2, dtype=torch.float32)
            else:
                right_axis = torch.empty(0, 2, dtype=torch.float32)
                right_mask = torch.empty(0, dtype=torch.bool)
            
            left_axis_padded.append(left_axis)
            right_axis_padded.append(right_axis)
            left_axis_masks.append(left_mask)
            right_axis_masks.append(right_mask)
        
        # Stack tensors
        left_axis_batch = torch.stack(left_axis_padded) if left_axis_padded else torch.empty(len(batch), 0, 2)
        right_axis_batch = torch.stack(right_axis_padded) if right_axis_padded else torch.empty(len(batch), 0, 2)
        left_axis_masks_batch = torch.stack(left_axis_masks) if left_axis_masks else torch.empty(len(batch), 0, dtype=torch.bool)
        right_axis_masks_batch = torch.stack(right_axis_masks) if right_axis_masks else torch.empty(len(batch), 0, dtype=torch.bool)
        
        return {
            'left_image': left_images,
            'right_image': right_images,
            'left_label': left_labels,
            'right_label': right_labels,
            'left_axis': left_axis_batch,
            'right_axis': right_axis_batch,
            'left_axis_mask': left_axis_masks_batch,
            'right_axis_mask': right_axis_masks_batch,
            'filename': filenames
        }
        
    except Exception as e:
        logger.error(f"Error in flexible_collate_fn: {e}")
        logger.error(f"Batch size: {len(batch)}")
        for i, sample in enumerate(batch):
            logger.error(f"Sample {i}: left_axis shape={sample.left_axis.shape}, "
                        f"right_axis shape={sample.right_axis.shape}, "
                        f"left_image shape={sample.left_image.shape}")
        raise