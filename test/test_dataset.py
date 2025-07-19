import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import cv2
from torch.utils.data import DataLoader

from src.dataset.core.dataset import SensingAreaDataset
from src.dataset.core.data_structures import SampleData, DatasetError
from config.data_config import CAMERA_CONFIG, IMG_PROC_DIR, LABEL_PROC_DIR, PROBE_PROC_DIR, LABEL_FILE
from src.dataset.utils import custom_collate_fn  # Import your custom collate function


class TestDatasetInitialization:
    """Test dataset initialization and configuration."""
    
    def test_valid_subset_initialization(self, mock_dataset_structure):
        """Test initialization with valid subsets."""
        for subset in ['train', 'val', 'test']:
            dataset = SensingAreaDataset(
                root=mock_dataset_structure,
                subset=subset,
                validate_structure=False
            )
            assert dataset.subset == subset
            assert len(dataset) >= 0
    
    def test_invalid_subset_raises_error(self, mock_dataset_structure):
        """Test that invalid subset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid subset 'invalid'"):
            SensingAreaDataset(
                root=mock_dataset_structure,
                subset='invalid'
            )
    
    def test_seed_initialization(self, mock_dataset_structure):
        """Test that seed is properly set."""
        dataset = SensingAreaDataset(
            root=mock_dataset_structure,
            subset='train',
            seed=42,
            validate_structure=False
        )
        assert dataset.seed == 42
    
    def test_lazy_loading_configuration(self, mock_dataset_structure):
        """Test lazy loading configuration."""
        dataset_lazy = SensingAreaDataset(
            root=mock_dataset_structure,
            subset='train',
            lazy_load=True,
            validate_structure=False
        )
        
        dataset_eager = SensingAreaDataset(
            root=mock_dataset_structure,
            subset='train',
            lazy_load=False,
            validate_structure=False
        )
        
        assert dataset_lazy.lazy_load == True
        assert dataset_eager.lazy_load == False


class TestDatasetStructureValidation:
    """Test dataset structure validation."""
    
    def test_structure_validation_success(self, complete_dataset_structure):
        """Test successful structure validation."""
        dataset = SensingAreaDataset(
            root=complete_dataset_structure,
            subset='train',
            validate_structure=True
        )
        # Should not raise any exception
        assert dataset is not None
    
    def test_structure_validation_failure(self, incomplete_dataset_structure):
        """Test structure validation failure."""
        with pytest.raises(FileNotFoundError):
            SensingAreaDataset(
                root=incomplete_dataset_structure,
                subset='train',
                validate_structure=True
            )
    
    def test_skip_structure_validation(self, incomplete_dataset_structure):
        """Test skipping structure validation."""
        dataset = SensingAreaDataset(
            root=incomplete_dataset_structure,
            subset='train',
            validate_structure=False
        )
        # Should initialize without error
        assert dataset is not None


class TestDatasetItemAccess:
    """Test dataset item access and indexing."""
    
    def test_len_method(self, populated_dataset):
        """Test __len__ method."""
        dataset, expected_count = populated_dataset
        assert len(dataset) == expected_count
        assert isinstance(len(dataset), int)
    
    def test_valid_getitem(self, populated_dataset):
        """Test __getitem__ with valid indices."""
        dataset, expected_count = populated_dataset
        
        if expected_count > 0:
            # Test first item
            sample = dataset[0]
            assert isinstance(sample, SampleData)
            self._validate_sample_structure(sample)
            
            # Test last item
            if expected_count > 1:
                sample = dataset[expected_count - 1]
                assert isinstance(sample, SampleData)
                self._validate_sample_structure(sample)
    
    def test_invalid_getitem_indices(self, populated_dataset):
        """Test __getitem__ with invalid indices."""
        dataset, expected_count = populated_dataset
        
        # Test negative index beyond range
        with pytest.raises(IndexError):
            dataset[-expected_count - 1]
        
        # Test positive index beyond range
        with pytest.raises(IndexError):
            dataset[expected_count]
        
        # Test very large index
        with pytest.raises(IndexError):
            dataset[1000000]
    
    def test_getitem_with_seed_reproducibility(self, populated_dataset):
        """Test reproducibility with seed."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        # Create two datasets with same seed
        dataset1 = SensingAreaDataset(
            root=dataset.root,
            subset=dataset.subset,
            seed=42,
            validate_structure=False
        )
        dataset2 = SensingAreaDataset(
            root=dataset.root,
            subset=dataset.subset,
            seed=42,
            validate_structure=False
        )
        
        # Same indices should produce identical results
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        
        # Compare tensors
        assert torch.equal(sample1.left_image, sample2.left_image)
        assert torch.equal(sample1.right_image, sample2.right_image)
    
    def _validate_sample_structure(self, sample: SampleData):
        """Validate sample data structure."""
        assert hasattr(sample, 'left_image')
        assert hasattr(sample, 'right_image')
        assert hasattr(sample, 'left_label')
        assert hasattr(sample, 'right_label')
        assert hasattr(sample, 'left_axis')
        assert hasattr(sample, 'right_axis')
        assert hasattr(sample, 'filename')
        
        # Validate tensor types
        assert isinstance(sample.left_image, torch.Tensor)
        assert isinstance(sample.right_image, torch.Tensor)
        assert isinstance(sample.left_label, torch.Tensor)
        assert isinstance(sample.right_label, torch.Tensor)
        assert isinstance(sample.left_axis, torch.Tensor)
        assert isinstance(sample.right_axis, torch.Tensor)
        assert isinstance(sample.filename, str)
        
        # Validate tensor shapes
        assert sample.left_image.dim() == 3  # C, H, W
        assert sample.right_image.dim() == 3  # C, H, W
        assert sample.left_label.dim() == 1 and sample.left_label.shape[0] == 2  # (x, y)
        assert sample.right_label.dim() == 1 and sample.right_label.shape[0] == 2  # (x, y)
        assert sample.left_axis.dim() == 2 and sample.left_axis.shape[1] == 2  # (N, 2)
        assert sample.right_axis.dim() == 2 and sample.right_axis.shape[1] == 2  # (N, 2)


class TestDatasetErrorHandling:
    """Test dataset error handling."""
    
    def test_getitem_with_corrupted_image(self, populated_dataset):
        """Test handling of corrupted images."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        # Mock corrupted image loading
        with patch.object(dataset.image_loader, 'load_image', side_effect=Exception("Corrupted image")):
            with pytest.raises(DatasetError, match="Failed to load sample"):
                dataset[0]
    
    def test_getitem_with_missing_label(self, populated_dataset):
        """Test handling of missing labels."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        # Mock missing label
        with patch.object(dataset.label_loader, 'get_label', side_effect=DatasetError("Label not found")):
            with pytest.raises(DatasetError, match="Failed to load sample"):
                dataset[0]
    
    def test_empty_dataset_handling(self, empty_dataset_structure):
        """Test handling of empty dataset."""
        dataset = SensingAreaDataset(
            root=empty_dataset_structure,
            subset='train',
            validate_structure=False
        )
        assert len(dataset) == 0


class TestDatasetUtilityMethods:
    """Test utility methods."""
    
    def test_get_sample_info(self, populated_dataset):
        """Test get_sample_info method."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        info = dataset.get_sample_info(0)
        
        assert isinstance(info, dict)
        assert 'filename' in info
        assert 'filename_no_ext' in info
        assert 'subset' in info
        assert 'left_img_path' in info
        assert 'right_img_path' in info
        assert 'left_axis_path' in info
        assert 'right_axis_path' in info
        assert 'transform_info' in info
        
        assert info['subset'] == dataset.subset
    
    def test_get_sample_info_invalid_index(self, populated_dataset):
        """Test get_sample_info with invalid index."""
        dataset, expected_count = populated_dataset
        
        with pytest.raises(IndexError):
            dataset.get_sample_info(expected_count)
    
    def test_get_subset_stats(self, populated_dataset):
        """Test get_subset_stats method."""
        dataset, expected_count = populated_dataset
        
        stats = dataset.get_subset_stats()
        
        assert isinstance(stats, dict)
        assert 'subset' in stats
        assert 'total_samples' in stats
        assert 'left_labels_count' in stats
        assert 'right_labels_count' in stats
        assert 'root_path' in stats
        assert 'subset_path' in stats
        assert 'transform_info' in stats
        
        assert stats['subset'] == dataset.subset
        assert stats['total_samples'] == expected_count
    
    def test_validate_sample_integrity(self, populated_dataset):
        """Test validate_sample_integrity method."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        integrity = dataset.validate_sample_integrity(0)
        
        assert isinstance(integrity, dict)
        required_keys = ['left_image', 'right_image', 'left_label', 'right_label', 'left_axis', 'right_axis']
        
        for key in required_keys:
            assert key in integrity
            assert isinstance(integrity[key], bool)
    
    def test_validate_all_samples(self, populated_dataset):
        """Test validate_all_samples method."""
        dataset, expected_count = populated_dataset
        
        issues = dataset.validate_all_samples()
        
        assert isinstance(issues, dict)
        expected_keys = [
            'missing_left_image', 'missing_right_image',
            'missing_left_label', 'missing_right_label',
            'missing_left_axis', 'missing_right_axis'
        ]
        
        for key in expected_keys:
            assert key in issues
            assert isinstance(issues[key], list)
    
    def test_get_camera_names(self, populated_dataset):
        """Test get_camera_names method."""
        dataset, _ = populated_dataset
        
        camera_names = dataset.get_camera_names()
        assert isinstance(camera_names, list)
        assert len(camera_names) > 0
        assert all(isinstance(name, str) for name in camera_names)
    
    def test_get_data_types(self, populated_dataset):
        """Test get_data_types method."""
        dataset, _ = populated_dataset
        
        data_types = dataset.get_data_types()
        assert isinstance(data_types, list)
        assert len(data_types) > 0
        assert all(isinstance(dtype, str) for dtype in data_types)

    def test_update_transform(self, populated_dataset):
        """Test update_transform method."""
        dataset, _ = populated_dataset
        
        # Test with None transform
        dataset.update_transform(None)
        assert dataset.transform_manager.transform is None
        
        # Test with a proper mock transform
        mock_transform = Mock()
        mock_transform.transforms = [Mock(), Mock()]  # Mock list of transforms
        
        dataset.update_transform(mock_transform)
        assert dataset.transform_manager.transform == mock_transform


class TestDatasetTransforms:
    """Test dataset transforms."""
    
    def test_no_transform(self, populated_dataset):
        """Test dataset with no transform."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        dataset.update_transform(None)
        sample = dataset[0]
        
        # Should still return valid SampleData
        assert isinstance(sample, SampleData)
    
    def test_with_mock_transform(self, populated_dataset):
        """Test dataset with mock transform."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        # Create a simple mock transform that just returns the input
        mock_transform = Mock()
        mock_transform.transforms = []  # Add the expected transforms attribute
        mock_transform.return_value = {
            'image': np.zeros((224, 224, 3), dtype=np.uint8),
            'keypoints': [[100.0, 100.0], [50.0, 50.0]]
        }
        mock_transform.get.return_value = {}
        
        with patch('albumentations.ReplayCompose.replay', return_value=mock_transform.return_value):
            dataset.update_transform(mock_transform)
            sample = dataset[0]
            
            assert isinstance(sample, SampleData)


class TestDatasetIntegration:
    """Test dataset integration with PyTorch components."""
    
    def test_dataloader_compatibility(self, populated_dataset):
        """Test compatibility with PyTorch DataLoader."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        # Use the custom collate function that handles SampleData
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
        
        for batch in dataloader:
            # With custom_collate_fn, batch is now a dictionary
            assert isinstance(batch, dict)
            assert 'left_image' in batch
            assert 'right_image' in batch
            assert 'left_label' in batch
            assert 'right_label' in batch
            assert 'left_axis' in batch
            assert 'right_axis' in batch
            assert 'filename' in batch
            
            # Check tensor shapes
            batch_size = len(batch['filename'])
            assert batch['left_image'].shape[0] == batch_size
            assert batch['right_image'].shape[0] == batch_size
            assert batch['left_label'].shape[0] == batch_size
            assert batch['right_label'].shape[0] == batch_size
            
            break  # Test just first batch
    
    def test_sample_data_backward_compatibility(self, populated_dataset):
        """Test SampleData backward compatibility features."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        sample = dataset[0]
        
        # Test dictionary-style access
        assert sample['left_image'] is not None
        assert sample['filename'] is not None
        
        # Test 'in' operator
        assert 'left_image' in sample
        assert 'nonexistent_key' not in sample
        
        # Test keys() method
        keys = list(sample.keys())
        assert 'left_image' in keys
        assert 'filename' in keys

    def test_custom_collate_function(self, populated_dataset):
        """Test custom collate function directly."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        # Create a small batch manually
        batch = [dataset[i] for i in range(min(2, expected_count))]
        
        # Test the collate function
        collated = custom_collate_fn(batch)
        
        assert isinstance(collated, dict)
        assert 'left_image' in collated
        assert 'right_image' in collated
        assert 'left_axis_mask' in collated
        assert 'right_axis_mask' in collated
        
        # Check batch dimension
        expected_batch_size = len(batch)
        assert collated['left_image'].shape[0] == expected_batch_size
        assert collated['right_image'].shape[0] == expected_batch_size


class TestDatasetPerformance:
    """Test dataset performance characteristics."""
    
    def test_repeated_access_consistency(self, populated_dataset):
        """Test that repeated access to same index returns consistent results."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Filenames should be identical
        assert sample1.filename == sample2.filename
        
        # For deterministic transforms, results should be identical
        if dataset.seed is not None:
            assert torch.equal(sample1.left_image, sample2.left_image)
    
    def test_memory_efficiency(self, populated_dataset):
        """Test memory efficiency (basic check)."""
        dataset, expected_count = populated_dataset
        if expected_count == 0:
            pytest.skip("No samples to test")
        
        import gc
        import sys
        
        # Get initial memory usage
        initial_objects = len(gc.get_objects())
        
        # Access multiple samples
        samples = [dataset[i] for i in range(min(5, expected_count))]
        
        # Force garbage collection
        del samples
        gc.collect()
        
        # Check that we're not leaking too many objects
        final_objects = len(gc.get_objects())
        assert final_objects - initial_objects < 1000  # Reasonable threshold


# Fixtures for testing
@pytest.fixture
def mock_dataset_structure():
    """Create a mock dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create basic directory structure
        for subset in ['train', 'val', 'test']:
            for camera in CAMERA_CONFIG.values():
                for data_type in [IMG_PROC_DIR, LABEL_PROC_DIR, PROBE_PROC_DIR]:
                    (root / subset / camera / data_type).mkdir(parents=True, exist_ok=True)
        
        yield root


@pytest.fixture
def complete_dataset_structure(mock_dataset_structure):
    """Create a complete dataset structure with sample files."""
    root = mock_dataset_structure
    
    # Create sample files
    for subset in ['train']:  # Just test with train for simplicity
        cameras = list(CAMERA_CONFIG.values())
        
        # Create sample images
        for camera in cameras:
            img_dir = root / subset / camera / IMG_PROC_DIR
            for i in range(3):
                img_path = img_dir / f"sample_{i:03d}.jpg"
                # Create a dummy image
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(img_path), dummy_img)
        
        # Create sample labels with format that works with the parser
        # The parser tries separators in order: ', ', ' ', '\t'
        for camera in cameras:
            label_dir = root / subset / camera / LABEL_PROC_DIR
            label_file = label_dir / LABEL_FILE  # Use LABEL_FILE instead of hardcoded name
            with open(label_file, 'w') as f:
                for i in range(3):
                    line = f"sample_{i:03d}.jpg, 50.0, 50.0"
                    f.write(line + "\n")
        
        # Create sample axis files
        for camera in cameras:
            axis_dir = root / subset / camera / PROBE_PROC_DIR
            for i in range(3):
                axis_file = axis_dir / f"sample_{i:03d}.txt"
                with open(axis_file, 'w') as f:
                    f.write("25.0 25.0\n75.0 75.0\n")
    
    yield root


@pytest.fixture
def incomplete_dataset_structure():
    """Create an incomplete dataset structure (missing directories)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create only partial structure
        for subset in ['train']:
            for camera in list(CAMERA_CONFIG.values())[:1]:  # Only first camera
                (root / subset / camera / IMG_PROC_DIR).mkdir(parents=True, exist_ok=True)
                # Missing other directories
        
        yield root


@pytest.fixture
def empty_dataset_structure(mock_dataset_structure):
    """Create an empty dataset structure."""
    # Return the mock structure without adding any files
    yield mock_dataset_structure


@pytest.fixture
def populated_dataset(complete_dataset_structure):
    """Create a populated dataset for testing."""
    dataset = SensingAreaDataset(
        root=complete_dataset_structure,
        subset='train',
        validate_structure=True
    )
    expected_count = 3  # Based on complete_dataset_structure
    yield dataset, expected_count


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])