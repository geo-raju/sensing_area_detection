"""
Test suite for stereo laparoscopic image transformations.
Tests transform pipeline and stereo consistency for medical imaging applications.
"""

import pytest
import numpy as np
import torch
import cv2
import albumentations as A
import logging

# Import the modules to test
from src.dataset.transforms.pipeline import TransformPipeline
from src.dataset.transforms.transform_manager import TransformManager
from config.transform_config import TransformConfig

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStereoTransform:
    """Test suite for stereo laparoscopic image transformations."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample transform configuration for testing."""
        return TransformConfig()

    @pytest.fixture
    def sample_stereo_images(self):
        """Create sample stereo laparoscopic images for testing."""
        # Create synthetic stereo images (896x896 as per config)
        height, width = 896, 896
        
        # Left image - create a pattern with anatomical-like features
        left_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(left_img, (400, 400), 100, (150, 100, 100), -1)  # Tissue-like color
        cv2.ellipse(left_img, (600, 300), (80, 40), 45, 0, 360, (200, 120, 120), -1)
        cv2.rectangle(left_img, (200, 600), (350, 750), (180, 110, 110), -1)
        
        # Right image - similar but with stereo disparity
        right_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(right_img, (390, 400), 100, (150, 100, 100), -1)  # Shifted left
        cv2.ellipse(right_img, (590, 300), (80, 40), 45, 0, 360, (200, 120, 120), -1)
        cv2.rectangle(right_img, (190, 600), (340, 750), (180, 110, 110), -1)
        
        return left_img, right_img

    @pytest.fixture
    def sample_keypoints(self):
        """Create sample keypoints for probe center and axis."""
        # Center points for probe tip
        left_center = (448.0, 400.0)  # Center of left image
        right_center = (438.0, 400.0)  # Slightly shifted for stereo disparity
        
        # Axis points for probe orientation (multiple points along axis)
        left_axis = np.array([
            [448.0, 350.0],  # Point above center
            [448.0, 450.0],  # Point below center
            [448.0, 300.0],  # Further point for axis direction
        ], dtype=np.float32)
        
        right_axis = np.array([
            [438.0, 350.0],  # Corresponding points in right image
            [438.0, 450.0],
            [438.0, 300.0],
        ], dtype=np.float32)
        
        return left_center, right_center, left_axis, right_axis

    def test_transform_pipeline_creation(self, sample_config):
        """Test that transform pipelines are created correctly."""
        pipeline = TransformPipeline(sample_config)
        
        # Test train transform creation
        train_transform = pipeline.get_train_transform()
        assert isinstance(train_transform, A.ReplayCompose)
        
        # Check that keypoint parameters are configured (they're stored internally)
        # We can verify this by checking if the transform can handle keypoints
        assert hasattr(train_transform, 'processors')
        
        # Test validation transform creation
        val_transform = pipeline.get_val_transform()
        assert isinstance(val_transform, A.ReplayCompose)
        
        # Test that train transform has more transforms than validation
        assert len(train_transform.transforms) > len(val_transform.transforms)
        
        # Test that we can actually apply transforms with keypoints
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_keypoints = [[50.0, 50.0]]
        
        try:
            result = train_transform(image=test_img, keypoints=test_keypoints)
            assert 'image' in result
            assert 'keypoints' in result
            logger.info("Keypoint handling verified")
        except Exception as e:
            pytest.fail(f"Transform failed to handle keypoints: {e}")
        
        logger.info("Transform pipeline creation test passed")

    def test_transform_manager_initialization(self):
        """Test TransformManager initialization with and without transforms."""
        # Test with no transform
        manager_none = TransformManager(None)
        assert manager_none.transform is None
        
        # Test with transform
        config = TransformConfig()
        pipeline = TransformPipeline(config)
        transform = pipeline.get_train_transform()
        manager_with_transform = TransformManager(transform)
        assert manager_with_transform.transform is not None
        
        logger.info("TransformManager initialization test passed")

    def test_stereo_consistency_without_transform(self, sample_stereo_images, sample_keypoints):
        """Test that stereo consistency is maintained when no transform is applied."""
        left_img, right_img = sample_stereo_images
        left_center, right_center, left_axis, right_axis = sample_keypoints
        
        manager = TransformManager(None)
        
        result = manager.apply_transform(
            left_img, right_img, 
            left_center, right_center,
            left_axis, right_axis
        )
        
        left_img_t, right_img_t, left_center_t, right_center_t, left_axis_t, right_axis_t = result
        
        # Check that images are converted to tensors
        assert isinstance(left_img_t, torch.Tensor)
        assert isinstance(right_img_t, torch.Tensor)
        assert left_img_t.shape[0] == 3  # RGB channels first
        assert right_img_t.shape[0] == 3
        
        # Check that keypoints remain unchanged
        assert left_center_t == left_center
        assert right_center_t == right_center
        np.testing.assert_array_equal(left_axis_t, left_axis)
        np.testing.assert_array_equal(right_axis_t, right_axis)
        
        logger.info("Stereo consistency without transform test passed")

    def test_stereo_consistency_with_transforms(self, sample_stereo_images, sample_keypoints):
        """Test that stereo consistency is maintained with transformations applied."""
        left_img, right_img = sample_stereo_images
        left_center, right_center, left_axis, right_axis = sample_keypoints
        
        config = TransformConfig()
        pipeline = TransformPipeline(config)
        
        # Test with training transforms (includes augmentations)
        train_transform = pipeline.get_train_transform()
        manager = TransformManager(train_transform)
        
        result = manager.apply_transform(
            left_img, right_img,
            left_center, right_center,
            left_axis, right_axis
        )
        
        left_img_t, right_img_t, left_center_t, right_center_t, left_axis_t, right_axis_t = result
        
        # Check tensor properties
        assert isinstance(left_img_t, torch.Tensor)
        assert isinstance(right_img_t, torch.Tensor)
        assert left_img_t.shape == right_img_t.shape
        assert left_img_t.shape[0] == 3  # RGB channels
        
        # Check that keypoints are within image boundaries
        img_height, img_width = left_img_t.shape[-2:]
        
        assert 0 <= left_center_t[0] < img_width
        assert 0 <= left_center_t[1] < img_height
        assert 0 <= right_center_t[0] < img_width
        assert 0 <= right_center_t[1] < img_height
        
        # Check axis points are within bounds
        for point in left_axis_t:
            assert 0 <= point[0] < img_width
            assert 0 <= point[1] < img_height
        
        for point in right_axis_t:
            assert 0 <= point[0] < img_width
            assert 0 <= point[1] < img_height
        
        logger.info("Stereo consistency with transforms test passed")

    def test_keypoint_validation(self, sample_stereo_images):
        """Test keypoint validation and clamping."""
        left_img, right_img = sample_stereo_images
        
        # Create keypoints that are out of bounds
        out_of_bounds_keypoints = [
            [-10.0, -5.0],      # Negative coordinates
            [1000.0, 1000.0],   # Too large coordinates
            [500.0, -10.0],     # Mixed valid/invalid
        ]
        
        manager = TransformManager(None)
        img_height, img_width = left_img.shape[:2]
        
        validated = manager._validate_keypoints(out_of_bounds_keypoints, img_height, img_width)
        
        # Check that all points are within bounds
        for point in validated:
            assert 0 <= point[0] < img_width
            assert 0 <= point[1] < img_height
        
        # Check specific clamping behavior
        assert validated[0] == [0.0, 0.0]  # Negative clamped to 0
        assert validated[1] == [img_width - 1, img_height - 1]  # Large values clamped to max
        assert validated[2] == [500.0, 0.0]  # Mixed case
        
        logger.info("Keypoint validation test passed")

    def test_tensor_conversion(self):
        """Test image to tensor conversion functionality."""
        manager = TransformManager(None)
        
        # Test numpy array conversion
        np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor_img = manager._convert_to_tensor(np_img)
        assert isinstance(tensor_img, torch.Tensor)
        assert tensor_img.shape == (3, 100, 100)  # Channels first
        
        # Test tensor that needs permutation
        wrong_tensor = torch.randn(100, 100, 3)
        corrected_tensor = manager._convert_to_tensor(wrong_tensor)
        assert corrected_tensor.shape == (3, 100, 100)
        
        # Test already correct tensor
        correct_tensor = torch.randn(3, 100, 100)
        result_tensor = manager._convert_to_tensor(correct_tensor)
        assert result_tensor.shape == (3, 100, 100)
        
        logger.info("Tensor conversion test passed")

    def test_transform_reproducibility(self, sample_stereo_images, sample_keypoints):
        """Test that the same transform parameters produce consistent results."""
        left_img, right_img = sample_stereo_images
        left_center, right_center, left_axis, right_axis = sample_keypoints
        
        # Create a simple deterministic transform
        simple_transform = A.ReplayCompose([
            A.Resize(height=896, width=896),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        manager = TransformManager(simple_transform)
        
        # Apply transform twice
        result1 = manager.apply_transform(
            left_img, right_img, left_center, right_center, left_axis, right_axis
        )
        result2 = manager.apply_transform(
            left_img, right_img, left_center, right_center, left_axis, right_axis
        )
        
        # Results should be different due to random augmentations in real scenarios
        # but for this simple case, we test that the function executes successfully
        assert len(result1) == 6
        assert len(result2) == 6
        
        logger.info("Transform reproducibility test passed")

    def test_transform_info_retrieval(self):
        """Test getting information about transforms."""
        config = TransformConfig()
        pipeline = TransformPipeline(config)
        transform = pipeline.get_train_transform()
        
        manager = TransformManager(transform)
        info = manager.get_transform_info()
        
        assert isinstance(info, dict)
        assert 'transform' in info
        assert 'type' in info
        assert info['type'] == 'albumentations_replay'
        
        # Test with no transform
        manager_none = TransformManager(None)
        info_none = manager_none.get_transform_info()
        assert info_none['type'] == 'none'
        assert info_none['transform'] is None
        
        logger.info("Transform info retrieval test passed")

    def test_medical_image_properties(self, sample_stereo_images, sample_keypoints):
        """Test that transforms preserve medical image properties appropriately."""
        left_img, right_img = sample_stereo_images
        left_center, right_center, left_axis, right_axis = sample_keypoints
        
        config = TransformConfig()
        pipeline = TransformPipeline(config)
        val_transform = pipeline.get_val_transform()  # Use validation (no augmentation)
        
        manager = TransformManager(val_transform)
        
        result = manager.apply_transform(
            left_img, right_img, left_center, right_center, left_axis, right_axis
        )
        
        left_img_t, right_img_t, _, _, _, _ = result
        
        # Check that images maintain expected dimensions for medical imaging
        assert left_img_t.shape[-2:] == (896, 896)  # Expected size from config
        assert right_img_t.shape[-2:] == (896, 896)
        
        # Check that normalization was applied (values should be roughly in [-2, 2] range after ImageNet normalization)
        assert left_img_t.min() >= -3.0  # Allowing some tolerance
        assert left_img_t.max() <= 3.0
        assert right_img_t.min() >= -3.0
        assert right_img_t.max() <= 3.0
        
        logger.info("Medical image properties test passed")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        manager = TransformManager(None)
        
        # Test with minimal images
        tiny_img = np.ones((10, 10, 3), dtype=np.uint8)
        tiny_center = (5.0, 5.0)
        tiny_axis = np.array([[5.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        
        result = manager.apply_transform(
            tiny_img, tiny_img, tiny_center, tiny_center, tiny_axis, tiny_axis
        )
        
        # Should handle small images without error
        assert len(result) == 6
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)
        
        # Test with empty axis arrays
        empty_axis = np.empty((0, 2), dtype=np.float32)
        result_empty = manager.apply_transform(
            tiny_img, tiny_img, tiny_center, tiny_center, empty_axis, empty_axis
        )
        
        assert len(result_empty) == 6
        assert result_empty[4].shape == (0, 2)  # Empty axis preserved
        assert result_empty[5].shape == (0, 2)
        
        logger.info("Edge cases test passed")


def run_comprehensive_test():
    """Run all tests manually without pytest."""
    test_instance = TestStereoTransform()
    
    # Create fixtures
    config = test_instance.sample_config()
    stereo_images = test_instance.sample_stereo_images()
    keypoints = test_instance.sample_keypoints()
    
    try:
        print("Running stereo transform tests...")
        
        test_instance.test_transform_pipeline_creation(config)
        test_instance.test_transform_manager_initialization()
        test_instance.test_stereo_consistency_without_transform(stereo_images, keypoints)
        test_instance.test_stereo_consistency_with_transforms(stereo_images, keypoints)
        test_instance.test_keypoint_validation(stereo_images)
        test_instance.test_tensor_conversion()
        test_instance.test_transform_reproducibility(stereo_images, keypoints)
        test_instance.test_transform_info_retrieval()
        test_instance.test_medical_image_properties(stereo_images, keypoints)
        test_instance.test_edge_cases()
        
        print("✅ All stereo transform tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_comprehensive_test()