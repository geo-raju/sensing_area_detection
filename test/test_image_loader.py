import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch

from src.dataset.core.data_structures import DatasetError
from src.dataset.loaders.image_loader import ImageLoader


class TestImageLoader:
    """Test suite for ImageLoader class."""
    
    def test_load_image_success(self):
        """Test successful image loading and RGB conversion."""
        # Create a mock BGR image (3x3x3 for simplicity)
        mock_bgr_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[128, 64, 32], [200, 100, 50], [75, 150, 225]],
            [[0, 0, 0], [255, 255, 255], [100, 100, 100]]
        ], dtype=np.uint8)
        
        # Expected RGB image after conversion
        expected_rgb = cv2.cvtColor(mock_bgr_image, cv2.COLOR_BGR2RGB)
        
        with patch('cv2.imread') as mock_imread, \
             patch.object(Path, 'exists', return_value=True):
            
            mock_imread.return_value = mock_bgr_image
            image_path = Path("test_image.jpg")
            
            result = ImageLoader.load_image(image_path)
            
            # Verify cv2.imread was called with correct path
            mock_imread.assert_called_once_with(str(image_path))
            
            # Verify the result is the RGB converted image
            np.testing.assert_array_equal(result, expected_rgb)
            
            # Verify the shape and dtype
            assert result.shape == (3, 3, 3)
            assert result.dtype == np.uint8

    def test_load_image_file_not_found(self):
        """Test FileNotFoundError when image file doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            image_path = Path("nonexistent_image.jpg")
            
            with pytest.raises(FileNotFoundError) as exc_info:
                ImageLoader.load_image(image_path)
            
            assert str(image_path) in str(exc_info.value)
            assert "Image not found" in str(exc_info.value)

    def test_load_image_cv2_imread_returns_none(self):
        """Test DatasetError when cv2.imread returns None (corrupted/unsupported file)."""
        with patch('cv2.imread', return_value=None), \
             patch.object(Path, 'exists', return_value=True):
            
            image_path = Path("corrupted_image.jpg")
            
            with pytest.raises(DatasetError) as exc_info:
                ImageLoader.load_image(image_path)
            
            assert str(image_path) in str(exc_info.value)
            assert "Could not load image" in str(exc_info.value)

    def test_load_image_various_formats(self):
        """Test loading images with different file extensions."""
        formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        for fmt in formats:
            with patch('cv2.imread', return_value=mock_image), \
                 patch.object(Path, 'exists', return_value=True):
                
                image_path = Path(f"test_image{fmt}")
                result = ImageLoader.load_image(image_path)
                
                # Should successfully load and convert any supported format
                assert isinstance(result, np.ndarray)
                assert result.shape == (100, 100, 3)

    def test_load_image_color_conversion_accuracy(self):
        """Test that BGR to RGB conversion is performed correctly."""
        # Create a known BGR image
        bgr_image = np.array([
            [[255, 0, 0]],  # Blue in BGR
            [[0, 255, 0]],  # Green in BGR  
            [[0, 0, 255]]   # Red in BGR
        ], dtype=np.uint8)
        
        with patch('cv2.imread', return_value=bgr_image), \
             patch.object(Path, 'exists', return_value=True):
            
            image_path = Path("color_test.jpg")
            result = ImageLoader.load_image(image_path)
            
            # After BGR to RGB conversion:
            # BGR [255, 0, 0] -> RGB [0, 0, 255] (Blue -> Red)
            # BGR [0, 255, 0] -> RGB [0, 255, 0] (Green -> Green)
            # BGR [0, 0, 255] -> RGB [255, 0, 0] (Red -> Blue)
            expected_rgb = np.array([
                [[0, 0, 255]],   # Red in RGB
                [[0, 255, 0]],   # Green in RGB
                [[255, 0, 0]]    # Blue in RGB
            ], dtype=np.uint8)
            
            np.testing.assert_array_equal(result, expected_rgb)

    def test_load_image_grayscale_handling(self):
        """Test handling of grayscale images loaded by cv2."""
        # cv2.imread loads grayscale as single channel, but with IMREAD_COLOR flag
        # it should still return 3 channels. Test the actual behavior.
        grayscale_3ch = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with patch('cv2.imread', return_value=grayscale_3ch), \
             patch.object(Path, 'exists', return_value=True):
            
            image_path = Path("grayscale.jpg")
            result = ImageLoader.load_image(image_path)
            
            assert result.shape == (100, 100, 3)
            # For grayscale loaded as 3-channel, BGR to RGB conversion 
            # should still work without issues
            assert isinstance(result, np.ndarray)

    @patch('cv2.imread')
    @patch.object(Path, 'exists')
    def test_load_image_integration(self, mock_exists, mock_imread):
        """Integration test with realistic image data."""
        mock_exists.return_value = True
        
        # Create a more realistic test image
        realistic_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = realistic_bgr
        
        image_path = Path("realistic_test.jpg")
        result = ImageLoader.load_image(image_path)
        
        # Verify method calls
        mock_exists.assert_called_once()
        mock_imread.assert_called_once_with(str(image_path))
        
        # Verify result properties
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8
        
        # Verify it's actually different from input (due to BGR->RGB conversion)
        # Unless the image is symmetric in R and B channels
        if not np.array_equal(realistic_bgr[:,:,0], realistic_bgr[:,:,2]):
            assert not np.array_equal(result, realistic_bgr)


# Additional fixtures for testing with actual files (optional)
@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary test image file."""
    # Create a small test image
    test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    temp_file = tmp_path / "test_image.jpg"
    
    # Save as BGR (OpenCV default)
    cv2.imwrite(str(temp_file), test_image)
    return temp_file


def test_load_image_with_real_file(temp_image_file):
    """Test with an actual temporary image file."""
    result = ImageLoader.load_image(temp_image_file)
    
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 3  # Height, Width, Channels
    assert result.shape[2] == 3    # RGB channels
    assert result.dtype == np.uint8