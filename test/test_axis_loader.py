import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

from src.dataset.core.data_structures import DatasetError
from src.dataset.loaders.axis_loader import AxisLoader


class TestAxisLoader:
    """Test suite for AxisLoader class."""
    
    def test_load_axis_points_nonexistent_file(self):
        """Test loading from a non-existent file returns empty array."""
        non_existent_path = Path("non_existent_file.txt")
        result = AxisLoader.load_axis_points(non_existent_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)
        assert result.dtype == np.float32
    
    def test_load_axis_points_empty_file(self):
        """Test loading from an empty file returns empty array."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            assert isinstance(result, np.ndarray)
            assert result.shape == (0, 2)
            assert result.dtype == np.float32
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_single_point(self):
        """Test loading a single point (2 values)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1.5 2.5")
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array([[1.5, 2.5]], dtype=np.float32)
            
            assert result.shape == (1, 2)
            assert result.dtype == np.float32
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_multiple_points_same_line(self):
        """Test loading multiple points from a single line."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1.0 2.0 3.0 4.0")
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            
            assert result.shape == (2, 2)
            assert result.dtype == np.float32
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_multiple_points_multiple_lines(self):
        """Test loading multiple points from multiple lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1.0 2.0\n3.0 4.0\n5.0 6.0")
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
            
            assert result.shape == (3, 2)
            assert result.dtype == np.float32
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_with_whitespace(self):
        """Test loading points with extra whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("  1.0   2.0  \n  3.0   4.0  \n")
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            
            assert result.shape == (2, 2)
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_negative_values(self):
        """Test loading points with negative values."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("-1.5 2.5\n3.0 -4.0")
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array([[-1.5, 2.5], [3.0, -4.0]], dtype=np.float32)
            
            assert result.shape == (2, 2)
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_odd_number_of_values_error(self):
        """Test that odd number of values raises DatasetError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1.0 2.0 3.0")  # 3 values, can't reshape to (N, 2)
            temp_path = Path(temp_file.name)
        
        try:
            with pytest.raises(DatasetError) as exc_info:
                AxisLoader.load_axis_points(temp_path)
            
            assert "Cannot reshape 3 points to (N, 2)" in str(exc_info.value)
            assert str(temp_path) in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_invalid_data_format(self):
        """Test that invalid data format raises DatasetError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("invalid text data")
            temp_path = Path(temp_file.name)
        
        try:
            with pytest.raises(DatasetError) as exc_info:
                AxisLoader.load_axis_points(temp_path)
            
            assert "Error reading axis points" in str(exc_info.value)
            assert str(temp_path) in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_file_permission_error(self):
        """Test that file permission error raises DatasetError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1.0 2.0")
            temp_path = Path(temp_file.name)
        
        try:
            # Remove read permissions
            os.chmod(temp_path, 0o000)
            
            with pytest.raises(DatasetError) as exc_info:
                AxisLoader.load_axis_points(temp_path)
            
            assert "Error reading axis points" in str(exc_info.value)
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)
    
    def test_load_axis_points_large_file(self):
        """Test loading a larger file with many points."""
        points_data = []
        expected_data = []
        
        # Generate 1000 points
        for i in range(1000):
            x, y = float(i), float(i * 2)
            points_data.append(f"{x} {y}")
            expected_data.append([x, y])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write('\n'.join(points_data))
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array(expected_data, dtype=np.float32)
            
            assert result.shape == (1000, 2)
            assert result.dtype == np.float32
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_scientific_notation(self):
        """Test loading points with scientific notation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1.5e-3 2.5e2\n3.0E-1 4.0E1")
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            expected = np.array([[1.5e-3, 2.5e2], [3.0e-1, 4.0e1]], dtype=np.float32)
            
            assert result.shape == (2, 2)
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    @patch('numpy.loadtxt')
    def test_load_axis_points_numpy_loadtxt_called_correctly(self, mock_loadtxt):
        """Test that numpy.loadtxt is called with correct parameters."""
        mock_loadtxt.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            
            mock_loadtxt.assert_called_once_with(temp_path, dtype=np.float32)
            assert result.shape == (1, 2)
        finally:
            os.unlink(temp_path)
    
    def test_load_axis_points_return_type_and_dtype(self):
        """Test that the return type is always numpy array with float32 dtype."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("1 2\n3 4")  # Integer values
            temp_path = Path(temp_file.name)
        
        try:
            result = AxisLoader.load_axis_points(temp_path)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.shape == (2, 2)
        finally:
            os.unlink(temp_path)


# Additional parametrized tests for edge cases
@pytest.mark.parametrize("file_content,expected_shape", [
    ("", (0, 2)),  # Empty file
    ("1.0 2.0", (1, 2)),  # Single point
    ("1.0 2.0\n3.0 4.0", (2, 2)),  # Two points on separate lines
    ("1.0 2.0 3.0 4.0", (2, 2)),  # Two points on same line
    ("1.0 2.0 3.0 4.0 5.0 6.0", (3, 2)),  # Three points on same line
])
def test_load_axis_points_parametrized(file_content, expected_shape):
    """Parametrized test for various valid input formats."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_path = Path(temp_file.name)
    
    try:
        result = AxisLoader.load_axis_points(temp_path)
        assert result.shape == expected_shape
        assert result.dtype == np.float32
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])