import pytest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch
from typing import Dict, Tuple

from src.dataset.loaders.label_loader import LabelLoader
from src.dataset.core.data_structures import DatasetError
from src.dataset.loaders.file_manager import FileManager


class TestLabelLoader:
    """Test suite for LabelLoader class."""
    
    @pytest.fixture
    def mock_file_manager(self):
        """Create a mock FileManager for testing."""
        file_manager = Mock(spec=FileManager)
        return file_manager
    
    @pytest.fixture
    def label_loader(self, mock_file_manager):
        """Create a LabelLoader instance for testing."""
        return LabelLoader(mock_file_manager)
    
    @pytest.fixture
    def sample_label_content(self):
        """Sample label file content for testing."""
        return """image001.jpg, 320.5, 240.7
image002.jpg, 415.2, 180.3
image003.jpg, 290.8, 350.1"""
    
    @pytest.fixture
    def mock_camera_config(self):
        """Mock camera configuration."""
        return {'left': 'left', 'right': 'right'}
    
    def test_init(self, mock_file_manager):
        """Test LabelLoader initialization."""
        loader = LabelLoader(mock_file_manager)
        assert loader.file_manager == mock_file_manager
        assert loader._center_points is None
    
    @patch('src.dataset.loaders.label_loader.CAMERA_CONFIG')
    def test_load_center_points_success(self, mock_camera_config, label_loader, sample_label_content):
        """Test successful loading of center points from both cameras."""
        mock_camera_config.values.return_value = ['left', 'right']
        
        # Mock directory and file paths
        left_dir = Path('/data/left/labels')
        right_dir = Path('/data/right/labels')
        left_file = left_dir / 'labels.txt'
        right_file = right_dir / 'labels.txt'
        
        label_loader.file_manager.get_directory.side_effect = [left_dir, right_dir]
        
        # Mock file existence and content
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=sample_label_content)):
            
            result = label_loader._load_center_points()
            
            assert 'left' in result
            assert 'right' in result
            assert len(result['left']) == 3
            assert len(result['right']) == 3
            assert result['left']['image001.jpg'] == (320.5, 240.7)
            assert result['right']['image002.jpg'] == (415.2, 180.3)
    
    @patch('src.dataset.loaders.label_loader.CAMERA_CONFIG')
    def test_load_center_points_missing_file(self, mock_camera_config, label_loader):
        """Test loading when label file doesn't exist."""
        mock_camera_config.values.return_value = ['left']
        
        left_dir = Path('/data/left/labels')
        label_loader.file_manager.get_directory.return_value = left_dir
        
        with patch.object(Path, 'exists', return_value=False):
            result = label_loader._load_center_points()
            
            assert result['left'] == {}
    
    @patch('src.dataset.loaders.label_loader.CAMERA_CONFIG')
    @patch('src.dataset.loaders.label_loader.logger')
    def test_load_center_points_with_exception(self, mock_logger, mock_camera_config, label_loader):
        """Test loading when an exception occurs."""
        mock_camera_config.values.return_value = ['left']
        
        left_dir = Path('/data/left/labels')
        label_loader.file_manager.get_directory.return_value = left_dir
        
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', side_effect=IOError("File read error")):
            
            result = label_loader._load_center_points()
            
            assert result['left'] == {}
            mock_logger.warning.assert_called_once()
    
    def test_parse_label_file_comma_separated(self, label_loader):
        """Test parsing label file with comma-separated values."""
        content = "image001.jpg, 320.5, 240.7\nimage002.jpg, 415.2, 180.3"
        
        with patch('builtins.open', mock_open(read_data=content)):
            result = label_loader._parse_label_file(Path('test.txt'))
            
            assert len(result) == 2
            assert result['image001.jpg'] == (320.5, 240.7)
            assert result['image002.jpg'] == (415.2, 180.3)
    
    def test_parse_label_file_space_separated(self, label_loader):
        """Test parsing label file with space-separated values."""
        content = "image001.jpg 320.5 240.7\nimage002.jpg 415.2 180.3"
        
        with patch('builtins.open', mock_open(read_data=content)):
            result = label_loader._parse_label_file(Path('test.txt'))
            
            assert len(result) == 2
            assert result['image001.jpg'] == (320.5, 240.7)
            assert result['image002.jpg'] == (415.2, 180.3)
    
    def test_parse_label_file_tab_separated(self, label_loader):
        """Test parsing label file with tab-separated values."""
        content = "image001.jpg\t320.5\t240.7\nimage002.jpg\t415.2\t180.3"
        
        with patch('builtins.open', mock_open(read_data=content)):
            result = label_loader._parse_label_file(Path('test.txt'))
            
            assert len(result) == 2
            assert result['image001.jpg'] == (320.5, 240.7)
            assert result['image002.jpg'] == (415.2, 180.3)
    
    def test_parse_label_file_with_empty_lines(self, label_loader):
        """Test parsing label file with empty lines."""
        content = "image001.jpg, 320.5, 240.7\n\nimage002.jpg, 415.2, 180.3\n\n"
        
        with patch('builtins.open', mock_open(read_data=content)):
            result = label_loader._parse_label_file(Path('test.txt'))
            
            assert len(result) == 2
            assert result['image001.jpg'] == (320.5, 240.7)
            assert result['image002.jpg'] == (415.2, 180.3)
    
    @patch('src.dataset.loaders.label_loader.logger')
    def test_parse_label_file_invalid_format(self, mock_logger, label_loader):
        """Test parsing label file with invalid format lines."""
        content = "image001.jpg, 320.5, 240.7\ninvalid_line\nimage002.jpg, 415.2, 180.3"
        
        with patch('builtins.open', mock_open(read_data=content)):
            result = label_loader._parse_label_file(Path('test.txt'))
            
            assert len(result) == 2
            assert result['image001.jpg'] == (320.5, 240.7)
            assert result['image002.jpg'] == (415.2, 180.3)
            mock_logger.warning.assert_called_once()
    
    @patch('src.dataset.loaders.label_loader.logger')
    def test_parse_label_file_invalid_coordinates(self, mock_logger, label_loader):
        """Test parsing label file with invalid coordinate values."""
        content = "image001.jpg, not_a_number, 240.7\nimage002.jpg, 415.2, 180.3"
        
        with patch('builtins.open', mock_open(read_data=content)):
            result = label_loader._parse_label_file(Path('test.txt'))
            
            assert len(result) == 1
            assert result['image002.jpg'] == (415.2, 180.3)
            mock_logger.warning.assert_called_once()
    
    def test_center_points_cached_property(self, label_loader, sample_label_content):
        """Test that center_points is cached after first access."""
        with patch.object(label_loader, '_load_center_points', return_value={'left': {}, 'right': {}}) as mock_load:
            # First access should call _load_center_points
            points1 = label_loader.center_points
            assert mock_load.call_count == 1
            
            # Second access should use cached value
            points2 = label_loader.center_points
            assert mock_load.call_count == 1
            assert points1 is points2
    
    def test_get_label_success(self, label_loader):
        """Test successful label retrieval."""
        # Mock the center_points property
        mock_points = {
            'left': {'image001.jpg': (320.5, 240.7)},
            'right': {'image001.jpg': (415.2, 180.3)}
        }
        
        with patch.object(type(label_loader), 'center_points', new_callable=lambda: mock_points):
            result = label_loader.get_label('image001.jpg', 'left')
            assert result == (320.5, 240.7)
            
            result = label_loader.get_label('image001.jpg', 'right')
            assert result == (415.2, 180.3)
    
    def test_get_label_invalid_camera(self, label_loader):
        """Test get_label with invalid camera name."""
        mock_points = {'left': {}, 'right': {}}
        
        with patch.object(type(label_loader), 'center_points', new_callable=lambda: mock_points):
            with pytest.raises(DatasetError, match="Invalid camera 'invalid'"):
                label_loader.get_label('image001.jpg', 'invalid')
    
    def test_get_label_filename_not_found(self, label_loader):
        """Test get_label when filename is not found."""
        mock_points = {
            'left': {'existing_image.jpg': (100.0, 200.0)},
            'right': {}
        }
        
        with patch.object(type(label_loader), 'center_points', new_callable=lambda: mock_points):
            with pytest.raises(DatasetError, match="Label not found for left camera: missing_image.jpg"):
                label_loader.get_label('missing_image.jpg', 'left')
    
    @patch('src.dataset.loaders.label_loader.CAMERA_CONFIG')
    @patch('src.dataset.loaders.label_loader.LABEL_PROC_DIR', 'processed_labels')
    @patch('src.dataset.loaders.label_loader.LABEL_FILE', 'labels.txt')
    def test_integration_full_workflow(self, mock_camera_config, label_loader, sample_label_content):
        """Integration test for the complete workflow."""
        mock_camera_config.values.return_value = ['left', 'right']
        
        # Mock directory structure
        left_dir = Path('/data/left/processed_labels')
        right_dir = Path('/data/right/processed_labels')
        
        label_loader.file_manager.get_directory.side_effect = [left_dir, right_dir]
        
        # Mock file existence and content
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=sample_label_content)):
            
            # Test that we can successfully get a label
            result = label_loader.get_label('image001.jpg', 'left')
            assert result == (320.5, 240.7)
            
            # Test caching works
            result2 = label_loader.get_label('image002.jpg', 'right')
            assert result2 == (415.2, 180.3)


if __name__ == '__main__':
    pytest.main([__file__])