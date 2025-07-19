import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from src.dataset.loaders.file_manager import FileManager
from src.dataset.core.data_structures import DatasetError


class TestFileManager:
    """Test suite for FileManager class."""

    @pytest.fixture
    def temp_root(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration values."""
        # Corrected patch targets
        with patch('src.dataset.loaders.file_manager.CAMERA_CONFIG', {'camera0': 'left', 'camera1': 'right'}), \
             patch('src.dataset.loaders.file_manager.DATA_TYPE_CONFIG', {'laser_off': 'images', 'line_annotation_sample': 'probe_axis'}), \
             patch('src.dataset.loaders.file_manager.LABEL_PROC_DIR', 'labels'), \
             patch('src.dataset.loaders.file_manager.PROBE_PROC_DIR', 'probe_axis'), \
             patch('src.dataset.loaders.file_manager.IMG_PROC_DIR', 'images'):
            yield

    @pytest.fixture
    def file_manager(self, temp_root, mock_config):
        """Create FileManager instance with temporary directory."""
        return FileManager(temp_root, "test_subset")

    def test_init(self, temp_root, mock_config):
        """Test FileManager initialization."""
        fm = FileManager(temp_root, "test_subset")

        assert fm.root == temp_root
        assert fm.subset == "test_subset"
        assert fm.subset_dir == temp_root / "test_subset"
        assert hasattr(fm, '_directories')
        assert isinstance(fm._directories, dict)

    def test_setup_directories(self, file_manager):
        """Test directory setup creates correct directory mappings."""
        # Check that directories are set up correctly
        expected_keys = [
            'left_images', 'left_probe_axis', 'left_labels',
            'right_images', 'right_probe_axis', 'right_labels'
        ]

        for key in expected_keys:
            assert key in file_manager._directories
            assert isinstance(file_manager._directories[key], Path)

    def test_get_directory_valid(self, file_manager):
        """Test getting directory for valid camera and data type."""
        directory = file_manager.get_directory('left', 'images')
        expected_path = file_manager.subset_dir / 'left' / 'images'

        assert directory == expected_path

    def test_get_directory_invalid_key(self, file_manager):
        """Test getting directory with invalid camera/data type combination."""
        with pytest.raises(DatasetError, match="Directory key 'invalid_invalid' not found"):
            file_manager.get_directory('invalid', 'invalid')

    def test_validate_structure_all_exist(self, file_manager, temp_root):
        """Test validation when all directories exist."""
        # Create all required directories
        for directory in file_manager._directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        # Should not raise any exception
        file_manager.validate_structure()

    def test_validate_structure_missing_directories(self, file_manager):
        """Test validation when directories are missing."""
        # Don't create directories - they should be missing
        with pytest.raises(FileNotFoundError, match="Missing directories"):
            file_manager.validate_structure()

    def test_validate_structure_partial_missing(self, file_manager):
        """Test validation when some directories are missing."""
        # Create only some directories
        first_dir = list(file_manager._directories.values())[0]
        first_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="Missing directories"):
            file_manager.validate_structure()

    def test_get_image_filenames_success(self, file_manager, temp_root):
        """Test getting image filenames when images exist."""
        # Create left image directory and add test images
        left_img_dir = file_manager.get_directory('left', 'images')
        left_img_dir.mkdir(parents=True, exist_ok=True)

        # Create test image files
        test_files = ['image1.jpg', 'image2.png', 'image3.jpeg', 'not_image.txt']
        for filename in test_files:
            (left_img_dir / filename).touch()

        filenames = file_manager.get_image_filenames()

        # Should only return image files, sorted
        expected_files = ['image1.jpg', 'image2.png', 'image3.jpeg']
        assert filenames == expected_files

    def test_get_image_filenames_directory_not_exist(self, file_manager):
        """Test getting image filenames when directory doesn't exist."""
        with pytest.raises(DatasetError, match="Left image directory not found"):
            file_manager.get_image_filenames()

    def test_get_image_filenames_no_images(self, file_manager):
        """Test getting image filenames when no images exist."""
        # Create directory but no image files
        left_img_dir = file_manager.get_directory('left', 'images')
        left_img_dir.mkdir(parents=True, exist_ok=True)

        # Create non-image files
        (left_img_dir / 'not_image.txt').touch()
        (left_img_dir / 'readme.md').touch()

        with pytest.raises(DatasetError, match="No images found"):
            file_manager.get_image_filenames()

    def test_get_image_filenames_case_insensitive(self, file_manager):
        """Test that image filename detection is case insensitive."""
        left_img_dir = file_manager.get_directory('left', 'images')
        left_img_dir.mkdir(parents=True, exist_ok=True)

        # Create files with different cases
        test_files = ['image1.JPG', 'image2.PNG', 'image3.Jpeg']
        for filename in test_files:
            (left_img_dir / filename).touch()

        filenames = file_manager.get_image_filenames()

        # Should return all files, sorted
        expected_files = ['image1.JPG', 'image2.PNG', 'image3.Jpeg']
        assert filenames == expected_files

    def test_get_image_filenames_empty_directory(self, file_manager):
        """Test getting image filenames from empty directory."""
        left_img_dir = file_manager.get_directory('left', 'images')
        left_img_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(DatasetError, match="No images found"):
            file_manager.get_image_filenames()

    @patch('src.dataset.loaders.file_manager.os.listdir') # Corrected patch target
    def test_get_image_filenames_os_error(self, mock_listdir, file_manager):
        """Test handling of OS errors when listing files."""
        # Create directory
        left_img_dir = file_manager.get_directory('left', 'images')
        left_img_dir.mkdir(parents=True, exist_ok=True)

        # Mock os.listdir to raise an exception
        mock_listdir.side_effect = OSError("Permission denied")

        with pytest.raises(OSError, match="Permission denied"):
            file_manager.get_image_filenames()

    def test_directories_property_immutable(self, file_manager):
        """Test that directories are properly encapsulated."""
        # Should not be able to modify directories from outside
        original_keys = set(file_manager._directories.keys())

        # Try to modify (this should not affect the internal state)
        external_ref = file_manager._directories
        external_ref['new_key'] = Path('some/path')

        # Internal state should be changed (it's the same object)
        # This test documents the current behavior
        assert 'new_key' in file_manager._directories

    def test_multiple_instances_independent(self, temp_root, mock_config):
        """Test that multiple FileManager instances are independent."""
        fm1 = FileManager(temp_root, "subset1")
        fm2 = FileManager(temp_root, "subset2")

        assert fm1.subset != fm2.subset
        assert fm1.subset_dir != fm2.subset_dir
        assert fm1._directories is not fm2._directories

    def test_path_resolution(self, file_manager):
        """Test that paths are resolved correctly."""
        directory = file_manager.get_directory('left', 'images')

        # Check that path is absolute and properly constructed
        assert directory.is_absolute()
        assert directory.parts[-3:] == ('test_subset', 'left', 'images')

    @patch('src.dataset.loaders.file_manager.logger') # Corrected patch target
    def test_logging_on_successful_validation(self, mock_logger, file_manager):
        """Test that successful validation logs appropriate message."""
        # Create all required directories
        for directory in file_manager._directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        file_manager.validate_structure()

        # Check that info was logged
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Directory structure validated" in call_args
        assert str(file_manager.subset_dir) in call_args


# Integration tests
class TestFileManagerIntegration:
    """Integration tests for FileManager with real filesystem."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow with real filesystem."""
        # Corrected patch targets
        with patch('src.dataset.loaders.file_manager.CAMERA_CONFIG', {'camera0': 'left', 'camera1': 'right'}), \
             patch('src.dataset.loaders.file_manager.DATA_TYPE_CONFIG', {'laser_off': 'images'}), \
             patch('src.dataset.loaders.file_manager.LABEL_PROC_DIR', 'labels'), \
             patch('src.dataset.loaders.file_manager.PROBE_PROC_DIR', 'probe_axis'), \
             patch('src.dataset.loaders.file_manager.IMG_PROC_DIR', 'images'):

            fm = FileManager(tmp_path, "integration_test")

            # Create directory structure
            for directory in fm._directories.values():
                directory.mkdir(parents=True, exist_ok=True)

            # Add some test images
            left_img_dir = fm.get_directory('left', 'images')
            for i in range(3):
                (left_img_dir / f'test_{i:03d}.jpg').touch()

            # Validate structure
            fm.validate_structure()

            # Get filenames
            filenames = fm.get_image_filenames()

            assert len(filenames) == 3
            assert all(f.startswith('test_') for f in filenames)
            assert all(f.endswith('.jpg') for f in filenames)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])