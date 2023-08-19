import unittest
from unittest.mock import patch, MagicMock, PropertyMock, Mock
from parameterized import parameterized
from app import App, IMG_DIR, DEBUG_MOCK_GEN_INFO
from PIL import Image

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = App()
        self.app.image_tags = []
        self.patcher_error = patch('custom_logging.logger.error', return_value=None)
        self.patcher_info = patch('custom_logging.logger.info', return_value=None)
        self.mock_error = self.patcher_error.start()
        self.mock_info = self.patcher_info.start()

    @patch('app.Image', autospec=True)
    def test_save_current_image(self, MockedImage):
        mock_image = MagicMock(spec=Image.Image)
        mock_image.save = MagicMock()
        MockedImage.new.return_value = mock_image
        self.app._image = mock_image
        saved_path = self.app._save_current_image()
        mock_image.save.assert_called_once()
        self.assertTrue(saved_path.startswith(IMG_DIR))
        self.assertTrue(saved_path.endswith('.png'))

    @parameterized.expand([
        (0, (1, 0, 250)),
        (249, (1, 249, 250)),
        (250, (2, 0, 500)),
        (749, (2, 499, 500)),
        (750, (3, 0, 750))
    ])
    def test_get_level_info(self, exp, expected_result):
        self.assertEqual(self.app.get_level_info(exp), expected_result)

    @patch('app.App.create_points_calculators', autospec=True)
    def test_create_points_calculators(self, MockedCalculators):
        MockedCalculators.return_value = (lambda x: x + 1, lambda x: x - 1)
        gained_points, lost_points = self.app.create_points_calculators()
        self.assertEqual(gained_points(0), 1)
        self.assertEqual(lost_points(0), -1)

    @parameterized.expand([
        ("Easy", ["tag1"], 1),
        ("Normal", ["tag1"], 1),
        ("Hard", ["tag1", "tag2", "tag3"], 6),
    ])
    def test_calculate_false_tags_count_based_on_difficulty_level(self, difficulty, tags, expected_count):
        # Backup the original property
        original_property = type(self.app).image_tags
        
        # Set the mocked property
        type(self.app).image_tags = PropertyMock(return_value=tags)
        self.app._difficulty_level = difficulty
        
        # Run the test
        result = self.app._calculate_false_tags_count_based_on_difficulty_level_()
        self.assertEqual(result, expected_count)
        
        # Restore the original property
        type(self.app).image_tags = original_property

    @patch('app.App._set_image', autospec=True)
    @patch('app.mock_generate_image', autospec=True)
    @patch('app.App._set_image_generation_info', autospec=True)
    @patch('app.App._set_image_generation_time', autospec=True)
    def test_mock_gen_image(self, mock_set_time, mock_set_info, mock_generate, mock_set_image):
        self.app._mock_gen_image()
        mock_generate.assert_called_once()
        mock_set_image.assert_called_once()
        mock_set_info.assert_called_once_with(self.app, DEBUG_MOCK_GEN_INFO)
        mock_set_time.assert_called_once()

    def test_mock_gen_tags(self):
        # Create an instance of your App class
        app_instance = App()
        
        # Mock the methods
        app_instance.wd14_tagger.mock_generate_tags = Mock()
        app_instance.wd14_tagger.mock_gen_false_tags = Mock()
        
        # Other test code...
        app_instance._mock_gen_tags()
        
        # Assert that mocks were called
        app_instance.wd14_tagger.mock_generate_tags.assert_called_once()
        app_instance.wd14_tagger.mock_gen_false_tags.assert_called_once()

    def test_generate_round(self):
        # Mocking the required methods directly on the App instance
        self.app._clear_round_data = Mock()
        self.app._set_difficulty_level = Mock()
        self.app._generate_image_func = Mock()
        self.app._generate_tags_func = Mock()
        self.app._apply_image_rating_filter = Mock()

        # Call the method you want to test
        prompt = "prompt"
        checkpoint = "checkpoint"
        content_filter_level = "MEDIUM"
        difficulty = "EASY"
        self.app.generate_round(prompt, checkpoint, content_filter_level, difficulty)

        # Assertions
        self.app._clear_round_data.assert_called_once()
        self.app._set_difficulty_level.assert_called_once_with(difficulty)
        self.app._generate_image_func.assert_called_once_with(prompt, checkpoint)
        self.app._generate_tags_func.assert_called_once()
        self.app._apply_image_rating_filter.assert_called_once_with(content_filter_level)

if __name__ == '__main__':
    unittest.main()
