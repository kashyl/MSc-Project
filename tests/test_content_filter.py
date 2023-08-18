import unittest
from unittest.mock import patch, Mock
from parameterized import parameterized
from PIL import Image
from content_filter import ContentFilter, RATING_LEVELS, GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING, BLURRED_IMG_CACHE_DIR, ImageFilter, GUIFiltersLabels, ImageRatings
import os

class TestContentFilter(unittest.TestCase):

    def setUp(self):
        self.cf = ContentFilter()

    # Cover all possible scenarios
    @parameterized.expand([
        (gui_label, image_rating, RATING_LEVELS[image_rating] > RATING_LEVELS[GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING[gui_label]])
        for gui_label in GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING.keys()
        for image_rating in RATING_LEVELS.keys()
    ])
    def test_GIVEN_GUI_label_and_image_rating_WHEN_is_rating_filtered_gui_THEN_return_expected_result(self, gui_label, image_rating, expected):
        self.assertEqual(self.cf.is_rating_filtered_gui(gui_label, image_rating), expected)

    @patch('content_filter.Image.open')
    @patch('os.path.exists', return_value=True)
    def test_GIVEN_path_to_cached_image_WHEN_get_blurred_image_from_path_THEN_return_blurred_image_from_cache(self, mock_exists, mock_open):
        self.cf.get_blurred_image_from_path('/path/to/image.jpg')
        mock_open.assert_called_once_with(os.path.join(BLURRED_IMG_CACHE_DIR, 'image.jpg'))

    @patch('content_filter.Image.open')
    @patch('os.path.exists', return_value=False)
    def test_GIVEN_path_to_uncached_image_WHEN_get_blurred_image_from_path_THEN_return_newly_blurred_image(self, mock_exists, mock_open):
        img_instance = Mock()
        blurred_instance = Mock()
        img_instance.filter.return_value = blurred_instance
        mock_open.return_value = img_instance
        
        self.cf.get_blurred_image_from_path('/path/to/image.jpg')
        
        args, _ = img_instance.filter.call_args
        self.assertIsInstance(args[0], ImageFilter.GaussianBlur)
        self.assertEqual(args[0].radius, 50)
        blurred_instance.save.assert_called_once()

    @patch.object(Image.Image, 'filter')
    def test_GIVEN_image_WHEN_blur_image_THEN_apply_gaussian_blur(self, mock_filter):
        img = Image.new('RGB', (50, 50))
        self.cf.blur_image(img)

        args, _ = mock_filter.call_args
        self.assertIsInstance(args[0], ImageFilter.GaussianBlur)
        self.assertEqual(args[0].radius, 50)

if __name__ == '__main__':
    unittest.main()
