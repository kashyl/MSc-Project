import unittest
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from unittest.mock import patch, Mock, MagicMock, call
from wd14_tagging.wd14_tagging import WD14Tagger, FILES, SUB_DIR, SUB_DIR_FILES, IMAGE_SIZE

class TestWD14Tagger(unittest.TestCase):

    def setUp(self):
        self.tagger = WD14Tagger(force_download=True, debug=False)
        self.tagger._general_tag_threshold = 0.3
        self.tagger.whitelisted_tags = ["good_tag", "another_tag"]
        self.tagger._all_general_tags_names_and_weights = [("good_tag", 0.5), ("bad_tag", 0.4), ("another_tag", 0.1)]
        self.tagger.tag_rename_mappings = {}
        # Sample image for testing
        self.test_image = Image.new("RGB", (100, 100))

    @patch('wd14_tagging.wd14_tagging.logger.info')
    @patch('wd14_tagging.wd14_tagging.load_model')
    @patch('wd14_tagging.wd14_tagging.hf_hub_download')
    def test_GIVEN_no_model_directory_WHEN_setting_up_THEN_directory_created(self, mock_hf_hub_download, mock_load_model, mock_logger_info):

        with patch('os.path.exists', return_value=False):
            self.tagger._setup_model()

        # Check if hf_hub_download was called for each file in FILES
        called_files = [call[0][1] for call in mock_hf_hub_download.call_args_list] # extract the file parameter from each call
        for file in FILES:
            self.assertIn(file, called_files)

        for file in SUB_DIR_FILES:
            self.assertIn(file, called_files)

        # check if load_model was called
        mock_load_model.assert_called_once_with(self.tagger._model_dir, compile=False)


    @patch('wd14_tagging.wd14_tagging.load_model')
    def test_GIVEN_model_directory_exists_WHEN_setting_up_without_force_THEN_no_overwrite(self, mock_load_model):

        self.tagger._force_download = False
        
        with patch('os.path.exists', return_value=True):
            self.tagger._setup_model()

        mock_load_model.assert_called_once_with(self.tagger._model_dir, compile=False)

    @patch('wd14_tagging.wd14_tagging.logger.info')
    @patch('wd14_tagging.wd14_tagging.load_model')
    @patch('wd14_tagging.wd14_tagging.hf_hub_download')
    def ttest_GIVEN_model_directory_exists_WHEN_setting_up_with_force_THEN_overwrite(self, mock_hf_hub_download, mock_load_model, mock_logger_info):

        self.tagger._force_download = True

        with patch('os.path.exists', return_value=True):
            self.tagger._setup_model()

        # Check if hf_hub_download was called for each file
        called_files = [call[0][1] for call in mock_hf_hub_download.call_args_list]
        for file in FILES:
            self.assertIn(file, called_files)


        mock_load_model.assert_called_once_with(self.tagger._model_dir, compile=False)

    @patch('wd14_tagging.wd14_tagging.load_model')
    def test_GIVEN_model_initialized_WHEN_setting_up_THEN_no_reinitialization(self, mock_load_model):
        
        self.tagger._model = Mock()
        
        self.tagger._setup_model()

        mock_load_model.assert_not_called()


    @patch('wd14_tagging.wd14_tagging.WD14Tagger._setup_model')
    @patch('wd14_tagging.wd14_tagging.json.load')
    @patch('builtins.open')
    @patch('wd14_tagging.wd14_tagging.csv.reader')
    @patch('os.path.join')
    @patch('wd14_tagging.wd14_tagging.EventHandler.notify_observers')
    def test_GIVEN_fresh_instance_WHEN_setting_up_tagger_THEN_initialize_successfully(
            self, mock_notify_observers, mock_path_join, mock_csv_reader, 
            mock_open, mock_json_load, mock_setup_model):

        # Mock file reading
        mock_json_load.side_effect = [{}, {}]  # Mock the result for white list and rename mappings
        mock_csv_reader.return_value = [["tag_id", "name", "category"], ["0", "name1", "9"], ["1", "name2", "0"]]

        self.tagger._setup_tagger()

        # Check if the event handler notify method was called
        mock_notify_observers.assert_called_with(0.0, "Initializing WD14 Tagger")

        # Check if variables were correctly initialized
        self.assertIsNotNone(self.tagger.whitelisted_tags)
        self.assertIsNotNone(self.tagger.tag_rename_mappings)
        self.assertIsNotNone(self.tagger._tags_rating)
        self.assertIsNotNone(self.tagger._tags_general)
        self.assertIsNone(self.tagger._all_tags_weights)
        self.assertIsNone(self.tagger._image_tags)
        self.assertIsNone(self.tagger._image_rating)
        self.assertIsNone(self.tagger._random_false_tags)

    @patch('wd14_tagging.wd14_tagging.WD14Tagger._setup_model')
    @patch('wd14_tagging.wd14_tagging.EventHandler.notify_observers')
    def test_GIVEN_existing_variables_WHEN_setting_up_tagger_THEN_variables_persist(self, mock_notify_observers, mock_setup_model):
        
        # Mocking already existing variables
        self.tagger.whitelisted_tags = {}
        self.tagger.tag_rename_mappings = {}
        self.tagger._tags_rating = ["tag1"]
        self.tagger._tags_general = ["tag2"]
        self.tagger._all_tags_weights = ["weight1"]
        self.tagger._image_tags = ["img_tag"]
        self.tagger._image_rating = 5
        self.tagger._random_false_tags = ["false_tag1"]

        self.tagger._setup_tagger()

        # Check that the event handler notify method was called
        mock_notify_observers.assert_called_with(0.0, "Initializing WD14 Tagger")

        # Assert that the variables were cleared appropriately
        self.assertIsNone(self.tagger._all_tags_weights)
        self.assertIsNone(self.tagger._image_tags)
        self.assertIsNone(self.tagger._image_rating)
        self.assertIsNone(self.tagger._random_false_tags)

    @patch('wd14_tagging.wd14_tagging.EventHandler.notify_observers')
    @patch('cv2.resize')
    @patch('numpy.pad')
    def test_GIVEN_image_WHEN_preprocessing_THEN_output_processed(self, mock_np_pad, mock_cv2_resize, mock_notify_observers):
        
        # Create a sample image
        image = Image.new('RGB', (50, 100), 'white')
        
        # Mock the return values for pad and resize (for simplicity)
        resized_image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)  # this is a mock image of size 448x448x3
        mock_cv2_resize.return_value = resized_image

        processed_image = self.tagger._preprocess_image(image)

        # 1 Ensure event_handler.notify_observers is called with the correct parameters
        mock_notify_observers.assert_called_once_with(0.1, "Pre-processing generated image")
        
        # 2 Ensure image is converted to "RGB"
        self.assertEqual(processed_image.shape[-1], 3)
        
        # 3 Check type and shape of the returned NumPy array
        self.assertEqual(processed_image.dtype, np.float32)
        self.assertEqual(processed_image.shape, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    @patch('wd14_tagging.wd14_tagging.EventHandler.notify_observers')
    def test_GIVEN_image_data_WHEN_calculating_weights_THEN_weights_determined(self, mock_notify_observers):

        # Given
        test_image_array = np.random.rand(1, 448, 448, 3) # a mock input image array

        # Mock the _model call to return a dummy TensorFlow tensor with weights
        mock_weights = tf.constant([[0.12345678, 0.87654321, 0.55555555]])
        self.tagger._model = MagicMock(return_value=mock_weights)

        # When
        self.tagger._calculate_tags_weights(test_image_array)

        # Then
        # 1. Ensure the model was called with the correct input
        self.tagger._model.assert_called_once_with(test_image_array, training=False)

        # 2. Ensure the weights are rounded correctly
        expected_rounded_weights = [0.12, 0.88, 0.56]
        self.assertListEqual(self.tagger._all_tags_weights, expected_rounded_weights)

        # 3. Ensure the event handler was called
        mock_notify_observers.assert_called_with(0.2, "Calculating tag weights")

    def test_GIVEN_tags_and_weights_WHEN_zipping_THEN_combined_correctly(self):
        self.tagger._tags_general = ["tag1", "tag2", "tag3"]
        self.tagger._all_tags_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        self.tagger._zip_general_tags_names_and_weights()

        expected_result = [("tag1", 0.5), ("tag2", 0.6), ("tag3", 0.7)]
        self.assertEqual(self.tagger._all_general_tags_names_and_weights, expected_result)

    @patch('wd14_tagging.wd14_tagging.EventHandler.notify_observers')
    def test_GIVEN_tags_weights_WHEN_determining_rating_THEN_rating_assigned(self, mock_notify):
        self.tagger._tags_rating = ["rating1", "rating2", "rating3", "rating4"]
        self.tagger._all_tags_weights = [0.1, 0.2, 0.5, 0.4, 0.9, 0.3, 0.2]

        self.tagger._determine_image_rating()

        mock_notify.assert_called_with(0.3, "Determining image rating")

        self.assertEqual(self.tagger._image_rating, "rating3")  # rating3 has the maximum weight of 0.5 among the first 4

    @patch('wd14_tagging.wd14_tagging.EventHandler.notify_observers')
    def test_GIVEN_tags_WHEN_processing_and_filtering_THEN_tags_filtered(self, mock_notify):
        self.tagger._process_and_filter_image_tags()
        mock_notify.assert_called_with(0.5, "Processing and filtering tags")
        expected_tags = [('good tag', 0.5)]
        self.assertEqual(self.tagger._image_tags, expected_tags)

    def test_GIVEN_string_with_underscores_WHEN_removing_underscores_THEN_underscores_removed(self):
        test_input = "test_tag"
        expected_output = "test tag"
        self.assertEqual(self.tagger._remove_underscores(test_input), expected_output)

    def test_GIVEN_string_emoji_WHEN_removing_underscores_THEN_underscores_retained(self):
        test_input = "U_U"
        expected_output = "U_U"  # Emojis should not change
        self.assertEqual(self.tagger._remove_underscores(test_input), expected_output)

    def test_GIVEN_list_of_strings_WHEN_removing_underscores_THEN_underscores_removed_from_all(self):
        test_input = ["test_tag1", "test_tag2", "^_^"]
        expected_output = ["test tag1", "test tag2", "^_^"]
        self.assertEqual(self.tagger._remove_underscores(test_input), expected_output)

    def test_GIVEN_list_of_tuples_WHEN_removing_underscores_THEN_underscores_removed_from_strings(self):
        test_input = [("test_tag1", 0.5), ("test_tag2", 0.4), ("^_^", 0.3)]
        expected_output = [("test tag1", 0.5), ("test tag2", 0.4), ("^_^", 0.3)]
        self.assertEqual(self.tagger._remove_underscores(test_input), expected_output)

    def test_GIVEN_tags_no_mapping_WHEN_applying_mappings_THEN_tags_unchanged(self):
        # Test with tags that don't require renaming
        tags_and_weights = [('dog', 0.9), ('cat', 0.5)]
        expected = [('dog', 0.9), ('cat', 0.5)]

        result = self.tagger.apply_tags_rename_mappings(tags_and_weights)
        self.assertEqual(result, expected)

    def test_GIVEN_tags_with_mapping_WHEN_applying_mappings_THEN_tags_renamed(self):
        # Test with tags that require renaming
        self.tagger.tag_rename_mappings = {'dog': 'canine', 'cat': 'feline'}
        
        tags_and_weights = [('dog', 0.9), ('cat', 0.5)]
        expected = [('canine', 0.9), ('feline', 0.5)]

        result = self.tagger.apply_tags_rename_mappings(tags_and_weights)
        self.assertEqual(result, expected)

    def test_GIVEN_mixed_tags_WHEN_applying_mappings_THEN_some_tags_renamed(self):
        # Test with a mix of tags: some require renaming, some don't
        self.tagger.tag_rename_mappings = {'dog': 'canine'}
        
        tags_and_weights = [('dog', 0.9), ('cat', 0.5), ('bird', 0.3)]
        expected = [('canine', 0.9), ('cat', 0.5), ('bird', 0.3)]

        result = self.tagger.apply_tags_rename_mappings(tags_and_weights)
        self.assertEqual(result, expected)

    def test_GIVEN_empty_tags_WHEN_applying_mappings_THEN_no_change(self):
        # Test with an empty list of tags
        tags_and_weights = []
        expected = []

        result = self.tagger.apply_tags_rename_mappings(tags_and_weights)
        self.assertEqual(result, expected)

    @patch('wd14_tagging.wd14_tagging.logger.info')
    def test_GIVEN_image_WHEN_generating_tags_THEN_tags_generated(self, mock_logger):
        call_sequence = []

        with patch.object(self.tagger, '_setup_model', side_effect=lambda: call_sequence.append('_setup_model')) as mock_load_model, \
            patch.object(self.tagger, '_preprocess_image', side_effect=lambda *args: call_sequence.append('_preprocess_image')) as mock_preprocess, \
            patch.object(self.tagger, '_calculate_tags_weights', side_effect=lambda *args: call_sequence.append('_calculate_tags_weights')) as mock_calculate_weights, \
            patch.object(self.tagger, '_zip_general_tags_names_and_weights', side_effect=lambda: call_sequence.append('_zip_general_tags_names_and_weights')) as mock_zip_tags, \
            patch.object(self.tagger, '_determine_image_rating', side_effect=lambda: call_sequence.append('_determine_image_rating')) as mock_determine_rating, \
            patch.object(self.tagger, '_process_and_filter_image_tags', side_effect=lambda: call_sequence.append('_process_and_filter_image_tags')) as mock_process_tags, \
            patch.object(self.tagger, '_set_image_tags', side_effect=lambda *args: call_sequence.append('_set_image_tags')) as mock_set_tags, \
            patch.object(self.tagger, 'apply_tags_rename_mappings', side_effect=lambda *args: call_sequence.append('apply_tags_rename_mappings')) as mock_apply_mappings:

            self.tagger.generate_tags(self.test_image)

            # Check the order of method calls
            expected_call_order = [
                '_setup_model',
                '_preprocess_image',
                '_calculate_tags_weights',
                '_zip_general_tags_names_and_weights',
                '_determine_image_rating',
                '_process_and_filter_image_tags',
                'apply_tags_rename_mappings',
                '_set_image_tags'
            ]
            self.assertEqual(call_sequence, expected_call_order)

if __name__ == '__main__':
    unittest.main()
