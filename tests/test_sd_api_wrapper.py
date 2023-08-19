import unittest
from unittest.mock import patch, Mock
from sd_api_wrapper import *
from PIL import Image

class TestSDApiWrapper(unittest.TestCase):

    def test_extract_base64_with_data_prefix(self):
        input_str = "data:image/png;base64,ABC123"
        expected_output = "ABC123"
        self.assertEqual(extract_base64(input_str), expected_output)

    def test_extract_base64_without_data_prefix(self):
        input_str = "ABC123"
        self.assertEqual(extract_base64(input_str), input_str)

    @patch('sd_api_wrapper.base64.b64decode')
    @patch('sd_api_wrapper.Image.open')
    def test_decode_base64_to_image_success(self, mock_open, mock_b64decode):
        mock_b64decode.return_value = b'valid_image_bytes'
        mock_image = Image.new('RGB', (60, 30))
        mock_image.close = Mock()
        mock_open.return_value = mock_image
        image, image_data = decode_base64_to_image("ABC123")
        self.assertIsInstance(image, Image.Image)
        self.assertIsInstance(image_data, bytes)
        self.addCleanup(mock_image.close)

    @patch('sd_api_wrapper.logger.warning')
    def test_decode_base64_to_image_failure(self, mock_logger_warning):
        decode_base64_to_image("invalid_base64")
        mock_logger_warning.assert_called_once()

    @patch('sd_api_wrapper.requests.post')
    def test_get_image_sd_info(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"info": "test_info"}
        mock_post.return_value = mock_response
        self.assertEqual(get_image_sd_info(b'12345'), "test_info")

    def test_add_generation_metadata(self):
        test_image = Image.new('RGB', (60, 30))
        metadata = "test_metadata"
        result_image = add_generation_metadata(test_image, metadata)
        self.assertIsInstance(result_image, Image.Image)
        self.addCleanup(test_image.close)
        self.addCleanup(result_image.close)

    @patch('sd_api_wrapper.os.listdir')
    @patch('sd_api_wrapper.Image.open')
    def test_get_first_image(self, mock_open, mock_listdir):
        mock_listdir.return_value = ['test1.png', 'test2.png']
        get_first_image()
        mock_open.assert_called_once()

    def test_mock_generate_image(self):
        image = mock_generate_image()
        self.assertIsInstance(image, Image.Image)
        self.addCleanup(image.close)

    def test_add_prefix_to_prompt(self):
        prompt = "test_prompt"
        prefix = "test_prefix"
        expected_output = "test_prefix, \n\ntest_prompt"
        self.assertEqual(add_prefix_to_prompt(prompt, prefix), expected_output)

    def test_create_payload(self):
        model_payload = {
            "example_key": "example_value"
        }
        model_prefixes = PromptPrefix(positive="pos:", negative="neg:")
        prompt = "test prompt"
        prompt_n = "test negative prompt"
        result = create_payload(model_payload, model_prefixes, prompt, prompt_n)
        expected_payload = {
            "example_key": "example_value",
            "prompt": "pos:, \n\ntest prompt",
            "negative_prompt": "neg:, \n\ntest negative prompt"
        }
        self.assertEqual(result, expected_payload)

    def test_as_percentage(self):
        result = as_percentage(0.7523)
        self.assertEqual(result, "75.23%")

    @patch('sd_api_wrapper.requests.get')
    @patch('sd_api_wrapper.requests.post')
    def test_update_model_options(self, mock_post, mock_get):
        model_option_payload = {"some_key": "some_value"}
        mock_get.return_value.json.return_value = {"existing_key": "existing_value"}
        update_model_options(model_option_payload)
        mock_post.assert_called_with(
            url=f'{SD_URL}/sdapi/v1/options',
            json={"existing_key": "existing_value", "some_key": "some_value"}
        )

if __name__ == "__main__":
    unittest.main()

