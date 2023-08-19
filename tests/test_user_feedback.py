import unittest
from unittest import mock
from user_feedback import UserFeedback

class TestUserFeedback(unittest.TestCase):

    def setUp(self):
        self.feedback = UserFeedback()
        self.patcher_error = mock.patch('custom_logging.logger.error', return_value=None)
        self.patcher_info = mock.patch('custom_logging.logger.info', return_value=None)
        self.mock_error = self.patcher_error.start()
        self.mock_info = self.patcher_info.start()

    def test_GIVEN_no_input_WHEN_generating_ticket_number_THEN_format_is_correct(self):
        ticket_number = self.feedback.generate_ticket_number()
        self.assertRegex(ticket_number, r"^TKT-\d{14}-[A-Z0-9]{5}$")

    def test_GIVEN_feedback_data_WHEN_formatting_email_THEN_output_contains_all_data(self):
        # sample data
        user_input_text = "Sample Text"
        image_gen_info = "Info"
        user_name = "JohnDoe"
        image_gen_time = "12:00"
        image_rating = "4/5"
        image_tags = "sky,grass"
        false_tags = "moon"
        difficulty_level = "Easy"
        user_selected_tags = "sky"

        expected_strs = [
            user_input_text, image_gen_info, user_name, image_gen_time, image_rating,
            image_tags, false_tags, difficulty_level, user_selected_tags
        ]

        email_content = self.feedback.format_feedback_email(user_input_text, image_gen_info, user_name, 
                                                            image_gen_time, image_rating, image_tags, 
                                                            false_tags, difficulty_level, user_selected_tags)

        for string in expected_strs:
            self.assertIn(string, email_content)

    @mock.patch("smtplib.SMTP_SSL")
    def test_GIVEN_valid_email_content_WHEN_sending_email_THEN_no_exception(self, mock_smtp):
        self.feedback.send_email("Sample Email Content")
        mock_smtp.assert_called_with('smtp.gmail.com', 465)
        
    @mock.patch("user_feedback.UserFeedback.send_email")
    @mock.patch("user_feedback.UserFeedback.format_feedback_email")
    def test_GIVEN_feedback_data_WHEN_processing_feedback_THEN_methods_called_in_order(self, mock_format, mock_send):
        # sample data
        user_input_text = "Sample Text"
        image_gen_info = "Info"

        self.feedback.process_feedback(user_input_text, image_gen_info)

        mock_format.assert_called_once()
        mock_send.assert_called_once()

if __name__ == "__main__":
    unittest.main()
