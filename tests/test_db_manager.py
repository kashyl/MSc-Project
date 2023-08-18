from unittest.mock import patch, Mock, MagicMock, ANY
from parameterized import parameterized
import unittest, bcrypt
from datetime import datetime
import json
from db_manager import DatabaseManager, DB, UserModel, QuestionModel, UserQuestionRelation, UserState, get_default_state, json_to_tags, QuestionKeys, GUIAlertType
from custom_logging import logger
from peewee import IntegrityError

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager()
        self.patcher_error = patch('custom_logging.logger.error', return_value=None)
        self.patcher_info = patch('custom_logging.logger.info', return_value=None)
        self.mock_error = self.patcher_error.start()
        self.mock_info = self.patcher_info.start()

    def tearDown(self):
        patch.stopall()
        DB.close()

    @patch.object(DatabaseManager, 'fetch_attempted_questions', return_value=[])
    @patch.object(UserModel, 'get', side_effect=UserModel.DoesNotExist)
    def test_GIVEN_non_existent_username_WHEN_get_user_state_invoked_THEN_default_state_returned(self, *_mocks):
        state = {}
        result = self.db_manager.get_user_state(state, "nonexistentuser")
        self.assertEqual(result, get_default_state())


    @patch.object(DatabaseManager, 'calculate_user_accuracy', return_value=0.8)
    @patch.object(DatabaseManager, 'format_accuracy_val', return_value="80%")
    @patch.object(DatabaseManager, 'fetch_attempted_questions', return_value=["q1", "q2"])
    def test_GIVEN_user_model_WHEN_get_user_state_invoked_THEN_user_state_returned(self, *_mocks):
        user = UserModel(username="testuser1", experience=10)
        expected_output = {
            UserState.NAME: "testuser1",
            UserState.EXP: 10,
            UserState.ACCURACY: "80%",
            UserState.ATTEMPTED_QUESTIONS: ["q1", "q2"],
            UserState.ATTEMPTED_COUNT: 2
        }
        result = self.db_manager.get_user_state({}, user)
        self.assertEqual(result, expected_output)

    @patch.object(UserModel, 'get', return_value=UserModel(username="testuser1", experience=10))
    @patch.object(DatabaseManager, 'calculate_user_accuracy', return_value=0.8)
    @patch.object(DatabaseManager, 'format_accuracy_val', return_value="80%")
    @patch.object(DatabaseManager, 'fetch_attempted_questions', return_value=["q1", "q2"])
    def test_GIVEN_user_name_string_WHEN_get_user_state_invoked_THEN_user_state_returned(self, *_mocks):
        expected_output = {
            UserState.NAME: "testuser1",
            UserState.EXP: 10,
            UserState.ACCURACY: "80%",
            UserState.ATTEMPTED_QUESTIONS: ["q1", "q2"],
            UserState.ATTEMPTED_COUNT: 2
        }
        result = self.db_manager.get_user_state({}, "testuser1")
        self.assertEqual(result, expected_output)

    @patch('custom_logging.logger.error')
    @patch.object(UserModel, 'get', side_effect=UserModel.DoesNotExist)
    def test_GIVEN_non_existent_user_name_string_WHEN_get_user_state_invoked_THEN_user_state_returned(self, *_mocks):
        result = self.db_manager.get_user_state({}, "nonexistentuser")
        self.assertEqual(result, get_default_state())
        logger.error.assert_called_once_with("Error when get_user_state: username nonexistentuser does not exist.")

    @parameterized.expand([
        (1, "100.0"),
        (0, "0.0"),
        (0.5, "50.0"),
        (1/3, "33.3"),
        (2/3, "66.7"),
        (0.876, "87.6")
    ])
    def test_GIVEN_accuracy_value_WHEN_format_accuracy_val_THEN_formatted_accuracy_returned(self, input_val, expected_output):
        db_manager = DatabaseManager()
        result = db_manager.format_accuracy_val(input_val)
        self.assertEqual(result, expected_output)


    @parameterized.expand([
        # Test Case: All correct answers
        (
            ["scenery", "bird", "sky"], # user_answers
            [["sky", 0.9], ["scenery", 0.88], ["bird", 0.67]], # correct_tags
            [["star (sky)", 0.83], ["cloud", 0.81]], # false_tags
            1.0 # expected accuracy
        ),
        # Test Case: None are correct answers
        (
            ["cloud", "star (sky)"], # user_answers
            [["sky", 0.9], ["scenery", 0.88], ["bird", 0.67]], # correct_tags
            [["star (sky)", 0.83], ["cloud", 0.81]], # false_tags
            0.0 # expected accuracy
        ),
        # Test Case: No user answers
        (
            [], # user_answers
            [["sky", 0.9], ["scenery", 0.88], ["bird", 0.67]], # correct_tags
            [["star (sky)", 0.83], ["cloud", 0.81]], # false_tags
            0.0 # expected accuracy
        ),
        # Test Case: Random mix of answers
        (
            ["scenery", "cloud", "bird"], # user_answers
            [["sky", 0.9], ["scenery", 0.88], ["bird", 0.67]], # correct_tags
            [["star (sky)", 0.83], ["cloud", 0.81]], # false_tags
            1/3 # expected accuracy
        )
    ])
    @patch.object(UserQuestionRelation, 'select')
    def test_GIVEN_user_answers_and_tags_WHEN_calculate_user_accuracy_invoked_THEN_returns_expected_accuracy(
        self, user_answers, correct_tags, false_tags, expected_accuracy, select_mock
        ):
        # Mocking attempted questions
        mock_question_1 = MagicMock(
            correct_tags=json.dumps(correct_tags), 
            false_tags=json.dumps(false_tags)
        )

        mock_relation_1 = MagicMock(
            question=mock_question_1,
            user_answers=json.dumps(user_answers)
        )

        mock_select_return = MagicMock()
        mock_select_return.where.return_value = [mock_relation_1]
        
        select_mock.return_value = mock_select_return

        db_manager = DatabaseManager()
        user = MagicMock()  # Mocked user object
        result = db_manager.calculate_user_accuracy(user)

        self.assertAlmostEqual(result, expected_accuracy, delta=0.001)

    @parameterized.expand([
        # User exists, experience increment positive
        ("existing_user", 10, 100, 110, True),
        # User exists, experience increment negative
        ("existing_user", -10, 100, 90, True),
        # User does not exist, experience increment positive
        ("non_existent_user", 10, None, None, False),
        # User does not exist, experience increment negative
        ("non_existent_user", -10, None, None, False)
    ])
    @patch.object(UserModel, 'get')  # Replace with the actual path to UserModel
    def test_GIVEN_username_and_experience_increment_WHEN_increment_user_experience_invoked_THEN_updates_experience_or_returns_false(
        self, user_username, experience_increment, initial_experience, expected_experience, expected_return, mock_get
    ):
        # Set up the mocked user if the user exists
        if initial_experience is not None:
            mock_user = MagicMock()
            mock_user.experience = initial_experience
            mock_get.return_value = mock_user
        else:
            mock_get.side_effect = UserModel.DoesNotExist

        db_manager = DatabaseManager()  # Assuming the function is in a class named DatabaseManager
        result = db_manager.increment_user_experience(user_username, experience_increment)

        # Assertions
        self.assertEqual(result, expected_return)
        if expected_experience:
            self.assertEqual(mock_user.experience, expected_experience)

    @patch.object(UserModel, 'get')
    @patch.object(UserQuestionRelation, 'select')
    def test_GIVEN_existing_user_WHEN_fetch_attempted_questions_THEN_return_questions_attempted_by_user(self, mock_select, mock_get):
        # Assuming the user exists and has attempted two questions
        mock_user = Mock()
        mock_get.return_value = mock_user

        mock_question_relation1 = Mock()
        mock_question_relation2 = Mock()

        # Mock data for two attempted questions
        mock_question_relation1.question.correct_tags = json.dumps([["tag1", 0.9]])
        mock_question_relation1.question.false_tags = json.dumps([["false_tag1", 0.5]])
        mock_question_relation1.user_answers = json.dumps(["tag1", "tag3"])

        mock_question_relation2.question.correct_tags = json.dumps([["tag2", 0.85]])
        mock_question_relation2.question.false_tags = json.dumps([["false_tag2", 0.45]])
        mock_question_relation2.user_answers = json.dumps(["tag2", "tag4"])

        mock_select.return_value.join.return_value.where.return_value = [mock_question_relation1, mock_question_relation2]

        questions = self.db_manager.fetch_attempted_questions('testuser')

        self.assertEqual(len(questions), 2)
        self.assertIn(QuestionKeys.ID, questions[0])
        self.assertEqual(questions[0][QuestionKeys.CORRECT_TAGS], [["tag1", 0.9]])
        self.assertEqual(questions[1][QuestionKeys.CORRECT_TAGS], [["tag2", 0.85]])

    @patch.object(UserModel, 'get')
    @patch.object(UserQuestionRelation, 'select')
    def test_GIVEN_existing_user_with_no_attempts_WHEN_fetch_attempted_questions_THEN_return_empty_list(self, mock_select, mock_get):
        # User exists but has not attempted any questions
        mock_user = Mock()
        mock_get.return_value = mock_user
        mock_select.return_value.join.return_value.where.return_value = []

        questions = self.db_manager.fetch_attempted_questions('testuser')

        self.assertEqual(len(questions), 0)

    @patch.object(UserModel, 'get')
    def test_GIVEN_non_existent_user_WHEN_fetch_attempted_questions_THEN_log_error_and_return_empty_list(self, mock_get):
        # User doesn't exist in the database
        mock_get.side_effect = UserModel.DoesNotExist

        questions = self.db_manager.fetch_attempted_questions('testuser')

        self.assertEqual(len(questions), 0)
        self.mock_error.assert_called_once_with('Fetch Attempted Questions: User "testuser" not found.')

    @patch.object(UserModel, 'get')
    @patch.object(QuestionModel, 'get')
    @patch.object(UserQuestionRelation, 'get_or_create')
    def test_GIVEN_user_and_question_without_existing_relation_WHEN_add_attempted_question_THEN_relation_created_with_correct_details(self, mock_get_or_create, mock_question_get, mock_user_get):
        # Mock setup
        mock_user = Mock(username="test_user")
        mock_question = Mock(id=1)
        mock_user_get.return_value = mock_user
        mock_question_get.return_value = mock_question
        mock_get_or_create.return_value = (Mock(user=mock_user, question=mock_question, user_answers='[]'), True)  # signifies relation was created

        # Call function
        self.db_manager.add_attempted_question("test_user", 1)

        # Assertions
        mock_get_or_create.assert_called_once_with(defaults=ANY, user=mock_user, question=mock_question)
        # 1. Check if the mock_info was called with a message that starts with the expected string
        call_args = self.mock_info.call_args[0][0]
        expected_start_of_message = 'Question 1 added to user "test_user" attempted questions at'
        self.assertTrue(call_args.startswith(expected_start_of_message))

        # 2. Check the remainder of the message
        remainder_of_message = call_args[len(expected_start_of_message):]
        self.assertIn('Mock name', remainder_of_message)

    @patch.object(UserModel, 'get')
    @patch.object(QuestionModel, 'get')
    @patch.object(UserQuestionRelation, 'get_or_create')
    def test_GIVEN_user_and_question_with_existing_relation_WHEN_add_attempted_question_THEN_timestamp_and_answers_updated(self, mock_get_or_create, mock_question_get, mock_user_get):
        # Mock setup
        mock_user = Mock(username="test_user")
        mock_question = Mock(id=1)
        mock_relation = Mock(user=mock_user, question=mock_question, attempted_at=datetime.now())
        mock_user_get.return_value = mock_user
        mock_question_get.return_value = mock_question
        mock_get_or_create.return_value = (mock_relation, False)  # signifies relation already exists

        # Call function
        self.db_manager.add_attempted_question("test_user", 1, ["answer1"])

        # Assertions
        mock_get_or_create.assert_called_once_with(defaults=ANY, user=mock_user, question=mock_question)
        mock_relation.save.assert_called_once()  # check if save method was called on relation
        self.assertEqual(mock_relation.user_answers, '["answer1"]')  # check updated user_answers
        self.mock_info.assert_called_once_with(ANY)  # the exact message can be asserted if required

    @patch.object(UserModel, 'get')
    @patch.object(QuestionModel, 'get')
    @patch.object(UserQuestionRelation, 'get_or_create')
    def test_GIVEN_user_and_question_with_no_provided_answers_WHEN_add_attempted_question_THEN_empty_answers_saved(self, mock_get_or_create, mock_question_get, mock_user_get):
        # Mock setup
        mock_user = Mock(username="test_user")
        mock_question = Mock(id=1)
        mock_user_get.return_value = mock_user
        mock_question_get.return_value = mock_question
        mock_get_or_create.return_value = (Mock(user=mock_user, question=mock_question, user_answers='[]'), True)  # signifies relation was created

        # Call function
        self.db_manager.add_attempted_question("test_user", 1)

        # Assertions
        mock_get_or_create.assert_called_once_with(defaults=ANY, user=mock_user, question=mock_question)
        call_args = self.mock_info.call_args[0][0]
        self.assertIn('Question 1 added to user "test_user" attempted questions at', call_args)

    @patch.object(UserModel, 'get')
    @patch.object(UserQuestionRelation, 'get')
    def test_GIVEN_valid_username_and_question_id_WHEN_fetch_question_user_answers_THEN_returns_correct_answers(
            self, mock_user_question_relation_get, mock_user_get):
        # Setting up the mocks
        mock_user = Mock()
        mock_user_get.return_value = mock_user

        mock_relation = Mock()
        mock_relation.user_answers = json.dumps(["tag1", "tag2"])
        mock_user_question_relation_get.return_value = mock_relation

        answers = self.db_manager.fetch_question_user_answers(1, "test_user")

        # Assertions
        self.assertEqual(answers, ["tag1", "tag2"])

    @patch.object(UserModel, 'get')
    @patch.object(UserQuestionRelation, 'get')
    def test_GIVEN_valid_username_but_no_question_attempt_WHEN_fetch_question_user_answers_THEN_returns_empty_list(
            self, mock_user_question_relation_get, mock_user_get):
        # Setting up the mocks
        mock_user = Mock()
        mock_user_get.return_value = mock_user
        mock_user_question_relation_get.side_effect = UserQuestionRelation.DoesNotExist

        answers = self.db_manager.fetch_question_user_answers(1, "test_user")

        # Assertions
        self.assertEqual(answers, [])
        self.mock_info.assert_called_once_with(f'User "test_user" did not attempt Question 1.')

    @patch.object(UserModel, 'get')
    def test_GIVEN_invalid_username_WHEN_fetch_question_user_answers_THEN_returns_empty_list_and_logs_error(self, mock_user_get):
        # Setting up the mocks
        mock_user_get.side_effect = UserModel.DoesNotExist

        answers = self.db_manager.fetch_question_user_answers(1, "invalid_user")

        # Assertions
        self.assertEqual(answers, [])
        self.mock_error.assert_called_once_with(f'User "invalid_user" not found in the database.')

    @patch.object(UserModel, 'get')
    @patch.object(UserQuestionRelation, 'get')
    def test_GIVEN_invalid_json_in_user_answers_WHEN_fetch_question_user_answers_THEN_raises_JSONDecodeError(self, mock_user_question_relation_get, mock_user_get):
        mock_user = Mock()
        mock_user_get.return_value = mock_user

        mock_relation = Mock()
        mock_relation.user_answers = "invalid_json_string"
        mock_user_question_relation_get.return_value = mock_relation

        with self.assertRaises(json.decoder.JSONDecodeError):
            self.db_manager.fetch_question_user_answers(1, "test_user")

    @patch.object(QuestionModel, 'get')
    def test_GIVEN_existing_question_id_WHEN_fetch_question_THEN_return_correct_question_data(self, mock_question_get):
        # Mocking the returned QuestionModel
        mock_question = Mock()
        mock_question.id = 1
        mock_question.image_file_path = "path/to/image.jpg"
        mock_question.generation_info = "test_info"
        mock_question.image_rating = 5
        mock_question.correct_tags = json.dumps([["tag1", 0.9], ["tag2", 0.8]])
        mock_question.false_tags = json.dumps([["false_tag1", 0.5]])
        mock_question.generation_time = "2023-08-18"
        mock_question_get.return_value = mock_question

        expected_data = {
            QuestionKeys.ID: 1,
            QuestionKeys.IMAGE_FILE_PATH: "path/to/image.jpg",
            QuestionKeys.GENERATION_INFO: "test_info",
            QuestionKeys.IMAGE_RATING: 5,
            QuestionKeys.CORRECT_TAGS: [["tag1", 0.9], ["tag2", 0.8]],
            QuestionKeys.FALSE_TAGS: [["false_tag1", 0.5]],
            QuestionKeys.GENERATION_TIME: "2023-08-18"
        }

        result = self.db_manager.fetch_question(1)
        self.assertEqual(result, expected_data)

    @patch.object(QuestionModel, 'get')
    def test_GIVEN_non_existent_question_id_WHEN_fetch_question_THEN_return_none_and_log_info(self, mock_question_get):
        # Mocking a scenario where question does not exist
        mock_question_get.side_effect = QuestionModel.DoesNotExist

        result = self.db_manager.fetch_question(-1)  # assuming -1 is a non-existent id
        self.assertIsNone(result)
        self.mock_info.assert_called_once_with(f'Question with ID -1 not found.')

    @patch.object(DatabaseManager, 'calculate_user_accuracy')
    @patch.object(DatabaseManager, 'format_accuracy_val')
    @patch.object(DatabaseManager, 'fetch_attempted_questions')
    def test_GIVEN_user_WHEN_fetch_user_leaderboard_data_THEN_user_leaderboard_data_returned(self, mock_attempted_questions, mock_format_accuracy, mock_calculate_accuracy):
        # Setup Mocks
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.experience = 100
        mock_calculate_accuracy.return_value = 0.85
        mock_format_accuracy.return_value = "85%"
        mock_attempted_questions.return_value = list(range(10))  # 10 questions

        # Call Method
        result = self.db_manager.fetch_leaderboard_data_for_user(mock_user)

        # Validate Results
        self.assertEqual(result, ["test_user", 100, "85%", 10])

    @patch.object(DatabaseManager, 'fetch_leaderboard_data_for_user')
    @patch.object(UserModel, 'select')
    def test_WHEN_fetch_all_users_leaderboard_data_THEN_leaderboard_data_returned(self, mock_select, mock_fetch_leaderboard_data_for_user):
        # Setup Mocks
        mock_user1 = Mock()
        mock_user2 = Mock()
        mock_user3 = Mock()
        
        mock_select.return_value.order_by.return_value = [mock_user1, mock_user2, mock_user3]
        
        user1_data = ["user1", 500, "95%", 50]
        user2_data = ["user2", 400, "90%", 40]
        user3_data = ["user3", 300, "85%", 30]

        mock_fetch_leaderboard_data_for_user.side_effect = [user1_data, user2_data, user3_data]

        # Call Method
        result = self.db_manager.fetch_all_users_leaderboard_data()

        # Validate Results
        self.assertEqual(result, [user1_data, user2_data, user3_data])

    @patch('db_manager.bcrypt.checkpw', return_value=True)
    @patch('db_manager.UserModel.get')
    def test_GIVEN_existing_user_and_correct_password_WHEN_login_THEN_successful_login_response(self, mock_user_get, mock_checkpw):
        # Mock user object
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "test_user"
        mock_user.password = "hashed_password"

        mock_user_get.return_value = mock_user

        # Mocked get_user_state method
        with patch.object(self.db_manager, 'get_user_state', return_value="some_state") as mock_get_user_state:

            state = "initial_state"
            response = self.db_manager.login("test_user", "test_password", state)

            mock_user_get.assert_called_once_with(UserModel.username == "test_user")
            mock_checkpw.assert_called_once_with("test_password".encode('utf-8'), "hashed_password".encode('utf-8'))
            mock_get_user_state.assert_called_once_with(state, mock_user)
            
            # Assertions for DBResponse object
            self.assertEqual(response.state, "some_state")  # assuming that get_user_state will return "some_state" for this mock setup
            self.assertEqual(response.message, "Logged in as test_user.")
            self.assertEqual(response.message_type, GUIAlertType.INFO)
            
            # Check if the log statement was executed correctly
            self.mock_info.assert_called_once_with('User "test_user" has logged in.')

    @patch.object(UserModel, 'get')
    @patch('db_manager.bcrypt.checkpw', return_value=False)
    def test_GIVEN_existing_user_and_incorrect_password_WHEN_login_THEN_unsuccessful_login_response(self, mock_checkpw, mock_user_get):
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.password = "hashed_password"
        mock_user_get.return_value = mock_user

        response = self.db_manager.login("test_user", "wrong_password", {})

        # Assertions for DBResponse object
        self.assertEqual(response.state, get_default_state())
        self.assertEqual(response.message, "Login unsuccessful. Please verify your credentials.")
        self.assertEqual(response.message_type, GUIAlertType.WARNING)

        self.mock_info.assert_called_once_with('Account Login: Invalid password entered for username "test_user".')


    @patch.object(UserModel, 'get', side_effect=UserModel.DoesNotExist)
    def test_GIVEN_non_existent_user_WHEN_login_THEN_unsuccessful_login_response(self, mock_user_get):
        response = self.db_manager.login("non_existent_user", "test_password", {})

        # Assertions for DBResponse object
        self.assertEqual(response.state, get_default_state())
        self.assertEqual(response.message, "Login unsuccessful. Please verify your credentials.")
        self.assertEqual(response.message_type, GUIAlertType.WARNING)

        self.mock_info.assert_called_once_with('Account Login: Username "non_existent_user" not found in the system.')

    @patch('db_manager.UserModel.create')
    @patch('db_manager.bcrypt.hashpw', return_value="hashed_password".encode('utf-8'))
    def test_GIVEN_new_user_WHEN_register_THEN_successful_registration(self, mock_hashpw, mock_create):
        # Mock user object
        mock_user = MagicMock()
        mock_user.username = "new_user"
        mock_create.return_value = mock_user

        # Mocked get_user_state method. You might want to update this with more specific behavior if needed.
        with patch.object(self.db_manager, 'get_user_state', return_value="some_state") as mock_get_user_state:

            state = "initial_state"
            response = self.db_manager.register("new_user", "test_password", state)

            mock_hashpw.assert_called_once_with("test_password".encode('utf-8'), ANY)
            mock_create.assert_called_once_with(username="new_user", password="hashed_password")
            mock_get_user_state.assert_called_once_with(state, mock_user)
            
            # Assertions for DBResponse object
            self.assertEqual(response.state, "some_state")
            self.assertEqual(response.message, "Account created. Logged in as new_user.")
            self.assertEqual(response.message_type, GUIAlertType.INFO)
            
            # Check if the log statement was executed correctly
            self.mock_info.assert_called_once_with('Account Registration: User "new_user" successfully created a new account.')

    @patch('db_manager.UserModel.create')
    @patch('db_manager.bcrypt.hashpw', return_value="hashed_password".encode('utf-8'))
    def test_GIVEN_existing_user_WHEN_register_THEN_username_already_taken(self, mock_hashpw, mock_create):
        # Mock the IntegrityError
        mock_create.side_effect = IntegrityError()

        state = "initial_state"
        response = self.db_manager.register("existing_user", "test_password", state)

        mock_hashpw.assert_called_once_with("test_password".encode('utf-8'), ANY)
        mock_create.assert_called_once_with(username="existing_user", password="hashed_password")
            
        # Assertions for DBResponse object
        self.assertEqual(response.state, get_default_state())  # <-- Modified this line
        self.assertEqual(response.message, 'Username "existing_user" is already taken.')
        self.assertEqual(response.message_type, GUIAlertType.WARNING)
            
        # Check if the log statement was executed correctly
        self.mock_info.assert_called_once_with('Account Registration: Username "existing_user" is already taken.')

if __name__ == '__main__':
    unittest.main()
