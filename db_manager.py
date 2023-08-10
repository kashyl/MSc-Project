import bcrypt, json, copy
from datetime import datetime
from peewee import (SqliteDatabase, Model, CharField, IntegerField, FloatField, TextField, 
                    ForeignKeyField, ManyToManyField, DateTimeField, IntegrityError, fn)
from enum import Enum
from custom_logging import logger


class UserState(Enum):
    NAME = 'user_name'
    EXP = 'total_experience_points'
    ACCURACY = 'accuracy_rating'
    ATTEMPTED_COUNT ='attempted_questions_count'
    ATTEMPTED_QUESTIONS = 'attempted_questions'

class GUIAlertType(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'

class DBResponse():
    def __init__(self, state, message=None, message_type: GUIAlertType=None):
        self.state = state
        self.message = message
        self.message_type = message_type

class QuestionKeys(Enum):
    ID = "id"
    IMAGE_FILE_PATH = "image_file_path"
    GENERATION_INFO = "generation_info"
    IMAGE_RATING = "image_rating"
    CORRECT_TAGS = "correct_tags"
    FALSE_TAGS = "false_tags"
    GENERATION_TIME = "generation_time"
    USER_ANSWERS = "user_submitted_answers"
    ATTEMPTED_TIME = "user_attempted_time"

def get_default_state():
    DEFAULT_STATE = {
        UserState.NAME: None, 
        UserState.EXP: None, 
        UserState.ACCURACY: None, 
        UserState.ATTEMPTED_COUNT: None,
        UserState.ATTEMPTED_QUESTIONS: []
    }
    return copy.deepcopy(DEFAULT_STATE)


DB = SqliteDatabase('app_data.db')
TABLE_USERS = 'users'
TABLE_QUESTIONS = 'questions'

class BaseModel(Model):
    class Meta:
        database = DB

class UserModel(BaseModel):
    username = CharField(unique=True)
    password = CharField()
    experience = IntegerField(default=0)

class QuestionModel(BaseModel):
    image_file_path = CharField()
    generation_info = TextField()
    image_rating = CharField()
    correct_tags = TextField()  # Store as JSON string. Ex: json.dumps([("tag1", 0.9), ("tag2", 0.8)])
    false_tags = TextField()    # Same as correct_tags
    generation_time = DateTimeField(default=fn.now())

class UserQuestionRelation(Model):
    user = ForeignKeyField(UserModel, backref='questions_attempted')
    question = ForeignKeyField(QuestionModel, backref='attempted_by')
    attempted_at = DateTimeField(default=datetime.now)  # automatically set the current date and time
    user_answers = TextField()  # User's answers serialized as JSON

    class Meta:
        database = DB
        indexes = (
            (('user', 'question'), True),  # Composite primary key
        )

# Utility function to convert tags to JSON
def tags_to_json(tags: list) -> str:
    return json.dumps(tags)

# Utility function to convert JSON to tags
def json_to_tags(json_str: str) -> list:
    return json.loads(json_str)

class DatabaseManager():
    def __init__(self):
        DB.connect()
        DB.create_tables([UserModel, QuestionModel, UserQuestionRelation])
        
    def get_user_state(self, state, user_model_or_name) -> dict:
        if isinstance(user_model_or_name, UserModel):
            user = user_model_or_name
        else:
            try:
                user = UserModel.get(UserModel.username == user_model_or_name)
            except UserModel.DoesNotExist:
                logger.error(f"Error when get_user_state: username {user_model_or_name} does not exist.")
                return get_default_state()  # return default state

        state[UserState.NAME] = user.username
        state[UserState.EXP] = user.experience
        accuracy_val = self.calculate_user_accuracy(user)
        state[UserState.ACCURACY] = self.format_accuracy_val(accuracy_val)

        state[UserState.ATTEMPTED_QUESTIONS] = self.fetch_attempted_questions(user.username)
        state[UserState.ATTEMPTED_COUNT] = len(state[UserState.ATTEMPTED_QUESTIONS])

        return state
    
    def fetch_leaderboard_data_for_user(self, user):
        # Calculate accuracy
        accuracy_val = self.calculate_user_accuracy(user)
        formatted_accuracy = self.format_accuracy_val(accuracy_val)

        # Calculate attempted questions count
        attempted_questions = self.fetch_attempted_questions(user.username)
        attempted_count = len(attempted_questions)

        return [
            user.username,
            user.experience,
            formatted_accuracy,
            attempted_count
        ]
    
    def fetch_all_users_leaderboard_data(self):
        """ Order by total experience points. """
        users = UserModel.select().order_by(UserModel.experience.desc())
        leaderboard_data = []

        for user in users:
            user_data = self.fetch_leaderboard_data_for_user(user)
            leaderboard_data.append(user_data)

        return leaderboard_data

    def register(self, username, password, state):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            new_user = UserModel.create(username=username, password=hashed_password.decode('utf-8'))
            logger.info(f'Account Registration: User "{new_user.username}" successfully created a new account.')

            state = self.get_user_state(state, new_user)

            return DBResponse(state, f"Account created. Logged in as {new_user.username}.", GUIAlertType.INFO)
        except IntegrityError:
            logger.info(f'Account Registration: Username "{username}" is already taken.')
            return DBResponse(get_default_state(), f'Username "{username}" is already taken.', GUIAlertType.WARNING)

    def login(self, username, password, state):
        try:
            user = UserModel.get(UserModel.username == username)
            if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                logger.info(f'User "{user.username}" has logged in.')

                state = self.get_user_state(state, user)

                return DBResponse(state, f"Logged in as {user.username}.", GUIAlertType.INFO)
            else:
                logger.info(f'Account Login: Invalid password entered for username "{username}".')
        except UserModel.DoesNotExist:
                logger.info(f'Account Login: Username "{username}" not found in the system.')

        return DBResponse(get_default_state(), f"Login unsuccessful. Please verify your credentials.", GUIAlertType.WARNING)

    def store_question(self, **data):
        question = QuestionModel.create(**data)
        logger.info(f'Question {question.id} data has been stored.')
        return question

    def fetch_question(self, question_id):
        # If needed, could add checks to validate if user has access to question data
        try:
            question = QuestionModel.get(QuestionModel.id == question_id)
            question_data = {
                QuestionKeys.ID: question.id,
                QuestionKeys.IMAGE_FILE_PATH: question.image_file_path,
                QuestionKeys.GENERATION_INFO: question.generation_info,
                QuestionKeys.IMAGE_RATING: question.image_rating,
                QuestionKeys.CORRECT_TAGS: json_to_tags(question.correct_tags),
                QuestionKeys.FALSE_TAGS: json_to_tags(question.false_tags),
                QuestionKeys.GENERATION_TIME: question.generation_time
            }   
            return question_data
        except QuestionModel.DoesNotExist:
            logger.info(f'Question with ID {question_id} not found.')
            return None

    def fetch_question_user_answers(self, question_id: int, username: str):
        try:
            user = UserModel.get(UserModel.username == username)
            user_question_relation = UserQuestionRelation.get(
                (UserQuestionRelation.user == user) &
                (UserQuestionRelation.question == question_id)
            )
            selected_tags = json.loads(user_question_relation.user_answers)  # assuming it's a list of tags
            return selected_tags
        except UserModel.DoesNotExist:
            logger.error(f'User "{username}" not found in the database.')
            return []
        except UserQuestionRelation.DoesNotExist:
            logger.info(f'User "{username}" did not attempt Question {question_id}.')
            return []

    def add_attempted_question(self, user_username, question_id, user_answers=None):
        user = UserModel.get(UserModel.username == user_username)
        question = QuestionModel.get(QuestionModel.id == question_id)
            
        # Default user_answers to an empty list if not provided
        if user_answers is None:
            user_answers = []
        
        # Convert user_answers list to JSON string for storage
        user_answers_json = json.dumps(user_answers)
            
        # Check if relation exists
        defaults = {
            'user': user,
            'question': question,
            'user_answers': user_answers_json
        }
        relation, created = UserQuestionRelation.get_or_create(defaults=defaults, user=user, question=question)

        if created:
            logger.info(f'Question {question_id} added to user "{user_username}" attempted questions at {relation.attempted_at}.')
        else:
            relation.attempted_at = datetime.datetime.now()  # Update the timestamp
            relation.user_answers = user_answers_json  # Update the user answers
            relation.save()  # Save the updated details to the database
            logger.info(f'Question {question_id} already existed in user "{user_username}" attempted questions. \
                        Updated attempted time to {relation.attempted_at}.')


    def fetch_attempted_questions(self, username):
        try:
            user = UserModel.get(UserModel.username == username)
            
            # Fetch questions attempted by this user
            attempted_relations = (UserQuestionRelation
                                .select(UserQuestionRelation, QuestionModel)
                                .join(QuestionModel, on=(QuestionModel.id == UserQuestionRelation.question))
                                .where(UserQuestionRelation.user == user))
                                
            # Convert to a list of dictionaries for easy consumption by other parts of the app
            result = []
            for relation in attempted_relations:
                question = relation.question
                question_data = {
                    QuestionKeys.ID: question.id,
                    QuestionKeys.IMAGE_FILE_PATH: question.image_file_path,
                    QuestionKeys.GENERATION_INFO: question.generation_info,
                    QuestionKeys.IMAGE_RATING: question.image_rating,
                    QuestionKeys.CORRECT_TAGS: json_to_tags(question.correct_tags),
                    QuestionKeys.FALSE_TAGS: json_to_tags(question.false_tags),
                    QuestionKeys.GENERATION_TIME: question.generation_time,
                    QuestionKeys.USER_ANSWERS: json_to_tags(relation.user_answers),
                    QuestionKeys.ATTEMPTED_TIME: relation.attempted_at
                }
                result.append(question_data)
            
            return result

        except UserModel.DoesNotExist:
            logger.error(f'Fetch Attempted Questions: User "{username}" not found.')
            return []

    def increment_user_experience(self, user_username, experience_increment):
        try:
            user = UserModel.get(UserModel.username == user_username)
            user.experience += experience_increment
            user.save()

            logger.info(f'User "{user_username}" gained {experience_increment} XP. Total XP: {user.experience}.')
            return True

        except UserModel.DoesNotExist:
            logger.error(f'Add XP error: User "{user_username}" does not exist.')
            return False

    @staticmethod
    def _calculate_accuracy(user_answers: list, correct_tags: list, false_tags: list) -> float:
        # Extracting only the tag names from the correct_tags and false_tags lists
        correct_tag_names = [tag[0] for tag in correct_tags]
        false_tag_names = [tag[0] for tag in false_tags]

        correct_selections = sum(1 for tag in user_answers if tag in correct_tag_names)
        false_selections = sum(1 for tag in user_answers if tag in false_tag_names)

        # Calculate net correct selections after accounting for incorrect ones
        net_correct = correct_selections - false_selections

        # If net correct selections is negative, clamp it to zero
        net_correct = max(net_correct, 0)

        # Calculate accuracy based on net correct selections and total number of correct tags
        if len(correct_tag_names) == 0:  # Avoid division by zero
            return 0.0

        return net_correct / len(correct_tag_names)

    def _calculate_accuracy_for_relation(self, relation: UserQuestionRelation) -> float:
        question = relation.question
        user_answers = json.loads(relation.user_answers)
        correct_tags = json.loads(question.correct_tags)
        false_tags = json.loads(question.false_tags)

        return self._calculate_accuracy(user_answers, correct_tags, false_tags)

    def calculate_single_question_accuracy_for_user(self, user: UserModel, question: QuestionModel) -> float:
        try:
            relation = UserQuestionRelation.get((UserQuestionRelation.user == user) & (UserQuestionRelation.question == question))
            return self._calculate_accuracy_for_relation(relation)
        except UserQuestionRelation.DoesNotExist:
            # The user has not attempted this question
            return 0.0

    def calculate_user_accuracy(self, user: UserModel) -> float:
        total_accuracy = 0.0
        attempted_questions = UserQuestionRelation.select().where(UserQuestionRelation.user == user)

        for relation in attempted_questions:
            accuracy_for_relation = self._calculate_accuracy_for_relation(relation)
            total_accuracy += accuracy_for_relation

        if len(attempted_questions) == 0:
            return 0.0
        return total_accuracy / len(attempted_questions)

    @staticmethod
    def format_accuracy_val(accuracy_value):
        return "{:.1f}".format(accuracy_value * 100)
