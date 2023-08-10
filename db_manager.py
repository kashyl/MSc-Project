import bcrypt
from peewee import Model, CharField, IntegerField, FloatField, SqliteDatabase, IntegrityError
from enum import Enum
import gradio
from custom_logging import logger

DB = SqliteDatabase('users.db')

class UserFields(Enum):
    NAME = 'username'
    EXP = 'experience'
    ACCURACY = 'accuracy'
    ANS_COUNT = 'questions'

DEFAULT_STATE = {UserFields.NAME: None, UserFields.EXP: None, UserFields.ACCURACY: None, UserFields.ANS_COUNT: None}

class UserModel(Model):
    username = CharField(unique=True)
    password = CharField()
    experience = IntegerField(default=0)
    accuracy = FloatField(default=0.0)
    questions = IntegerField(default=0)

    class Meta:
        database = DB

class GUIAlertType(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'

class DBResponse():
    def __init__(self, state, message=None, message_type: GUIAlertType=None):
        self.state = state
        self.message = message
        self.message_type = message_type

class DatabaseManager():
    def __init__(self):
        DB.connect()
        DB.create_tables([UserModel])
        
    def register(self, username, password, state):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            new_user = UserModel.create(username=username, password=hashed_password.decode('utf-8'))
            logger.info(f'Account Registration: User "{new_user.username}" successfully created a new account.')
            state[UserFields.NAME] = new_user.username
            state[UserFields.EXP] = new_user.experience
            state[UserFields.ACCURACY] = new_user.accuracy
            state[UserFields.ANS_COUNT] = new_user.questions

            return DBResponse(state, f"Account created. Logged in as {new_user.username}.", GUIAlertType.INFO)
        except IntegrityError:
            logger.info(f'Account Registration: Username "{username}" is already taken.')
            return DBResponse(DEFAULT_STATE, f'Username "{username}" is already taken.', GUIAlertType.WARNING)

    def login(self, username, password, state):
        try:
            user = UserModel.get(UserModel.username == username)
            if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                logger.info(f'User {user.username} has logged in.')
                state[UserFields.NAME] = user.username
                state[UserFields.EXP] = user.experience
                state[UserFields.ACCURACY] = user.accuracy
                state[UserFields.ANS_COUNT] = user.questions

                return DBResponse(state, f"Logged in as {user.username}.", GUIAlertType.INFO)
            else:
                logger.info(f'Account Login: Invalid password entered for username "{username}".')
        except UserModel.DoesNotExist:
                logger.info(f'Account Login: Username "{username}" not found in the system.')

        return DBResponse(DEFAULT_STATE, f"Login unsuccessful. Please verify your credentials.", GUIAlertType.WARNING)


    # def interact_with_system(self, some_input, state):
    #     if 'username' in state:
    #         user = User.get(User.username == state['username'])
    #         user.experience += 10
    #         user.save()
    #         state['experience'] = user.experience
    #         return f"Experience updated, new experience: {state['experience']}"
    #     else:
    #         return "Please login first"
