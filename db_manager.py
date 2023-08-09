import bcrypt
from peewee import Model, CharField, IntegerField, FloatField, SqliteDatabase, IntegrityError
from custom_logging import logger

DB = SqliteDatabase('users.db')

DEFAULT_STATE = {'username': None, 'experience': None, 'accuracy': None, 'questions': None}

class User(Model):
    username = CharField(unique=True)
    password = CharField()
    experience = IntegerField(default=0)
    accuracy = FloatField(default=0.0)
    questions = IntegerField(default=0)

    class Meta:
        database = DB

class DatabaseManager():
    def __init__(self):
        DB.connect()
        DB.create_tables([User])
        
    def register(self, username, password, state):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            new_user = User.create(username=username, password=hashed_password.decode('utf-8'))
            logger.info(f'Account Registration: User "{new_user.username}" successfully created a new account.')
            state['username'] = new_user.username
            state['experience'] = new_user.experience
            state['accuracy'] = new_user.accuracy
            state['questions'] = new_user.questions

            return state
        except IntegrityError:
            logger.info(f'Account Registration: Username "{username}" is already taken.')

        return DEFAULT_STATE

    def login(self, username, password, state):
        try:
            user = User.get(User.username == username)
            if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                logger.info(f'User {user.username} has logged in.')
                state['username'] = user.username
                state['experience'] = user.experience
                state['accuracy'] = user.accuracy
                state['questions'] = user.questions

                return state
            else:
                logger.info(f'Account Login: Invalid password entered for username "{username}".')
        except User.DoesNotExist:
                logger.info(f'Account Login: Username "{username}" not found in the system.')

        return DEFAULT_STATE

    # def interact_with_system(self, some_input, state):
    #     if 'username' in state:
    #         user = User.get(User.username == state['username'])
    #         user.experience += 10
    #         user.save()
    #         state['experience'] = user.experience
    #         return f"Experience updated, new experience: {state['experience']}"
    #     else:
    #         return "Please login first"
