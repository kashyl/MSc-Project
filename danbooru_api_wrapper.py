import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('DANBOORU_API_KEY')

# TODO: funcs to get tag description from the tag wiki