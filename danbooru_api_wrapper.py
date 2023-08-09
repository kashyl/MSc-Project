import os, requests, re, json
from dotenv import load_dotenv
from wd14_tagging.wd14_tagging import TAGS_RENAME_MAP_PATH

load_dotenv()

API_KEY = os.getenv('DANBOORU_API_KEY')
API_USER = os.getenv('DANBOORU_USER_NAME')
API_URL = "https://danbooru.donmai.us/"

class DanbooruApi():
    def __init__(self):
        # Depending on the API's requirement, you might pass the API key as a parameter
        self.params = {
            "login": API_USER,
            "api_key": API_KEY
        }
        # Load the renaming mapping from the JSON file
        with open(TAGS_RENAME_MAP_PATH, 'r') as file:
            self.tags_renaming = json.load(file)
            self.reverse_tags_renaming = {v: k for k, v in self.tags_renaming.items()}  # Reverse the dictionary for lookup

    def convert_tag_for_api(self, tag_name: str) -> str:
        # 1. Replace whitespace with underscores
        tag_name = tag_name.replace(" ", "_")

        # 2. Check if the tag exists in the reverse mapping
        return self.reverse_tags_renaming.get(tag_name, tag_name)  # Use the original name if not found in the reversed mapping

    @staticmethod
    def format_wiki_text(text):   
        # Convert headers
        text = re.sub(r'h4\.', '', text)  # Simply remove header indicators for now
       
        # Format links in {{...}} format with bold
        text = re.sub(r'\{\{([^\}]+)\}\}', r'<b>\1</b>', text)
                
        # Format links in [[...]] format with bold
        text = re.sub(r'\[\[([^\]]+)\]\]', r'<b>\1</b>', text)

        # Convert newline characters to line breaks for HTML
        text = text.replace('\r\n', '<br>')

        return text
    
    def strip_sections(self, tag_info: str) -> str:
        # This regex pattern matches "See Also" or "Related tags" followed by any characters 
        # until the end of the string, effectively removing the entire sections.
        stripped_info = re.sub(r'(?i)(See Also|Related tags).*$', '', tag_info, flags=re.DOTALL).strip()
        return stripped_info
    
    def get_tag_wiki(self, tag_name):
        """Returns the tag wiki for a given tag name."""
        tag_name = self.convert_tag_for_api(tag_name)   # restore underscores, revert renaming
        response = requests.get(
            f"{API_URL}wiki_pages/{tag_name}.json",
            params=self.params
        )

        response_data = response.json()
        
        if response.status_code != 200 or not response_data:
            return None
        
        wiki_title = response_data["title"]
        wiki_body = self.format_wiki_text(response_data["body"])
        wiki_body = self.strip_sections(wiki_body)
        
        return wiki_title, wiki_body
