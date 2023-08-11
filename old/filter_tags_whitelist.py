import json
import ast

# File paths
python_path = 'tag_categories.py'
json_path = 'tags_whitelist.json'

# Load content of tag_categories.py
with open(python_path, 'r') as f:
    python_file_content = f.read()

# Use ast.literal_eval to safely evaluate strings containing python lists
namespace = {}
exec(python_file_content, {}, namespace)

# Get sensitive words
sensitive_words = namespace.get("sensitive_words", [])
sensitive_words_to_keep = namespace.get("sensitive_words_to_keep", [])

# Function to check if a tag contains any sensitive words
def contains_sensitive(tag):
    for word in sensitive_words:
        if word in tag and word not in sensitive_words_to_keep:
            return True
    return False

# Load the tags from the JSON file
with open(json_path, 'r') as f:
    json_tags = json.load(f)

# Filter the tags
cleaned_json_tags = [tag for tag in json_tags if not contains_sensitive(tag)]

# Display the removed tags from JSON file
removed_json_tags = [tag for tag in json_tags if tag not in cleaned_json_tags]
print("Removed tags from JSON file:")
for tag in removed_json_tags:
    print(tag)

# Save the cleaned tags back to the JSON file
with open(json_path, 'w') as f:
    json.dump(cleaned_json_tags, f, indent=4)
