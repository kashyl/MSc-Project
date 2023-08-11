# Import necessary modules
import ast

path = 'tag_categories.py'

# Define lists of tags
list_names = [
    "hair_related", "eye_related", "clothing_accessories", "actions_poses",
    "emotions_expressions", "locations_backgrounds", "humans_people",
    "animals_animal_related", "nature", "other"
]

# Load content of tag_categories.py
with open(path, 'r') as f:
    file_content = f.read()

# Use ast.literal_eval to safely evaluate strings containing python lists
namespace = {}
exec(file_content, {}, namespace)

# Get sensitive words
sensitive_words = namespace.get("sensitive_words", [])
sensitive_words_to_keep = namespace.get("sensitive_words_to_keep", [])

# Function to check if a tag contains any sensitive words
def contains_sensitive(tag):
    for word in sensitive_words:
        # Check if word is not in sensitive_words_to_keep before flagging
        if word in tag and word not in sensitive_words_to_keep:
            return True
    return False

removed_tags = []  # Initialize an empty list to store removed tags

# Update the lists
for list_name in list_names:
    tags = namespace.get(list_name, [])
    clean_tags = [tag for tag in tags if not contains_sensitive(tag)]
    # Find the difference between the original tags and cleaned tags
    removed_from_current_list = [tag for tag in tags if tag not in clean_tags]
    removed_tags.extend(removed_from_current_list)
    namespace[list_name] = clean_tags

# Write the cleaned lists back to the file
with open(path, 'w') as f:
    for list_name in list_names:
        f.write(f"{list_name} = {namespace[list_name]}\n\n")
    
    # Add back sensitive_words and sensitive_words_to_keep to the file
    f.write(f"sensitive_words = {sensitive_words}\n\n")
    f.write(f"sensitive_words_to_keep = {sensitive_words_to_keep}\n")

# Display the removed tags
print("Removed tags:")
for tag in removed_tags:
    print(tag)
