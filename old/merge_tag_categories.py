import re

# Read content from file
with open("categories.txt", "r") as infile:
    file_content = infile.read()

# Regular expression to match list assignments
pattern = r"(\w+)\s*=\s*\[((?:[^\[]|\[(?:[^\[]|\[(?:[^\[]|\[(?:[^\[]|\[.*?\])*?\])*?\])*?\])*?)\]"

# Dictionary to hold merged lists
merged_lists = {}

# Find all matches using the regular expression
for match in re.findall(pattern, file_content, re.DOTALL):
    list_name = match[0].strip()
    items_str = '[' + match[1].strip() + ']'
    items = eval(items_str)

    # Merge lists
    if list_name in merged_lists:
        merged_lists[list_name].extend(items)
    else:
        merged_lists[list_name] = items

# Write merged lists to a text file
with open("output.txt", "w") as outfile:
    for name, items in merged_lists.items():
        formatted_str = f"{name} = {items}\n\n"
        outfile.write(formatted_str)
