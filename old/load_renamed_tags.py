def load_renamed_tags_dict(fpath):
    dictionary = {}  
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            # Split each line into key and value based on the " -> " delimiter
            key, value = line.strip().split(" -> ")
            
            dictionary[key] = value
    return dictionary

# Load mappings from the outputs.txt file
mapping_dict = load_renamed_tags_dict('renamed_tags.txt')
print(mapping_dict)