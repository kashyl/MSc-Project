import csv
from itertools import islice

def extract_strings_from_columnB(csv_filename):
    values = []
    
    with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in islice(reader, 5, None):
            # Check if the data in column B (index 1) is a string
            if len(row) > 1 and isinstance(row[1], str):
                if row[1]: values.append(row[1])
                
    return values

def save_to_textfile(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

# Test the functions
excel_file = "accepted_tags.csv"
text_file = "output.txt"

result = extract_strings_from_columnB(excel_file)
save_to_textfile(result, text_file)
