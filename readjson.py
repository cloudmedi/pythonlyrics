import json

# Open the JSON file and load its contents into a Python dictionary
with open('Lyrics_SezenAksu.json') as json_file:
    data = json.load(json_file)

# Access the specific variable (e.g., 'variable_name') from the dictionary
variable_value = data['songs'][2]['lyrics']
variable = variable_value.split("Lyrics")
lyrics_part = ''.join(variable[1:])

embed_variable = lyrics_part.split("Embed")
lyrics_part = embed_variable[0].strip()
# print(variable_value[:])
print(lyrics_part)
