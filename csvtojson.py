import pandas as pd
import os
import glob

directory = '/Users/alkinkabul/Downloads/ezan_vakit'

files = glob.glob(os.path.join(directory, "*"))

print(files[0].split("ezan_vakit/")[1])
for i in files:
    print("/Users/alkinkabul/Downloads/ezan_vakit/xlsx (1).xlsx")
    modified_string = i.replace("~$", "")
    csv_data = pd.read_excel(modified_string)
    # 31 Aralık 2024 Salı'nın verilerini oku
    print(csv_data.loc[0][0])
    specific_row = csv_data.loc[0][0].split()

    old_name = i
    new_name = specific_row[0]+".xlsx"
    print(new_name)

    try:
        os.rename(old_name, new_name)
    except FileExistsError:
        print("File already Exists")
        print("Removing existing file")
        # skip the below code if you don't' want to forcefully rename
        os.remove(new_name)
        # rename it
        os.rename(old_name, new_name)
        print('Done renaming a file')

# enclosing inside try-except
