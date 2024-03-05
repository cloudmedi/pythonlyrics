import pandas as pd
import os
import glob

directory = 'ezan_cities/'

files = glob.glob(os.path.join(directory, "*"))

print(files[0])
for i in files:
    modified_string = i.replace("~$", "")
    csv_data = pd.read_excel(modified_string)
    # 31 Aralık 2024 Salı'nın verilerini oku

    specific_row = csv_data.loc[0][0].split()
    new_name = specific_row[0] + ".xlsx"

    print(print(new_name))

    df = csv_data.iloc[2:]
    df.to_excel(new_name, index=False)


