import pandas as pd

def clean():

  file_path = 'Dataset/data.json'

  df = pd.read_json(file_path)

  df.loc[df['Type'] == '', 'Type'] = 'abnormal'

  df.to_csv('Dataset/cleaned_data.csv', index=False)
