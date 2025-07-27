import pandas as pd
import os, json

# Train set
def get_classification_task_train_df() -> pd.DataFrame:
    df = pd.read_json('Dataset/train/cls.json', orient='index')
    df = df.reset_index()
    df.columns = ['Path', 'Label']
    df['Path'] = df['Path'].apply(lambda x: os.path.join('./Dataset/train/imgs/', x))
    return df

def get_i2i_task_train_df() -> pd.DataFrame:
    df = pd.read_json('Dataset/train/i2i.json', orient='index')
    df = df.reset_index()
    df.columns = ['Path1', 'Path2']
    df['Path1'] = df['Path1'].apply(lambda x: f'./Dataset/train/imgs/{x}')
    df['Path2'] = df['Path2'].apply(lambda x: f'./Dataset/train/imgs/{x}')
    return df

def get_t2i_task_train_df() -> pd.DataFrame:
    df = pd.read_json('Dataset/train/t2i.json', orient='index')
    df = df.reset_index()
    df.columns = ['Caption', 'Path']
    df['Path'] = df['Path'].apply(lambda x: f'./Dataset/train/imgs/{x}')
    return df

###################################################################################################

# Test set (for making submission)
def get_classification_task_test_df() -> pd.DataFrame:
    df = pd.read_csv('Dataset/test/cls.csv', header=None, names=['Path'])
    df['Path'] = df['Path'].apply(lambda x: f'./Dataset/test/imgs/{x}')
    return df

def get_i2i_task_test_df() -> pd.DataFrame:
    df = pd.read_csv('./Dataset/test/i2i.csv', header=None, names=['Path'])
    df['Path'] = df['Path'].apply(lambda x: f'./Dataset/test/imgs/{x}')
    return df

def get_t2i_task_test_df() -> pd.DataFrame:
    df = pd.read_csv('./Dataset/test/t2i.csv', header=None, names=['Caption'])
    return df

###################################################################################################

# Public set
def get_public_df() -> pd.DataFrame:
    df = pd.read_json('./Dataset/public/data.json')
    df.loc[df['Type'] == '', 'Type'] = 'abnormal'
    df["Path"] = df["Path"].str.replace("_image", "_Image", regex=False)
    df['Path'] = df['Path'].apply(lambda x: f'./Dataset/public/images/{x}')
    return df