import os
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import DatasetDict, Dataset
import pandas as pd
import json
from transformers import pipeline
from sklearn.model_selection import train_test_split
from datasets import Dataset

class DataPreparation:
  def __init__(self):
    self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")
    self.model = SentenceTransformer("all-mpnet-base-v2")

  def preprocess_data(self) -> pd.DataFrame:
    with open('Dataset/train/t2i.json', 'r', encoding='utf-8') as file:
      t2i_data = json.load(file)
    t2i_df = pd.DataFrame(list(t2i_data.items()), columns=['DescriptionEN', 'Path'])
    t2i_df['Path'] = t2i_df['Path'].apply(lambda x: f"Dataset/train/imgs/{x}")

    with open('Dataset/public/data.json', 'r', encoding='utf-8') as file:
      public_data = json.load(file)
    public_df = pd.DataFrame(public_data)
    public_df = public_df[['Path', 'DescriptionEN']]
    public_df['Path'] = public_df['Path'].str.replace(r'_image(\d+)', r'_Image\1', regex=True)
    public_df['Path'] = public_df['Path'].apply(lambda x: f"Dataset/public/images/{x}")

    return pd.concat([t2i_df, public_df], ignore_index=True)

  def detect_and_translate(self, df: pd.DataFrame) -> pd.DataFrame:
    def translate_if_vietnamese(text):
      if any(char in text for char in "áàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"):
        return self.translator(text)[0]['translation_text']
      return text
    df['DescriptionEN'] = df['DescriptionEN'].apply(translate_if_vietnamese)
    return df

  def generate_embeddings(self, descriptions: list) -> np.ndarray:
    return self.model.encode(descriptions)

  def compute_negative_pairs(self, embeddings: np.ndarray) -> list:
    similarities = self.model.similarity(embeddings, embeddings)
    similarities_argsorted = np.argsort(similarities.numpy(), axis=1)
    negative_pair_index_list = []

    for i in range(len(similarities)):
      j = 0
      index = int(similarities_argsorted[i][j])
      while index in negative_pair_index_list:
        j += 1
        index = int(similarities_argsorted[i][j])
      negative_pair_index_list.append(index)

    return negative_pair_index_list

  def validate_dataframe(self, df: pd.DataFrame):
    def contains_vietnamese(text):
      return any(char in text for char in "áàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ")

    vietnamese_rows = df[df['DescriptionEN'].apply(contains_vietnamese)]
    if not vietnamese_rows.empty:
      print("Warning: The dataframe contains Vietnamese text in the following rows:")
      print(vietnamese_rows)
    else:
      print("No Vietnamese text detected in the dataframe.")

    inaccessible_paths = df[~df['Path'].apply(os.path.exists)]
    if not inaccessible_paths.empty:
      print("Warning: The following paths are inaccessible:")
      print(inaccessible_paths['Path'])
    else:
      print("All paths are accessible.")

  def train_test_split(self, df: pd.DataFrame, train_frac: float, valid_frac: float, test_frac: float):

    # First split train+valid and test
    temp_frac = train_frac + valid_frac
    df_temp, df_test = train_test_split(
      df, test_size=test_frac, random_state=42
    )

    # Split train and valid
    valid_ratio = valid_frac / temp_frac
    df_train, df_valid = train_test_split(
      df_temp, test_size=valid_ratio, random_state=42
    )

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(df_valid.reset_index(drop=True))
    test_ds = Dataset.from_pandas(df_test.reset_index(drop=True))
    
    return DatasetDict({
        'train': train_ds,
        'valid': valid_ds,
        'test': test_ds
    })



