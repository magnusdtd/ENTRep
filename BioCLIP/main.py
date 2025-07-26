from BioCLIP.make_submission import make_submission_t2i_task
from BioCLIP.evaluator import ImageToTextEvaluator, TextToImageEvaluator
from BioCLIP.data_preparation import DataPreparation
import pandas as pd
import os

def main():
  data_preparation = DataPreparation()
  image_to_text_df = data_preparation.preprocess_data()
  image_to_text_df = data_preparation.detect_and_translate(image_to_text_df)
  data_preparation.validate_dataframe(image_to_text_df)
  image_to_text_df['Path'] = image_to_text_df['Path'].apply(lambda x: os.path.join("/kaggle/working/", x))
  image_to_text_df.head()
  queries = image_to_text_df['DescriptionEN'].to_list()

  text_to_image_evaluator = TextToImageEvaluator(
    df=image_to_text_df,
    queries=queries,
    model_name='hf-hub:magnusdtd/bio-clip-cls-ft',
    model_path='',
    path_column='Path',
    caption_column='DescriptionEN'
  )

  # Evaluate recall at k
  recall_at_k = text_to_image_evaluator.get_recall_at_k(k=10)
  print(f"Recall at k: {recall_at_k}")

  make_submission_t2i_task(
    model_name="hf-hub:magnusdtd/bio-clip-cls-ft",
    model_path="",
    test_file_path="Dataset/test/t2i.csv",
    image_folder_path="Dataset/test/imgs"
  )

if __name__ == "__main__":
  main()