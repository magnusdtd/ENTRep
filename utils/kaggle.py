from utils.file import File
import os

class Kaggle(File):
  def __init__(self):
    """
    This function will create paths to run the notebook in Kaggle. 
    This function uses the train dataset instead of the public dataset.
    """
    super().__init__()

    copy_paths = {
      "/kaggle/working/ENTRep": "/kaggle/working",
      "/kaggle/input/entrep-train-dataset/train": "/kaggle/working/Dataset/train",
      "/kaggle/input/entrep-public-test/test": "/kaggle/working/Dataset/test",
      "/kaggle/input/entrep-public-dataset": "/kaggle/working/Dataset/public",
      "/kaggle/input/entrep-model-weights/pytorch/default/1/model_weights": "/kaggle/working/model_weights"
    }

    for src, dst in copy_paths.items():
      File.copy_files(src, dst)

    current_path = os.getcwd()
    print("Current path:", current_path)
